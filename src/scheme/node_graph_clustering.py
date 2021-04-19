import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import faiss

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels

class NodeGraphClusteringModel(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, proj_dim, drop_rate, node_temperature, graph_temperature):
        self.encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        self.projector = torch.nn.Linear(emb_dim, proj_dim)
        self.node_soft_embedding = torch.nn.Linear(num_node_clusters, proj_dim)

        self.node_centroids = None
        self.node2cluster = None
        self.node_density = None
                
        self.graph_centroids = None
        self.graph2cluster = None
        self.graph_density = None

    def compute_features_node(self, x, edge_index, edge_attr):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        features_node = self.projector(out)
        
        return features_node

    def compute_features_graph(self, x, edge_index, edge_attr, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        features_node = self.projector(out)
        
        logits_node_proto = torch.mm(features_node, self.node_centroids.T)
        probs_node_proto = torch.softmax(logits_node_proto, dim=1)
        
        out = self.node_soft_embedding(probs_node_proto)
        features_graph = global_mean_pool(out, batch)
        
        return features_graph
        

    def compute_logits_and_labels(self, x, edge_index, edge_attr, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        features_node = self.projector(out)
        
        _ = get_contrastive_logits_and_labels(features_node)
        logits_node_contrastive, labels_node_contrastive = _
        
        logits_node_proto = torch.mm(features_node, self.node_centroids.T)
        probs_node_proto = torch.softmax(logits_node_proto, dim=1)
        labels_node_proto = self.node2cluster[dataset_node_idx]

        out = self.node_soft_embedding(probs_node_proto)
        features_graph = global_mean_pool(out, batch)
        logits_graph_proto = torch.mm(features_graph, self.graph_centroids)
        labels_graph_proto = self.graph2cluster[dataset_graph_idx]
        
        logits_and_labels = {
            "node_contrastive": [logits_node_contrastive, labels_node_contrastive],
            "node_proto": [logits_node_proto, labels_node_proto],
            "graph_proto": [logits_graph_proto, labels_graph_proto],            
        }
        
        return logits_and_labels


class NodeGraphClusteringScheme:
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, num_clusters, use_density_rescaling, use_euclidean_clustering):
        self.num_clusters = num_clusters
        self.use_density_rescaling = use_density_rescaling
        self.use_euclidean_clustering = use_euclidean_clustering

        self.proto_temperature = 0.2
        self.contrastive_temperature = 0.04
        self.mask_rate = 0.3

        self.graph_clus_verbose = True
        self.graph_clus_niter = 20
        self.graph_clus_nredo = 1
        self.graph_clus_seed = 0
        self.graph_clus_max_points_per_centroid = 1000
        self.graph_clus_min_points_per_centroid = 10

        self.node_clus_verbose = True
        self.node_clus_niter = 20
        self.node_clus_nredo = 1
        self.node_clus_seed = 0
        self.node_clus_max_points_per_centroid = 100
        self.node_clus_min_points_per_centroid = 10

        self.graph_centroids = None
        self.graph2cluster = None
        self.graph_cluster_density = None
        
        self.node_centroids = None
        self.node2cluster = None
        self.node_cluster_density = None

    @staticmethod
    def collate_fn(data_list):
        data_list = [elem for elem in data_list if elem is not None]
        data_list = list(zip(*data_list))
        data_list = [data for inner_data_list in data_list for data in inner_data_list]

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = Data()
        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.batch_num_nodes = []
        batch.batch_size = len(data_list)
        batch.num_views = 2

        cumsum_node = 0
        for i, data in enumerate(data_list):
            num_nodes = data.x.size(0)
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            batch.batch_num_nodes.append(torch.LongTensor([num_nodes]))

            for key in keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node

                batch[key].append(item)

            cumsum_node += num_nodes

        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_num_nodes = torch.LongTensor(batch.batch_num_nodes)
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        return batch.contiguous()

    def transform(self, data):
        x, edge_index, edge_attr = mask_nodes(
            data.x.clone(), data.edge_index.clone(), data.edge_attr.clone(), mask_rate=0.3
        )
        if x.size(0) == 0:
            return None

        data0 = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dataset_graph_idx=data.dataset_graph_idx,
            dataset_node_idx=data.dataset_node_idx,
        )

        x, edge_index, edge_attr = mask_nodes(
            data.x.clone(), data.edge_index.clone(), data.edge_attr.clone(), mask_rate=0.3
        )
        if x.size(0) == 0:
            return None

        data1 = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dataset_graph_idx=data.dataset_graph_idx,
            dataset_node_idx=data.dataset_node_idx,
        )

        return data0, data1

    def assign_cluster(self, loader, model, device):
        print("Collecting graph features for clustering...")
        model.eval()
        features = None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model["encoder"](batch.x, batch.edge_index, batch.edge_attr)
                out = model["projector"](out)
                out = global_mean_pool(out, batch.batch)

                if features is None:
                    features = torch.zeros(len(loader.dataset), out.size(1)).to(device)

                features[batch.dataset_graph_idx] = out

            features = torch.nn.functional.normalize(features, p=2, dim=1)

        features = features.cpu().numpy()
        
        node_clus_result, node_clus_statistics = run_clustering(node_features)
        model.node_centroids = node_clus_result["centroids"]
        model.node2cluster = node_clus_result["item2cluster"]
        model.node_density = node_clus_result["density"]
        
        graph_clus_result, graph_clus_statistics = run_clustering(graph_features)
        model.graph_centroids = graph_clus_result["centroids"]
        model.graph2cluster = graph_clus_result["item2cluster"]
        model.graph_density = graph_clus_result["density"]
        
        return statistics

    def train_step(self, batch, model, optim, device):
        model.train()
        batch = batch.to(device)
        
        forward_result = model.compute_logits_and_labels(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
        
        loss_cum = 0.0
        losses = dict()
        accs = dict()
        for key in forward_result:
            logits, labels = forward_result[key]
            loss = self.criterion(logits, labels)
            acc = compute_accuracy(logits, labels)
            
            loss_cum += loss
            
            statistics[f"{key}/loss"] = loss.detach()
            statistics[f"{key}/acc"] = acc
            
        optim.zero_grad()
        loss_sum.backward()
        optim.step()

        return statistics
