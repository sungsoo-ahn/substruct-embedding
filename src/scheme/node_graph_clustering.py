import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering

class NodeGraphClusteringModel(torch.nn.Module):
    def __init__(self, use_density_rescaling):
        super(NodeGraphClusteringModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.proj_dim = 100
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
        self.criterion = torch.nn.CrossEntropyLoss()
        self.use_density_rescaling = use_density_rescaling

        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.projector = torch.nn.Linear(self.emb_dim, self.proj_dim)
        
        self.node_centroids = None
        self.node2cluster = None
        self.node_density = None

        self.graph_centroids = None
        self.graph2cluster = None
        self.graph_density = None
        
    def compute_ema_features_node(self, x, edge_index, edge_attr, batch):
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        features_node = torch.nn.functional.normalize(out, p=2, dim=1)
        
        return features_node

    def compute_ema_features_graph(self, x, edge_index, edge_attr, batch):        
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        features_node = torch.nn.functional.normalize(out, p=2, dim=1)

        features_graph = global_mean_pool(features_node, batch)
        features_graph = torch.nn.functional.normalize(features_graph, p=2, dim=1)

        return features_graph

    def compute_ema_features_all(self, x, edge_index, edge_attr, batch):
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        features_node = torch.nn.functional.normalize(out, p=2, dim=1)

        features_graph = global_mean_pool(features_node, batch)
        features_graph = torch.nn.functional.normalize(features_graph, p=2, dim=1)

        return features_node, features_graph

    def compute_logits_and_labels(
        self, x, edge_index, edge_attr, batch, dataset_node_idx, dataset_graph_idx,
        ):
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        features_node = torch.nn.functional.normalize(out, p=2, dim=1)

        # Subsample nodes for fast computation
        node_mask = torch.bernoulli(torch.zeros(x.size(0) // 2), p=0.1).bool().to(x.device)
        node_mask = torch.cat([node_mask, node_mask], axis=0)
        sampled_feature_nodes = features_node[node_mask]
        
        _ = get_contrastive_logits_and_labels(sampled_feature_nodes)
        logits_node_contrastive, labels_node_contrastive = _
        logits_node_contrastive /= self.contrastive_temperature
        
        features_graph = global_mean_pool(features_node, batch)
        features_graph = torch.nn.functional.normalize(features_graph, p=2, dim=1)

        _ = get_contrastive_logits_and_labels(features_graph)
        logits_graph_contrastive, labels_graph_contrastive = _
        logits_graph_contrastive /= self.contrastive_temperature
        
        logits_and_labels = {
            "node_contrastive": [logits_node_contrastive, labels_node_contrastive],
            "graph_contrastive": [logits_graph_contrastive, labels_graph_contrastive],
        }

        if self.node_centroids is not None:
            batch_active = self.node_active[dataset_node_idx]
            node_mask = (batch_active > 0)
            sampled_feature_nodes = features_node[node_mask]
            logits_node_proto = torch.mm(sampled_feature_nodes, self.node_centroids.T)
            logits_node_proto /= self.proto_temperature
            if self.use_density_rescaling:
                logits_node_proto /= self.node_density.unsqueeze(0)
            
            labels_node_proto = self.node2cluster[batch_active[batch_active > 0] - 1]
            
            logits_and_labels["node_proto"] = [logits_node_proto, labels_node_proto]
        
        if self.graph_centroids is not None:
            logits_graph_proto = torch.mm(features_graph, self.graph_centroids.T)
            logits_graph_proto /= self.proto_temperature
            if self.use_density_rescaling:
                logits_graph_proto /= self.graph_density.unsqueeze(0)

            labels_graph_proto = self.graph2cluster[dataset_graph_idx]
            logits_and_labels["graph_proto"] = [logits_graph_proto, labels_graph_proto]
        
        return logits_and_labels
    

class NodeGraphClusteringScheme:
    def __init__(self, num_clusters, use_euclidean_clustering):
        self.num_clusters = num_clusters
        self.clus_use_euclidean_clustering = use_euclidean_clustering
        
        self.clus_verbose = True
        self.clus_niter = 20
        self.clus_nredo = 1
        self.clus_seed = 0
        self.clus_max_points_per_centroid = 1000
        self.clus_min_points_per_centroid = 10

    def assign_cluster(self, loader, model, device):
        print("Collecting graph features for clustering...")
        model.eval()
        node_features = None
        graph_features = None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                features_node, features_graph = model.compute_ema_features_all(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    )

                if node_features is None:
                    node_features = torch.zeros(
                        loader.dataset.num_nodes, features_node.size(1)
                        ).to(device)
                
                if graph_features is None:
                    graph_features = torch.zeros(
                        len(loader.dataset), features_graph.size(1)
                        ).to(device)

                node_features[batch.dataset_node_idx] = features_node
                graph_features[batch.dataset_graph_idx] = features_graph

        node_active = torch.zeros(loader.dataset.num_nodes)
        node_active = torch.bernoulli(node_active, p=0.1).long().to(device)
        node_active[node_active > 0] = (torch.arange(node_active.sum()).to(device) + 1)
        model.node_active = node_active
        
        node_features = node_features[node_active > 0]
        node_features = node_features.cpu().numpy()
        
        graph_features = graph_features.cpu().numpy()
        
        node_clus_result, node_statistics = run_clustering(
            node_features,
            self.num_clusters,
            self.clus_verbose,
            self.clus_niter,
            self.clus_nredo,
            self.clus_seed,
            self.clus_max_points_per_centroid,
            self.clus_min_points_per_centroid,
            self.clus_use_euclidean_clustering,
            0
            )
        model.node_centroids = node_clus_result["centroids"].to(device)
        model.node2cluster = node_clus_result["item2cluster"].to(device)
        model.node_density = node_clus_result["density"].to(device)

        graph_clus_result, graph_statistics = run_clustering(
            graph_features,
            self.num_clusters,
            self.clus_verbose,
            self.clus_niter,
            self.clus_nredo,
            self.clus_seed,
            self.clus_max_points_per_centroid,
            self.clus_min_points_per_centroid,
            self.clus_use_euclidean_clustering,
            0
            )
        model.graph_centroids = graph_clus_result["centroids"].to(device)
        model.graph2cluster = graph_clus_result["item2cluster"].to(device)
        model.graph_density = graph_clus_result["density"].to(device)
               
        statistics = dict()
        for key in node_statistics:
            statistics[f"node_{key}"] = node_statistics[key]

        for key in graph_statistics:
            statistics[f"graph_{key}"] = graph_statistics[key]
         
        return statistics

    def train_step(self, batch, model, optim, device):
        model.train()
        batch = batch.to(device)
        
        logits_and_labels = model.compute_logits_and_labels(
            batch.x, 
            batch.edge_index, 
            batch.edge_attr, 
            batch.batch, 
            batch.dataset_node_idx, 
            batch.dataset_graph_idx, 
            )
        
        loss_cum = 0.0
        statistics = dict()
        for key in logits_and_labels:
            logits, labels = logits_and_labels[key]
            loss = model.criterion(logits, labels)
            acc = compute_accuracy(logits, labels)                
            
            loss_cum += loss
            
            statistics[f"{key}/loss"] = loss.detach()
            statistics[f"{key}/acc"] = acc
            
        optim.zero_grad()
        loss_cum.backward()
        optim.step()
        
        #model.update_ema_encoder()

        return statistics