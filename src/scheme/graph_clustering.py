import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import faiss

from model import NodeEncoder
from util import compute_accuracy


class GraphClusteringScheme:
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, num_clusters, use_density_rescaling, use_euclidean_clustering):
        self.num_clusters = num_clusters
        self.use_density_rescaling = use_density_rescaling
        self.use_euclidean_clustering = use_euclidean_clustering

        self.temperature = 0.2
        self.mask_rate = 0.3

        self.clus_verbose = True
        self.clus_niter = 20
        self.clus_nredo = 1
        self.clus_seed = 0
        self.clus_max_points_per_centroid = 1000
        self.clus_min_points_per_centroid = 10

        self.centroids = None
        self.node2cluster = None
        self.density = None
                
    @staticmethod
    def collate_fn(data_list):
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
        num_nodes = data.x.size(0)
        num_mask_nodes = min(int(self.mask_rate * num_nodes), 1)
        mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
        
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[mask_nodes] = True
        
        data = Data(
            x=data.x.clone(), edge_index=data.edge_index.clone(), edge_attr=data.edge_attr.clone(),
            dataset_graph_idx = data.dataset_graph_idx.clone(),
            dataset_node_idx = data.dataset_node_idx.clone(),
            )
        data.y = data.x[:, 0].clone()
        data.x[mask_nodes] = 0
        data.node_mask = node_mask
        
        return data

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        projector = torch.nn.Linear(emb_dim, 100)
        atom_classifier = torch.nn.Linear(100, 120)
        models = torch.nn.ModuleDict({
            "encoder": encoder, 
            "projector": projector,
            "atom_classifier": atom_classifier
            })
        return models

    def assign_cluster(self, loader, models, device):
        print("Collecting graph features for clustering...")
        models.eval()
        features = None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = models["encoder"](batch.x, batch.edge_index, batch.edge_attr)
                out = models["projector"](out)
                out = global_mean_pool(out, batch.batch)
                
                if features is None:
                    features = torch.zeros(len(loader.dataset), out.size(1)).to(device)
                
                features[batch.dataset_graph_idx] = out
            
            features = torch.nn.functional.normalize(features, p=2, dim=1)
                
        features = features.cpu().numpy()
        
        d = features.shape[1]
        clus = faiss.Clustering(d, self.num_clusters)
        clus.verbose = self.clus_verbose
        clus.niter = self.clus_niter
        clus.nredo = self.clus_nredo
        clus.seed = self.clus_seed
        clus.max_points_per_centroid = self.clus_max_points_per_centroid
        clus.min_points_per_centroid = self.clus_min_points_per_centroid
        clus.spherical = (not self.use_euclidean_clustering)

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = device
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(features, index)

        # for each sample, find cluster distance and assignments
        D, I = index.search(features, 1)
        g2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(self.num_clusters, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(self.num_clusters)]
        for im, i in enumerate(g2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(self.num_clusters)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        # clamp extreme values for stability
        density = density.clip(np.percentile(density, 10), np.percentile(density, 90))

        # scale the mean to temperature
        density = self.temperature * density / density.mean()

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(device)
        self.centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

        self.graph2cluster = torch.LongTensor(g2cluster).to(device)
        self.density = torch.Tensor(density).to(device)
        
        obj = clus.iteration_stats.at(clus.iteration_stats.size()-1).obj
        bincount = torch.bincount(self.graph2cluster)
        statistics = {"obj": obj, "bincount": bincount}
        return statistics
        
    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)

        out = models["encoder"](batch.x, batch.edge_index, batch.edge_attr)
        features_node = models["projector"](out)
        features_node = torch.nn.functional.normalize(features_node, p=2, dim=1)

        features_graph = global_mean_pool(features_node, batch.batch)
        features_graph = torch.nn.functional.normalize(features_graph, p=2, dim=1)
        
        logits_node = models["atom_classifier"](features_node[batch.node_mask])
        labels_node = batch.y[batch.node_mask]
        
        loss = loss_node = self.criterion(logits_node, labels_node)
        acc_node = compute_accuracy(logits_node, labels_node)

        if self.centroids is not None:
            logits_proto = torch.mm(features_graph, self.centroids.T)
            if self.use_density_rescaling:
                logits_proto /= self.density.unsqueeze(0)
                            
            labels_proto = self.graph2cluster[batch.dataset_graph_idx]
            loss_proto = self.criterion(logits_proto, labels_proto)
            acc_proto = compute_accuracy(logits_proto, labels_proto)
            loss += loss_proto

        optim.zero_grad()
        loss.backward()
        optim.step()

        statistics = {
            "loss": loss,
            "loss_node": loss_node.detach(), 
            "acc_node": acc_node,
            }
        if self.centroids is not None:
            statistics.update({
                "loss_proto": loss_proto.detach(),
                "acc_proto": acc_proto,
            })

        return statistics