import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering

class NodeClusteringModel(torch.nn.Module):
    def __init__(self, use_density_rescaling):
        super(NodeClusteringModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.proj_dim = 100
        self.drop_rate = 0.0
        self.proto_temperature = 0.2
        self.contrastive_temperature = 0.04
        self.ema_rate = 0.995                
        self.criterion = torch.nn.CrossEntropyLoss()
        self.use_density_rescaling = use_density_rescaling

        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.ema_encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        for param, ema_param in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            ema_param.data.copy_(param.data)
            ema_param.requires_grad = False
        
        self.projector = torch.nn.Linear(self.emb_dim, self.proj_dim)
        self.ema_projector = torch.nn.Linear(self.emb_dim, self.proj_dim)
        for param, ema_param in zip(self.projector.parameters(), self.ema_projector.parameters()):
            ema_param.data.copy_(param.data)
            ema_param.requires_grad = False

        self.node_centroids = None
        self.node2cluster = None
        self.node_density = None
        
    def update_ema_encoder(self):
        for param, ema_param in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)
    
        for param, ema_param in zip(self.projector.parameters(), self.ema_projector.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)

    def compute_ema_features_node(self, x, edge_index, edge_attr, batch):
        out = self.ema_encoder(x, edge_index, edge_attr)
        out = self.ema_projector(out)
        features_node = torch.nn.functional.normalize(out, p=2, dim=1)
        
        return features_node
    
    def compute_ema_features_graph(self, x, edge_index, edge_attr, batch):
        out = self.ema_encoder(x, edge_index, edge_attr)
        out = self.ema_projector(out)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        features_graph = global_mean_pool(out, batch)
        features_graph = torch.nn.functional.normalize(features_graph, p=2, dim=1)
        
        return features_graph
    
    
    def compute_logits_and_labels(self, x, edge_index, edge_attr, batch, dataset_node_idx):
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        features_node = torch.nn.functional.normalize(out, p=2, dim=1)
        
        _ = get_contrastive_logits_and_labels(features_node)
        logits_node_contrastive, labels_node_contrastive = _
        logits_node_contrastive /= self.contrastive_temperature
        
        logits_and_labels = {
            "node_contrastive": [logits_node_contrastive, labels_node_contrastive],
        }
        
        if self.node_centroids is not None:
            logits_node_proto = torch.mm(features_node, self.node_centroids.T)
            logits_node_proto /= self.proto_temperature
            if self.use_density_rescaling:
                logits_node_proto /= self.node_density.unsqueeze(0)
                
            labels_node_proto = self.node2cluster[dataset_node_idx]
            logits_and_labels["node_proto"] = [logits_node_proto, labels_node_proto]
        
        return logits_and_labels
    

class NodeClusteringScheme:
    def __init__(self, num_clusters, use_euclidean_clustering):
        self.num_clusters = num_clusters
        self.clus_use_euclidean_clustering = use_euclidean_clustering

        self.proto_temperature = 0.2
        self.contrastive_temperature = 0.04
        
        self.clus_verbose = True
        self.clus_niter = 20
        self.clus_nredo = 1
        self.clus_seed = 0
        self.clus_max_points_per_centroid = 1000
        self.clus_min_points_per_centroid = 10

        self.centroids = None
        self.node2cluster = None
        self.density = None

    def assign_cluster(self, loader, model, device):
        print("Collecting graph features for clustering...")
        model.eval()
        node_features = None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model.compute_ema_features_node(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    )

                if node_features is None:
                    node_features = torch.zeros(loader.dataset.num_nodes, out.size(1)).to(device)

                node_features[batch.dataset_node_idx] = out

        node_features = node_features.cpu().numpy()
        
        clus_result, statistics = run_clustering(
            node_features,
            self.num_clusters,
            self.clus_verbose,
            self.clus_niter,
            self.clus_nredo,
            self.clus_seed,
            self.clus_max_points_per_centroid,
            self.clus_min_points_per_centroid,
            self.clus_use_euclidean_clustering,
            device
            )
        model.node_centroids = clus_result["centroids"].to(device)
        model.node2cluster = clus_result["item2cluster"].to(device)
        model.node_density = clus_result["density"].to(device)
        
        return statistics

    def train_step(self, batch, model, optim, device):
        model.train()
        batch = batch.to(device)
        
        logits_and_labels = model.compute_logits_and_labels(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.dataset_node_idx
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
        
        model.update_ema_encoder()

        return statistics