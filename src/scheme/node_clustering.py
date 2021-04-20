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
        self.proto_temperature = 0.01
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

    def compute_ema_features(self, x, edge_index, edge_attr, batch):
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
    
    def compute_logits_and_labels(
        self, x, edge_index, edge_attr, batch, dataset_node_idx
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
        
        logits_and_labels = {
            "node_contrastive": [logits_node_contrastive, labels_node_contrastive],
        }

        if self.node_centroids is not None:
            batch_active = self.node_active[dataset_node_idx]
            node_mask = (batch_active > 0)
            sampled_feature_nodes = features_node[node_mask]
            logits_node_proto = torch.mm(sampled_feature_nodes, self.node_centroids.T)
            logits_node_proto /= self.proto_temperature
            if self.use_density_rescaling:
                logits_node_proto /= self.node_density.unsqueeze(0)
            
            tmp = (batch_active[batch_active > 0] - 1)
            labels_node_proto = self.node2cluster[tmp]
            
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
                out = model.compute_ema_features(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    )

                if node_features is None:
                    node_features = torch.zeros(loader.dataset.num_nodes, out.size(1)).to(device)

                node_features[batch.dataset_node_idx] = out

        node_active = torch.zeros(loader.dataset.num_nodes)
        node_active = torch.bernoulli(node_active, p=0.1).long().to(device)
        node_active[node_active > 0] = (torch.arange(node_active.sum()).to(device) + 1)
        model.node_active = node_active
        
        node_features = node_features[node_active > 0]
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
            0
            )
        model.node_centroids = clus_result["centroids"].to(device)
        model.node2cluster = clus_result["item2cluster"].to(device)
        model.node_density = clus_result["density"].to(device)
                
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