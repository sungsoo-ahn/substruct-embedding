import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering

class GraphClusteringModel(torch.nn.Module):
    def __init__(self, use_density_rescaling):
        super(GraphClusteringModel, self).__init__()
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

        self.graph_centroids = None
        self.graph2cluster = None
        self.graph_density = None
        
    def update_ema_encoder(self):
        for param, ema_param in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)
    
        for param, ema_param in zip(self.projector.parameters(), self.ema_projector.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)

    def compute_ema_features_graph(self, x, edge_index, edge_attr, batch):
        out = self.ema_encoder(x, edge_index, edge_attr)
        out = self.ema_projector(out)
        features_graph = global_mean_pool(out, batch)
        features_graph = torch.nn.functional.normalize(features_graph, p=2, dim=1)
        
        return features_graph
        
        
    def compute_logits_and_labels(self, x, edge_index, edge_attr, batch, dataset_graph_idx):
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        features_graph = global_mean_pool(out, batch)
        features_graph = torch.nn.functional.normalize(features_graph, p=2, dim=1)
        
        _ = get_contrastive_logits_and_labels(features_graph)
        logits_graph_contrastive, labels_graph_contrastive = _
        logits_graph_contrastive /= self.contrastive_temperature
        
        logits_and_labels = {
            "graph_contrastive": [logits_graph_contrastive, labels_graph_contrastive],
        }
        
        if self.graph_centroids is not None:
            logits_graph_proto = torch.mm(features_graph, self.graph_centroids)
            logits_graph_proto /= self.proto_temperature
            if self.use_density_rescaling:
                logits /= self.graph_density
                
            labels_graph_proto = self.graph2cluster[dataset_graph_idx]
            logits_and_labels["graph_proto"] = [logits_graph_proto, labels_graph_proto]
        
        return logits_and_labels
    

class GraphClusteringScheme:
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
        graph_features = None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model.compute_ema_features_graph(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    )

                if graph_features is None:
                    graph_features = torch.zeros(len(loader.dataset), out.size(1)).to(device)

                graph_features[batch.dataset_graph_idx] = out

        graph_features = graph_features.cpu().numpy()
        
        clus_result, statistics = run_clustering(
            graph_features,
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
        model.graph_centroids = clus_result["centroids"].to(device)
        model.graph2cluster = clus_result["item2cluster"].to(device)
        model.graph_density = clus_result["density"].to(device)
        
        return statistics

    def train_step(self, batch, model, optim, device):
        model.train()
        batch = batch.to(device)
        
        logits_and_labels = model.compute_logits_and_labels(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.dataset_graph_idx
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

