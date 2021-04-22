import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, run_clustering

class NodeGraphClusteringNoAugModel(torch.nn.Module):
    def __init__(self, use_linear_projection):
        super(NodeGraphClusteringNoAugModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.proj_dim = 100
        self.drop_rate = 0.0
        self.proto_temperature = 0.1
        self.entropy_coef = 1.0
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        if use_linear_projection:
            self.projector = torch.nn.Linear(self.emb_dim, self.proj_dim)
        else:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim, self.proj_dim)
            )

        self.node_centroids = None
        self.node2cluster = None
        self.node_density = None
        
        self.graph_centroids = None
        self.graph2cluster = None
        self.graph_density = None
        
    def compute_node_features(self, batch):        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)

        return node_features

    def compute_graph_features(self, batch):        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)

        graph_features = global_mean_pool(node_features, batch.batch)
        graph_features = torch.nn.functional.normalize(graph_features, p=2, dim=1)

        return graph_features

    def compute_features(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)

        graph_features = global_mean_pool(node_features, batch.batch)
        graph_features = torch.nn.functional.normalize(graph_features, p=2, dim=1)

        return node_features, graph_features

    def compute_logits_and_labels(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)

        graph_features = global_mean_pool(node_features, batch.batch)
        graph_features = torch.nn.functional.normalize(graph_features, p=2, dim=1)

        logits_and_labels = dict()
        losses = dict()
        if self.node_centroids is not None:
            batch_active = self.node_active[batch.dataset_node_idx]
            node_mask = (batch_active > 0)
            sampled_feature_nodes = node_features[node_mask]
            logits_node_proto = torch.mm(sampled_feature_nodes, self.node_centroids.T)
            logits_node_proto /= self.proto_temperature
            
            tmp = (batch_active[batch_active > 0] - 1)
            labels_node_proto = self.node2cluster[tmp]
            
            logits_and_labels["node_proto"] = [logits_node_proto, labels_node_proto]
            
            probs_node_proto = torch.softmax(logits_node_proto, dim=1)
            losses["node_proto_entropy"] = self.entropy_coef * (
                (probs_node_proto * probs_node_proto.log()).sum(dim=1).mean(dim=0)
            )
        
        
        if self.graph_centroids is not None:
            logits_graph_proto = torch.mm(graph_features, self.graph_centroids.T)
            logits_graph_proto /= self.proto_temperature
            labels_graph_proto = self.graph2cluster[batch.dataset_graph_idx]
            logits_and_labels["graph_proto"] = [logits_graph_proto, labels_graph_proto]
            
            probs_graph_proto = torch.softmax(logits_graph_proto, dim=1)
            losses["graph_proto_entropy"] = self.entropy_coef * (
                (probs_graph_proto * probs_graph_proto.log()).sum(dim=1).mean(dim=0)
            )
        
        return logits_and_labels, losses
    

class NodeGraphClusteringNoAugScheme:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        
        self.clus_verbose = True
        self.clus_niter = 20
        self.clus_nredo = 1
        self.clus_seed = 0
        self.clus_max_points_per_centroid = 500
        self.clus_min_points_per_centroid = 10
        self.clus_use_euclidean_clustering = False
        
    def assign_cluster(self, loader, model):
        print("Collecting graph features for clustering...")
        model.eval()
        node_features = None
        graph_features = None
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(0)
                out0, out1 = model.compute_features(batch)

                if node_features is None:
                    node_features = torch.zeros(loader.dataset.num_nodes, out0.size(1)).cuda()
                
                if graph_features is None:
                    graph_features = torch.zeros(len(loader.dataset), out1.size(1)).cuda()

                node_features[batch.dataset_node_idx] = out0
                graph_features[batch.dataset_graph_idx] = out1

        node_active = torch.zeros(loader.dataset.num_nodes)
        node_active = torch.bernoulli(node_active, p=0.1).long().cuda()
        node_active[node_active > 0] = (torch.arange(node_active.sum()).cuda() + 1)
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
        model.node_centroids = node_clus_result["centroids"].cuda()
        model.node2cluster = node_clus_result["item2cluster"].cuda()
        model.node_density = node_clus_result["density"].cuda()

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
        model.graph_centroids = graph_clus_result["centroids"].cuda()
        model.graph2cluster = graph_clus_result["item2cluster"].cuda()
        model.graph_density = graph_clus_result["density"].cuda()
               
        statistics = dict()
        for key in node_statistics:
            statistics[f"node_{key}"] = node_statistics[key]

        for key in graph_statistics:
            statistics[f"graph_{key}"] = graph_statistics[key]
         
        return statistics

    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)
        
        logits_and_labels, losses = model.compute_logits_and_labels(batch)        
        
        loss_cum = 0.0
        statistics = dict()
        for key in logits_and_labels:
            logits, labels = logits_and_labels[key]
            loss = model.criterion(logits, labels)
            acc = compute_accuracy(logits, labels)                
            loss_cum += loss
            
            statistics[f"{key}/loss"] = loss.detach()
            statistics[f"{key}/acc"] = acc
            
        for key in losses:
            loss = losses[key]
            loss_cum += loss
            statistics[f"{key}/loss"] = loss.detach()
                        
        if len(logits_and_labels.keys()) > 0:
            optim.zero_grad()
            loss_cum.backward()
            optim.step()
        
        return statistics