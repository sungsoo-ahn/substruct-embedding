import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering

class NodeClusteringModel(torch.nn.Module):
    def __init__(self, use_linear_projection):
        super(NodeClusteringModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.proj_dim = 100
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
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

    def compute_node_features(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)

        return node_features

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

    def compute_logits_and_labels(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.projector(out)
        features_node = torch.nn.functional.normalize(out, p=2, dim=1)

        # Subsample nodes for fast computation
        node_mask = torch.bernoulli(torch.zeros(batch.x.size(0) // 2), p=0.1).bool().cuda()
        node_mask = torch.cat([node_mask, node_mask], axis=0)
        sampled_feature_nodes = features_node[node_mask]

        _ = get_contrastive_logits_and_labels(sampled_feature_nodes)
        logits_node_contrastive, labels_node_contrastive = _
        logits_node_contrastive /= self.contrastive_temperature

        logits_and_labels = {
            "node_contrastive": [logits_node_contrastive, labels_node_contrastive],
        }

        if self.node_centroids is not None:
            batch_active = self.node_active[batch.dataset_node_idx]
            node_mask = (batch_active > 0)
            sampled_feature_nodes = features_node[node_mask]
            logits_node_proto = torch.mm(sampled_feature_nodes, self.node_centroids.T)
            logits_node_proto /= self.proto_temperature

            tmp = (batch_active[batch_active > 0] - 1)
            labels_node_proto = self.node2cluster[tmp]

            logits_and_labels["node_proto"] = [logits_node_proto, labels_node_proto]

        return logits_and_labels


class NodeClusteringScheme:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

        self.proto_temperature = 0.2
        self.contrastive_temperature = 0.04

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

        del model.node_centroids
        del model.node2cluster
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(0)
                out = model.compute_node_features(batch)

                if node_features is None:
                    node_features = torch.zeros(loader.dataset.num_nodes, out.size(1))

                node_features[batch.dataset_node_idx] = out.cpu()

        node_active = torch.zeros(loader.dataset.num_nodes)
        node_active = torch.bernoulli(node_active, p=0.1).long()
        node_active[node_active > 0] = (torch.arange(node_active.sum()) + 1)
        model.node_active = node_active

        node_features = node_features[node_active > 0]
        node_features = node_features.numpy()

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
        model.node_centroids = clus_result["centroids"].cuda()
        model.node2cluster = clus_result["item2cluster"].cuda()

        return statistics

    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)

        logits_and_labels = model.compute_logits_and_labels(batch)

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

        return statistics