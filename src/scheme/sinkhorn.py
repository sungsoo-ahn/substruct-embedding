import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering

class mask

class SinkhornLayer(torch.nn.Module):
    def __init__(self, eps, num_iters):
        super(SinkhornLayer, self).__init__()
        self.eps = eps
        self.num_iters = num_iters

    def forward(K, mask):
        """
        dot: n x in_size x out_size
        mask: n x in_size
        output: n x in_size x out_size
        """
        batch_size, in_size, out_size = K.shape
        def min_eps(u, v, dim):
            Z = (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps
            return -torch.logsumexp(Z, dim=dim)
        # K: batch_size x in_size x out_size
        u = K.new_zeros((batch_size, in_size))
        v = K.new_zeros((batch_size, out_size))
        a = torch.ones_like(u).fill_(out_size / in_size)
        if mask is not None:
            a = out_size / mask.float().sum(1, keepdim=True)
        a = torch.log(a)
        for _ in range(self.num_iters):
            u = self.eps * (a + min_eps(u, v, dim=-1)) + u
            if mask is not None:
                u = u.masked_fill(~mask, -1e8)
            v = self.eps * min_eps(u, v, dim=1) + v

        output = torch.exp(
            (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
        output = output / out_size
        return (output * K).sum(dim=[1, 2])

class SinkhornModel(torch.nn.Module):
    def __init__(self, use_linear_projection):
        super(ClusteringBottleneckModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.proj_dim = 100
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
        self.sinkhorn_epsilon = 1.0
        self.num_sinkhon_iters = 10

        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.projector = torch.nn.Linear(self.emb_dim, self.proj_dim)
        self.sinkhorn = SinkhornLayer(self.sinkhorn_epsilon, self.num_sinkhon_iters)

        self.centroids = nn.Parameter(torch.randn(self.num_centroids, self.emb_dim))

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
        centroid_features = torch.nn.functional.normalize(self.centroids, p=2, dim=1)

        node_features0, node_features1 = torch.chunk(node_features, 2, dim=0)
        dist_mat0 = torch.einsum('nc,mc->nm', [node_features0, centroid_features])
        dist_mat1 = torch.einsum('nc,mc->nm', [node_features1, centroid_features])
        sinkhorn_coupling0 = self.sinkhorn(dist_mat0.unsqueeze(0))
        sinkhorn_coupling1 = self.sinkhorn(dist_mat1.unsqueeze(0))

        sinkhorn_loss = (
            torch.sum(sinkhorn_coupling1*dist_mat0) + torch.sum(sinkhorn_coupling0*dist_mat1)
        )

        """
        stacked_node_features = batch.sinkhorn_mask.clone().float()
        stacked_node_features[stacked_node_features>0] = node_features
        stacked_node_features = stacked_node_features.view(batch.batch_size, -1)
        stacked_node_features0, stacked_node_features1 = torch.chunk(
            stacked_node_features, 2, dim=0
            )

        stacked_dist_mat0 = torch.einsum(
            'bnc, mc->bnm', [stacked_node_features0, centroid_features]
            )
        stacked_dist_mat1 = torch.einsum(
            'bnc, mc->bnm', [stacked_node_features1, centroid_features]
            )
        stacked_sinkhorn_mask = batch.sinkhorn_mask.view(batch.batch_size, -1)
        stacked_sinkhorn_mask0, stacked_sinkhorn_mask1 = torch.chunk(
            stacked_sinkhorn_mask, 2, dim=0
            )
        stacked_sinkhorn_coupling0 = self.sinkhorn(stacked_dist_mat0, mask=stacked_sinkhorn_mask0)
        stacked_sinkhorn_coupling1 = self.sinkhorn(stacked_dist_mat1, mask=stacked_sinkhorn_mask1)

        stacked_sinkhorn_loss = -torch.sum(
            stacked_sinkhorn_coupling1 * stacked_dist_mat0
            stacked_sinkhorn_coupling0 * stacked_dist_mat1
            )
        stacked_sinkhorn_loss /= batch.batch_size

        return sinkhorn_loss, stacked_sinkhorn_loss
        """
        return sinkhorn_loss

class SinkhornScheme:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

        self.clus_verbose = True
        self.clus_niter = 20
        self.clus_nredo = 1
        self.clus_seed = 0
        self.clus_max_points_per_centroid = 500
        self.clus_min_points_per_centroid = 10
        self.clus_use_euclidean_clustering = False


    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)

        loss = sinkhorn_loss = model.compute_loss(batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        statistics = {"loss": loss.detach().item(), "sinkhorn_loss": sinkhorn_loss.detach().item()}

        return statistics