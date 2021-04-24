import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool
import torch.nn as nn

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering

class SinkhornLayer(torch.nn.Module):
    def __init__(self, num_iters):
        super(SinkhornLayer, self).__init__()
        self.num_iters = num_iters

    def forward(self, dot, mask=None):
        n, in_size, out_size = dot.shape
        K = torch.exp(dot)
        # K: n x in_size x out_size
        u = K.new_ones((n, in_size))
        v = K.new_ones((n, out_size))
        a = float(out_size / in_size)
        if mask is not None:
            mask = mask.float()
            a = out_size / mask.sum(1, keepdim=True)
        for _ in range(self.num_iters):
            u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
            if mask is not None:
                u = u * mask
            v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
        K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
        #K /= out_size
        return K

class SinkhornModel(torch.nn.Module):
    def __init__(self, num_centroids):
        super(SinkhornModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.eps = 0.01
        self.num_sinkhon_iters = 10
        self.num_centroids = num_centroids
        self.use_stacked_sinkhorn = False
        self.queue_size = 50000

        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.projector = torch.nn.Linear(self.emb_dim, self.emb_dim)
        self.sinkhorn = SinkhornLayer(self.num_sinkhon_iters)

        self.centroids = nn.Parameter(torch.randn(self.num_centroids, self.emb_dim))
    
        self.register_buffer("queue", torch.randn(self.emb_dim, self.queue_size))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        self.queue = torch.cat([self.queue, keys.T], dim=1)[:, -self.queue_size:]
            
    def compute_graph_features(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)

        graph_features = global_mean_pool(node_features, batch.batch)
        graph_features = torch.nn.functional.normalize(graph_features, p=2, dim=1)
        
        return graph_features

    def compute_loss(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)
        centroid_features = torch.nn.functional.normalize(self.centroids, p=2, dim=1)

        node_features0, node_features1 = torch.chunk(node_features, 2, dim=0)
        node_features0 = torch.cat([node_features0, self.queue.T], dim=0)
        node_features1 = torch.cat([node_features1, self.queue.T], dim=0)
        score_mat0 = torch.einsum('nc,mc->nm', [node_features0, centroid_features]) / self.eps
        score_mat1 = torch.einsum('nc,mc->nm', [node_features1, centroid_features]) / self.eps

        with torch.no_grad():
            sinkhorn_coupling0 = self.sinkhorn(score_mat0.unsqueeze(0)).squeeze(0)
            sinkhorn_coupling1 = self.sinkhorn(score_mat1.unsqueeze(0)).squeeze(0)

        log_probs0 = torch.nn.functional.log_softmax(score_mat0, dim=1)
        log_probs1 = torch.nn.functional.log_softmax(score_mat1, dim=1)
        
        sinkhorn_loss = (
            - (sinkhorn_coupling1*log_probs0).sum(1).mean(0)
            - (sinkhorn_coupling0*log_probs1).sum(1).mean(0)
        )

        self.dequeue_and_enqueue(node_features0)

        if self.use_stacked_sinkhorn:
            stacked_node_features = batch.sinkhorn_mask.clone().float()
            stacked_node_features[stacked_node_features>0] = node_features
            stacked_node_features = stacked_node_features.view(batch.batch_size, -1)
            stacked_node_features0, stacked_node_features1 = torch.chunk(
                stacked_node_features, 2, dim=0
                )

            stacked_score_mat0 = torch.einsum(
                'bnc, mc->bnm', [stacked_node_features0, centroid_features]
                ) / self.eps
            stacked_score_mat1 = torch.einsum(
                'bnc, mc->bnm', [stacked_node_features1, centroid_features]
                ) / self.eps
            stacked_sinkhorn_mask = batch.sinkhorn_mask.view(batch.batch_size, -1)
            stacked_sinkhorn_mask0, stacked_sinkhorn_mask1 = torch.chunk(
                stacked_sinkhorn_mask, 2, dim=0
                )
            stacked_sinkhorn_coupling0 = self.sinkhorn(
                stacked_score_mat0, mask=stacked_sinkhorn_mask0
                )
            stacked_sinkhorn_coupling1 = self.sinkhorn(
                stacked_score_mat1, mask=stacked_sinkhorn_mask1
                )

            stacked_sinkhorn_loss = torch.sum(
                stacked_sinkhorn_coupling1*torch.exp(stacked_score_mat0)
                + stacked_sinkhorn_coupling0*torch.exp(stacked_score_mat1)
                )
            stacked_sinkhorn_loss /= batch.batch_size

            return sinkhorn_loss, stacked_sinkhorn_loss

        else:
            return sinkhorn_loss

class SinkhornScheme:
    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)

        loss = sinkhorn_loss = model.compute_loss(batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        statistics = {"loss": loss.detach().item(), "sinkhorn_loss": sinkhorn_loss.detach().item()}

        return statistics