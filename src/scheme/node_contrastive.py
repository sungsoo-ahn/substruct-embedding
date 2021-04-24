import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy

from torch_geometric.utils import to_dense_adj

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
        return K

class NodeContrastiveModel(torch.nn.Module):
    def __init__(self):
        super(NodeContrastiveModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.temperature = 0.05
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.sinkhorn = SinkhornLayer(10)
    
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

        contrastive_logits = torch.mm(node_features, node_features.T) / self.temperature
        contrastive_log_probs = torch.log_softmax(contrastive_logits, dim=1)
        
        with torch.no_grad():
            sinkhorn_coupling = self.sinkhorn(contrastive_logits.unsqueeze(0)).squeeze(0)
            
        loss = -(sinkhorn_coupling*contrastive_log_probs).sum(1).mean(0)
        
        adj = to_dense_adj(batch.edge_index, max_num_nodes = batch.x.size(0)).squeeze(0)
        adj = adj + torch.eye(adj.size(0), dtype=torch.long).cuda()
        adj = (torch.mm(adj, adj) > 0).float()
        contrastive_targets = adj
        contrastive_targets /= contrastive_targets.sum(dim=0).unsqueeze(dim=0)
        loss += -(contrastive_targets * contrastive_log_probs).sum(dim=1).mean(dim=0)
        
        return loss

class NodeContrastiveScheme:
    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)

        loss = model.compute_loss(batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        statistics = {"loss": loss.detach().item()}

        return statistics