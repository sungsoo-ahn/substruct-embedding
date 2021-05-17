import torch
import numpy as np
from model import GNN, GCNConv
from torch_geometric.nn import global_mean_pool

def build_projector(emb_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(emb_dim, emb_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(emb_dim, emb_dim),
        )
    

class Model(torch.nn.Module):
    def __init__(self, use_double_projector, use_dangling_mask):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.contrastive_temperature = 0.04
        
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        
        self.projector0 = build_projector(self.emb_dim)
        
        self.use_double_projector = use_double_projector        
        if self.use_double_projector:
            self.projector1 = build_projector(self.emb_dim)
        else:
            self.projector1 = self.projector0

        self.use_dangling_mask = use_dangling_mask
        if self.use_dangling_mask:
            self.dangling_projector0 = build_projector(self.emb_dim)

            if self.use_double_projector:
                self.dangling_projector1 = build_projector(self.emb_dim)
            else:
                self.dangling_projector1 = self.dangling_projector0

        self.epsilon = 1e-10
        self.max_iter = 10

    def compute_logits_and_labels(self, batch0, batch1):
        batch0 = batch0.to(0)
        repr = self.encoder(batch0.x, batch0.edge_index, batch0.edge_attr)
        out = global_mean_pool(repr, batch0.batch)
        out = self.projector0(out)
        if self.use_dangling_mask:
            dangling_out = self.dangling_projector0(repr[batch0.dangling_mask])
            out = out + dangling_out
        
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)

        batch1 = batch1.to(0)
        repr = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        out = global_mean_pool(repr, batch1.batch)
        out = self.projector1(out)
        if self.use_dangling_mask:
            dangling_out = self.dangling_projector1(repr[batch1.dangling_mask])
            out = out + dangling_out

        features1 = torch.nn.functional.normalize(out, p=2, dim=1)

        sim_mat = torch.exp(torch.matmul(features0, features1.t()) / self.contrastive_temperature)
        probs = self.sinkhorn(sim_mat.unsqueeze(0)).squeeze(0)
        
        labels = torch.arange(probs.size(0)).to(0)

        return probs, labels

    def compute_accuracy(self, probs, labels):
        acc = float(torch.sum(torch.max(probs, dim=1)[1] == labels)) / probs.size(0)
        return acc
    
    def criterion(self, probs, labels):
        binary_labels = torch.eye(probs.size(0)).to(0)        
        loss = torch.nn.functional.binary_cross_entropy(probs.view(-1), binary_labels.view(-1))
        return loss

    def sinkhorn(self, s, nrows=None, ncols=None):
        batch_size = s.shape[0]

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s = s + self.epsilon

        for i in range(self.max_iter):
            if i % 2 == 1:
                # column norm
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        return s