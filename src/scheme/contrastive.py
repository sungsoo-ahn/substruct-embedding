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
    def __init__(self, proj_type):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.contrastive_temperature = 0.04

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        
        self.projector0 = build_projector(self.emb_dim)
        
        self.proj_type = proj_type
        if self.proj_type == 0:
            self.projector0 = build_projector(self.emb_dim)
            self.projector1 = build_projector(self.emb_dim)
            self.dangling_projector0 = build_projector(self.emb_dim)
            self.dangling_projector1 = build_projector(self.emb_dim)
        
        elif self.proj_type == 1:
            self.projector = build_projector(self.emb_dim)
            self.dangling_projector = build_projector(self.emb_dim)
            self.predictor = build_projector(self.emb_dim)

    def compute_logits_and_labels(self, batch0, batch1):
        batch0 = batch0.to(0)
        repr = self.encoder(batch0.x, batch0.edge_index, batch0.edge_attr)
        out = global_mean_pool(repr, batch0.batch)
        if self.proj_type == 0:
            out = self.projector0(out)
            dangling_out = self.dangling_projector0(repr[batch0.dangling_mask])
            out = out + dangling_out
        elif self.proj_type == 1:
            out = self.projector(out)
            dangling_out = self.dangling_projector(repr[batch0.dangling_mask])
            out = out + dangling_out
            out = self.predictor(out)
        
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)

        batch1 = batch1.to(0)
        repr = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        out = global_mean_pool(repr, batch1.batch)
        if self.proj_type == 0:
            out = self.projector1(out)
            dangling_out = self.dangling_projector1(repr[batch1.dangling_mask])
            out = out + dangling_out
        elif self.proj_type == 1:
            out = self.projector(out)
            dangling_out = self.dangling_projector(repr[batch1.dangling_mask])
            out = out + dangling_out

        features1 = torch.nn.functional.normalize(out, p=2, dim=1)

        logits = torch.matmul(features0, features1.t()) / self.contrastive_temperature
        labels = torch.arange(logits.size(0)).to(0)

        return logits, labels

    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
        return acc

