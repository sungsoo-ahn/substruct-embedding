import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool

def to_dense_adj(edge_index):
    num_nodes = edge_index.max().item() + 1
    dense_adj = torch.sparse.LongTensor(
        edge_index, torch.ones(edge_index.size(1)), torch.Size([num_nodes, num_nodes])
        ).to_dense()
    return dense_adj

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

    def compute_logits_and_labels(self, batch):
        batch = batch.to(0)
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.frag_batch)
        out = self.projector(out)
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)
        out = self.predictor(out)
        features1 = torch.nn.functional.normalize(out, p=2, dim=1)

        logits = torch.matmul(features0, features1.t()) / self.contrastive_temperature
        labels = to_dense_adj(batch.dangling_edge_index)

        return logits, labels
    
    #def criterion(self, logits, labels):
        

    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(target[torch.max(pred, dim=1)[1]] == 1)) / pred.size(0)
        return acc

