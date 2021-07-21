import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import uniform

num_bond_type = 6
num_bond_direction = 3 

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.contrastive_temperature = 0.04

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def compute_logits_and_labels(self, batch):        
        batch = batch.to(0)

        batch0, batch1 = batch    
        out0 = self.encoder(batch0.x, batch0.edge_index, batch0.edge_attr)
        out0 = self.projector(out0)
        out0 = global_mean_pool(out0)
        features0 = torch.nn.functional.normalize(out0, p=2, dim=1)

        out1 = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        out1 = self.projector(out1)
        out1 = global_mean_pool(out1)
        features1 = torch.nn.functional.normalize(out1, p=2, dim=1)

        logits = torch.matmul(features0, features1.t()) / self.contrastive_temperature
        targets = torch.arange(logits.size(0)).to(0)

        return logits, targets
    
    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
        return acc
