import torch
import numpy as np
from model import GNN, GCNConv
from torch_geometric.nn import global_mean_pool


class Model(torch.nn.Module):
    def __init__(self, use_double_encoder):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.use_double_encoder = use_double_encoder
        self.contrastive_temperature = 0.04

        self.encoder0 = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        if self.use_double_encoder:
            self.encoder1 = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        else:
            self.encoder1 = self.encoder0

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )

        self.masked_frag_embedding = torch.nn.Parameter(torch.empty(self.emb_dim).normal_())

    def compute_logits_and_labels(self, batch0, batch1):
        batch0 = batch0.to(0)
        out = self.encoder0(batch0.x, batch0.edge_index, batch0.edge_attr)
        out = global_mean_pool(out, batch0.batch)
        out = self.projector(out)
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)

        batch1 = batch1.to(0)
        out = self.encoder1(batch1.x, batch1.edge_index, batch1.edge_attr)
        out = global_mean_pool(out, batch1.batch)
        out = self.projector(out)
        features1 = torch.nn.functional.normalize(out, p=2, dim=1)

        logits = torch.matmul(features0, features1.t()) / self.contrastive_temperature
        labels = torch.arange(logits.size(0)).to(0)

        return logits, labels

