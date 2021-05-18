import torch
import numpy as np
from model import GNN, GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_sparse import coalesce


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.contrastive_temperature = 0.04

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, 1),
        )


    def compute_logits_and_labels(self, batchs):
        batch0, batch1 = batchs
        batch0 = batch0.to(0)
        out = self.encoder(batch0.x, batch0.edge_index, batch0.edge_attr)
        out = global_mean_pool(out, batch0.batch)
        pos_logits = self.classifier(out)
        
        batch1 = batch1.to(0)
        out = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        out = global_mean_pool(out, batch1.batch)        
        neg_logits = self.classifier(out)
        
        logits = torch.cat([pos_logits, neg_logits], dim=0).squeeze(1)
        labels = torch.cat(
            [torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))], dim=0
        ).to(0)

        return logits, labels
    

    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.eq(pred > 0, target > 0.5)).long()) / pred.size(0)
        return acc
