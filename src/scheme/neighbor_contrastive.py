import torch
import numpy as np
from torch_sparse.coalesce import coalesce
from model import GNN, GINConv
from torch_geometric.nn import global_mean_pool


def build_projector(emb_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(emb_dim, emb_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(emb_dim, emb_dim),
        )
    

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.contrastive_temperature = 0.04

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        
        self.projector = build_projector(self.emb_dim)
        self.predictor = GINConv(self.emb_dim, add_self_loop=False)

    def compute_logits_and_labels(self, batch):
        batch = batch.to(0)
        u_index = torch.cat([batch.frag_edge_index[0], batch.frag_edge_index[1]], dim=0)
        v_index = torch.cat([batch.frag_edge_index[1], batch.frag_edge_index[0]], dim=0)
        uv_index = torch.stack([u_index, v_index], dim=0)
        uv_edge_attr = torch.cat([batch.dangling_edge_attr, batch.dangling_edge_attr], dim=0)
        
        #print(uv_index)
        #print(batch.frag_batch)
        #assert False
        #uv_index, uv_edge_attr = coalesce(uv_index, uv_edge_attr, batch.x.size(0), batch.x.size(0))
        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.frag_batch)
        out = self.projector(out)
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)

        
        out = self.predictor(out, uv_index, uv_edge_attr)
        features1 = torch.nn.functional.normalize(out, p=2, dim=1)
         
        logits = torch.matmul(features1, features0.t()) / self.contrastive_temperature
        labels = torch.arange(logits.size(0)).to(0)

        return logits, labels

    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
        return acc

