import torch
import numpy as np
from torch_sparse.coalesce import coalesce
from model import GNN, GCNConv
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
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        
        self.projector = build_projector(self.emb_dim)
        self.predictor = GCNConv(self.emb_dim, add_self_loop=False)
        self.classifier = torch.nn.Linear(self.emb_dim, 1)
        
    def compute_logits_and_labels(self, batch):
        batch = batch.to(0)
        u_index = torch.cat([batch.frag_edge_index[0], batch.frag_edge_index[1]], dim=0)
        v_index = torch.cat([batch.frag_edge_index[1], batch.frag_edge_index[0]], dim=0)
        uv_index = torch.stack([u_index, v_index], dim=0)
        uv_edge_attr = torch.cat([batch.dangling_edge_attr, batch.dangling_edge_attr], dim=0)
                
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.frag_batch)
        out = self.projector(out)
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)
        
        out = self.predictor(out, uv_index, uv_edge_attr)
        features1 = torch.nn.functional.normalize(out, p=2, dim=1)
        features2 = torch.roll(features1, shifts=1, dims=0)

        pos_logits = self.classifier(torch.max(features0, features1))
        neg_logits = self.classifier(torch.max(features0, features2))                    

        logits = torch.cat([pos_logits, neg_logits], dim=0).squeeze(1)
        
        labels = torch.cat(
            [torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))], dim=0
        ).to(0)

        return logits, labels

    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.eq(pred > 0, target > 0.5)).long()) / pred.size(0)
        return acc    
    
    

