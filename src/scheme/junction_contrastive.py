import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.BCEWithLogitsLoss()
                                
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, 1),
        )        
        
    def compute_logits_and_labels(self, batch):
        batch = batch.to(0)
                
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        lower_features = global_mean_pool(out, batch.lower_batch)
        
        out = global_mean_pool(lower_features, batch.upper_batch)
        logits0 = self.classifier(out).squeeze(1)
        
        swap_idx0 = torch.cumsum(batch.upper_num_nodes, dim=0) - 1
        swap_idx1 = swap_idx0 + 1
        swap_idx1[-1] = 0
        lower_features[swap_idx0] = lower_features[swap_idx1]
        out = global_mean_pool(lower_features, batch.upper_batch)
        logits1 = self.classifier(out).squeeze(1)
    
        logits = torch.cat([logits0, logits1], dim=0)
        labels = torch.cat([torch.zeros(logits0.size(0)), torch.ones(logits1.size(0))], dim=0)
        labels = labels.to(0)
        
        return logits, labels