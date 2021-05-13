import torch
import numpy as np
from model import GNN, GCNConv
from torch_geometric.nn import global_mean_pool
        

class Model(torch.nn.Module):
    def __init__(self, aggr, use_relation):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.aggr = aggr
        self.use_relation = use_relation
        self.contrastive_temperature = 0.04
        
        if self.aggr == "cat":
            self.feat_dim = 2*self.emb_dim
        else:
            self.feat_dim = self.emb_dim
                        
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
                
        self.masked_frag_embedding = torch.nn.Parameter(torch.empty(self.emb_dim).normal_())
        
    def compute_logits_and_labels(self, batch):
        batch = batch.to(0)
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
        out = self.projector(out)
        features = torch.nn.functional.normalize(out, p=2, dim=1)
        features0, features1 = torch.split(features, features.size(0) // 2)
    
        logits = torch.matmul(features0, features1.t()) / self.contrastive_temperature
        labels = torch.arange(logits.size(0)).to(0)
        
        return logits, labels
        
        