import torch
import numpy as np
from model import GNN, GCNConv
from torch_geometric.nn import global_mean_pool

class SuperEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(SuperEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.layers = torch.nn.ModuleList(
            [GCNConv(self.emb_dim, aggr="add"), GCNConv(self.emb_dim, aggr="add")]
        )
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(emb_dim)])
        self.relus = torch.nn.ModuleList([torch.nn.ReLU()])
    
    def forward(self, x, edge_index, edge_attr):
        out = self.layers[0](x, edge_index, edge_attr)
        #out = self.batch_norms[0](out)
        out = self.relus[0](out)
        out = self.layers[1](out, edge_index, edge_attr)
        return out
        

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
        self.super_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
        
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
                
        self.masked_frag_embedding = torch.nn.Parameter(torch.empty(self.emb_dim).normal_())
        
    def compute_logits_and_labels(self, batch, super_batch):
        batch = batch.to(0)
        super_batch = super_batch.to(0)
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
                
        target_features = self.projector(out[super_batch.mask])
        target_features = torch.nn.functional.normalize(target_features, p=2, dim=1)
        
        out[super_batch.mask] = 0.0
        out = global_mean_pool(out, super_batch.batch)
        pred_features = self.super_encoder(out)
        pred_features = torch.nn.functional.normalize(pred_features, p=2, dim=1)
        
        logits = torch.matmul(pred_features, target_features.t()) / self.contrastive_temperature
        labels = torch.arange(logits.size(0)).to(0)
                
        return logits, labels
        
        