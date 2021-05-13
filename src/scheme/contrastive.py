import torch
import numpy as np
from model import GINConv, GNN, GCNConv
from torch_geometric.nn import global_mean_pool

class SuperEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(SuperEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.layers = torch.nn.ModuleList(
            [GCNConv(self.emb_dim, aggr="add"), GCNConv(self.emb_dim, aggr="add")]
        )
        self.relus = torch.nn.ModuleList([torch.nn.ReLU()])
    
    def forward(self, x, edge_index, edge_attr):
        out = self.layers[0](x, edge_index, edge_attr)
        out = self.relus[0](out)
        out = self.layers[1](out, edge_index, edge_attr)
        return out
        

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.contrastive_temperature = 0.04
                                
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.junction_encoder = SuperEncoder(self.emb_dim)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
                
        self.mask_embedding = torch.nn.Parameter(torch.empty(self.emb_dim).normal_())
        
        self.empty_edge_index = torch.empty(2, 0, dtype=torch.long).cuda()
        self.empty_edge_attr = torch.empty(0, 2, dtype=torch.long).cuda()
        
    def compute_logits_and_labels(self, batch, frag_batch, junction_batch):
        batch = batch.to(0)
        frag_batch = frag_batch.to(0)
        junction_batch = junction_batch.to(0)
        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
        out = self.junction_encoder(out, self.empty_edge_index, self.empty_edge_attr)
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)
                                
        out = self.encoder(frag_batch.x, frag_batch.edge_index, frag_batch.edge_attr)
        out = global_mean_pool(out, frag_batch.junction_batch)
        out[junction_batch.mask > 0.5] = self.mask_embedding
        out = self.junction_encoder(out, self.empty_edge_index, self.empty_edge_attr)
        out = global_mean_pool(out, junction_batch.batch)
        features1 = torch.nn.functional.normalize(out, p=2, dim=1)
    
        logits = torch.matmul(features0, features1.t()) / self.contrastive_temperature
        labels = torch.arange(logits.size(0)).to(0)
        
        return logits, labels