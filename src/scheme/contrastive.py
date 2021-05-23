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
        self.dangling_projector = torch.nn.Linear(self.emb_dim, self.emb_dim)        
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, self.emb_dim * self.emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, self.emb_dim* self.emb_dim)

        embedding_size = (num_bond_type * self.emb_dim * self.emb_dim)
        uniform(embedding_size, self.edge_embedding1.weight)
        embedding_size = (num_bond_direction * self.emb_dim * self.emb_dim)
        uniform(embedding_size, self.edge_embedding2.weight)

    def compute_logits_and_labels(self, batch):        
        batch = batch.to(0)
                
        u_index = batch.dangling_edge_index[0]
        v_index = batch.dangling_edge_index[1]
        uv_edge_attr = batch.drop_edge_attr
        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        dangling_out = out[batch.dangling_mask]
        dangling_out = self.dangling_projector(dangling_out)
        
        frag_out = global_mean_pool(out, batch.frag_batch)
        frag_out = torch.repeat_interleave(frag_out, batch.frag_num_nodes, dim=0)
        frag_out = frag_out[batch.dangling_mask]
        
        out = dangling_out + frag_out
        
        out0 = out[u_index]
        predict_mat = (
            self.edge_embedding1(uv_edge_attr[:,0]) + self.edge_embedding2(uv_edge_attr[:,1])
        ).view(-1, self.emb_dim, self.emb_dim)
        
        out0 = torch.bmm(out0.unsqueeze(1), predict_mat).squeeze(1)
        features0 = torch.nn.functional.normalize(out0, p=2, dim=1)

        out1 = out[v_index]
        features1 = torch.nn.functional.normalize(out1, p=2, dim=1)

        logits = torch.matmul(features0, features1.t()) / self.contrastive_temperature
        targets = torch.arange(logits.size(0)).to(0)

        return logits, targets
    
    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
        return acc
