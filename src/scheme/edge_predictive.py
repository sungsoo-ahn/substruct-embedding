import random
import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool

num_bond_type = 6
num_bond_direction = 3 

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(5*self.emb_dim, 5*self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(5*self.emb_dim, 1),
        )
            
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, self.emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, self.emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def compute_logits_and_labels(self, batch):
        batch = batch.to(0)
                
        u_index = batch.dangling_edge_index[0]
        v_index = batch.dangling_edge_index[1]
        uv_edge_attr = batch.drop_edge_attr
        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        
        frag_out = global_mean_pool(out, batch.frag_batch)
        frag_out = torch.repeat_interleave(frag_out, batch.frag_num_nodes, dim=0)
        frag_out = frag_out[batch.dangling_mask]
        
        dangling_out = out[batch.dangling_mask]
        
        frag_out0 = frag_out[u_index]
        frag_out1 = frag_out[v_index]
        shift_k = random.choice(range(1, frag_out0.size(0)))
        frag_out2 = torch.roll(frag_out1, shifts=shift_k, dims=0)
        
        dangling_out0 = dangling_out[u_index]
        dangling_out1 = dangling_out[v_index]
        dangling_out2 = torch.roll(dangling_out1, shifts=shift_k, dims=0)
        
        edge_embeddings = (
            self.edge_embedding1(uv_edge_attr[:,0]) + self.edge_embedding2(uv_edge_attr[:,1])
        )

        out01 = torch.cat(
            [frag_out0, dangling_out0, frag_out1, dangling_out1, edge_embeddings], dim=1
            )
        out02 = torch.cat(
            [frag_out0, dangling_out0, frag_out2, dangling_out2, edge_embeddings], dim=1
            )
        
        pos_logits = self.classifier(out01)
        neg_logits = self.classifier(out02)                    

        logits = torch.cat([pos_logits, neg_logits], dim=0).squeeze(1)
        
        labels = torch.cat(
            [torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))], dim=0
        ).to(0)

        return logits, labels
    
    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.eq(pred > 0, target > 0.5)).long()) / pred.size(0)
        return acc    
    
    
