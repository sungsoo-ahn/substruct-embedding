import random
import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import uniform

num_bond_type = 6
num_bond_direction = 3 

class Model(torch.nn.Module):
    def __init__(self, version):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        if version == 0:
            self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
            self.dangling_projector = torch.nn.Linear(self.emb_dim, self.emb_dim) 
            self.edge_embedding1 = torch.nn.Embedding(num_bond_type, self.emb_dim * self.emb_dim)
            self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, self.emb_dim* self.emb_dim)

            embedding_size = (num_bond_type * self.emb_dim * self.emb_dim)
            uniform(embedding_size, self.edge_embedding1.weight)
            embedding_size = (num_bond_direction * self.emb_dim * self.emb_dim)
            uniform(embedding_size, self.edge_embedding2.weight)
            
            self.compute_logits_and_labels = self.compute_logits_and_labels_v0
            
        elif version == 1:
            self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
            
            self.edge_embedding1 = torch.nn.Embedding(
                num_bond_type, 4 * self.emb_dim * self.emb_dim
                )
            self.edge_embedding2 = torch.nn.Embedding(
                num_bond_direction, 4 * self.emb_dim* self.emb_dim
                )

            embedding_size = (4 * num_bond_type * self.emb_dim * self.emb_dim)
            uniform(embedding_size, self.edge_embedding1.weight)
            embedding_size = (4 * num_bond_direction * self.emb_dim * self.emb_dim)
            uniform(embedding_size, self.edge_embedding2.weight)
            
            self.compute_logits_and_labels = self.compute_logits_and_labels_v1
            

    def compute_logits_and_labels_v0(self, batch):
        batch = batch.to(0)
        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        dangling_out = out[batch.dangling_mask]
        
        frag_out = global_mean_pool(out, batch.frag_batch)
        frag_out = frag_out[batch.frag_batch][batch.dangling_mask]
        
        dangling_out = self.dangling_projector(dangling_out)
        out = dangling_out + frag_out
        
        out0 = out[batch.dangling_edge_index[0]]
        predict_mat = (
            self.edge_embedding1(batch.drop_edge_attr[:,0]) 
            + self.edge_embedding2(batch.drop_edge_attr[:,1])
        ).view(-1, self.emb_dim, self.emb_dim)        

        out0 = torch.bmm(out0.unsqueeze(1), predict_mat).squeeze(1)

        out1 = out[batch.dangling_edge_index[1]]
        
        shift_k = random.choice(range(1, out1.size(0)))
        out2 = torch.roll(out1, shifts=shift_k, dims=0)
        
        out01 = torch.sum(out0*out1, dim=1)
        out02 = torch.sum(out0*out2, dim=1)
        
        logits = torch.cat([out01, out02], dim=0)
        
        labels = torch.cat(
            [torch.ones(out01.size(0)), torch.zeros(out02.size(0))], dim=0
        ).to(0)

        return logits, labels
    
    def compute_logits_and_labels_v1(self, batch):
        batch = batch.to(0)
        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        dangling_out = out[batch.dangling_mask]
        
        frag_out = global_mean_pool(out, batch.frag_batch)
        frag_out = frag_out[batch.frag_batch][batch.dangling_mask]
        
        out = torch.cat([dangling_out, frag_out], dim=1)
        
        out0 = out[batch.dangling_edge_index[0]]
        predict_mat = (
            self.edge_embedding1(batch.drop_edge_attr[:,0]) 
            + self.edge_embedding2(batch.drop_edge_attr[:,1])
        ).view(-1, 2 * self.emb_dim, 2 * self.emb_dim)        
        out0 = torch.bmm(out0.unsqueeze(1), predict_mat).squeeze(1)

        out1 = out[batch.dangling_edge_index[1]]
        
        shift_k = random.choice(range(1, out1.size(0)))
        out2 = torch.roll(out1, shifts=shift_k, dims=0)
        
        out01 = torch.sum(out0*out1, dim=1)
        out02 = torch.sum(out0*out2, dim=1)
        
        logits = torch.cat([out01, out02], dim=0)
        
        labels = torch.cat(
            [torch.ones(out01.size(0)), torch.zeros(out02.size(0))], dim=0
        ).to(0)

        return logits, labels
    
    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.eq(pred > 0, target > 0.5)).long()) / pred.size(0)
        return acc    
    
    
