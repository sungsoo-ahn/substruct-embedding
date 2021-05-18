import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool

def to_dense_adj(edge_index):
    edge_index = edge_index.cpu()
    num_nodes = edge_index.max().item() + 1
    dense_adj = torch.sparse.LongTensor(
        edge_index, torch.ones(edge_index.size(1)), torch.Size([num_nodes, num_nodes])
        ).to_dense().to(0)
    dense_adj = dense_adj + dense_adj.t()
    return dense_adj

def build_projector(emb_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(emb_dim, emb_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(emb_dim, emb_dim),
        )

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
        
        self.projector = build_projector(self.emb_dim)
        self.dangling_projector = build_projector(self.emb_dim)
        
        self.classifier = torch.nn.Linear(self.emb_dim, 1)        
        self.predictor = build_projector(self.emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, self.emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, self.emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def compute_logits_and_labels(self, batch):        
        batch = batch.to(0)
                
        u_index = torch.cat([batch.frag_edge_index[0], batch.frag_edge_index[1]], dim=0)
        v_index = torch.cat([batch.frag_edge_index[1], batch.frag_edge_index[0]], dim=0)
        uv_edge_attr = torch.cat([batch.dangling_edge_attr, batch.dangling_edge_attr], dim=0)
        
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        dangling_out = out[batch.dangling_mask]
        dangling_out = self.dangling_projector(dangling_out)
        
        frag_out = global_mean_pool(out, batch.frag_batch)
        frag_out = self.projector(frag_out)
        frag_out = torch.repeat_interleave(frag_out, batch.frag_num_nodes, dim=0)
        frag_out = frag_out[batch.dangling_mask]
        
        out = dangling_out + frag_out
        
        out0 = out[u_index]
        out0 = self.predict(out0, uv_edge_attr)
        features0 = torch.nn.functional.normalize(out0, p=2, dim=1)

        out1 = out[v_index]
        features1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        features2 = torch.roll(features1, shifts=1, dims=0)

        pos_logits = self.classifier(torch.max(features0, features1))
        neg_logits = self.classifier(torch.max(features0, features2))                    

        logits = torch.cat([pos_logits, neg_logits], dim=0).squeeze(1)
        
        labels = torch.cat(
            [torch.ones(pos_logits.size(0)), torch.zeros(neg_logits.size(0))], dim=0
        ).to(0)

        return logits, labels
    
    def predict(self, x, edge_attr):
        edge_embeddings = (
            self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        )
        out = x + edge_embeddings
        out = self.predictor(out)
        
        return out
    
    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.eq(pred > 0, target > 0.5)).long()) / pred.size(0)
        return acc    
    
    
