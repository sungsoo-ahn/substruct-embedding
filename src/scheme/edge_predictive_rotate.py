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
        
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(2*self.emb_dim, 2*self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.emb_dim, self.emb_dim),
        )
            
        hidden_dim = self.emb_dim // 2

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, hidden_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, hidden_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        
        
        self.gamma = 1.0
        self.epsilon = 2.0
        self.embedding_range = (self.gamma + self.epsilon) / hidden_dim

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

        out0 = torch.cat([frag_out0, dangling_out0], dim=1)
        out1 = torch.cat([frag_out1, dangling_out1], dim=1)
        out2 = torch.cat([frag_out2, dangling_out2], dim=1)

        out0 = self.projector(out0)
        out1 = self.projector(out1)
        out2 = self.projector(out2)
        
        pos_score = self.compute_score(out0, edge_embeddings, out1)
        neg_score = self.compute_score(out0, edge_embeddings, out2)
                
        logits = torch.cat([pos_score, neg_score], dim=0)
        
        labels = torch.cat(
            [torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))], dim=0
        ).to(0)

        return logits, labels
    
    def criterion(self, pred, target):
        loss = -torch.nn.functional.logsigmoid((target * 2 - 1) * pred)
        return loss.mean()
    
    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.eq(pred > 0, target > 0.5)).long()) / pred.size(0)
        return acc

    def compute_score(self, head, relation, tail):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)

        phase_relation = relation / (self.embedding_range / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma - score.sum(dim = 1)
        return score