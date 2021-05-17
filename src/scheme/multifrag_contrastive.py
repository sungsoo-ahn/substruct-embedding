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

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.contrastive_temperature = 0.04

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        
        self.projector = build_projector(self.emb_dim)
        self.dangling_projector = build_projector(self.emb_dim)
        self.predictor = build_projector(self.emb_dim)

    def compute_logits_and_labels(self, batch):        
        batch = batch.to(0)
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        dangling_out = out[batch.dangling_mask]
        dangling_out = self.dangling_projector(dangling_out)
        
        frag_out = global_mean_pool(out, batch.frag_batch)
        frag_out = torch.repeat_interleave(frag_out, batch.frag_num_nodes, dim=0)
        frag_out = frag_out[batch.dangling_mask]
        frag_out = self.projector(frag_out)
        
        out = dangling_out + frag_out
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)
        out = self.predictor(out)
        features1 = torch.nn.functional.normalize(out, p=2, dim=1)

        logits = torch.matmul(features0, features1.t()) / self.contrastive_temperature
        targets = to_dense_adj(batch.dangling_edge_index)

        return logits, targets
    
    def criterion(self, logits, targets):
        mask = targets.float()        
        log_prob = logits - torch.logsumexp(logits, dim=1).unsqueeze(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()

        return loss

    def compute_accuracy(self, pred, target):
        acc = target.gather(1, torch.max(pred, dim=1)[1].view(-1, 1)).float().mean()
        return acc

