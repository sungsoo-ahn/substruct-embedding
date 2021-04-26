import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering



class MoCoModel(torch.nn.Module):
    def __init__(self, pool_type):
        super(MoCoModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.temperature = 0.07
        self.ema_rate = 0.999
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pool_type = pool_type
    
        self.queue_size = 65536
    
        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )            
        
        self.ema_encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.ema_projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )
        
        for param, ema_param in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            ema_param.data.copy_(param.data)
            ema_param.requires_grad = False
        
        for param, ema_param in zip(self.projector.parameters(), self.ema_projector.parameters()):
            ema_param.data.copy_(param.data)
            ema_param.requires_grad = False
            
        self.register_buffer("queue", torch.randn(self.emb_dim, self.queue_size))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_ema_encoder(self):
        for param, ema_param in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)
    
        for param, ema_param in zip(self.projector.parameters(), self.ema_projector.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        #assert self.queue_size % batch_size == 0  # for simplicity

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        next_ptr = min(self.queue_size, ptr + batch_size)
        self.queue[:, ptr:next_ptr] = keys[:next_ptr-ptr, :].T
        ptr = next_ptr % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def compute_graph_features(self, batch):        
        out = self.ema_encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = self.ema_projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)
        graph_features = global_mean_pool(node_features, batch.batch)
        graph_features = torch.nn.functional.normalize(graph_features, p=2, dim=1)

        return graph_features
    
    def compute_logits_and_labels(self, query_batch, key_batch):
        batch_size = query_batch.batch_size
        batch_num_nodes = key_batch.batch_num_nodes
        
        out = self.encoder(query_batch.x, query_batch.edge_index, query_batch.edge_attr)
        out = self.projector(out)
        out = self.pool_graphs(out, query_batch)
        query_node_features = torch.nn.functional.normalize(out, p=2, dim=1)
        
        with torch.no_grad():
            self.update_ema_encoder()
            out = self.ema_encoder(query_batch.x, query_batch.edge_index, query_batch.edge_attr)
            out = self.ema_projector(out)
            out = self.pool_graphs(out, key_batch)
            key_node_features = torch.nn.functional.normalize(out, p=2, dim=1)
            
        inter_logits = torch.einsum('nc,kc->nk', [query_node_features, key_node_features])
        intra_logits =  torch.einsum(
            'nc,ck->nk', [query_node_features, self.queue.clone().detach()]
            )
        
        logits = torch.cat([inter_logits, intra_logits], dim=1)
        logits /= self.temperature
        labels = torch.arange(logits.size(0), dtype=torch.long).cuda()

        self.dequeue_and_enqueue(key_node_features)
        
        logits_and_labels = {"contrastive": [logits, labels]}
        
        return logits_and_labels

    def pool_graphs(self, h, batch):
        if self.pool_type == "mean":
            out = global_mean_pool(h, batch.batch)
        elif self.pool_type == "mask":
            out = h[batch.pool_mask]
        
        return out

class MoCoScheme:
    def __init__(self):
        pass
     
    def train_step(self, batch0, batch1, model, optim):
        model.train()
        batch0 = batch0.to(0)
        batch1 = batch1.to(0)
        
        logits_and_labels = model.compute_logits_and_labels(batch0, batch1)        
        
        loss_cum = 0.0
        statistics = dict()
        for idx, key in enumerate(logits_and_labels):
            logits, labels = logits_and_labels[key]
            loss = model.criterion(logits, labels)
            acc = compute_accuracy(logits, labels)                
            
            loss_cum += loss
            
            statistics[f"loss/{key}"] = loss.detach()
            statistics[f"acc/{key}"] = acc
        
        statistics[f"loss/total"] = loss_cum.detach()
        
        if "subgraph_mask" in batch0.keys:
            statistics[f"stats/subgraph_ratio"] = (
                torch.sum(batch0.subgraph_mask.float()) / batch0.x.size(0)
                )
        
        optim.zero_grad()
        loss_cum.backward()
        optim.step()
        
        return statistics