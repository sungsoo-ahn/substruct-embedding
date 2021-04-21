import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering



class DoubleMoCoModel(torch.nn.Module):
    def __init__(self):
        super(DoubleMoCoModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.temperature = 0.07
        self.ema_rate = 0.995
        self.criterion = torch.nn.CrossEntropyLoss()
    
        self.queue_size = 65536
    
        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )
        
        self.sub_encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.sub_projector = torch.nn.Sequential(
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
        
        self.ema_sub_encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.ema_sub_projector = torch.nn.Sequential(
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
            
        for param, ema_param in zip(self.sub_encoder.parameters(), self.ema_sub_encoder.parameters()):
            ema_param.data.copy_(param.data)
            ema_param.requires_grad = False
        
        for param, ema_param in zip(self.sub_projector.parameters(), self.ema_sub_projector.parameters()):
            ema_param.data.copy_(param.data)
            ema_param.requires_grad = False

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue", torch.randn(self.emb_dim, self.queue_size))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("sub_queue", torch.randn(self.emb_dim, self.queue_size))
        self.sub_queue = torch.nn.functional.normalize(self.sub_queue, dim=0)
        
    @torch.no_grad()
    def update_ema_encoder(self):
        for param, ema_param in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)
    
        for param, ema_param in zip(self.projector.parameters(), self.ema_projector.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)

        for param, ema_param in zip(self.sub_encoder.parameters(), self.ema_sub_encoder.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)
    
        for param, ema_param in zip(self.sub_projector.parameters(), self.ema_sub_projector.parameters()):
            ema_param.data = ema_param.data * self.ema_rate + param.data * (1. - self.ema_rate)
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, sub_keys):
        batch_size = keys.size(0)
        assert self.queue_size % batch_size == 0  # for simplicity

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.sub_queue[:, ptr:ptr + batch_size] = sub_keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

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
        
        sample_node_indices = (
            (batch_num_nodes.float() * torch.rand(batch_size).cuda()).long() + 1
        )
        sample_node_indices = batch_num_nodes - sample_node_indices
        
        out = self.encoder(query_batch.x, query_batch.edge_index, query_batch.edge_attr)
        if "subgraph_mask" in query_batch.keys:
            out = out[query_batch.subgraph_mask]
        
        out = out[sample_node_indices]
        out = self.projector(out)
        query_node_features = torch.nn.functional.normalize(out, p=2, dim=1)
        
        out = self.sub_encoder(key_batch.x, key_batch.edge_index, key_batch.edge_attr)
        out = out[sample_node_indices]
        key_node_features = torch.nn.functional.normalize(out, p=2, dim=1)
        
        with torch.no_grad():
            self.update_ema_encoder()
            out = self.ema_encoder(query_batch.x, query_batch.edge_index, query_batch.edge_attr)
            if "subgraph_mask" in query_batch.keys:
                out = out[query_batch.subgraph_mask]
            
            out = self.ema_projector(out)
            ema_query_node_features = torch.nn.functional.normalize(out, p=2, dim=1)
            
            out = self.ema_sub_encoder(key_batch.x, key_batch.edge_index, key_batch.edge_attr)
            out = self.ema_sub_projector(out)
            ema_key_node_features = torch.nn.functional.normalize(out, p=2, dim=1)
            
            
        inter_logits = torch.einsum('nc,mc->nm', [query_node_features, ema_key_node_features])
        intra_logits =  torch.einsum(
            'nc,ck->nk', [query_node_features, self.queue.clone().detach()]
            )
        
        logits = torch.cat([inter_logits, intra_logits], dim=1)
        logits /= self.temperature
        labels = sample_node_indices
        
        sub_inter_logits = torch.einsum('nc,mc->nm', [key_node_features, ema_query_node_features])
        sub_intra_logits =  torch.einsum(
            'nc,ck->nk', [key_node_features, self.sub_queue.clone().detach()]
            )
        
        sub_logits = torch.cat([sub_inter_logits, sub_intra_logits], dim=1)
        sub_logits /= self.temperature
        sub_labels = sample_node_indices

        self.dequeue_and_enqueue(
            ema_key_node_features[sample_node_indices],
            ema_query_node_features[sample_node_indices], 
            )
        
        logits_and_labels = {
            "1_contrastive": [logits, labels],
            "2_sub_contrastive": [sub_logits, sub_labels]
            }
        
        return logits_and_labels
        

class DoubleMoCoScheme:
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
            
            statistics[f"0_loss/{key}"] = loss.detach()
            statistics[f"1_acc/{key}"] = acc
        
        statistics[f"0_loss/0_total"] = loss_cum.detach()
        
        if "subgraph_mask" in batch0.keys:
            statistics[f"stats/0_subgraph_ratio"] = (
                torch.sum(batch0.subgraph_mask.float()) / batch0.x.size(0)
                )
        
        optim.zero_grad()
        loss_cum.backward()
        optim.step()
        
        return statistics