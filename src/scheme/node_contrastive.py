import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels, run_clustering

class NodeContrastiveModel(torch.nn.Module):
    def __init__(self, use_linear_projector):
        super(NodeContrastiveModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.proj_dim = 100
        self.drop_rate = 0.0
        self.contrastive_temperature = 0.04
        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        
        if not use_linear_projector:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim, self.emb_dim),
            )
        else:
            self.projector = torch.nn.Linear(self.emb_dim, self.emb_dim)
        
        self.node_centroids = None
        self.node2cluster = None
        self.node_density = None
        
    def compute_graph_features(self, x, edge_index, edge_attr, batch):        
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)

        graph_features = global_mean_pool(node_features, batch)
        graph_features = torch.nn.functional.normalize(graph_features, p=2, dim=1)

        return graph_features
    
    def compute_logits_and_labels(
        self, x, edge_index, edge_attr, batch, pool_mask
        ):
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        node_features = torch.nn.functional.normalize(out, p=2, dim=1)

        logits_and_labels = dict()
        
        #node_mask = torch.bernoulli(torch.zeros(x.size(0) // 2), p=0.1).bool().to(x.device)
        #node_mask = torch.cat([node_mask, node_mask], axis=0)
        pooled_node_features = node_features[pool_mask]
        _ = get_contrastive_logits_and_labels(pooled_node_features)
        logits_node_contrastive, labels_node_contrastive = _
        logits_node_contrastive /= self.contrastive_temperature

        logits_and_labels["node_contrastive"] =[logits_node_contrastive, labels_node_contrastive]

        return logits_and_labels
    

class NodeContrastiveScheme:
    def __init__(self):
        self.contrastive_temperature = 0.04
        
        self.centroids = None
        self.node2cluster = None
        self.density = None
    
    def train_step(self, batch, model, optim, device):
        model.train()
        batch = batch.to(device)
        
        logits_and_labels = model.compute_logits_and_labels(
            batch.x, 
            batch.edge_index, 
            batch.edge_attr, 
            batch.batch,  
            batch.pool_mask,
            )
        
        loss_cum = 0.0
        statistics = dict()
        for key in logits_and_labels:
            logits, labels = logits_and_labels[key]
            loss = model.criterion(logits, labels)
            acc = compute_accuracy(logits, labels)                
            
            loss_cum += loss
            
            statistics[f"{key}/loss"] = loss.detach()
            statistics[f"{key}/acc"] = acc
        
        if len(logits_and_labels.keys()) > 0:
            optim.zero_grad()
            loss_cum.backward()
            optim.step()
        
        statistics["num_nodes"] = batch.x.size(0)
        statistics["num_pooled_nodes"] = batch.pool_mask.long().sum()
        
        return statistics