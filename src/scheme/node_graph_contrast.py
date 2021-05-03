import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import GNN
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels

class NodeGraphContrastiveModel(torch.nn.Module):
    def __init__(self, logit_sample_ratio):
        super(NodeGraphContrastiveModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.proj_dim = 100
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector = torch.nn.Linear(self.emb_dim, self.proj_dim)
                
        self.logit_sample_ratio = logit_sample_ratio
        
    def compute_logits_and_labels(
        self, x, edge_index, edge_attr, batch, 
        ):
        out = self.encoder(x, edge_index, edge_attr)
        out = self.projector(out)
        features_node = torch.nn.functional.normalize(out, p=2, dim=1)

        # Subsample nodes for fast computation
        node_mask = (torch.bernoulli(torch.zeros(x.size(0) // 2), p=self.logit_sample_ratio) > 0.5).to(0)
        node_mask = torch.cat([node_mask, node_mask], dim=0)
        sampled_feature_nodes = features_node[node_mask]
        
        _ = get_contrastive_logits_and_labels(sampled_feature_nodes)
        logits_node_contrastive, labels_node_contrastive = _
        logits_node_contrastive /= self.contrastive_temperature
        
        features_graph = global_mean_pool(features_node, batch)
        features_graph = torch.nn.functional.normalize(features_graph, p=2, dim=1)

        _ = get_contrastive_logits_and_labels(features_graph)
        logits_graph_contrastive, labels_graph_contrastive = _
        logits_graph_contrastive /= self.contrastive_temperature
        
        logits_and_labels = {
            "node_contrastive": [logits_node_contrastive, labels_node_contrastive],
            "graph_contrastive": [logits_graph_contrastive, labels_graph_contrastive],
        }
        
        return logits_and_labels
    

class NodeGraphContrastiveScheme:
    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)
        
        logits_and_labels = model.compute_logits_and_labels(
            batch.x, 
            batch.edge_index, 
            batch.edge_attr, 
            batch.batch, 
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
            
        optim.zero_grad()
        loss_cum.backward()
        optim.step()
        
        #model.update_ema_encoder()

        return statistics