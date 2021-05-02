import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

from model import GNN
from scheme.util import compute_accuracy
from data.collate import collate

def collate_scaffold(data_list_):
    data_list = []
    scaffold_data_list = []    
    scaffold_y_list = []
    graph_contrast_labels = []
    for data_ in data_list_:
        data = Data(
            x=data_.x, 
            edge_index=data_.edge_index, 
            edge_attr=data_.edge_attr
            )
        data.scaffold_y = data_.scaffold_y
        data.scaffold_mask = data_.scaffold_mask
        scaffold_data = Data(
            x=data_.scaffold_x, 
            edge_index=data_.scaffold_edge_index, 
            edge_attr=data_.scaffold_edge_attr
            )
        
        scaffold_y = data.scaffold_y.item()

        data_list.append(data)
        if scaffold_y not in scaffold_y_list:
            scaffold_y_list.append(scaffold_y)
            scaffold_data_list.append(data)
            
        graph_contrast_labels.append(scaffold_y_list.index(scaffold_y))
    
    batch = collate(data_list)
    scaffold_batch = collate(scaffold_data_list)
    batch.graph_contrast_labels = torch.LongTensor(graph_contrast_labels)
    
    offsets = [0] + torch.cumsum(scaffold_batch.batch_num_nodes, dim=0).tolist()
    node_contrast_labels = [
        torch.arange(offsets[idx], offsets[idx+1]) for idx in graph_contrast_labels
        ]
    batch.node_contrast_labels = torch.cat(node_contrast_labels, dim=0)
    
    return batch, scaffold_batch

class ScaffoldGraphContrastModel(torch.nn.Module):
    def __init__(self):
        super(ScaffoldGraphContrastModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.temperature = 0.1
        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )            
        
        self.mask_scaffold_features = True
        
    def compute_logits_and_labels(self, batch, scaffold_batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        node_features = self.projector(out)
        if self.mask_scaffold_features:
            scaffold_mask = (batch.scaffold_mask > 0.5)
            graph_features = global_mean_pool(
                node_features[scaffold_mask], batch.batch[scaffold_mask]
                )
        else:
            graph_features = global_mean_pool(node_features, batch.batch)
        
        out = self.encoder(scaffold_batch.x, scaffold_batch.edge_index, scaffold_batch.edge_attr)
        scaffold_node_features = self.projector(out)
        scaffold_graph_features = global_mean_pool(scaffold_node_features, scaffold_batch.batch)
                
        # compute logits
        graph_features = torch.nn.functional.normalize(graph_features, p=2, dim=1)
        scaffold_graph_features = torch.nn.functional.normalize(scaffold_graph_features, p=2, dim=1)
        logits = (torch.matmul(graph_features, scaffold_graph_features.t()) / self.temperature)
        labels = batch.graph_contrast_labels
                
        return logits, labels
    
class ScaffoldNodeContrastModel(ScaffoldGraphContrastModel):
    def compute_logits_and_labels(self, batch, scaffold_batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        node_features = self.projector(out)
        scaffold_mask = (batch.scaffold_mask > 0.5)
        node_features = node_features[scaffold_mask]
        
        out = self.encoder(scaffold_batch.x, scaffold_batch.edge_index, scaffold_batch.edge_attr)
        scaffold_node_features = self.projector(out)
                
        # compute logits
        node_features = torch.nn.functional.normalize(node_features, p=2, dim=1)
        scaffold_node_features = torch.nn.functional.normalize(scaffold_node_features, p=2, dim=1)
        logits = (torch.matmul(node_features, scaffold_node_features.t()) / self.temperature)
        labels = batch.node_contrast_labels
                
        return logits, labels

class ScaffoldContrastScheme:
    def train_step(self, batch, scaffold_batch, model, optim):
        model.train()
        batch = batch.to(0)
        scaffold_batch = scaffold_batch.to(0)
        
        logits, labels = model.compute_logits_and_labels(batch, scaffold_batch)
        loss = model.criterion(logits, labels)
        acc = compute_accuracy(logits, labels)
        
        statistics = dict()
        statistics["loss"] = loss.detach()
        statistics["acc"] = acc

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        return statistics

    def eval_epoch(self, loader, model):
        model.eval()
        avg_loss = 0.0
        avg_acc = 0.0
        for batch, scaffold_batch in loader:
            batch = batch.to(0)
            scaffold_batch = scaffold_batch.to(0)
            
            with torch.no_grad():
                logits, labels = model.compute_logits_and_labels(batch, scaffold_batch)
                loss = model.criterion(logits, labels)
                acc = compute_accuracy(logits, labels)
            
            avg_loss += loss / len(loader)
            avg_acc += acc / len(loader)
            
        statistics = dict()
        statistics["loss"] = avg_loss
        statistics["acc"] = avg_acc
        
        return statistics
    
            
            
        
            
            