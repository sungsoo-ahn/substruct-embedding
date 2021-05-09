import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )

    def compute_features(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
        out = self.projector(out)
        features = torch.nn.functional.normalize(out, p=2, dim=1)
        
        return features
    
    def compute_logits_and_labels(self, batchs):
        batch0, batch1 = batchs
        batch0 = batch0.to(0)
        batch1 = batch1.to(0)

        features0 = self.compute_features(batch0)
        features1 = self.compute_features(batch1)
        
        features = torch.cat([features0, features1])
        
        result = dict()
        result["sample"] = self.get_logits_and_labels(features)
            
        return result

    def get_logits_and_labels(self, features):
        similarity_matrix = torch.matmul(features, features.t())

        batch_size = similarity_matrix.size(0) // 2

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        mask = (torch.eye(labels.shape[0]) > 0.5).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[(labels > 0.5)].view(labels.shape[0], -1)
        negatives = similarity_matrix[(labels < 0.5)].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.contrastive_temperature
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        return logits, labels