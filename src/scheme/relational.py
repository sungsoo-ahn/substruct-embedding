import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool


class Model(torch.nn.Module):
    def __init__(self, aggr, use_relation):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
        self.criterion = torch.nn.CrossEntropyLoss()
        self.aggr = aggr
        self.use_relation = use_relation

        if self.use_relation:
            pred_dim = 5
        else:
            pred_dim = 2

        if self.aggr == "cat":
            self.feat_dim = 2*self.emb_dim
        else:
            self.feat_dim = self.emb_dim
            
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.feat_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),            
            torch.nn.Linear(self.emb_dim, pred_dim),
        )
        
        if self.use_relation:
            self.pos_labels = torch.tensor(
                [
                    [0, 2, 2, 1, 1, 1],
                    [2, 0, 3, 1, 1, 1],
                    [2, 3, 0, 1, 1, 1],
                    [1, 1, 1, 0, 2, 2],
                    [1, 1, 1, 2, 0, 3],
                    [1, 1, 1, 2, 3, 0],
                ], dtype=torch.long
            )
        else:
            self.pos_labels = torch.full((6, 6), 0, dtype=torch.long)
            
        self.neg_labels = torch.full((6, 6), pred_dim - 1, dtype=torch.long)

        self.pos_labels = self.pos_labels.to(0)
        self.neg_labels = self.neg_labels.to(0)
        
    def compute_features(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
        #out = self.projector(out)
        #out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out

        #out = self.projector(out)
        
    def compute_logits_and_labels(self, batch):
        batch = batch.to(0)
        features = self.compute_features(batch)        
        batch_size = int(features.size(0) / 6)

        features = features.view(batch_size, 6, self.emb_dim)
        features = torch.transpose(features, 0, 1)
        
        if self.aggr in ["plus", "minus", "max"]:
            features0 = features.unsqueeze(0)#.expand(6, 6, batch_size, self.emb_dim)
            features1 = features.unsqueeze(1)#.expand(6, 6, batch_size, self.emb_dim)
            features2 = torch.roll(features1, 1, dims=2)
            if self.aggr == "plus":
                pos_features = features0 + features1
                neg_features = features0 + features2      
            elif self.aggr == "minus":
                pos_features = features0 - features1
                neg_features = features0 - features2      
            if self.aggr == "max":
                pos_features = torch.max(features0, features1)
                neg_features = torch.max(features0, features2)
        
        elif self.aggr == "cat":
            features0 = features.unsqueeze(0).expand(6, 6, batch_size, self.emb_dim)
            features1 = features.unsqueeze(1).expand(6, 6, batch_size, self.emb_dim)
            features2 = torch.roll(features1, 1, dims=2)
            
            pos_features = torch.cat([features0, features1], dim=3)
            neg_features = torch.cat([features0, features2], dim=3)
        
        
        features = torch.cat([pos_features, neg_features], dim=2).view(-1, self.feat_dim)
        logits = self.classifier(features)
        
        pos_labels = self.pos_labels.view(6, 6, 1).expand(6, 6, batch_size)
        neg_labels = self.neg_labels.view(6, 6, 1).expand(6, 6, batch_size)
        labels = torch.cat([pos_labels, neg_labels], dim=2).view(-1)

        return logits, labels

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
