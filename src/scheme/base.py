import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool

def compute_accuracy(pred, target):
    acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
    return acc


def get_contrastive_logits_and_labels(features):
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
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    return logits, labels


class GraphContrastiveModel(torch.nn.Module):
    def __init__(self, zero_pool=False):
        super(GraphContrastiveModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
        self.criterion = torch.nn.CrossEntropyLoss()
        self.zero_pool = zero_pool
        
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector0 = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
        
        if self.zero_pool:
            self.projector1 = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim, self.emb_dim),
            )

    def compute_logits_and_labels(self, batch0, batch1):
        out = self.encoder(batch0.x, batch0.edge_index, batch0.edge_attr)
        out0 = global_mean_pool(out, batch0.batch)
        out0 = self.projector0(out0)
        if self.zero_pool:
            out1 = out[batch0.x[:, 2] == 1]
            out1 = self.projector1(out1)
            out = out0 + out1
        else:
            out = out0
        
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)
        
        out = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        out0 = global_mean_pool(out, batch1.batch)
        out0 = self.projector0(out0)
        if self.zero_pool:
            out1 = out[batch1.x[:, 2] == 1]
            out1 = self.projector1(out1)
            out = out0 + out1
        else:
            out = out0
            
        features1 = torch.nn.functional.normalize(out, p=2, dim=1)

        features = torch.cat([features0, features1])
        logits, labels = get_contrastive_logits_and_labels(features)
        
        logits /= self.contrastive_temperature

        return logits, labels
    
class NodeContrastiveModel(GraphContrastiveModel):
    def compute_logits_and_labels(self, batch0, batch1):
        out = self.encoder(batch0.x, batch0.edge_index, batch0.edge_attr)
        out = out[batch0.x[:, 0] > 0]
        out = self.projector(out)
        features0 = torch.nn.functional.normalize(out, p=2, dim=1)
        
        out = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        out = out[batch1.x[:, 0] > 0]
        out = self.projector(out)
        features1 = torch.nn.functional.normalize(out, p=2, dim=1)
        
        features = torch.cat([features0, features1])
        logits, labels = get_contrastive_logits_and_labels(features)
        
        logits /= self.contrastive_temperature

        return logits, labels

class BaseScheme:
    def train_step(self, batch0, batch1, model, optim):
        model.train()
        batch0 = batch0.to(0)
        batch1 = batch1.to(0)

        logits, labels = model.compute_logits_and_labels(batch0, batch1)

        statistics = dict()
        loss = model.criterion(logits, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        acc = compute_accuracy(logits, labels)
        statistics["loss"] = loss.detach()
        statistics["acc"] = acc

        return statistics