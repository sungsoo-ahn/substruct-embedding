import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import GNN
from scheme.util import compute_accuracy, get_contrastive_logits_and_labels

class MotifContrastiveModel(torch.nn.Module):
    def __init__(self):
        super(MotifContrastiveModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def compute_logits_and_labels(self, batch0, batch1):
        out = self.encoder(batch0.x, batch0.edge_index, batch0.edge_attr)
        out = global_mean_pool(out, batch0.batch)
        graph_features0 = torch.nn.functional.normalize(out, p=2, dim=1)
        out = self.projector(out)

        out = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        out = global_mean_pool(out, batch1.batch)
        out = self.projector(out)
        graph_features1 = torch.nn.functional.normalize(out, p=2, dim=1)

        logits = torch.mm(graph_features0, graph_features1.t())
        logits /= self.contrastive_temperature
        labels = torch.arange(logits.size(0)).to(0)

        logits_and_labels = {"motif_contrastive": [logits, labels]}

        return logits_and_labels


class MotifContrastiveScheme:
    def train_step(self, batch0, batch1, model, optim):
        model.train()
        batch0 = batch0.to(0)
        batch1 = batch1.to(0)

        logits_and_labels = model.compute_logits_and_labels(batch0, batch1)

        loss_cum = 0.0
        statistics = dict()
        for key in logits_and_labels:
            logits, labels = logits_and_labels[key]
            loss = model.criterion(logits, labels)
            acc = compute_accuracy(logits, labels)

            loss_cum += loss

            statistics[f"{key}/loss"] = loss.detach()
            statistics[f"{key}/acc"] = acc
            statistics[f"{key}/logit_size"] = logits.size(0)


        optim.zero_grad()
        loss_cum.backward()
        optim.step()

        return statistics