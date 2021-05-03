import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import GNN
from scheme.util import compute_accuracy

class GroupContrastModel(torch.nn.Module):
    def __init__(self, reweight, drop_rate):
        super(GroupContrastModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = drop_rate
        self.temperature = 0.01

        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim)
        )

        self.reweight = reweight

    def compute_logits_and_labels(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = out[batch.group_y > 0]
        features = self.projector(out)

        labels = batch.group_y[batch.group_y > 0]

        # compute logits
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        logits = (torch.matmul(features, features.t()) / self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        return logits, labels

    def criterion(self, logits, labels):
        #
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().cuda()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(mask.size(0)).view(-1, 1).cuda(), 0
        )
        mask = mask * logits_mask

        maskmask = (torch.sum(mask, dim=1) > 0)
        mask = mask[maskmask]
        logits = logits[maskmask]
        logits_mask = logits_mask[maskmask]

        if self.reweight:
            pos_logits_mask = logits_mask * mask
            neg_logits_mask = logits_mask * (1 - mask)

            logits = (
                (logits - pos_logits_mask.sum(1, keepdim=True).log()) * pos_logits_mask
                + (logits - neg_logits_mask.sum(1, keepdim=True).log()) * neg_logits_mask
                )

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -self.temperature * mean_log_prob_pos.mean()

        return loss

class GroupContrastScheme:
    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)

        logits, labels = model.compute_logits_and_labels(batch)
        loss = model.criterion(logits, labels)

        statistics = dict()
        statistics["loss"] = loss.detach()
        statistics["num_labels"] = labels.max() + 1

        optim.zero_grad()
        loss.backward()
        optim.step()

        return statistics

    def eval_epoch(self, loader, model):
        model.eval()
        avg_loss = 0.0
        for batch in loader:
            batch = batch.to(0)
            with torch.no_grad():
                logits, labels = model.compute_logits_and_labels(batch)
                loss = model.criterion(logits, labels)

            avg_loss += loss / len(loader)

        statistics = dict()
        statistics["loss"] = loss

        return statistics