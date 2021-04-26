import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import NodeEncoder
from scheme.util import compute_accuracy

class MaskContrastModel(torch.nn.Module):
    def __init__(self, use_mlp):
        super(MaskContrastModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.temperature = 0.1
        self.criterion = torch.nn.CrossEntropyLoss()

        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        if use_mlp:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim, self.emb_dim)
            )            
        else:
            self.projector = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def compute_logits_and_labels(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = out[batch.mask]
        features = self.projector(out)
        labels = batch.y[batch.mask]
                
        # compute logits
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        logits = (torch.matmul(features, features.T) / self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        return logits, labels

    def criterion(self, logits, labels):
        #
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().cuda()
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(mask.size(0)).view(-1, 1).cuda(), 0
        )
        mask = mask * logits_mask

        maskmask = (torch.sum(mask, dim=1) > 0)
        mask = mask[maskmask]
        logits = logits[maskmask]
        logits_mask = logits_mask[maskmask]
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = -self.temperature * mean_log_prob_pos.mean()
        
        return loss

class MaskContrastScheme:
    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)

        logits, labels = model.compute_logits_and_labels(batch)
        loss = model.criterion(logits, labels)
        
        statistics = dict()
        statistics["loss"] = loss.detach()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        statistics["num_masked_nodes"] = batch.mask.long().sum()

        return statistics