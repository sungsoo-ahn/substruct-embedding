import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import GNN
from scheme.util import compute_accuracy

class MaskContrastModel(torch.nn.Module):
    def __init__(self):
        super(MaskContrastModel, self).__init__()
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
        
        self.mask_features = True
        
    def compute_logits_and_labels(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        if self.mask_features:
            out = out[batch.mask]
        
        features = self.projector(out)
        if self.mask_features:
            labels = batch.y[batch.mask]
                
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
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = -self.temperature * mean_log_prob_pos.mean()
        
        return loss

class RobustMaskContrastModel(MaskContrastModel):
    def __init__(self, gce_coef):
        super(RobustMaskContrastModel, self).__init__()
        self.gce_coef = gce_coef

    def criterion(self, logits, labels):
        #
        labels = labels.contiguous().view(-1, 1)
        numer_mask = torch.eq(labels, labels.t()).float().cuda()
        
        # mask-out self-contrast cases
        denom_mask = torch.scatter(
            torch.ones_like(numer_mask), 1, torch.arange(numer_mask.size(0)).view(-1, 1).cuda(), 0
        )
        numer_mask = numer_mask * denom_mask

        # mask out where numerator is zero
        invalid_numer_mask = (torch.sum(numer_mask, dim=1) > 0)
        logits = logits[invalid_numer_mask]
        numer_mask = numer_mask[invalid_numer_mask]
        denom_mask = denom_mask[invalid_numer_mask]
        
        # compute log_prob
        exp_logits = torch.exp(logits) * denom_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        prob = (log_prob * self.gce_coef).exp()
        # compute mean of log-likelihood over positive
        #probs = numer_exp_logits / denom_exp_logits.sum(1, keepdim=True)
        loss = -((numer_mask * prob).sum(dim=1) / numer_mask.sum(dim=1)).log().mean()
        loss *= self.temperature / self.gce_coef
        return loss

class MaskBalancedContrastModel(MaskContrastModel):
    def __init__(self, balance_k):
        super(MaskBalancedContrastModel, self).__init__()
        self.balance_k = balance_k
        
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
        
        perm = torch.randperm(mask.size(1))
        mask = mask[:, perm]
        logits = logits[:, perm]
        balance_mask = (torch.cumsum(mask, dim=1) < self.balance_k + 1).float()
        mask = balance_mask * mask
        
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
