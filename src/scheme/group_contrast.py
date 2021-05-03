import random
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from model import GNN
from scheme.util import compute_accuracy

class GroupContrastModel(torch.nn.Module):
    def __init__(self, self_contrast, atom_contrast, group_contrast, logit_sample_ratio, drop_rate):
        super(GroupContrastModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = drop_rate
        self.temperature = 0.04
        
        self.atom_contrast = atom_contrast
        self.group_contrast = group_contrast
        self.self_contrast = self_contrast
        
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)

        if self.atom_contrast:
            self.atom_projector = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim, self.emb_dim)
            )           
        
        if self.group_contrast:
            self.group_projector = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim, self.emb_dim)
            )
        
        if self.self_contrast:
            self.self_projector = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim, self.emb_dim)
            )

        self.logit_sample_ratio = logit_sample_ratio

    def compute_logits_and_labels(self, batch0, batch1):
        out0 = self.encoder(batch0.x, batch0.edge_index, batch0.edge_attr)
        out1 = self.encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        out = torch.cat([out0, out1], dim=0)
                
        logits_dict = dict()
        labels_dict = dict()
        
        if self.atom_contrast:
            atom_labels = torch.cat([batch0.atom_y, batch1.atom_y], dim=0)
            atom_features = self.atom_projector(out)
            atom_features = torch.nn.functional.normalize(atom_features, p=2, dim=1)
            
            if self.logit_sample_ratio < 1.0:
                mask0 = torch.rand(batch0.x.size(0)).to(0) < self.logit_sample_ratio
                mask = torch.cat([mask0, mask0], dim=0)
                atom_features = atom_features[mask]
                atom_labels = atom_labels[mask]
                
            atom_logits = torch.matmul(atom_features, atom_features.t()) / self.temperature
            
            logits_dict["atom"] = atom_logits
            labels_dict["atom"] = atom_labels
        
        if self.group_contrast:
            group_labels = torch.cat([batch0.group_y, batch1.group_y], dim=0)
            group_labels = group_labels[group_labels > -1]
            
            group_features = self.group_projector(out[group_labels > -1])
            group_features = torch.nn.functional.normalize(group_features, p=2, dim=1)
            
            if self.logit_sample_ratio < 1.0:
                mask0 = torch.rand(batch0.x.size(0)).to(0) < self.logit_sample_ratio
                mask = torch.cat([mask0, mask0], dim=0)
                group_features = group_features[mask]
                group_labels = group_labels[mask]
            
            group_logits = torch.matmul(group_features, group_features.t()) / self.temperature
            
            logits_dict["group"] = group_logits
            labels_dict["group"] = group_labels
            
        if self.self_contrast:
            self_labels = torch.cat(
                [torch.arange(batch0.x.size(0)), torch.arange(batch0.x.size(0))], dim=0
                ).to(0)
            
            self_features = self.self_projector(out)
            self_features = torch.nn.functional.normalize(self_features, p=2, dim=1)

            if self.logit_sample_ratio < 1.0:
                mask0 = torch.rand(batch0.x.size(0)).to(0) < self.logit_sample_ratio
                mask = torch.cat([mask0, mask0], dim=0)
                self_features = self_features[mask]
                self_labels = self_labels[mask]

            self_logits = torch.matmul(self_features, self_features.t()) / self.temperature
            
            logits_dict["self"] = self_logits
            labels_dict["self"] = self_labels
                    
        return logits_dict, labels_dict

    def criterion(self, logits, labels):
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

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

class GroupContrastScheme:
    def train_step(self, batch0, batch1, model, optim):
        model.train()
        batch0 = batch0.to(0)
        batch1 = batch1.to(0)

        logits_dict, labels_dict = model.compute_logits_and_labels(batch0, batch1)
        
        statistics = dict()
        loss = 0.0
        for key in logits_dict:
            logits = logits_dict[key]
            labels = labels_dict[key]
            key_loss = model.criterion(logits, labels)
            
            loss += key_loss
            
            statistics[f"{key}/loss"] = key_loss.detach()

        statistics[f"total/loss"] = loss.detach()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        return statistics