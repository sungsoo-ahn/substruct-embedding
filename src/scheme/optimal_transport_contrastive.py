import random
import numpy as np
import torch
from torch_geometric.nn import global_add_pool

from scheme.contrastive import ContrastiveScheme
from model import NodeEncoder

def log_sinkhorn_iterations(self, Z, log_mu, log_nu, num_iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(num_iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)
        
    return Z + u.unsqueeze(1) + v.unsqueeze(0)

def compute_log_coupling(score, num_sinkhorn_iters, device):
    m, n = score.shape
    one = score.new_tensor(1)
    ms, ns = (m*one).to(device), (n*one).to(device)
        
    norm = - (ms + ns).log()
    log_mu = norm.expand(m)
    log_nu = norm.expand(n)

    log_coupling = self.log_sinkhorn_iterations(score, log_mu, log_nu, num_sinkhorn_iters)
    log_coupling = log_coupling - norm
    
    return log_coupling


class OptimalTransportContrastiveScheme(ContrastiveScheme):
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, temperature, num_sinkhorn_iters):
        self.temperature = temperature
        self.num_sinkhorn_iters = num_sinkhorn_iters
 
    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        models = torch.nn.ModuleDict({"encoder": encoder, "head": head})
        return models
    
    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)

        out = models["encoder"](batch.x, batch.edge_index, batch.edge_attr)
        out = models["head"](out)
        out = torch.nn.functional.normalize(out, dim=1)
        nodewise_score = torch.matmul(out, out.T)
        
        log_coupling = compute_log_coupling(nodewise_score, self.num_sinkhorn_iters, device)
        coupling = log_coupling.exp()
        
        normalizer = global_add_pool(coupling, batch.batch)
        normalizer = global_add_pool(normalizer.T, batch.batch)
        
        graphwise_score = nodewise_score * renormalized_coupling
        graphwise_score = global_add_pool(graphwise_score, batch.batch)
        graphwise_score = global_add_pool(graphwise_score.T, batch.batch)
        graphwise_score = graphwise_score / normalizer
        
        logis, labels = self.get_logits_and_labels(graphwise_score)
        loss = self.criterion(logits, labels)       
        acc = compute_accuracy(logits, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        coupling_label = torch.arange(batch.x.size(0)).to(device)
        coupling_acc = compute_accuracy(coupling, coupling_label)

        statistics = {"loss": loss.detach(), "acc": acc, "coupling_acc": coupling_acc}

        return statistics