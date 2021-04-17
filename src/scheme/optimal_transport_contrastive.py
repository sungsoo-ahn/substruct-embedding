import random
import numpy as np
import torch
from torch_geometric.nn import global_add_pool, global_max_pool

from scheme.contrastive import ContrastiveScheme
from model import NodeEncoder
from util import compute_accuracy



def log_sinkhorn_iterations(Z, log_mu, log_nu, batch, batch_num_nodes, num_iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(num_iters):
        tmp = Z + torch.repeat_interleave(v, batch_num_nodes, dim=0)
        tmp = global_add_pool(tmp.T.exp(), batch).log().T        
        u = log_mu - tmp

        tmp = Z + torch.repeat_interleave(u, batch_num_nodes, dim=1)
        tmp = global_add_pool(tmp.exp(), batch).log()
        v = log_nu - tmp

    return (
        Z
        + torch.repeat_interleave(u, batch_num_nodes, dim=1)
        + torch.repeat_interleave(v, batch_num_nodes, dim=0)
    )


def compute_log_coupling(score, batch, batch_num_nodes, num_sinkhorn_iters, device):
    batch_size = batch_num_nodes.size(0)

    norm = -(batch_num_nodes.unsqueeze(0) + batch_num_nodes.unsqueeze(1)).float().log()
    log_mu = torch.repeat_interleave(norm, batch_num_nodes, dim=0) #batch_num_nodes x batch_size
    log_nu = torch.repeat_interleave(norm, batch_num_nodes, dim=1) #batch_size x batch_num_nodes

    log_coupling = log_sinkhorn_iterations(
        score, log_mu, log_nu, batch, batch_num_nodes, num_sinkhorn_iters
    )
    norm = torch.repeat_interleave(norm, batch_num_nodes, dim=0)
    norm = torch.repeat_interleave(norm, batch_num_nodes, dim=1)
    log_coupling = log_coupling - norm

    return log_coupling

"""
def org_log_sinkhorn_iterations(Z, log_mu, log_nu, batch, batch_num_nodes, num_iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(num_iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1) #batch_size x batch_size + 1 x batch_size
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)

    return Z + u.unsqueeze(1) + v.unsqueeze(0)

def org_compute_log_coupling(score, batch, batch_num_nodes, num_sinkhorn_iters, device):
    batch_size = batch_num_nodes.size(0)

    m, n = score.shape
    one = score.new_tensor(1)
    ms, ns = (m*one).to(device), (n*one).to(device)

    norm = - (ms + ns).log()
    log_mu = norm.expand(m)
    log_nu = norm.expand(n)

    log_coupling = org_log_sinkhorn_iterations(
        score, log_mu, log_nu, batch, batch_num_nodes, num_sinkhorn_iters
    )
    log_coupling = log_coupling - norm

    return log_coupling
"""

class OptimalTransportContrastiveScheme(ContrastiveScheme):
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, transform, temperature, num_sinkhorn_iters):
        super(OptimalTransportContrastiveScheme, self).__init__(transform, temperature)
        self.num_sinkhorn_iters = num_sinkhorn_iters

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim),
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

        #with torch.no_grad():
        log_coupling = compute_log_coupling(
            nodewise_score, batch.batch, batch.batch_num_nodes, self.num_sinkhorn_iters, device
        )
        
        coupling = log_coupling.exp()
        #normalizer = global_add_pool(coupling, batch.batch)
        #normalizer = global_add_pool(normalizer.T, batch.batch)
        #normalizer = torch.repeat_interleave(normalizer, batch.batch_num_nodes, dim=0)
        #normalizer = torch.repeat_interleave(normalizer, batch.batch_num_nodes, dim=1)
        #coupling = coupling / normalizer

        coupling_label = torch.arange(coupling.size(0)).to(device)
        coupling_acc = compute_accuracy(coupling, coupling_label)

        graphwise_score = nodewise_score * coupling
        graphwise_score = global_add_pool(graphwise_score, batch.batch)
        graphwise_score = global_add_pool(graphwise_score.T, batch.batch)
        graphwise_score = graphwise_score

        logits, labels = self.get_logits_and_labels(graphwise_score, device)
        loss = self.criterion(logits, labels)
        acc = compute_accuracy(logits, labels)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 0.5)
        optim.step()

        statistics = {"loss": loss.detach(), "acc": acc, "coupling_acc": coupling_acc}

        return statistics
