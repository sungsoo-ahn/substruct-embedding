import random
import torch
from model import NodeEncoder
from torch_geometric.data import Data
import numpy as np
import sys, os
from torch_geometric.nn import global_add_pool


def subgraph(x, edge_index, edge_attr, sub_num):
    node_num, _ = x.size()
    _, edge_num = edge_index.size()

    edge_index_np = edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index_np[1][edge_index_np[0] == idx_sub[0]]])
    
    count = 0
    while len(idx_sub) < sub_num:
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(
            set([n for n in edge_index_np[1][edge_index_np[0] == idx_sub[-1]]])
            )
                

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}
    edge_mask = np.array(
        [
            n
            for n in range(edge_num)
            if (edge_index_np[0, n] in idx_nondrop and edge_index_np[1, n] in idx_nondrop)
        ]
    )

    edge_index = edge_index.numpy()
    edge_index = [
        [idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]]
        for n in range(edge_num)
        if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)
    ]
    try:
        edge_index = torch.tensor(edge_index).transpose_(0, 1)
        x = x[idx_nondrop]
        edge_attr = edge_attr[edge_mask]
    except:
        return None, None, None

    return x, edge_index, edge_attr, idx_nondrop


class OptimalTransportContrastiveScheme:
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, sub_num, temperature):
        self.sub_num = sub_num
        self.temperature = temperature

    def transform(self, data):
        if data.x.size(0) < self.sub_num:
            return None
                   
        x0, edge_index0, edge_attr0 = data.x, data.edge_index, data.edge_attr
        x1, edge_index1, edge_attr1, mask_node_idxs = subgraph(
            x0.clone(), edge_index0.clone(), edge_attr0.clone(), self.sub_num
        )
        if x1 is None:
            return None
                
        x0[mask_node_idxs] = 0
        
        mask = torch.zeros(x0.size(0), dtype=torch.bool)
        mask[mask_node_idxs] = True


        data = Data()
        data.x0 = x0
        data.edge_index0 = edge_index0
        data.edge_attr0 = edge_attr0
        data.x1 = x1
        data.edge_index1 = edge_index1
        data.edge_attr1 = edge_attr1
        data.mask = mask

        return data

    def sample_node_indices(self, data):
        num_nodes = data.x.size()[0]
        sample_size = int(num_nodes * self.node_mask_rate + 1)
        node_indices = list(random.sample(range(num_nodes), sample_size))
        return node_indices

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim),
        )
        models = torch.nn.ModuleDict({"encoder": encoder, "head": head})
        return models

    def get_logits_and_labels(self, similarity_matrix, temperature, device):
        batch_size = similarity_matrix.size(0) // 2

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        return logits, labels

    def get_similarity_matrix(self, node_dist_matrix, num_iters):
        node_dist_cols = torch.split(node_dist_matrix, self.sub_num, dim=0)
        batch_size = len(node_dist_cols)
        node_dist_rows = [
            torch.split(node_dist_col, self.sub_num, dim=1) for node_dist_col in node_dist_cols
            ]
        node_dists = [node_dist for node_dist_row in node_dist_rows for node_dist in node_dist_row]
        node_dists = torch.stack(node_dists, dim=0)

        weights = self.log_optimal_transport(node_dists, num_iters)
        
        similarities = torch.sum(torch.sum(weights * node_dists, dim=2), dim=1)
        similarities = similarities.view(batch_size, batch_size)
        return similarities


    def log_sinkhorn_iterations(self, Z, log_mu, log_nu, iters):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)


    def log_optimal_transport(self, scores, iters):
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)

        """
        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                            torch.cat([bins1, alpha], -1)], 1)
        """
        
        couplings = scores
        
        norm = - (ms + ns).log()
        log_mu = norm.expand(m)
        log_nu = norm.expand(n)
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm
        return torch.exp(Z)

    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
        return acc

    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)

        emb0 = models["encoder"](batch.x0, batch.edge_index0, batch.edge_attr0)
        emb0 = models["head"](emb0[batch.mask])
        emb1 = models["encoder"](batch.x1, batch.edge_index1, batch.edge_attr1)
        emb1 = models["head"](emb1)

        emb0 = torch.nn.functional.normalize(emb0, dim=1)
        emb1 = torch.nn.functional.normalize(emb1, dim=1)
        node_dist_matrix = -torch.matmul(emb0, emb1.T)
        
        logits = self.get_similarity_matrix(node_dist_matrix, 100) / self.temperature
        labels = torch.arange(batch.batch_size).to(device)
        loss = self.criterion(logits, labels)

        with torch.no_grad():
            acc = self.compute_accuracy(logits, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        statistics = {"loss": loss.detach(), "acc": acc}

        return statistics

    @staticmethod
    def collate_fn(data_list):
        data_list = [data for data in data_list if data is not None]
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = Data()

        for key in keys:
            batch[key] = []

        batch.batch0 = []
        batch.batch_num_nodes0 = []
        batch.batch1 = []
        batch.batch_num_nodes1 = []
        batch.batch_size = len(data_list)

        cumsum_node0 = 0
        cumsum_node1 = 0
        for i, data in enumerate(data_list):
            num_nodes0 = data.x0.size(0)
            batch.batch0.append(torch.full((num_nodes0,), i, dtype=torch.long))
            batch.batch_num_nodes0.append(torch.LongTensor([num_nodes0]))

            num_nodes1 = data.x1.size(0)
            batch.batch1.append(torch.full((num_nodes1,), i, dtype=torch.long))
            batch.batch_num_nodes1.append(torch.LongTensor([num_nodes1]))

            for key in keys:
                item = data[key]
                if key in ["edge_index0"]:
                    item = item + cumsum_node0
                if key in ["edge_index1"]:
                    item = item + cumsum_node1

                batch[key].append(item)

            cumsum_node0 += num_nodes0
            cumsum_node1 += num_nodes1

        batch.batch0 = torch.cat(batch.batch0, dim=-1)
        batch.batch_num_nodes0 = torch.LongTensor(batch.batch_num_nodes0)
        batch.batch1 = torch.cat(batch.batch1, dim=-1)
        batch.batch_num_nodes1 = torch.LongTensor(batch.batch_num_nodes1)
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        return batch.contiguous()
