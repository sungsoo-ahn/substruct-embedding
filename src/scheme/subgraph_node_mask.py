import itertools
import random
import torch
from torch_geometric.data import Data
from torch_cluster import random_walk
from model import NodeEncoder, NodeEncoderWithHead
from torch_geometric.nn import global_mean_pool

class SubgraphNodeMaskScheme:
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, walk_length_rate):
        self.walk_length_rate = walk_length_rate

    def transform(self, data):
        masked_node_indices = self.sample_node_indices(data)

        node_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
        node_mask[masked_node_indices] = True

        edge_mask = torch.any(
            data.edge_index.view(2, -1, 1) == masked_node_indices.view(1, 1, -1), dim=2
        )
        edge_mask = torch.all(edge_mask, dim=0)

        data.x_masked = data.x[node_mask].clone()
        data.x[node_mask] = 0

        edge_index_masked = data.edge_index[:, edge_mask]
        edge_index_masked = edge_index_masked.view(-1, 1) == masked_node_indices.view(1, -1)
        data.edge_index_masked = edge_index_masked.nonzero(as_tuple=False)[:, 1].view(2, -1)
        data.edge_attr_masked = data.edge_attr[edge_mask, :]
        
        data.node_mask = node_mask

        return data

    def sample_node_indices(self, data):
        num_atoms = data.x.size(0)
        start_idx = torch.tensor([random.choice(range(num_atoms))])
        walk_length = int(num_atoms * self.walk_length_rate + 1)
        node_indices = random_walk(
            data.edge_index[0, :], data.edge_index[1, :], start_idx, walk_length
        )[0]
        node_indices = torch.unique(node_indices)
        return node_indices

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(emb_dim, emb_dim)
            )
        target_encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        models = torch.nn.ModuleDict({
            "encoder": encoder, "head": head, "target_encoder": target_encoder
            })
        return models

    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)

        emb = models["encoder"](batch.x, batch.edge_index, batch.edge_attr)
        preds = models["head"](emb[batch.node_mask])
        targets = models["target_encoder"](
            batch.x_masked, batch.edge_index_masked, batch.edge_attr_masked
        )
        subgraph_reps = global_mean_pool(targets, batch.batch_masked)
        targets = targets + torch.repeat_interleave(
            subgraph_reps, batch.batch_num_nodes_masked, dim=0
            )
        
        logits = torch.mm(preds, targets.T) / preds.size(1)
        dummy_targets = torch.arange(logits.size(0)).to(device)
        loss = self.criterion(logits, dummy_targets)
        
        with torch.no_grad():
            acc = self.compute_accuracy(logits, dummy_targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        statistics = {
            "loss": loss.detach(), 
            "acc": acc, 
            "num_masked_nodes": torch.sum(batch.node_mask) / batch.batch_size,
            "unque_subgraph_ratio": float(len(set(batch.smiles))) / len(batch.smiles),
            }

        return statistics

    def compute_accuracy(self, preds, targets):
        acc = float(torch.sum(torch.max(preds, dim=1)[1] == targets)) / preds.size(0)
        return acc
