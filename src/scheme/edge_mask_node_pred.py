import random
import torch
from model import NodeEncoder

NUM_ATOM_TYPES = 120

class EdgeMaskNodePredScheme:
    criterion = torch.nn.CrossEntropyLoss()
    def __init__(self, edge_mask_rate):
        self.edge_mask_rate = edge_mask_rate
        
    def transform(self, data):
        masked_edge_indices = self.sample_edge_indices(data)
        edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
        edge_mask[masked_edge_indices] = True
        
        node_indices_masked = torch.unique(torch.cat(
            [data.edge_index[0, edge_mask], data.edge_index[1, edge_mask]], dim=0
            ))
        node_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
        node_mask[node_indices_masked] = True

        data.node_mask = node_mask
        data.x_masked = data.x[node_mask].clone()            
        data.x[node_mask] = 0
        
        
        return data

    def sample_edge_indices(self, data):
        num_edges = data.edge_index.size(1) // 2
        sample_size = int(num_edges * self.edge_mask_rate)
        edge_indices = list(random.sample(range(num_edges), sample_size))
        edge_indices = [idx * 2 for idx in edge_indices]
        return edge_indices

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Linear(emb_dim, NUM_ATOM_TYPES-1)
        models = torch.nn.ModuleDict({"encoder": encoder, "head": head})
        return models
    
    def compute_accuracy(self, preds, targets):
        acc = float(torch.sum(torch.max(preds, dim=1)[1] == targets)) / preds.size(0)
        return acc

    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)
        targets = (batch.x_masked - 1)[:, 0]
        
        emb = models["encoder"](batch.x, batch.edge_index, batch.edge_attr)
        preds = models["head"](emb[batch.node_mask])
        loss = self.criterion(preds, targets)
        
        with torch.no_grad():
            acc = self.compute_accuracy(preds, targets)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        statistics = {"loss": loss.detach(), "acc": acc, "num_masked_nodes": torch.sum(batch.node_mask) / batch.batch_size}
        
        return statistics     
