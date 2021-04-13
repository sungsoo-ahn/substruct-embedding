import random
import torch
from model import NodeEncoder
from torch_geometric.data import Data

NUM_ATOM_TYPES = 120

class NodeMaskScheme:
    criterion = torch.nn.CrossEntropyLoss()
    def __init__(self, node_mask_rate):
        self.node_mask_rate = node_mask_rate
        
    def transform(self, data):
        masked_node_indices = self.sample_node_indices(data)
        mask = torch.zeros(data.x.size(0), dtype=torch.bool)
        mask[masked_node_indices] = True
        
        data.mask_target = data.x[mask].clone()[:, 0] - 1
        data.x[mask] = 0
        data.node_mask = mask
        
        return data

    def sample_node_indices(self, data):
        num_nodes = data.x.size()[0]
        sample_size = int(num_nodes * self.node_mask_rate + 1)
        node_indices = list(random.sample(range(num_nodes), sample_size))
        return node_indices

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Linear(emb_dim, NUM_ATOM_TYPES-1)
        models = torch.nn.ModuleDict({"encoder": encoder, "head": head})
        return models
    
    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
        return acc

    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)
        
        emb = models["encoder"](batch.x, batch.edge_index, batch.edge_attr)
        pred = models["head"](emb[batch.node_mask])
        loss = self.criterion(pred, batch.mask_target)
        
        with torch.no_grad():
            acc = self.compute_accuracy(pred, batch.mask_target)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        statistics = {"loss": loss.detach(), "acc": acc, "num_masked_nodes": torch.sum(batch.node_mask) / batch.batch_size}
        
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

        batch.batch = []
        batch.batch_size = len(data_list)
        batch.batch_num_nodes = []

        cumsum_node = 0
        for i, data in enumerate(data_list):            
            num_nodes = data.x.size(0)
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            batch.batch_num_nodes.append(torch.LongTensor([num_nodes]))

            for key in keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node
                    
                batch[key].append(item)

            cumsum_node += num_nodes

        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_num_nodes = torch.LongTensor(batch.batch_num_nodes)
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        
        return batch.contiguous()
