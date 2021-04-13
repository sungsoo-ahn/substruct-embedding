import random
import torch
from model import NodeEncoder

NUM_ATOM_TYPES = 120

class EdgeMaskScheme:
    criterion = torch.nn.CrossEntropyLoss()
    def __init__(self, edge_mask_rate, edge_attr_mask):
        self.edge_mask_rate = edge_mask_rate
        self.edge_attr_mask = edge_attr_mask
        
    def transform(self, data):
        masked_edge_indices = self.sample_edge_indices(data)
        edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
        edge_mask[masked_edge_indices] = True
        
        data.node_indices_masked0 = data.edge_index[0, edge_mask]
        data.node_indices_masked1 = data.edge_index[1, edge_mask]

        data.x_masked0 = torch.index_select(data.x, 0, data.node_indices_masked0)
        data.x_masked1 = torch.index_select(data.x, 0, data.node_indices_masked1)
                    
        data.x[data.node_indices_masked0] = 0
        data.x[data.node_indices_masked1] = 0

        masked_edge_indices_all = masked_edge_indices + [idx+1 for idx in masked_edge_indices]
        if self.edge_attr_mask:
            edge_mask_all = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
            edge_mask_all[masked_edge_indices_all] = True
            data.edge_attr[edge_mask_all, :] = 0       
        
        return data

    def sample_edge_indices(self, data):
        num_edges = data.edge_index.size(1) // 2
        sample_size = int(num_edges * self.edge_mask_rate)
        edge_indices = list(random.sample(range(num_edges), sample_size))
        edge_indices = [idx * 2 for idx in edge_indices]
        return edge_indices

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = NodeEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Linear(emb_dim, (NUM_ATOM_TYPES-1) * 100)
        models = torch.nn.ModuleDict({"encoder": encoder, "head": head})
        return models
    
    def compute_accuracy(self, preds, targets):
        acc = float(torch.sum(torch.max(preds, dim=1)[1] == targets)) / preds.size(0)
        return acc

    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)
        targets = (batch.x_masked0 - 1)[:, 0] * (NUM_ATOM_TYPES -1) + (batch.x_masked1 - 1)[:, 0]
        
        emb = models["encoder"](batch.x, batch.edge_index, batch.edge_attr)
        out = models["head"](emb)
        
        out0 = torch.index_select(out, 0, batch.node_indices_masked0)
        out0 = out0.view(-1, (NUM_ATOM_TYPES-1), 100)
        out1 = torch.index_select(out, 0, batch.node_indices_masked1)
        out1 = out1.view(-1, (NUM_ATOM_TYPES-1), 100)
        
        preds = torch.bmm(out0, out1.transpose(1, 2)).view(-1, (NUM_ATOM_TYPES-1)**2) / 100
                
        loss = self.criterion(preds, targets)
        
        with torch.no_grad():
            acc = self.compute_accuracy(preds, targets)
        
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 1.0)
        optim.step()
                
        statistics = {
            "loss": loss.detach(), "acc": acc,
            "masked_ratio": (batch.node_indices_masked0.size(0) / batch.edge_index.size(1))
            }
        
        return statistics     

    def collate_fn(self, data_list):
        data_list = [data for data in data_list if data is not None]
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = Data()

        for key in keys:
            batch[key] = []

        batch.batch_size = len(data_list)
        batch.batch_masked = []
        batch.batch_num_nodes = []
        batch.batch_num_nodes_masked = []

        cumsum_node = 0
        cumsum_node_masked = 0
        batch_size = 0
        for i, data in enumerate(data_list):
            if data is None:
                continue
            
            batch_size += 1
            
            num_nodes = data.x.size(0)
            if "x_masked" in keys:
                num_nodes_masked = data.x_masked.size(0)
            else:
                num_nodes_masked = 0

            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            batch.batch_masked.append(torch.full((num_nodes_masked,), i, dtype=torch.long))

            batch.batch_num_nodes.append(num_nodes)
            batch.batch_num_nodes_masked.append(num_nodes_masked)

            for key in keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node
                elif key in ["edge_index_masked"]:
                    item = item + cumsum_node_masked
                elif key in ["node_indices_masked0", "node_indices_masked1"]:
                    item = item + cumsum_node
                    
                batch[key].append(item)


            cumsum_node += num_nodes
            cumsum_node_masked += num_nodes_masked

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch_size = batch_size
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_masked = torch.cat(batch.batch_masked, dim=-1)
        batch.batch_num_nodes = torch.LongTensor(batch.batch_num_nodes)
        batch.batch_num_nodes_masked = torch.LongTensor(batch.batch_num_nodes_masked)
        
        return batch.contiguous()