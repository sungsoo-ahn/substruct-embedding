import random
import torch
from model import GraphEncoder
from torch_geometric.data import Data
import numpy as np

class StructContrastiveScheme:
    criterion = torch.nn.CrossEntropyLoss()
    def __init__(self, aug_rate, temperature):
        self.aug_ratio = aug_rate
        self.temperature = temperature
        
    def transform(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x0, edge_index0, edge_attr0 = self.transform_data(
            x.clone(), edge_index.clone(), edge_attr.clone(), self.aug_ratio
            )
        x1, edge_index1, edge_attr1 = self.transform_data(
            x.clone(), edge_index.clone(), edge_attr.clone(), self.aug_ratio
            )
        
        data = Data()
        data.x0 = x0
        data.edge_index0 = edge_index0
        data.edge_attr0 = edge_attr0
        data.x1 = x1
        data.edge_index1 = edge_index1
        data.edge_attr1 = edge_attr1
        
        return data
    
    def transform_data(self, x, edge_index, edge_attr, aug_ratio):
        num_nodes = x.size(0)
        num_mask_nodes = int(aug_ratio*num_nodes)

        mask_node_indices = list(random.sample(range(num_nodes), num_mask_nodes))
        x[mask_node_indices] = 0

        num_edges = edge_index.size(1) // 2
        num_add_edges = int(0.5 * aug_ratio * num_edges)
        num_mask_edges = int(0.5 * aug_ratio * num_edges)
        
        mask_edge_indices = list(random.sample(range(num_edges), num_mask_edges))
        mask_edge_indices = [
            [idx*2 for idx in mask_edge_indices] + [idx*2+1 for idx in mask_edge_indices]
        ]
        edge_attr[mask_edge_indices] = 0
        
        new_uv_list = list(zip(
            random.sample(range(num_nodes), num_add_edges),
            random.sample(range(num_nodes), num_add_edges)
            ))
        new_edge_index = []
        for u, v in new_uv_list:
            new_edge_index.append([u, v])
            new_edge_index.append([v, u])
        
        new_edge_index = torch.LongTensor(new_edge_index).T
        edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        edge_attr = torch.cat(
            [edge_attr, torch.zeros(2*num_add_edges, 2, dtype=torch.long)], dim=0
            )
        
        return x, edge_index, edge_attr
        

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = GraphEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
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
    
    def compute_accuracy(self, pred, target):
        acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
        return acc

    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)
        
        emb0 = models["encoder"](batch.x0, batch.edge_index0, batch.edge_attr0, batch.batch0)
        emb0 = models["head"](emb0)
        emb1 = models["encoder"](batch.x1, batch.edge_index1, batch.edge_attr1, batch.batch1)
        emb1 = models["head"](emb1)
        
        emb = torch.cat([emb0, emb1], dim=0)        
        emb = torch.nn.functional.normalize(emb, dim=1)
        
        similarity_matrix = torch.matmul(emb, emb.T)
        logits, labels = self.get_logits_and_labels(
            similarity_matrix, self.temperature, device
            )
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