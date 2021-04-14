import random
import numpy as np
import torch
from torch_geometric.data import Data

from model import GraphEncoder
from util import compute_accuracy

from data.transform import subgraph, drop_nodes, random_transform

class ContrastiveScheme:
    criterion = torch.nn.CrossEntropyLoss()
    def __init__(self, transform, temperature):
        self.transform = transform
        self.temperature = temperature
        
    @staticmethod
    def transform(self, data):
        data0 = random_transform(data)
        if data0 is None:
            return None
        
        data1 = random_transform(data)
        if data1 is None:
            return None
        
        return data0, data1

    @staticmethod
    def collate_fn(data_list):
        data_list = [elem for elem in data_list if elem is not None]
        data_list = list(zip(*data_list))
        data_list = [data for inner_data_list in data_list for data in inner_data_list]
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = Data()
        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.batch_num_nodes = []
        batch.batch_size = len(data_list)
        batch.num_views = 2
        
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

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = GraphEncoder(num_layers, emb_dim, drop_rate)
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
        graphwise_score = torch.matmul(out, out.T)
        
        logits, labels = self.get_logits_and_labels(graphwise_score, self.temperature, device)
        loss = self.criterion(logits, labels)
        
        with torch.no_grad():
            acc = self.compute_accuracy(logits, labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        statistics = {"loss": loss.detach(), "acc": acc}
        
        return statistics

    def get_logits_and_labels(self, similarity_matrix, temperature, device):
        batch_size = similarity_matrix.size(0) // 2        
        
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        return logits, labels
