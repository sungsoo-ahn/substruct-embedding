import numpy as np
import torch
from torch_geometric.data import Data

from model import NodeEncoder
from util import compute_accuracy


class NodeContrastiveScheme:
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, transform, temperature):
        self._transform = transform
        self.temperature = temperature

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

    def transform(self, data):
        x, edge_index, edge_attr = self._transform(
            data.x.clone(), data.edge_index.clone(), data.edge_attr.clone()
        )
        if x.size(0) == 0:
            return None

        data0 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        x, edge_index, edge_attr = self._transform(
            data.x.clone(), data.edge_index.clone(), data.edge_attr.clone()
        )
        if x.size(0) == 0:
            return None

        data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data0, data1

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

        out0, out1 = torch.chunk(out, 2, dim=0)
        out0 = torch.split(out0, batch.batch_num_nodes.tolist()[:batch.batch_size // 2])
        out1 = torch.split(out1, batch.batch_num_nodes.tolist()[:batch.batch_size // 2])

        loss = 0.0
        acc = 0.0        
        for idx in range(batch.batch_size // 2):            
            logits = torch.matmul(out0[idx], out1[idx].T) / self.temperature
            labels = torch.arange(out0[idx].size(0)).to(device)
            loss_ = self.criterion(logits, labels)
            acc_ = compute_accuracy(logits, labels)
            
            loss += loss_ / (batch.batch_size // 2)
            acc += acc_ / (batch.batch_size // 2)
                            
        optim.zero_grad()
        loss.backward()
        optim.step()

        statistics = {"loss": loss.detach(), "acc": acc}

        return statistics