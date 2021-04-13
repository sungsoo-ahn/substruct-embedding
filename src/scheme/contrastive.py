import random
import torch
from model import GraphEncoder
from torch_geometric.data import Data
import numpy as np

def subgraph(x, edge_index, edge_attr, aug_ratio):
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index_np = edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index_np[1][edge_index_np[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index_np[1][edge_index_np[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
    edge_mask = np.array([n for n in range(edge_num) if (edge_index_np[0, n] in idx_nondrop and edge_index_np[1, n] in idx_nondrop)])

    edge_index = edge_index.numpy()
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        edge_index = torch.tensor(edge_index).transpose_(0, 1)
        x = x[idx_nondrop]
        edge_attr = edge_attr[edge_mask]
    except:
        pass

    return x, edge_index, edge_attr


def drop_nodes(x, edge_index, edge_attr, aug_ratio):
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = edge_index.numpy()
    edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    try:
        edge_index = torch.tensor(edge_index).transpose_(0, 1)
        x = x[idx_nondrop]
        edge_attr = edge_attr[edge_mask]
    except:
        pass
    
    return x, edge_index, edge_attr

def random_transform(x, edge_index, edge_attr, aug_ratio):
    n = np.random.randint(2)
    if n == 0:
        x, edge_index, edge_attr = drop_nodes(x, edge_index, edge_attr, aug_ratio)
    elif n == 1:
        x, edge_index, edge_attr = subgraph(x, edge_index, edge_attr, aug_ratio)
    
    return x, edge_index, edge_attr

class ContrastiveScheme:
    criterion = torch.nn.CrossEntropyLoss()
    def __init__(self, aug_rate, temperature):
        self.aug_ratio = aug_rate
        self.temperature = temperature
        
    def transform(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x0, edge_index0, edge_attr0 = random_transform(
            x.clone(), edge_index.clone(), edge_attr.clone(), self.aug_ratio
            )
        x1, edge_index1, edge_attr1 = random_transform(
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