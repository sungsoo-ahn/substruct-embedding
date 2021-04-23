import torch
from torch_geometric.data import Data

def collate(data_list):
    keys = [set(data.keys) for data in data_list]
    keys = list(set.union(*keys))

    batch = Data()
    for key in keys:
        batch[key] = []

    batch.batch = []
    batch.batch_num_nodes = []
    batch.batch_size = len(data_list)
    batch.sinkhorn_mask = []
    
    cumsum_node = 0
    max_num_nodes = max([data.x.size(0) for data in data_list])
    for i, data in enumerate(data_list):
        num_nodes = data.x.size(0)
        batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
        batch.batch_num_nodes.append(torch.LongTensor([num_nodes]))
        
        sinkhorn_mask = torch.zeros(max_num_nodes, dtype=torch.bool)
        sinkhorn_mask[:num_nodes] = True
        batch.sinkhorn_mask.append(sinkhorn_mask)
        
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

def collate_twice(data_list):
    data_list0, data_list1 = map(list, zip(*data_list))
    return collate(data_list0), collate(data_list1)

def collate_cat(data_list):
    data_list = [data for data_ in zip(*data_list) for data in data_]
    return collate(data_list)

    