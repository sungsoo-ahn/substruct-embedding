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

def contrastive_collate(data_list):
    data_list = [elem for elem in data_list if elem is not None]
    data_list = list(zip(*data_list))
    data_list = [data for inner_data_list in data_list for data in inner_data_list]
    return collate(data_list)

    