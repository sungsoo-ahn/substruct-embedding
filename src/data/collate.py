import torch
from torch_geometric.data import Data

def collate(data_list):
    keys = [set(data.keys) for data in data_list]
    keys = list(set.union(*keys))
    assert 'batch' not in keys

    batch = Data()

    for key in keys:
        batch[key] = []
    batch.batch = []
    batch.batch_num_nodes = []

    cumsum_node = 0
    cumsum_edge = 0

    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
        for key in data.keys:
            item = data[key]
            if key == 'edge_index':
                item = item + cumsum_node
            batch[key].append(item)

        cumsum_node += num_nodes
        cumsum_edge += data.edge_index.shape[1]

        batch.batch_num_nodes.append(torch.tensor([num_nodes]))

    for key in keys:
        batch[key] = torch.cat(
            batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))

    batch.batch_num_nodes = torch.cat(batch.batch_num_nodes, dim=0)

    batch.batch = torch.cat(batch.batch, dim=-1)

    return batch.contiguous()

def double_collate(data_list):
    data_list0 = [data_tuple[0] for data_tuple in data_list if data_tuple[0] is not None]
    data_list1 = [data_tuple[1] for data_tuple in data_list if data_tuple[1] is not None]

    return collate(data_list0), collate(data_list1)