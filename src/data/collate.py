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

def super_collate(data_list):
    data_list = [data for data in data_list if data[0] is not None]
    
    super_data_list = [data[1] for data in data_list]
    data_list = [data for data_tuple, _ in data_list for data in data_tuple]
    return collate(data_list), collate(super_data_list)

def double_super_collate(data_list):
    data_list = [data for data in data_list if data[0] is not None]    
    super_data_list = [data[2] for data in data_list] + [data[3] for data in data_list]
    data_list = (
        [data for data_tuple, _, _, _ in data_list for data in data_tuple] 
        + [data for _, data_tuple, _, _ in data_list for data in data_tuple] 
    )
        
    return collate(data_list), collate(super_data_list)
