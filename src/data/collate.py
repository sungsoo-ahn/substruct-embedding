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
        
    for key in keys:
        batch[key] = torch.cat(
            batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))
    
    batch.batch = torch.cat(batch.batch, dim=-1)
    
    batch.lower_batch = []
    batch.upper_batch = []
    lower_offset, upper_offset = 0, 0
    for data in data_list:
        batch.upper_batch.append(
            torch.full((data.upper_num_nodes.item(), ), upper_offset, dtype=torch.long)
            )
        upper_offset += 1
        for num_nodes in data.lower_num_nodes.tolist():
            batch.lower_batch.append(torch.full((num_nodes, ), lower_offset, dtype=torch.long))
            lower_offset += 1
                
    batch.upper_batch = torch.cat(batch.upper_batch, dim=-1)
    batch.lower_batch = torch.cat(batch.lower_batch, dim=-1)
        
    return batch.contiguous()