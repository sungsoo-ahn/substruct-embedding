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
    if "node2junctionnode" in data_list[0].keys:    
        batch.junction_batch = []
        offset = 0
        for i, data in enumerate(data_list):
            batch.junction_batch.append(offset + data.node2junctionnode)
            offset += data.node2junctionnode.max().item() + 1
            
        batch.junction_batch = torch.cat(batch.junction_batch, dim=-1)
        
    return batch.contiguous()

def junction_collate(data_list):
    data_list, frag_data_list, junction_data_list = map(list, zip(*data_list))    
    return collate(data_list), collate(frag_data_list), collate(junction_data_list)