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

def collate_twice(data_list):
    data_list0, data_list1 = map(list, zip(*data_list))
    return collate(data_list0), collate(data_list1)

def collate_cat(data_list):
    data_list = [data for data_ in zip(*data_list) for data in data_]
    return collate(data_list)

def collate_nodemask(data_list, mask_rate):
    batch = collate(data_list)
    
    num_nodes = batch.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))    

    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[mask_nodes] = True
    batch.y = batch.x[:, 0].clone()
    batch.x[mask_nodes, 0] = 0
    batch.mask=mask
    
    return batch    

def collate_edgemask(data_list, mask_rate):
    batch = collate(data_list)
    
    num_edges = batch.edge_index.size(1)
    num_mask_edges = max(int(mask_rate * num_nodes), 1)
    mask_edges = list(sorted(random.sample(range(num_edges), num_mask_edges)))    

    labels0 = batch.edge_attr[mask_edges, 0]
    labels1 = batch.x[batch.edge_attr]

    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[mask_nodes] = True
    batch.y = batch.x[:, 0].clone()
    batch.x[mask_nodes, 0] = 0
    batch.mask=mask
    
    return batch

def collate_balanced_nodemask(data_list):
    batch = collate(data_list)
    
    node_label = batch.x[:, 0]
    label_bincount = torch.bincount(node_label)

    
def collate_balanced_edgemask(data_list):
    asd

    