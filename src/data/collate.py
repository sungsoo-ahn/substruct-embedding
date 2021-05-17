import torch
from torch_geometric.data import Data

def collate(data_list):
    data_list = [data for data in data_list if data is not None]
    
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
        batch[key] = torch.cat(batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))

    batch.batch_num_nodes = torch.cat(batch.batch_num_nodes, dim=0)

    batch.batch = torch.cat(batch.batch, dim=-1)
            
    return batch.contiguous()

def multifrag_collate(data_list):
    data_list = [data for data in data_list if data is not None]
    
    keys = [set(data.keys) for data in data_list]
    keys = list(set.union(*keys))
    assert 'batch' not in keys

    batch = Data()

    for key in keys:
        batch[key] = []
    batch.batch = []
    batch.frag_batch = []
    batch.batch_num_nodes = []

    cumsum_node = 0
    cumsum_dangling_node = 0
    cumsum_edge = 0
    cumsum_frag = 0
    frag_cnt = 0
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        
        num_frags = data.frag_num_nodes.size(0)
        num_dangling_nodes = data.dangling_mask.long().sum().item()
        batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
        batch.frag_batch.append(
            torch.repeat_interleave(
                torch.arange(cumsum_frag, cumsum_frag+num_frags), data.frag_num_nodes
                )
            )
        for key in data.keys:
            item = data[key]
            if key == 'edge_index':
                item = item + cumsum_node
            elif key == "dangling_edge_index":
                item = item + cumsum_dangling_node
                
            batch[key].append(item)

        cumsum_node += num_nodes
        cumsum_dangling_node += num_dangling_nodes
        cumsum_edge += data.edge_index.shape[1]
        cumsum_frag += num_frags
        
        batch.batch_num_nodes.append(torch.tensor([num_nodes]))

    for key in keys:
        if key == "dangling_edge_index":
            batch[key] = torch.cat(batch[key], dim=1)
        else:
            batch[key] = torch.cat(batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))

    batch.batch_num_nodes = torch.cat(batch.batch_num_nodes, dim=0)

    batch.batch = torch.cat(batch.batch, dim=-1)
    batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)           
            
    return batch.contiguous()

def double_collate(data_list):
    data_list0 = [data_tuple[0] for data_tuple in data_list if data_tuple[0] is not None]
    data_list1 = [data_tuple[1] for data_tuple in data_list if data_tuple[1] is not None]

    return collate(data_list0), collate(data_list1)

def merge_collate(data_list):
    data_list0 = [data_tuple[0] for data_tuple in data_list if data_tuple[0] is not None]
    data_list1 = [data_tuple[1] for data_tuple in data_list if data_tuple[1] is not None]
    
    batch0 = collate(data_list0)
    batch1 = collate(data_list1)
    
    offset = batch0.x.size(0)
    x = torch.cat([batch0.x, batch1.x], dim=0)

    dangling_u = batch0.dangling_mask.nonzero().squeeze(1)
    dangling_v = batch1.dangling_mask.nonzero().squeeze(1) + offset
    rolled_dangling_v = torch.roll(dangling_v, shifts=1, dims=0)
    pos_dangling_edge_index = torch.cat(
        [
            torch.stack([dangling_u, dangling_v], dim=0),
            torch.stack([dangling_v, dangling_u], dim=0),
        ],
        dim=1,
    )
    neg_dangling_edge_index = torch.cat(
        [
            torch.stack([dangling_u, rolled_dangling_v], dim=0),
            torch.stack([rolled_dangling_v, dangling_u], dim=0),
        ],
        dim=1,
    )

    edge_index = torch.cat([batch0.edge_index, batch1.edge_index + offset], dim=1)
    pos_edge_index = torch.cat([edge_index, pos_dangling_edge_index], dim=1)
    neg_edge_index = torch.cat([edge_index, neg_dangling_edge_index], dim=1)
    
    dangling_edge_attr = torch.zeros_like(pos_dangling_edge_index).t()
    edge_attr = torch.cat([batch0.edge_attr, batch1.edge_attr, dangling_edge_attr], dim=0)

    pos_batch, node2posnode = torch.sort(torch.cat([batch0.batch, batch1.batch], dim=0))
    posnode2node = torch.empty_like(node2posnode)
    posnode2node[node2posnode] = torch.arange(node2posnode.size(0))
    
    neg_batch1 = batch1.batch + 1
    neg_batch1[neg_batch1 == neg_batch1.max()] = 0
    neg_batch, node2negnode = torch.sort(torch.cat([batch0.batch, neg_batch1], dim=0))
    negnode2node = torch.empty_like(node2negnode)
    negnode2node[node2negnode] = torch.arange(node2negnode.size(0))
    
    pos_x = x[posnode2node]
    neg_x = x[negnode2node]
    
    pos_edge_index = node2posnode[pos_edge_index]
    neg_edge_index = node2negnode[neg_edge_index]

    num_nodes = x.size(0)
    pos_edge_index, pos_edge_attr = coalesce(pos_edge_index, edge_attr, num_nodes, num_nodes)
    neg_edge_index, neg_edge_attr = coalesce(neg_edge_index, edge_attr, num_nodes, num_nodes)

    pos_batch = Data(pos_x, pos_edge_index, pos_edge_attr)
    pos_batch.batch = pos_batch
    neg_batch = Data(neg_x, neg_edge_index, neg_edge_attr)
    neg_batch.batch = neg_batch

    return pos_batch, neg_batch