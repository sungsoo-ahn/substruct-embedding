import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def mask_nodes(x, edge_index, edge_attr, mask_rate):
    num_nodes = x.size(0)
    num_mask_nodes = 1
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    
    x = x.clone()
    x[mask_nodes] = 0
    
    return x, edge_index, edge_attr


def mask_samepool(data, mask_rate, pool_rate):
    x0 = data.x.clone()
    x1 = data.x.clone()
    
    num_nodes = x0.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes0 = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    x0[mask_nodes0] = 0
    mask_nodes1 = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    x1[mask_nodes1] = 0
    
    num_pool_nodes = max(int(pool_rate * num_nodes), 1)
    pool_mask_nodes = list(sorted(random.sample(range(num_nodes), num_pool_nodes)))
    pool_mask = torch.zeros(num_nodes, dtype=torch.bool)
    pool_mask[pool_mask_nodes] = True
    
    data0 = Data(
        x=x0,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        pool_mask=pool_mask,
    )
    data1 = Data(
        x=x1,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        pool_mask=pool_mask,
    )
    
    return data0, data1

def mask_diffpool(data, mask_rate, pool_rate):
    x0 = data.x.clone()
    x1 = data.x.clone()
    
    num_nodes = x0.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes0 = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    x0[mask_nodes0] = 0
    mask_nodes1 = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    x1[mask_nodes1] = 0
    
    num_pool_nodes = max(int(pool_rate * num_nodes), 1)
    pool_mask_nodes0 = list(sorted(random.sample(range(num_nodes), num_pool_nodes)))
    pool_mask0 = torch.zeros(num_nodes, dtype=torch.bool)
    pool_mask0[pool_mask_nodes0] = True
    
    pool_mask_nodes1 = list(sorted(random.sample(range(num_nodes), num_pool_nodes)))
    pool_mask1 = torch.zeros(num_nodes, dtype=torch.bool)
    pool_mask1[pool_mask_nodes1] = True
    
    data0 = Data(
        x=x0,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        pool_mask=pool_mask,
    )
    data1 = Data(
        x=x1,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        pool_mask=pool_mask,
    )
    
    return data0, data1
