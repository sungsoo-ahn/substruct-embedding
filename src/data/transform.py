import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_cluster import random_walk

def mask_data(data, mask_rate=0.15):
    num_nodes = data.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))    
    
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[mask_nodes] = True
    x = data.x.clone()
    y = data.x[:, 0].clone()
    x[mask_nodes, 0] = 0
     
    data = Data(
        x=x, 
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        )
    data.y = y
    data.mask=mask
    
    return data    

def double_mask_data(data, mask_rate=0.15):
    data0 = mask_data(data, mask_rate=0.15) 
    data1 = mask_data(data, mask_rate=0.15) 
    return data0, data1