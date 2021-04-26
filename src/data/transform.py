import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_cluster import random_walk

def mask_data(data, atom_bincount=None, mask_rate=0.15):
    if atom_bincount is not None:
        weights = 1 / atom_bincount[data.x[:, 0]]
        weights /= np.sum(weights)        
    else:
        weights = np.ones(data.x.size(0))
        weights /= np.sum(weights)
    
    num_nodes = data.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = np.random.choice(range(num_nodes), size=num_mask_nodes, replace=False, p=weights).tolist()
    #list(sorted(random.sample(range(num_nodes), num_mask_nodes)))    
    
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[mask_nodes] = True
    x = data.x.clone()
    y = data.x[:, 0].clone()
    x[mask_nodes, 0] = 0
     
    data = Data(
        x=x,
        y=y,
        mask=mask,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
    )
    
    return data    