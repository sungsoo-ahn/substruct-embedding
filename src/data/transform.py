import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_cluster import random_walk

def mask_data(data, mask_rate=0.0):
    num_nodes = data.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    
    x = data.x.clone()
    x[mask_nodes, 0] = 0
    
    data = Data(
        x=x,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    
    return data    

def mask_data_twice(data, mask_rate=0.0):
    data0 = mask_data(data, mask_rate=mask_rate)
    data1 = mask_data(data, mask_rate=mask_rate)
    return data0, data1

