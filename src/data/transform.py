import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_cluster import random_walk

num_atom_types = 120
num_bond_types = 6

def _clone_data(data):
    data = Data(
        x=data.x.clone(), 
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        )
    return data

def _mask_data(data, mask_rate):
    data.y = data.x[:, 0].clone()
    
    num_nodes = data.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))    
    
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[mask_nodes] = True
    
    data.x[mask_nodes, 0] = 0
    data.mask = mask
    
    return data

def _mask_edge_data(data, mask_rate):
    data.y = data.x[:, 0].clone()
    
    sorted_edge_index = torch.sort(data.edge_index, dim=0)[0]
    sorted_edge_atoms = data.y[sorted_edge_index]
    edge_y = (
        (num_atom_types ** 2) * data.edge_attr[:, 0] 
        + num_atom_types * (sorted_edge_atoms[0] - 1) 
        + (sorted_edge_atoms[1] - 1)
        + num_atom_types
    )
    data.edge_y = edge_y

    num_unique_edges = data.edge_attr.size(0) // 2
    num_mask_edges = max(int(mask_rate * num_unique_edges), 1)
    unique_mask_edges = list(sorted(list(random.sample(range(num_unique_edges), num_mask_edges))))
    mask_edges = list(sorted([2*e for e in unique_mask_edges] + [2*e+1 for e in unique_mask_edges]))
    mask_nodes = torch.unique(data.edge_index[0, mask_edges])
    
    data.x[mask_nodes, 0] = 0
    data.edge_attr[mask_edges, 0] = 0
    
    num_nodes = data.x.size(0)
    num_edges = data.edge_attr.size(0)
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    edge_mask = torch.zeros(num_edges, dtype=torch.bool)
    node_mask[mask_nodes] = True
    edge_mask[[2*e for e in unique_mask_edges]] = True
    
    data.mask = node_mask
    data.edge_mask = edge_mask
    
    return data


def mask_data(data, mask_rate):
    data = _clone_data(data)
    data = _mask_data(data, mask_rate)
    
    return data

def double_mask_data(data, mask_rate):
    data0 = _clone_data(data)
    data0 = _mask_data(data0, mask_rate)
    
    data1 = _clone_data(data)
    data1 = _mask_data(data1, mask_rate)
    
    return data0, data1

def mask_edge_data(data, mask_rate):
    data = _clone_data(data)
    data = _mask_edge_data(data, mask_rate)
    
    return data