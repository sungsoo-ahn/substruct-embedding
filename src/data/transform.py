import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_cluster import random_walk

num_atom_types = 120
num_bond_types = 6

def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False,
             num_nodes=None):
    
    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1

        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.size(0), device=device)

    mask = (n_mask[edge_index[0]].long() + n_mask[edge_index[1]].long() == 2)
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr

def _clone_data(data):
    new_data = Data(
        x=data.x.clone(), 
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        )

    new_data.atom_y = data.x[:, 0]
    try:
        new_data.group_y = data.group_y.clone()
    except:
        pass
    
    return new_data

def _disconnect_motif(data):
    subgraph_nodes = (data.group_y == sample_y).nonzero().squeeze(1)
    
    edge_mask = data.group_y[data.edge_index[0]] == data.group_y[data.edge_index[1]]
    edge_index = data.edge_index[:, edge_mask]
    edge_attr = data.edge_attr[edge_mask, :]
    subgraph_x = data.x[subgraph_nodes]
    new_data = Data(
        x=subgraph_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    new_data.group_y = data.group_y
    
    return new_data


def _extract_motif(data, keep_all, drop_scaffold):
    if keep_all:
        subgraph_nodes = (data.group_y != 0)
    else:
        if drop_scaffold:
            if data.group_y.max().item() == 0:
                return None
            
            sample_y = random.choice(range(data.group_y.max().item())) + 1
        else:
            sample_y = random.choice(range(data.group_y.max().item() + 1))
            
        subgraph_nodes = (data.group_y == sample_y).nonzero().squeeze(1)
         
    edge_index, edge_attr = subgraph(
        subgraph_nodes, 
        data.edge_index, 
        data.edge_attr, 
        relabel_nodes=True, 
        num_nodes=data.x.size(0)
        )
    
    subgraph_x = data.x[subgraph_nodes]
    new_data = Data(
        x=subgraph_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    new_data.group_y = data.group_y[subgraph_nodes]
    
    return new_data

def _mask_data(data, mask_rate):    
    num_nodes = data.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))    
    
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[mask_nodes] = True
    
    data.x[mask_nodes, 0] = 0
    data.mask = mask
    
    return data

def _mask_edge_data(data, mask_rate):
    data.y = data.x[:, 0]
    
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


def mask_data(data, mask_rate=0.15):
    data = _clone_data(data)
    data = _mask_data(data, mask_rate)
    
    return data

def double_mask_data(data, mask_rate=0.15):
    data0 = _clone_data(data)
    data0 = _mask_data(data0, mask_rate)
    
    data1 = _clone_data(data)
    data1 = _mask_data(data1, mask_rate)
    
    return data0, data1

def mask_edge_data(data, mask_rate=0.15):
    data = _clone_data(data)
    data = _mask_edge_data(data, mask_rate)
    
    return data

def extract_motif_data(data, keep_all, drop_scaffold):
    motif_data = _extract_motif(_clone_data(data), keep_all, drop_scaffold)
    if motif_data is None:
        return None, None
    
    return data, motif_data