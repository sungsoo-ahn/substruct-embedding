import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_cluster import random_walk
from torch_sparse import coalesce

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

def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index, edge_attr, p):
    row, col = edge_index

    row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    mask = edge_index.new_full((row.size(0), ), 1 - p, dtype=torch.float)
    mask = torch.bernoulli(mask).to(torch.bool)

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    edge_index = torch.stack([torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    
    return edge_index, edge_attr


def get_undirected_edge_index(data):
    return data.edge_index[(data.edge_index[0] < data.edge_index[1])]

def get_intra_edge_mask(data):
    edge_mask = (data.frag_y[data.edge_index[0]] == data.frag_y[data.edge_index[1]])
    return edge_mask

def get_inter_edge_mask(data):
    edge_mask = (data.frag_y[data.edge_index[0]] != data.frag_y[data.edge_index[1]])
    return edge_mask   

def clone_data(data):
    new_data = Data(
        x=data.x.clone(), 
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        )

    new_data.frag_y = data.frag_y.clone()
    return new_data

def subgraph_data(data, subgraph_nodes):
    x = data.x[subgraph_nodes]
    edge_index, edge_attr = subgraph(
        subgraph_nodes, 
        data.edge_index, 
        data.edge_attr, 
        relabel_nodes=True, 
        num_nodes=data.x.size(0)
        )
    new_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return new_data

def _fragment(data, p):
    if data.frag_y.max() == 0:
        return data
    
    inter_edge_mask = data.frag_y[data.edge_index[0]] != data.frag_y[data.edge_index[1]]
    inter_edge_index = data.edge_index[:, inter_edge_mask]
    inter_edge_attr = data.edge_attr[inter_edge_mask, :]
    
    intra_edge_index = data.edge_index[:, ~inter_edge_mask]
    intra_edge_attr = data.edge_attr[~inter_edge_mask, :]
    
    undirected_mask = (inter_edge_index[0] < inter_edge_index[1])
    undirected_inter_edge_index = inter_edge_index[:, undirected_mask]
    undirected_inter_edge_attr = inter_edge_attr[undirected_mask, :]
    
    num_idxs = undirected_inter_edge_index.size(1)
    num_drops = max(int(p * num_idxs), 1)
    drop_idxs = random.sample(range(num_idxs), num_drops)
    drop_mask = torch.zeros(num_idxs).to(torch.bool)
    drop_mask[drop_idxs] = True
    keep_mask = (drop_mask == False)
        
    row, col = undirected_inter_edge_index[:, keep_mask]
    new_inter_edge_index = torch.stack(
        [torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0
        )
    new_inter_edge_attr = torch.cat([
        undirected_inter_edge_attr[keep_mask, :],
        undirected_inter_edge_attr[keep_mask, :],
        ], dim=0)
    
    edge_index = torch.cat([intra_edge_index, new_inter_edge_index], dim=1)
    edge_attr = torch.cat([intra_edge_attr, new_inter_edge_attr], dim=0)
    num_nodes = data.x.size(0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
        
    new_data = Data(
        x=data.x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    
    return new_data

def _sample_fragment(data, p):
    data = _fragment(data, p)
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(data.edge_index.t().tolist())
    subgraph_nodes = list(random.choice(list(nx.connected_components(nx_graph))))
    
    return subgraph_data(data, subgraph_nodes)

def fragment(data, p):
    if data.frag_y.max() == 0:
        return data, data

    data0 = _fragment(clone_data(data), p)
    data1 = _fragment(clone_data(data), p)
    
    return data0, data1    

def sample_fragment(data, p):
    if data.frag_y.max() == 0:
        return data, data

    data0 = _sample_fragment(clone_data(data), p)
    data1 = _sample_fragment(clone_data(data), p)
    
    return data0, data1    

def partition_fragment(data):
    if data.frag_y.max() == 0:
        return None, None
    
    data = _fragment(data, 0.0)
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(data.edge_index.t().tolist())
    subgraph_nodes0, subgraph_nodes1 = map(list, list(nx.connected_components(nx_graph)))
    
    data0 = subgraph_data(data, subgraph_nodes0)
    data1 = subgraph_data(data, subgraph_nodes1)
    
    return data0, data1