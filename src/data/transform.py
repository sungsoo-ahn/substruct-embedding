import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_sparse import coalesce


def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False, num_nodes=None):
    device = edge_index.device
    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1

        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.size(0), device=device)

    mask = n_mask[edge_index[0]].long() + n_mask[edge_index[1]].long() == 2
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr


def clone_data(data):
    new_data = Data(
        x=data.x.clone(), edge_index=data.edge_index.clone(), edge_attr=data.edge_attr.clone(),
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
        num_nodes=data.x.size(0),
    )
    new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,)
    return new_data


def sequential_fragment(data):
    if data.frag_y.max() == 0:
        return None
    
    row, col = data.edge_index
    inter_edge_index = data.edge_index[:, data.frag_y[row] != data.frag_y[col]]
    frag_edge_index = data.frag_y[inter_edge_index]
    
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(frag_edge_index.t().tolist())
    
    triplets = []
    while len(nx_graph.edges) > 0:
        nx_graph = nx_graph.copy()
        u, v = random.choice(list(nx_graph.edges))
        nx_graph.remove_edge(u, v)
        
        subgraph_nodes0, subgraph_nodes1 = nx.connected_components(nx_graph)
        triplets.append([set(nx_graph.nodes), subgraph_nodes0, subgraph_nodes1])
        
        nx_graph = nx_graph.subgraph(random.choice([subgraph_nodes0, subgraph_nodes1]))
        
    triplet = random.choice(triplets)
    
    data_triplet = []
    for frag_ys in triplet:
        subgraph_nodes = torch.sum(
            (data.frag_y.unsqueeze(1) == torch.tensor(list(frag_ys))
             ).long(), dim=1).nonzero().squeeze(1)
        
        data_triplet.append(subgraph_data(data, subgraph_nodes))
        
    return data_triplet

def double_sequential_fragment(data):
    if data.frag_y.max() == 0:
        return None
     
    data_triplet0 = sequential_fragment(clone_data(data))
    data_triplet1 = sequential_fragment(clone_data(data))
    
    data_list = data_triplet0 + data_triplet1
    
    return data_list