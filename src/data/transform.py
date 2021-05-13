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
    new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return new_data

def get_mask(data, mask_p):
    num_nodes = data.x.size(0)
    num_mask = max(1, int(mask_p * num_nodes))
    mask_idx = random.sample(range(num_nodes), num_mask)
    mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    mask[mask_idx] = True
    return mask

def mask_data(data, mask_p):
    num_nodes = data.x.size(0)
    num_mask = int(mask_p * num_nodes)
    if num_mask == 0:
        return data
    
    data.x = data.x.clone()
    mask_idx = random.sample(range(num_nodes), num_mask)
    data.x[mask_idx, 0] = 0
     
    return data

def fragment(data, mask_p):
    if data.frag_y.max() == 0:
        frag_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        frag_data.node2junctionnode = torch.zeros(data.x.size(0), dtype=torch.long)
        
        junction_data = Data(
            x=torch.empty(1, dtype=torch.long), 
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 2, dtype=torch.long),
            )
        junction_data.mask = torch.zeros([1])
        return frag_data, junction_data
      
    num_nodes = data.x.size(0)
    
    row, col = data.edge_index
    inter_edge_index = data.edge_index[:, data.frag_y[row] != data.frag_y[col]]
    
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(data.edge_index.t().tolist())
    
    num_drop_edges = random.choice(range(1, inter_edge_index.size(1) + 1))
    drop_edge_index = random.sample(inter_edge_index.t().tolist(), num_drop_edges)
    nx_graph.remove_edges_from(drop_edge_index)
    connected_components = list(map(list, nx.connected_components(nx_graph)))
    num_components = len(connected_components)
    
    node2newnode = torch.zeros(num_nodes, dtype=torch.long)
    newnode2node = torch.zeros(num_nodes, dtype=torch.long)
    node2junctionnode = torch.zeros(num_nodes, dtype=torch.long)
    offset = 0
    for junctionnode, component in enumerate(connected_components):
        node2newnode[component] = torch.arange(offset, offset+len(component))
        newnode2node[offset:offset+len(component)] = torch.tensor(component)
        node2junctionnode[component] = junctionnode
        offset += len(component)
        
    ###
    frag_x = data.x.clone()
    frag_edge_index = data.edge_index.clone()
    frag_edge_attr = data.edge_attr.clone()
    
    frag_x = frag_x[newnode2node]
    frag_edge_index = node2newnode[frag_edge_index]
    frag_edge_index, frag_edge_attr = coalesce(frag_edge_index, frag_edge_attr, num_nodes, num_nodes)
        
    frag_data = Data(x=frag_x, edge_index=frag_edge_index, edge_attr=frag_edge_attr)
    frag_data.node2junctionnode = node2junctionnode[newnode2node]
    
    ###
    junction_edge_index = data.edge_index.clone()
    junction_edge_attr = data.edge_attr.clone()
    
    junction_edge_index = node2junctionnode[junction_edge_index]
    mask = junction_edge_index[0] != junction_edge_index[1]
    junction_edge_index = junction_edge_index[:, mask]
    junction_edge_attr = junction_edge_attr[mask, :]
    
    junction_edge_index, junction_edge_attr = coalesce(
        junction_edge_index, junction_edge_attr, num_components, num_components
        )
    
    junction_data = Data(
        x=torch.empty(num_components, dtype=torch.long), 
        edge_index=junction_edge_index, 
        edge_attr=junction_edge_attr
        )
    junction_data.mask = torch.empty(num_components)
    junction_data.mask.bernoulli_(mask_p)
    
    return frag_data, junction_data

def double_fragment(data, mask_p):
    frag_data, junction_data = fragment(data, mask_p=mask_p)
    
    return data, frag_data, junction_data
    