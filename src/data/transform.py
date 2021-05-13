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

def contract(data, contract_p):
    if data.frag_y.max() == 0:
        return data
    
    num_nodes = data.x.size(0)
    x = data.x.clone()
    frag_y = data.frag_y.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()

    num_frags = data.frag_y.max().item()+1
    #num_contract = min(np.random.binomial(num_frags, contract_p), num_frags-1)
    num_contract = np.random.binomial(num_frags, contract_p)
    contract_frags = random.sample(range(num_frags), num_contract)
    frag_nodes = list(range(num_nodes, num_nodes + num_contract))
    
    keepnode_mask = torch.ones(num_nodes, dtype=torch.bool)
    for frag_node, frag in zip(frag_nodes, contract_frags):
        keepnode_mask[data.frag_y == frag] = False
        edge_index[data.frag_y[data.edge_index] == frag] = frag_node
    
    frag_x = torch.zeros(num_contract, x.size(1), dtype=torch.long)
    x = torch.cat([x[keepnode_mask], frag_x], dim=0)
    
    selfloop_mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, selfloop_mask]
    edge_attr = edge_attr[selfloop_mask, :]
    
    num_keepnodes = keepnode_mask.long().sum()
    node2newnode = -torch.ones(data.num_nodes, dtype=torch.long)
    node2newnode[keepnode_mask] = torch.arange(num_keepnodes)
    node2newnode = torch.cat(
        [node2newnode, torch.arange(num_keepnodes, num_keepnodes+num_contract)], dim=0
        )
    edge_index = node2newnode[edge_index]
    
    edge_index, edge_attr = coalesce(
        edge_index, edge_attr, num_keepnodes+num_contract, num_keepnodes+num_contract
        )
    
    #print(edge_index)
    #assert False
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def contract_once(data, contract_p):
    data0 = data
    data1 = contract(data, contract_p)
    
    return data0, data1

def contract_both(data, contract_p):
    data0 = contract(data, contract_p)
    data1 = contract(data, contract_p)
    
    return data0, data1
    
    