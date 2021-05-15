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
            n_idx[subset] = torch.arange(subset.long().sum().item(), device=device)
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
    if mask_p == 0.0:
        return data
    
    num_nodes = data.x.size(0)
    num_mask = int(mask_p * num_nodes)
    if num_mask == 0:
        return data
    
    data.x = data.x.clone()
    mask_idx = random.sample(range(num_nodes), num_mask)
    data.x[mask_idx, 0] = 120
     
    return data

def fragment(data, drop_p, choose_pair):
    if data.frag_y.max() == 0:
        if choose_pair:
            return None, None
        else:
            return data
    
    num_nodes = data.x.size(0)
    num_frags = data.frag_y.max() + 1
    x = data.x.clone()
    frag_y = data.frag_y.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()

    # 
    frag_edge_index = frag_y[edge_index]
    frag_edge_index = frag_edge_index[:, frag_edge_index[0] != frag_edge_index[1]]
    
    nxgraph = nx.Graph()
    nxgraph.add_edges_from(frag_edge_index.t().tolist())
    num_drop_edges = max(1, int(len(nxgraph.edges()) * drop_p))
    drop_edges = random.sample(list(nxgraph.edges()), num_drop_edges)
    nxgraph.remove_edges_from(drop_edges)
    
    if choose_pair:
        u, v = random.choice(drop_edges)
        
        connected_frag_ys0 = list(nx.node_connected_component(nxgraph, u))
        connected_frag_ys1 = list(nx.node_connected_component(nxgraph, v))
        
        keepfrag_mask0 = torch.zeros(num_frags, dtype=torch.bool)
        keepfrag_mask0[connected_frag_ys0] = True
        keepnode_mask0 = keepfrag_mask0[frag_y]
        
        keepfrag_mask1 = torch.zeros(num_frags, dtype=torch.bool)
        keepfrag_mask1[connected_frag_ys1] = True
        keepnode_mask1 = keepfrag_mask1[frag_y]
               
        x0 = x[keepnode_mask0]
        edge_index0, edge_attr0 = subgraph(
            keepnode_mask0, edge_index, edge_attr, relabel_nodes=True, num_nodes=x.size(0)
            )
        
        x1 = x[keepnode_mask1]
        edge_index1, edge_attr1 = subgraph(
            keepnode_mask1, edge_index, edge_attr, relabel_nodes=True, num_nodes=x.size(0)
            )
        
        data0 = Data(x=x0, edge_index=edge_index0, edge_attr=edge_attr0)
        data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)
        
        uv_edge = edge_index[:, frag_y[edge_index[0]] == u]
        uv_edge = uv_edge[:, frag_y[uv_edge[1]] == v]
        uu, vv = uv_edge.tolist()

        dangling_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dangling_mask[uu] = True
        dangling_mask[vv] = True
        data0.dangling_mask = dangling_mask[keepnode_mask0]
        
        data1.dangling_mask = dangling_mask[keepnode_mask1]
        
        return data0, data1
        
    else:
        connected_frag_ys = list(random.choice(list(nx.connected_components(nxgraph))))
        keepfrag_mask = torch.zeros(num_frags, dtype=torch.bool)
        keepfrag_mask[connected_frag_ys] = True
        keepnode_mask = keepfrag_mask[frag_y]
        
        x = x[keepnode_mask]
        edge_index, edge_attr = subgraph(
            keepnode_mask, edge_index, edge_attr, relabel_nodes=True, num_nodes=x.size(0)
            )
    
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
        return data

def fragment_data(data, drop_p, type):
    if type == "once":
        frag_data0 = data
        frag_data1 = fragment(data, drop_p, False)
    elif type == "twice":
        frag_data0 = fragment(data, drop_p, False)
        frag_data1 = fragment(data, drop_p, False)
    elif type == "pair":
        frag_data0, frag_data1 = fragment(data, drop_p, True)
    
    return frag_data0, frag_data1  