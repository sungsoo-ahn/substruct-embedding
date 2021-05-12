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

def fragment(data, mask_p=0, min_num_frags=0, max_num_frags=100):
    if data.frag_y.max() < min_num_frags:
        return None, None
        
    row, col = data.edge_index
    inter_edge_index = data.edge_index[:, data.frag_y[row] != data.frag_y[col]]
    inter_edge_attr = data.edge_attr[data.frag_y[row] != data.frag_y[col], :]
    frag_edge_index = data.frag_y[inter_edge_index]
    
    edge_list = [(u, v, {"edge_idx": idx}) for idx, (u, v) in enumerate(frag_edge_index.t().tolist())]
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edge_list)
    num_drop_edges = random.choice(range(min_num_frags-1, min(max_num_frags, len(nx_graph.edges)+1)))
    if num_drop_edges == 0:
        super_data = Data(
            x=torch.empty(1, dtype=torch.long), 
            edge_index=torch.empty(2, 0, dtype=torch.long), 
            edge_attr=torch.empty(0, 2, dtype=torch.long),
            )
        super_data.mask = torch.zeros(1, dtype=torch.bool)
        
        return [data], super_data
    
    drop_edges = random.sample(list(nx_graph.edges(data=True)), num_drop_edges)
    
    nx_graph = nx_graph.copy()
    nx_graph.remove_edges_from([(u, v) for u, v, _ in drop_edges])
    connected_components = list(map(list, nx.connected_components(nx_graph)))
            
    frag_data_list = []
    for component_idx, component in enumerate(connected_components):
        subgraph_nodes = torch.sum(
            (data.frag_y.unsqueeze(1) == torch.tensor(component)
             ).long(), dim=1).nonzero().squeeze(1)        
                
        frag_data_list.append(subgraph_data(data, subgraph_nodes))
        
        for node in component:
            nx_graph.nodes[node]["component_idx"] = component_idx
            
    node2component_idx = {u: nx_graph.nodes[u]["component_idx"] for u in nx_graph.nodes}
    super_edge_index = torch.tensor(
        [[node2component_idx[u], node2component_idx[v]] for u, v, _ in drop_edges]
        + [[node2component_idx[v], node2component_idx[u]] for u, v, _ in drop_edges],
        dtype=torch.long
    ).t()
    
    super_edge_attr = inter_edge_attr[[edge[2]["edge_idx"] for edge in drop_edges]]
    super_edge_attr = torch.cat([super_edge_attr, super_edge_attr], dim=0)

    num_super_nodes = len(connected_components)
    super_edge_index, super_edge_attr = coalesce(
        super_edge_index, super_edge_attr, num_super_nodes, num_super_nodes
        )
    
    #print(data.x.size())
    #print(super_edge_index)
    #assert False
    
    super_data = Data(
        x=torch.empty(num_super_nodes, dtype=torch.long), 
        edge_index = super_edge_index,
        edge_attr = super_edge_attr,
        )
    
    #if mask_p > 0.0:
    #    frag_data_list = [mask_data(data, mask_p) for data in frag_data_list]
        
    super_data.mask = get_mask(super_data, 0.0)
    
    return frag_data_list, super_data

def double_fragment(data, mask_p=0, min_num_frags=0, max_num_frags=100):
    data0, super_data0 = fragment(data, mask_p=mask_p, min_num_frags=min_num_frags)
    data1, super_data1 = fragment(data, mask_p=mask_p, min_num_frags=min_num_frags)
    
    return data0, data1, super_data0, super_data1
    