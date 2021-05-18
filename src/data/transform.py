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

def fragment(data, drop_p, min_num_nodes, aug_x):
    if data.frag_y.max() == 0:
        return None, None
    
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
    
    u, v = random.choice(drop_edges)
    connected_frag_ys0 = list(nx.node_connected_component(nxgraph, u))
    connected_frag_ys1 = list(nx.node_connected_component(nxgraph, v))
    
    if len(connected_frag_ys0) + len(connected_frag_ys1) < min_num_nodes:
        return None, None
    
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
    
    if aug_x:
        data0.x = torch.cat([data0.x, data0.dangling_mask.unsqueeze(1).long()], dim=1)
        data1.x = torch.cat([data1.x, data1.dangling_mask.unsqueeze(1).long()], dim=1)
    
    return data0, data1

def multi_fragment(data, mask_p):
    if data.frag_y.max() == 0:
        return None
      
    num_nodes = data.x.size(0)
    
    ### sample edges to drop
    # get undirected edge_index edge_attr
    uniq_mask = data.edge_index[0] < data.edge_index[1]
    uniq_edge_index = data.edge_index[:, uniq_mask]
    uniq_edge_attr = data.edge_attr[uniq_mask, :]
    
    # get inter edge_index edge_attr
    inter_mask = data.frag_y[uniq_edge_index[0]] != data.frag_y[uniq_edge_index[1]]
    uniq_inter_edge_index = uniq_edge_index[:, inter_mask]
    uniq_inter_edge_attr = uniq_edge_attr[inter_mask, :]
    
    # get intra edge_index edge_attr
    intra_mask = data.frag_y[uniq_edge_index[0]] == data.frag_y[uniq_edge_index[1]]
    uniq_intra_edge_index = uniq_edge_index[:, intra_mask]
    uniq_intra_edge_attr = uniq_edge_attr[intra_mask, :]
    
    # get drop edges
    num_uniq_inter_edges = uniq_inter_edge_index.size(1)
    num_drops = max(1, int(num_uniq_inter_edges * mask_p))
    drop_idxs = random.sample(range(num_uniq_inter_edges), num_drops)
    drop_edge_index = uniq_inter_edge_index[:, drop_idxs]   
    drop_edge_attr = uniq_inter_edge_attr[drop_idxs, :]
    
    # get keep edge_index edge_attr
    keep_idxs = [idx for idx in range(num_uniq_inter_edges) if idx not in drop_idxs]
    uniq_inter_edge_index = uniq_inter_edge_index[:, keep_idxs]
    uniq_inter_edge_attr = uniq_inter_edge_attr[keep_idxs, :]
    
    # create new data with dropped edges
    new_x = data.x.clone()
    
    new_uniq_edge_index0 = torch.cat([uniq_intra_edge_index, uniq_inter_edge_index], dim=1)
    new_uniq_edge_index1 = torch.roll(new_uniq_edge_index0, shifts=1, dims=0)
    new_edge_index = torch.cat([new_uniq_edge_index0, new_uniq_edge_index1], dim=1)
    
    new_uniq_edge_attr = torch.cat([uniq_intra_edge_attr, uniq_inter_edge_attr])
    new_edge_attr = torch.cat([new_uniq_edge_attr, new_uniq_edge_attr], dim=0)
    
    # extract connected fragments
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(uniq_intra_edge_index.t().tolist())
    nx_graph.add_edges_from(uniq_inter_edge_index.t().tolist())
    frag_nodes_list = list(map(list, nx.connected_components(nx_graph)))
        
    frag_num_nodes = torch.tensor([len(frag_nodes) for frag_nodes in frag_nodes_list])
        
    # create mapping to sort fragments
    newnode2node = torch.tensor(
        [node for frag_nodes in frag_nodes_list for node in frag_nodes], dtype=torch.long
        )
    node2newnode = torch.empty_like(newnode2node)
    node2newnode[newnode2node] = torch.arange(num_nodes)
    
    node2frag = torch.zeros(num_nodes, dtype=torch.long)
    for idx, frag_nodes in enumerate(frag_nodes_list):
        node2frag[frag_nodes] = idx
    
    # relabel 
    new_x = new_x[newnode2node]
    new_edge_index = node2newnode[new_edge_index]
    new_edge_index, new_edge_attr = coalesce(new_edge_index, new_edge_attr, num_nodes, num_nodes)
    frag_edge_index = node2frag[drop_edge_index]
    drop_edge_index = node2newnode[drop_edge_index]
    
    # create dangling_mask
    dangling_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dangling_mask[drop_edge_index[0]] = True
    dangling_mask[drop_edge_index[1]] = True
    
    # create dangling_adj
    num_dangling_nodes = dangling_mask.long().sum().item()
    node2dangling_node = torch.full((num_nodes, ), -1, dtype=torch.long)
    node2dangling_node[dangling_mask] = torch.arange(num_dangling_nodes)
    dangling_edge_index = node2dangling_node[drop_edge_index]
    
    # create new data
    new_data = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
    new_data.frag_num_nodes = frag_num_nodes
    new_data.dangling_mask = dangling_mask
    new_data.dangling_edge_index = dangling_edge_index
    new_data.frag_edge_index = frag_edge_index
        
    new_data.dangling_edge_attr = drop_edge_attr

    return new_data