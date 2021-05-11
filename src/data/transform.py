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


def full_to_uniq(edge_index, edge_attr):
    row, col = edge_index
    return edge_index[:, row < col], edge_attr[row < col, :]


def uniq_to_full(uniq_edge_index, uniq_edge_attr):
    row, col = uniq_edge_index
    edge_index = torch.stack([torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0)
    edge_attr = torch.cat([uniq_edge_attr, uniq_edge_attr], dim=0)
    return edge_index, edge_attr

def _get_inter_and_intra(data):
    inter_edge_mask = data.frag_y[data.edge_index[0]] != data.frag_y[data.edge_index[1]]
    inter_edge_index = data.edge_index[:, inter_edge_mask]
    inter_edge_attr = data.edge_attr[inter_edge_mask, :]

    intra_edge_index = data.edge_index[:, ~inter_edge_mask]
    intra_edge_attr = data.edge_attr[~inter_edge_mask, :]

    return inter_edge_index, inter_edge_attr, intra_edge_index, intra_edge_attr


def _fragment(data, edge_sample_func, return_dangling_edge_index=False):
    if data.frag_y.max() == 0:
        return data

    # 
    _ = _get_inter_and_intra(data)
    inter_edge_index, inter_edge_attr, intra_edge_index, intra_edge_attr = _

    # 
    uniq_inter_edge_index, uniq_inter_edge_attr = full_to_uniq(inter_edge_index, inter_edge_attr)

    #
    drop_idxs = edge_sample_func(uniq_inter_edge_index)
    if len(drop_idxs) == 0:
        return data
    
    drop_mask = torch.zeros(uniq_inter_edge_index.size(1)).to(torch.bool)
    drop_mask[drop_idxs] = True
    keep_mask = drop_mask == False

    #
    dangling_row, dangling_col = uniq_inter_edge_index[:, drop_mask]
    dangling_nodes = torch.cat([dangling_row, dangling_col], dim=0)
    fake_nodes = torch.arange(dangling_nodes.size(0)) + data.x.size(0)
    uniq_dangling_edge_index = torch.stack([dangling_nodes, fake_nodes], dim=0)
    uniq_dangling_edge_attr = torch.cat(
        [uniq_inter_edge_attr[drop_mask, :], uniq_inter_edge_attr[drop_mask, :]], dim=0
    )
    
    #
    row, col = torch.cat([uniq_inter_edge_index[:, keep_mask], uniq_dangling_edge_index], dim=1)
    inter_edge_index = torch.stack(
        [torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0
    )

    #
    inter_edge_attr = torch.cat(
        [
            uniq_inter_edge_attr[keep_mask, :],
            uniq_dangling_edge_attr,
            uniq_inter_edge_attr[keep_mask, :],
            uniq_dangling_edge_attr,
        ],
        dim=0,
    )

    #
    edge_index = torch.cat([intra_edge_index, inter_edge_index], dim=1)
    edge_attr = torch.cat([intra_edge_attr, inter_edge_attr], dim=0)
    num_nodes = data.x.size(0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

    #
    dangling_x = torch.zeros(dangling_nodes.size(0), data.x.size(1), dtype=torch.long)
    x = torch.cat([data.x, dangling_x], dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if return_dangling_edge_index:
        return data, uniq_dangling_edge_index
    else:
        return data

def _sample_fragment0(data, p):
    def _sample_func(uniq_inter_edge_index):
        num_uniq_edges = uniq_inter_edge_index.size(1)
        num_drops = max(int(p * num_uniq_edges), 1)
        return random.sample(range(num_uniq_edges), num_drops)
    
    data = _fragment(data, _sample_func)
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(data.edge_index.t().tolist())
    subgraph_nodes = list(random.choice(list(nx.connected_components(nx_graph))))

    return subgraph_data(data, subgraph_nodes)


def _sample_fragment1(data):
    def _sample_func(uniq_inter_edge_index):
        num_uniq_edges = uniq_inter_edge_index.size(1)
        num_drops = random.choice(range(num_uniq_edges + 1))
        if num_drops == 0:
            return []
        else:
            return random.sample(range(num_uniq_edges), num_drops)
    
    data = _fragment(data, _sample_func)
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(data.edge_index.t().tolist())
    subgraph_nodes = list(max(list(nx.connected_components(nx_graph)), key=len))

    return subgraph_data(data, subgraph_nodes)

def _sample_for_combine_fragment(data):
    def _sample_func(uniq_inter_edge_index):
        num_uniq_edges = uniq_inter_edge_index.size(1)
        return [random.choice(range(num_uniq_edges))]
    
    data, dangling_edge_index = _fragment(data, _sample_func, return_dangling_edge_index=True)
    
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(data.edge_index.t().tolist())
    
    idx = random.choice(range(2))
    subgraph_nodes = list(nx.connected_component(nx_graph, dangling_edge_index[0, idx].item()))
    return subgraph_data(data, subgraph_nodes), dangling_edge_index[:, idx]
    

def sample_data0(data, p):
    if data.frag_y.max() == 0:
        return data, data

    data0 = _sample_fragment0(clone_data(data), p)
    data1 = _sample_fragment0(clone_data(data), p)

    return data0, data1


def sample_data1(data):
    if data.frag_y.max() == 0:
        return data, data

    data0 = _sample_fragment1(clone_data(data))
    data1 = _sample_fragment1(clone_data(data))

    return data0, data1

def sample_data2(data):
    if data.frag_y.max() == 0:
        return data, data

    data0 = _sample_fragment0(clone_data(data), p)
    data1 = _sample_fragment0(clone_data(data))

    return data0, data1


def _combine(data0, data1, uniq_dangling_edge_index0, uniq_dangling_edge_index1):
    x0 = data0.x
    uniq_edge_index0, uniq_edge_attr0 = full_to_uniq(data0.edge_index, data0.edge_attr)

    x1 = data1.x
    uniq_edge_index1, uniq_edge_attr1 = full_to_uniq(data1.edge_index, data1.edge_attr)

    dangling_node0, fake_node0 = uniq_dangling_edge_index0.squeeze(1).tolist()
    dangling_node1, fake_node1 = uniq_dangling_edge_index1.squeeze(1).tolist() + x0.size(0)

    x = torch.cat([x0[:fake_node0], x1[:fake_node1]], dim=0)
    x = torch.cat([x, x[fake_node0 + 1 : fake_node1]], dim=0)

    uniq_edge_index = torch.cat([uniq_edge_index0, uniq_edge_index1 + x0.size(0)], dim=1)
    uniq_edge_index[uniq_edge_index == fake_node0] = dangling_node1 + x0.size(0)
    uniq_edge_index[uniq_edge_index == fake_node1 + x0.size(0)] = dangling_node0
    uniq_edge_attr = torch.cat([uniq_edge_attr0, uniq_edge_attr1], dim=0)

    edge_index, edge_attr = uniq_to_full(uniq_edge_index, uniq_edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, x.size(0), x.size(0))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def sample_for_combine_data(data):
    if data.frag_y.max() == 0:
        return None, None

    data, dangling_edge_index = _sample_fragment2(clone_data(data))
    
    return data, data, dangling_edge_index