import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def drop_nodes(x, edge_index, edge_attr, aug_severity):
    drop_rate = [0.1, 0.2, 0.3][aug_severity]
        
    num_nodes = x.size(0)
    num_keep_nodes = min(int((1 - drop_rate) * num_nodes), num_nodes - 1)
    keep_nodes = list(sorted(random.sample(range(num_nodes), num_keep_nodes)))

    x = x[keep_nodes].clone()
    edge_index, edge_attr = subgraph(
        keep_nodes, edge_index, edge_attr=edge_attr, relabel_nodes=True, num_nodes=num_nodes,
    )

    return x, edge_index, edge_attr


def mask_nodes(x, edge_index, edge_attr, mask_rate):
    num_nodes = x.size(0)
    num_mask_nodes = min(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    
    x = x.clone()
    x[mask_nodes] = 0
    
    return x, edge_index, edge_attr

def realmask_nodes(x, edge_index, edge_attr, mask_rate):
    num_nodes = x.size(0)
    num_mask_nodes = min(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    
    x = x.clone()
    x[mask_nodes] = 0
    
    return x, edge_index, edge_attr

def mask_data(data):
    x, edge_index, edge_attr = mask_nodes(
        data.x.clone(), data.edge_index.clone(), data.edge_attr.clone(), mask_rate=0.3
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dataset_graph_idx=data.dataset_graph_idx,
        dataset_node_idx=data.dataset_node_idx,
    )
    
    return data


def realmask_data(data):
    x, edge_index, edge_attr = realmask_nodes(
        data.x.clone(), data.edge_index.clone(), data.edge_attr.clone(), mask_rate=0.1
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        dataset_graph_idx=data.dataset_graph_idx,
        dataset_node_idx=data.dataset_node_idx,
    )
    
    return data

def mask_data_twice(data):
    data0 = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    data1 = mask_data(data)
    
    return data0, data1

def realmask_data_twice(data):
    data0 = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    data1 = realmask_data(data)
    
    return data0, data1