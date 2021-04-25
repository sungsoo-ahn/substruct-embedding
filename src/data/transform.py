import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_cluster import random_walk

def randomwalk_subgraph_data(data):
    walk_ratio = 1.0
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone() 
    
    start = torch.tensor(random.choice(range(x.size(0))))
    walk_length = int(x.size(0) * walk_ratio)
    walk_nodes = random_walk(edge_index[0], edge_index[1], start, walk_length)
    keep_nodes = torch.unique(walk_nodes)

    subgraph_mask = torch.zeros(x.size(0), dtype=torch.bool)
    subgraph_mask[keep_nodes] = True

    x = x[keep_nodes].clone()
    edge_index, edge_attr = subgraph(
        keep_nodes, 
        edge_index.clone(), 
        edge_attr=edge_attr.clone(), 
        relabel_nodes=True, 
        num_nodes=x.size(0),
    )
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.subgraph_mask = torch.ones(x.size(0), dtype=torch.bool)
    return subgraph_mask, data


def khop_subgraph_data(data):
    num_hops = 10
    num_nodes = data.x.size(0)
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone() 
    
    start = torch.tensor([random.choice(range(num_nodes))])
    keep_nodes = k_hop_subgraph(start, num_hops, edge_index, num_nodes=num_nodes)[0]
    keep_nodes = torch.sort(keep_nodes)[0]
    
    subgraph_mask = torch.zeros(num_nodes, dtype=torch.bool)
    subgraph_mask[keep_nodes] = True

    x = x[keep_nodes]
    edge_index, edge_attr = subgraph(
        keep_nodes, 
        edge_index, 
        edge_attr=edge_attr, 
        relabel_nodes=True, 
        num_nodes=num_nodes,
    )
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.subgraph_mask = torch.ones(x.size(0), dtype=torch.bool)
    return subgraph_mask, data


def randomwalk_subgraph_data_twice(data):
    data0 = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    subgraph_mask, data1 = randomwalk_subgraph_data(data)
    data0.subgraph_mask = subgraph_mask
    
    return data0, data1


def khop_subgraph_data_twice(data):
    data0 = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    subgraph_mask, data1 = khop_subgraph_data(data)
    data0.subgraph_mask = subgraph_mask
    
    return data0, data1

def mask_data(data, mask_rate=0.1):
    num_nodes = data.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    
    x = data.x.clone()
    x[mask_nodes, 0] = 0
    
    data = Data(
        x=x,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    
    return data    


def mask_data_and_node_label(data, mask_rate=0.1):
    num_nodes = data.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    
    x = data.x.clone()
    x[mask_nodes, 0] = 0
    
    y = data.x[:, 0].clone()
    
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[mask_nodes] = True
    
    data = Data(
        x=x,
        y=y,
        node_mask=node_mask,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    
    return data

def mask_data_and_rw_label(data, walk_length, mask_rate=0.1):   
    num_nodes = data.x.size(0)
    num_mask_nodes = max(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    
    x = data.x.clone()
    x[mask_nodes, 0] = 0
    
    if walk_length > 0:
        start = torch.arange(num_nodes)
        rw_nodes = random_walk(data.edge_index[0], data.edge_index[1], start, walk_length)
    else:
        rw_nodes = torch.arange(num_nodes).unsqueeze(1)
    
    z = data.x[:, 0][rw_nodes].clone()

    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[mask_nodes] = True
    
    z_mask = node_mask[rw_nodes]
    
    data = Data(
        x=x,
        z=z,
        z_mask=z_mask,
        node_mask=node_mask,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    
    return data
    

def mask_data_twice(data, mask_rate=0.1):
    """
    data0 = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone(),
        dataset_graph_idx=data.dataset_graph_idx.clone(),
        dataset_node_idx=data.dataset_node_idx.clone(),
    )
    """
    data0 = mask_data(data, mask_rate=0.1)
    data1 = mask_data(data, mask_rate=0.1)
    
    return data0, data1

