import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_sparse import coalesce

def fragment(data):
    if data.frag_y.max() == 0:
        frag_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        frag_data.upper_num_nodes = torch.tensor([1])
        frag_data.lower_num_nodes = torch.tensor([data.x.size(0)])
        
        return frag_data
      
    num_nodes = data.x.size(0)
    
    row, col = data.edge_index
    inter_edge_index = data.edge_index[:, data.frag_y[row] != data.frag_y[col]]
    
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(data.edge_index.t().tolist())
    
    num_drop_edges = random.choice(range(0, inter_edge_index.size(1) + 1))
    drop_edge_index = random.sample(inter_edge_index.t().tolist(), num_drop_edges)
    nx_graph.remove_edges_from(drop_edge_index)
    connected_components = list(map(list, nx.connected_components(nx_graph)))
    random.shuffle(connected_components)
    
    node2newnode = torch.zeros(num_nodes, dtype=torch.long)
    offset = 0
    for component in connected_components:
        node2newnode[component] = torch.arange(offset, offset+len(component))
        offset += len(component)
        
    ###
    frag_x = data.x.clone()
    frag_edge_index = data.edge_index.clone()
    frag_edge_attr = data.edge_attr.clone()
    
    frag_x[node2newnode] = frag_x
    frag_edge_index = node2newnode[frag_edge_index]
    frag_edge_index, frag_edge_attr = coalesce(frag_edge_index, frag_edge_attr, num_nodes, num_nodes)
        
    frag_data = Data(x=frag_x, edge_index=frag_edge_index, edge_attr=frag_edge_attr)
    frag_data.upper_num_nodes = torch.tensor([len(connected_components)])
    frag_data.lower_num_nodes = torch.tensor([len(component) for component in connected_components])
        
    return frag_data