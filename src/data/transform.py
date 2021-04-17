import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

"""
def subgraph(data, aug_severity):
    aug_rate = [0.1, 0.2, 0.3, 0.4, 0.5][aug_severity]

    x, edge_index, edge_attr = data.x.clone(), data.edge_index.clone(), data.edge_attr.clone()
    
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    sub_num = int(node_num * aug_rate)

    edge_index_np = edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index_np[1][edge_index_np[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index_np[1][edge_index_np[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
    edge_mask = np.array([n for n in range(edge_num) if (edge_index_np[0, n] in idx_nondrop and edge_index_np[1, n] in idx_nondrop)])

    edge_index_np = edge_index.numpy()
    edge_index_np = [[idx_dict[edge_index_np[0, n]], idx_dict[edge_index_np[1, n]]] for n in range(edge_num) if (not edge_index_np[0, n] in idx_drop) and (not edge_index_np[1, n] in idx_drop)]
    try:
        edge_index = torch.tensor(edge_index_np).transpose_(0, 1)
        x = x[idx_nondrop]
        edge_attr = edge_attr[edge_mask]
    except:
        pass

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data
"""

def drop_nodes(x, edge_index, edge_attr, aug_severity):
    drop_rate = [0.1, 0.2, 0.3][aug_severity]
        
    num_nodes = x.size(0)
    num_keep_nodes = min(int((1-drop_rate) * num_nodes), num_nodes - 1)
    keep_nodes = list(sorted(random.sample(range(num_nodes), num_keep_nodes)))

    x = x[keep_nodes].clone()
    edge_index, edge_attr = subgraph(
        keep_nodes, 
        edge_index, 
        edge_attr=edge_attr, 
        relabel_nodes=True, 
        num_nodes=num_nodes,
        )

    return x, edge_index, edge_attr


def mask_nodes(x, edge_index, edge_attr, aug_severity):
    mask_rate = [0.3, 0.6, 0.9][aug_severity]
    num_nodes = x.size(0)
    num_mask_nodes = min(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))
    
    x = x.clone()
    x[mask_nodes] = 0
    
    return x, edge_index, edge_attr


def compose(transforms):
    def composed_transform(x, edge_index, edge_attr):
        for transform in transforms:
            x, edge_index, edge_attr = transform(x, edge_index, edge_attr)
        
        return x, edge_index, edge_attr
    
    return composed_transform

