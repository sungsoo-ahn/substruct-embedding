from torch_geometric.data import Data

def subgraph(data, aug_severity):
    aug_ratio = [0.1, 0.2, 0.3, 0.4, 0.5][aug_severity]

    x, edge_index, edge_attr = data.x.clone(), data.edge_index.clone(), data.edge_attr.clone()
    
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    sub_num = int(node_num * aug_ratio)

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


def drop_nodes(data, aug_severity):
    aug_ratio = [0.1, 0.2, 0.3, 0.4, 0.5][aug_severity]
    
    x, edge_index, edge_attr = data.x.clone(), data.edge_index.clone(), data.edge_attr.clone()

    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_drop = idx_perm[:drop_num]
    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    edge_index_np = edge_index.numpy()
    edge_mask = np.array([n for n in range(edge_num) if not (edge_index_np[0, n] in idx_drop or edge_index_np[1, n] in idx_drop)])

    edge_index_np = [[idx_dict[edge_index_np[0, n]], idx_dict[edge_index_np[1, n]]] for n in range(edge_num) if (not edge_index_np[0, n] in idx_drop) and (not edge_index_np[1, n] in idx_drop)]
    try:
        edge_index = torch.tensor(edge_index_np).transpose_(0, 1)
        x = x[idx_nondrop]
        edge_attr = edge_attr[edge_mask]
    except:
        pass
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def random_transform(data, aug_severity):
    n = np.random.randint(2)
    if n == 0:
        data = drop_nodes(data, aug_severity)
    elif n == 1:
        data = subgraph(data, aug_severity)
    
    return data

