import random
import torch
from torch_cluster import random_walk
import networkx as nx
import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from data.util import graph_data_obj_to_nx_simple, nx_to_graph_data_obj_simple, graph_data_obj_to_smarts, mol_to_graph_data_obj_simple



def reset_idxes(nx_graph):
    mapping = {}
    for new_idx, old_idx in enumerate(nx_graph.nodes()):
        mapping[old_idx] = new_idx
    nx_graph = nx.relabel_nodes(nx_graph, mapping, copy=True)
    return nx_graph

class AddSubStruct:
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, data):
        subdata = self.sample_subdata(data)
        
        data.sub_x = subdata.x
        data.sub_edge_attr = subdata.edge_attr
        data.sub_edge_index = subdata.edge_index
        
        return data
    
    def sample_subdata(self, data):
        raise NotImplementedError

class AddRandomWalkSubStruct(AddSubStruct):
    def __init__(self, min_walk_length, max_walk_length):
        self.min_walk_length = min_walk_length
        self.max_walk_length = max_walk_length

    def sample_subdata(self, data):
        nx_graph = graph_data_obj_to_nx_simple(data)
        root_node = random.sample(range(data.num_nodes), 1)[0]
        walk_length = random.randint(self.min_walk_length, self.max_walk_length)
        
        randomwalk_nodes = random_walk(
            data.edge_index[0], 
            data.edge_index[1],
            torch.tensor([root_node]), 
            walk_length=walk_length
        ).squeeze(0)
        inducing_nodes = torch.unique(randomwalk_nodes).tolist()
        induced_nx_graph = nx_graph.subgraph(inducing_nodes)
        induced_nx_graph = reset_idxes(induced_nx_graph)
        subdata = nx_to_graph_data_obj_simple(induced_nx_graph)

        return subdata

class AddMurckoSubStruct(AddSubStruct):
    def __init__(self):
        pass
    
    def sample_subdata(self, data):
        scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=data.smiles, includeChirality=True
            )
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        subdata = mol_to_graph_data_obj_simple(scaffold_mol)
        return subdata