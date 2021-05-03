import os
from collections import defaultdict
import pandas as pd
import re
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition

import torch
from torch_geometric.data import Data, InMemoryDataset
from itertools import repeat

from data.dataset import MoleculeDataset, mol_to_graph_data_obj_simple
from data.splitter import generate_scaffold

def double_collate(data_list):
    data_list0, data_list1 = zip(*data_list)
    batch0 = collate(data_list0)
    batch1 = collate(data_list1)
    return batch0, batch1

def collate(data_list):
    keys = [set(data.keys) for data in data_list]
    keys = list(set.union(*keys))
    assert 'batch' not in keys

    batch = Data()

    for key in keys:
        batch[key] = []
    batch.batch = []
    batch.batch_num_nodes = []
    
    cumsum_node = 0
    cumsum_edge = 0
    cumsum_group = 0

    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
        num_groups = data.group_y.max() + 1
        
        for key in data.keys:
            item = data[key]
            if key == 'edge_index':
                item = item + cumsum_node
            elif key == 'group_y':
                item[item > -1] = item[item > -1] + cumsum_group
                
            batch[key].append(item)

        cumsum_node += num_nodes
        cumsum_edge += data.edge_index.shape[1]
        cumsum_group += num_groups
        
        batch.batch_num_nodes.append(torch.tensor([num_nodes]))

    for key in keys:
        batch[key] = torch.cat(
            batch[key], dim=data_list[0].cat_dim(key, batch[key][0])
            )
    
    batch.batch_num_nodes = torch.cat(batch.batch_num_nodes, dim=0)
    
    batch.batch = torch.cat(batch.batch, dim=-1)
    
    return batch.contiguous()


class GroupDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.data, self.slices = torch.load(os.path.join(root, "geometric_data_processed.pt"))

def main():
    dataset_name = "zinc_standard_agent"
    print("Loading dataset...")
    dataset = MoleculeDataset("../resource/dataset/" + dataset_name, dataset=dataset_name)
    smiles_list = pd.read_csv(
                '../resource/dataset/' + dataset_name + '/processed/smiles.csv', header=None
                )[0].tolist()

    scaffold2idxs = defaultdict(list)
    data_list = []
    for idx, smiles in tqdm(list(enumerate(smiles_list))):
        data = dataset[idx]
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetIntProp("SourceAtomIdx", atom.GetIdx())
        
        core = MurckoScaffold.GetScaffoldForMol(mol)
        groups, unmatched = rdRGroupDecomposition.RGroupDecompose([core], [mol], asSmiles=False)
        if len(groups) == 0:
            continue
        
        groups = groups[0]
        
        group_y = torch.full((data.x.size(0), ), -1, dtype=torch.long)
        
        label_cnt = 1
        for key in groups:
            group_mol = groups[key]
            atom_idxs = [
                atom.GetIntProp("SourceAtomIdx") 
                for atom in group_mol.GetAtoms() 
                if atom.HasProp("SourceAtomIdx")
                ]
            if key == "Core":
                group_y[atom_idxs] = 0
            else:
                group_y[atom_idxs] = label_cnt
                label_cnt += 1
        
        data.group_y = group_y
        data_list.append(data)
            
    print(len(smiles_list))
    print(len(data_list))        
    
    processed_dir = "../resource/dataset/zinc_group"
    os.makedirs(processed_dir, exist_ok=True)
    
    data, slices = dataset.collate(data_list)
    torch.save((data, slices), os.path.join(processed_dir, "geometric_data_processed.pt"))                
    
if __name__ == "__main__":
    main()
