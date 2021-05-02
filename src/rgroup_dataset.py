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

class RGroupData(Data):
    def cat_dim(self, key, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # `*index*` and `*face*` should be concatenated in the last dimension,
        # everything else in the first dimension.
        return -1 if bool(re.search('(index|face)', key)) else 0

class ScaffoldDataset(InMemoryDataset):
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

    def get(self, idx):
        data = ScaffoldData()
        
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.cat_dim(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        
        return data

    def len(self):
        for item in self.slices.values():
            return len(item) - 1

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
