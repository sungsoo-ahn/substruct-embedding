import os
from collections import defaultdict
import pandas as pd
import re
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch_geometric.data import Data, InMemoryDataset
from itertools import repeat

from data.dataset import MoleculeDataset, mol_to_graph_data_obj_simple
from data.splitter import generate_scaffold

class ScaffoldData(Data):
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
    new_data_list = []
    new_smiles_list = []
    scaffold_data_list = []
    scaffold_smiles_list = []
    unique_scaffold_smiles_list = dict()
    unique_cnt = 0

    for idx, smiles in (list(enumerate(smiles_list))):
        data = dataset[idx]
        mol = Chem.MolFromSmiles(smiles)
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetProp('atomNote', str(idx))

        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if len(scaffold_mol.GetAtoms()) == 0:
            continue

        scaffold_idxs = [int(atom.GetProp('atomNote')) for atom in scaffold_mol.GetAtoms()]

        scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
        scaffold_smiles_list.append(smiles)
        if scaffold_smiles not in unique_scaffold_smiles_list:
            unique_scaffold_smiles_list[scaffold_smiles] = unique_cnt
            unique_cnt += 1

        scaffold_data = mol_to_graph_data_obj_simple(scaffold_mol)

        new_data = ScaffoldData()
        new_data.x = data.x
        new_data.edge_index = data.edge_index
        new_data.edge_attr = data.edge_attr

        new_data.scaffold_x = scaffold_data.x
        new_data.scaffold_edge_index = scaffold_data.edge_index
        new_data.scaffold_edge_attr = scaffold_data.edge_attr

        new_data.scaffold_mask = torch.zeros(data.x.size(0))
        new_data.scaffold_mask[scaffold_idxs] = True
        new_data.scaffold_y = torch.tensor([unique_scaffold_smiles_list[scaffold_smiles]])

        new_data_list.append(new_data)

        print(len(new_data_list))

    processed_dir = "../resource/dataset/zinc_scaffold"
    os.makedirs(processed_dir, exist_ok=True)
    scaffold_smiles_series = pd.Series(scaffold_smiles_list)
    scaffold_smiles_series.to_csv(
        os.path.join(processed_dir, 'scaffold_smiles.csv'), index=False, header=False
        )

    unique_scaffold_smiles_series = pd.Series(unique_scaffold_smiles_list)
    unique_scaffold_smiles_series.to_csv(
        os.path.join(processed_dir, 'unique_scaffold_smiles.csv'), index=False, header=False
        )

    data, slices = dataset.collate(new_data_list)
    torch.save((data, slices), os.path.join(processed_dir, "geometric_data_processed.pt"))

if __name__ == "__main__":
    main()
