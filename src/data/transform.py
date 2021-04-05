import re
import random
import torch
from torch_cluster import random_walk
from torch_geometric.data import Data
import networkx as nx
import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from data.util import (
    graph_data_obj_to_nx_simple,
    nx_to_graph_data_obj_simple,
    graph_data_obj_to_smarts,
    mol_to_graph_data_obj_simple,
)


class CutLinkerBond:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data0):
        mol0 = Chem.AllChem.MolFromSmiles(data0.smiles)
        cut_bond_idxs0 = mol0.GetSubstructMatches(Chem.MolFromSmarts("[!R][R]"))
        #cut_bond_idxs1 = mol0.GetSubstructMatches(Chem.MolFromSmarts("[!R][!R]"))
        #cut_bond_idxs = list(cut_bond_idxs0) + list(cut_bond_idxs1)
        cut_bond_idxs = cut_bond_idxs0            
        if len(cut_bond_idxs) == 0:
            return None
                
        cut_bond_idx = random.choice(cut_bond_idxs)
        bond = mol0.GetBondBetweenAtoms(*cut_bond_idx)
        mol_frags = Chem.FragmentOnBonds(mol0, [bond.GetIdx()])
        mol_frags = Chem.rdmolops.GetMolFrags(mol_frags, asMols=True)
        mol1 = max(mol_frags, default=mol0, key=lambda m: m.GetNumAtoms())
        smiles1 = Chem.MolToSmiles(mol1)
        smiles1 = re.sub('\[[0-9]+\*\]', '', smiles1)
        mol1 = Chem.MolFromSmiles(smiles1)
        if mol1 is None:
            return None
                
        match_idxs = mol0.GetSubstructMatch(mol1)
        if len(match_idxs) == 0:
            return None
        
        mask = torch.zeros(data0.x.size(0))
        mask[list(match_idxs)] = 1.0
        
        try:        
            data1 = mol_to_graph_data_obj_simple(mol1)
        except:
            return None
                            
        new_data = Data(
            x0=data0.x,
            edge_attr0=data0.edge_attr,
            edge_index0=data0.edge_index,
            x1=data1.x,
            edge_attr1=data1.edge_attr,
            edge_index1=data1.edge_index,
            mask=mask
        )

        return new_data


class CutLinkerAndMaskAtom(CutLinkerBond):
    def __init__(self, mask_rate=0.3):
        self.mask_rate = mask_rate

    def __call__(self, data0):
        data = super(CutLinkerAndMaskAtom, self).__call__(data0)
        if data is None:
            return None
        
        num_atoms = data.x1.size()[0]
        sample_size = int(num_atoms * self.mask_rate + 1)
        masked_atom_indices = random.sample(range(num_atoms), sample_size)
        data.x1[masked_atom_indices] = 0.0
        
        return data