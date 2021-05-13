import os
from collections import defaultdict
import pandas as pd
import re
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition

import torch
from torch_geometric.data import Data, InMemoryDataset
from itertools import repeat

from data.dataset import MoleculeDataset
from MACCS import smartsPatts
import json
import pickle

class MotifDataset(InMemoryDataset):
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
    FG_SMARTS_LIST = []
    with open("./functional_groups.csv", "r") as f:
        for line in f.readlines():
            if len(line) and line[0] != '#':
                splitL = line.split('\t')
                if len(splitL) >= 3:
                    smarts = Chem.MolFromSmarts(splitL[2])
                    FG_SMARTS_LIST.append(smarts)

    MACCS_SMARTS_LIST = []
    for idx in range(1, 167):
        patt = smartsPatts[idx][0]
        if patt == '?':
            continue
        try:
            MACCS_SMARTS_LIST.append(Chem.MolFromSmarts(patt))
        except:
            print(f"MACCS {idx} failed")    


    dataset_name = "zinc_standard_agent"
    smiles_list = pd.read_csv(
                '../resource/dataset/' + dataset_name + '/processed/smiles.csv', header=None
                )[0].tolist()
    
    motifs_list = defaultdict(list)
    for idx, smiles in tqdm(list(enumerate(smiles_list))):
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            atom.SetIntProp("SourceAtomIdx", atom.GetIdx())
        
        
        # BRICS
        brics_motif_nodes = []
        brics_bonds = BRICS.FindBRICSBonds(mol)
        fragged_mol = BRICS.BreakBRICSBonds(mol, bonds=brics_bonds)
        frags = Chem.GetMolFrags(fragged_mol, asMols=True)        
        for frag in frags:
            frag_nodes = [
                atom.GetIntProp("SourceAtomIdx") 
                for atom in frag.GetAtoms() 
                if atom.HasProp("SourceAtomIdx")
                ]
            brics_motif_nodes.append(frag_nodes)
        
        # Murcko scaffold + R-group
        murcko_core_motif_nodes = []
        murcko_rgroup_motif_nodes = []
        
        core = MurckoScaffold.GetScaffoldForMol(mol)
        groups, _ = rdRGroupDecomposition.RGroupDecompose([core], [mol], asSmiles=False)
        if len(groups) > 0:    
            for key, group_mol in groups[0].items():
                group_nodes = [
                    atom.GetIntProp("SourceAtomIdx") 
                    for atom in group_mol.GetAtoms() 
                    if atom.HasProp("SourceAtomIdx")
                    ]
                if key == "Core":
                    murcko_core_motif_nodes.append(group_nodes)
                else:
                    murcko_rgroup_motif_nodes.append(group_nodes)
                            
        # MACCS key
        maccs_motif_nodes = []
        for smarts in MACCS_SMARTS_LIST:
            matches = mol.GetSubstructMatches(smarts)
            for nodes in matches:
                if len(nodes) > 1:
                    maccs_motif_nodes.append(list(nodes))
                
        fg_motif_nodes = []
        for smarts in FG_SMARTS_LIST:
            matches = mol.GetSubstructMatches(smarts)
            for nodes in matches:
                if len(nodes) > 1:
                    fg_motif_nodes.append(list(nodes))
            
        motifs_list["brics"].append(brics_motif_nodes)
        motifs_list["murcko_core"].append(murcko_core_motif_nodes)
        motifs_list["murcko_rgroup"].append(murcko_rgroup_motif_nodes)
        motifs_list["maccs"].append(maccs_motif_nodes)
        motifs_list["fg"].append(fg_motif_nodes)

    with open("../resource/dataset/motif_list.pkl", "wb") as f:
        pickle.dump(motifs_list, f)
    
if __name__ == "__main__":
    main()
