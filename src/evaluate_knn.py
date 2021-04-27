import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier

from model import GNN
from data.dataset import MoleculeDataset
from data.splitter import scaffold_split
from torch_geometric.nn import global_mean_pool

def compute_all_features(featurizer, loader):
    features = []
    labels = []
    num_tasks = loader.dataset.num_tasks
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(0)
            features_ = featurizer(batch)
            features.append(features_.cpu().numpy())
            labels.append(batch.y.reshape(-1, num_tasks).cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    is_valid = labels ** 2 > 0
    labels = ((labels + 1) / 2).astype(int)

    return features, labels, is_valid


def evaluate_knn(featurizer, train_dataset, test_dataset):
   
    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=False, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=8
    )

    train_features, train_labels, train_is_valid = compute_all_features(featurizer, train_loader)
    test_features, test_labels, test_is_valid = compute_all_features(featurizer, test_loader)

    roc_list = []
    for idx in range(train_loader.dataset.num_tasks):
        if np.sum(test_labels[test_is_valid[:, idx], idx] == 1) == 0:
            continue
        elif np.sum(test_labels[test_is_valid[:, idx], idx] == 0) == 0:
            continue
        
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(train_features[train_is_valid[:, idx]], train_labels[train_is_valid[:, idx], idx])

        probs = neigh.predict_proba(test_features[test_is_valid[:, idx]])    
        score = roc_auc_score(test_labels[test_is_valid[:, idx], idx], probs[:, 1])
        roc_list.append(score)

    acc = sum(roc_list) / len(roc_list)
    return {"acc": acc}

def get_eval_datasets(names=None):
    if names is None:
        names = ["tox21", "hiv", "muv", "bace", "bbbp", "toxcast", "sider", "clintox"]
    
    datasets = dict()
    for name in names:
        print(f"Loading {name} dataset...")
        dataset = MoleculeDataset("../resource/dataset/" + name, dataset=name)
        train_dataset, _, test_dataset = scaffold_split(
            dataset,
            dataset.smiles_list,
            null_value=0,
            frac_train=0.9,
            frac_valid=0.0,
            frac_test=0.1,
        )
        
        datasets[name] = {"train": train_dataset, "test": test_dataset}
    
    return datasets