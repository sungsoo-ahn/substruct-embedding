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

from model import GraphEncoder
from data.dataset import MoleculeDataset
from data.splitter import scaffold_split


criterion = nn.BCEWithLogitsLoss(reduction="none")


def compute_all_features(model, loader, device):
    features = []
    labels = []
    num_tasks = loader.dataset.num_tasks
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            features_ = torch.nn.functional.normalize(features_, dim=1)
            features.append(features_.cpu().numpy())
            labels.append(batch.y.reshape(-1, num_tasks).cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    is_valid = labels ** 2 > 0
    labels = ((labels + 1) / 2).astype(int)

    return features, labels, is_valid


def evaluate_knn(model, train_dataset, test_dataset, device):
    model.eval()
    
    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=False, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=8
    )

    train_features, train_labels, train_is_valid = compute_all_features(model, train_loader, device)
    test_features, test_labels, test_is_valid = compute_all_features(model, test_loader, device)

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

def get_eval_datasets():
    datasets = dict()
    for name in ["tox21", "hiv", "muv", "bace", "bbbp", "toxcast", "sider", "clintox"]:
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

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tox21")
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--num_runs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(0)

    model = GraphEncoder(
        num_layers=args.num_layers, emb_dim=args.emb_dim, drop_rate=args.drop_rate,
    )
    if not args.model_path == "":
        model.load_state_dict(torch.load(args.model_path))

    model.to(device)

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    datasets = get_eval_datasets()
    for name in datasets:
        statistics = evaluate_knn(
            model, datasets[name]["train"], datasets[name]["test"], device
            )
        acc = statistics["acc"]
        print(f"{name} accuracy: {acc}")

if __name__ == "__main__":
    main()
    
"""
{'acc': 0.6445872773245684}
{'acc': 0.660085169663377}
{'acc': 0.5209195128150834}
{'acc': 0.7462180490349504}
{'acc': 0.5486593364197532}
{'acc': 0.5525562127338279}
{'acc': 0.5888277434529402}
{'acc': 0.5086604629340007}
"""

"""
{'acc': 0.6702627560789853}
{'acc': 0.6692771200679812}
{'acc': 0.5384504556925603}
{'acc': 0.767866458007303}
{'acc': 0.6362847222222223}
{'acc': 0.5698420844776477}
{'acc': 0.5869091210072744}
{'acc': 0.5804395613943627}
"""