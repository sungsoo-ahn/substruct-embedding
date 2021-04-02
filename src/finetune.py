import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

import neptune

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(model, optimizer, loader, device):
    model.train()

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype)
        )

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_scores = []

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    return sum(roc_list) / len(roc_list)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="tox21",
    )
    parser.add_argument(
        "--model_path", type=str, default="",
    )
    parser.add_argument(
        "--runseed", type=int, default=0,
    )
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    num_tasks = {
        "tox21": 12,
        "hiv": 1,
        "pcba": 128,
        "muv": 17,
        "bace": 1,
        "bbbp": 1,
        "toxcast": 617,
        "sider": 27,
        "clintox": 2,
    }.get(args.dataset)

    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset,
        dataset.smiles_list,
        null_value=0,
        frac_train=FRAC_TRAIN,
        frac_valid=FRAC_VALID,
        frac_test=FRAC_TEST,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    vali_loader = DataLoader(
        valid_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # set up model
    model = GNN_graphpred()
    if not args.model_path == "":
        model.from_pretrained(args.model_path)

    model.to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [
        {"params": model.gnn.parameters()},
        {"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale},
    ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    neptune.init(project_qualified_name="sungsoo.ahn/self-sup-graph")
    neptune.create_experiment(name="finetune", params=vars(args))
    neptune.append_tag(args.dataset)
    neptune.append_tag(args.model_path)

    best_vali_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(EPOCHS):
        train_statistics = train(model, optimizer, train_loader, device)
        vali_statistics = evaluate(model, vali_loader, device)
        test_statistics = evaluate(model, test_loader, device)

        if vali_acc > best_vali_acc:
            best_vali_acc = vali_acc
            best_test_acc = test_acc

        neptune.log_metric("val/rocauc", vali_acc)
        neptune.log_metric("val/best_rocauc", best_vali_acc)
        neptune.log_metric("test/rocauc", test_acc)
        neptune.log_metric("test/best_rocauc", best_test_acc)

    with open("result.csv", "a") as f:
        f.write(f"{args.dataset}, {args.model_path}, {best_vali_acc}, {best_test_acc}\n")


if __name__ == "__main__":
    main()
