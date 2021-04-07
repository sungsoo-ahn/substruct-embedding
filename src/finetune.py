import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import neptune.new as neptune

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader

from model import GraphEncoderWithHead
from data.dataset import MoleculeDataset
from data.splitter import scaffold_split

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

    return {"loss": loss.detach()}

def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_scores = []

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.reshape(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    acc = sum(roc_list) / len(roc_list)
    return {"acc": acc}


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tox21")
    parser.add_argument("--model_path", type=str, default="")
    
    parser.add_argument("--num_epochs", type=float, default=100)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--neptune_mode", type=str, default="sync")

    args = parser.parse_args()

    device = torch.device(0)
    
    run = neptune.init(
        project="sungsahn0215/relation-embedding", name="finetune", mode=args.neptune_mode
        )
    run["parameters"] = vars(args)

    for runseed in range(args.num_runs):
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        torch.cuda.manual_seed_all(args.runseed)

        dataset = MoleculeDataset("../resource/dataset/" + args.dataset, dataset=args.dataset)
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset,
            dataset.smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        vali_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

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

        model = GraphEncoderWithHead(
            num_head_layers=1,
            head_dim=num_tasks,
            num_encoder_layers=args.num_layers,
            emb_dim=args.emb_dim,
            drop_rate=args.drop_rate,
        )
        if not args.model_path == "":
            model.encoder.load_state_dict(torch.load(args.model_path))

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_vali_acc = 0.0
        best_test_acc = 0.0
        for epoch in range(args.num_epochs):
            train_statistics = train(model, optimizer, train_loader, device)
            for key, val in train_statistics.items():
                run[f"train/{key}/run{runseed}"].log(val)

            vali_statistics = evaluate(model, vali_loader, device)
            for key, val in vali_statistics.items():
                run[f"vali/{key}/run{runseed}"].log(val)

            test_statistics = evaluate(model, test_loader, device)
            for key, val in test_statistics.items():
                run[f"test/{key}/run{runseed}"].log(val)

            if vali_statistics["acc"] > best_vali_acc:
                best_vali_acc = vali_statistics["acc"]
                best_test_acc = test_statistics["acc"]

            run[f"vali/best_acc/run{runseed}"].log(best_vali_acc)
            run[f"test/best_acc/run{runseed}"].log(best_test_acc)

        run[f"vali/final_acc/run{runseed}"] = best_vali_acc
        run[f"test/final_acc/run{runseed}"] = best_test_acc

if __name__ == "__main__":
    main()
