import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import neptune.new as neptune

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader

from model import GNN
from data.dataset import MoleculeDataset
from data.splitter import scaffold_split, random_split
from torch_geometric.nn import global_mean_pool

import old_model

def extract_tsr_dataset(model, loader, device):
    x_dataset = []
    y_dataset = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = global_mean_pool(out, batch.batch)
            x_dataset.append(out.cpu())
            y_dataset.append(batch.y)

    x_dataset = torch.cat(x_dataset, dim=0)
    y_dataset = torch.cat(y_dataset, dim=0)
    tsr_dataset = torch.utils.data.TensorDataset(x_dataset, y_dataset.view(x_dataset.size(0), -1))
    
    return tsr_dataset

def train_classification(model, optimizer, loader, device):
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.train()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        y = y.view(pred.shape).to(torch.float64)
        
        # Whether y is non-null or not.
        is_valid = y ** 2 > 1e-6
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


def evaluate_classification(model, loader, device):
    model.eval()
    y_true = []
    y_scores = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x)
            
        y = y.view(pred.shape).to(torch.float64)

        y_true.append(y.reshape(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 1e-6 and np.sum(y_true[:, i] == -1) > 1e-6:
            is_valid = y_true[:, i] ** 2 > 1e-6
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    score = sum(roc_list) / len(roc_list)
    return {"score": score}

def train_regression(model, optimizer, loader, device):
    criterion = nn.MSELoss(reduction="none")
    model.train()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        
        # Loss matrix
        loss = criterion(pred, y).mean()
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
    return {"loss": loss.detach()}


def evaluate_regression(model, loader, device):
    criterion = nn.MSELoss(reduction="none")
    model.eval()
    mses = []
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x)
        
        mse = criterion(pred, y)
        mses.append(mse.cpu())

    rmse = torch.cat(mses, dim=0).mean() ** 0.5
    
    return {"score": -rmse}

def evaluate_regression_mae(model, loader, device):
    criterion = nn.L1Loss(reduction="none")
    model.eval()
    maes = []
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = model(x)
        
        mae = criterion(pred.double(), y)
        maes.append(mae.cpu())

    mae = torch.cat(maes, dim=0).mean() ** 0.5
    
    return {"score": -mae}

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=[
        "freesolv", 
        "esol", 
        "sider", 
        "bace", 
        "bbbp", 
        "clintox", 
        "lipophilicity", 
        "tox21",
        "qm7",
        "toxcast", 
        "qm8",
        "hiv", 
        "muv", 
        ])
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--num_atom_type", type=int, default=120)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_scale", type=float, default=1.0)

    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--split_type", type=str, default="scaffold")
    parser.add_argument("--num_runs", type=int, default=5)

    args = parser.parse_args()

    device = torch.device(0)

    run = neptune.init(
        project="sungsahn0215/ssg-finetune", name="finetune_linear"
    )
    run["parameters"] = vars(args)

    dataset2vali_best_score_list = {dataset: [] for dataset in args.datasets}
    dataset2test_best_score_list = {dataset: [] for dataset in args.datasets}
    dataset2last_vali_score_list = {dataset: [] for dataset in args.datasets}
    dataset2last_test_score_list = {dataset: [] for dataset in args.datasets}

    for runseed in range(args.num_runs):
        for dataset_name in args.datasets:
            torch.manual_seed(runseed)
            np.random.seed(runseed)
            torch.cuda.manual_seed_all(runseed)

            if dataset_name in [
                "bace", "bbbp", "sider", "clintox", "tox21", "toxcast", "hiv", "muv"
                ]:
                task = "classification"
                train = train_classification
                evaluate = evaluate_classification
                
            elif dataset_name in ["esol", "freesolv", "lipophilicity"]:
                task = "regression"
                train = train_regression
                evaluate = evaluate_regression
            
            elif dataset_name in ["qm7", "qm8"]:
                task = "regression_mae"
                train = train_regression
                evaluate = evaluate_regression_mae
            
                
            dataset = MoleculeDataset("../resource/dataset/" + dataset_name, dataset=dataset_name)
            smiles_list = pd.read_csv(
                '../resource/dataset/' + dataset_name + '/processed/smiles.csv', header=None
                )[0].tolist()
        
            if args.split_type == "scaffold":
                train_dataset, valid_dataset, test_dataset = scaffold_split(
                    dataset,
                    smiles_list,
                    null_value=0,
                    frac_train=0.8,
                    frac_valid=0.1,
                    frac_test=0.1,
                )
            
            elif args.split_type == "random":
                train_dataset, valid_dataset, test_dataset = random_split(
                    dataset,
                    null_value=0,
                    frac_train=0.8,
                    frac_valid=0.1,
                    frac_test=0.1,
                )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=1024, 
                shuffle=True, 
                num_workers=args.num_workers,
            )
            vali_loader = DataLoader(
                valid_dataset, 
                batch_size=1024, 
                shuffle=False, 
                num_workers=args.num_workers
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=1024, 
                shuffle=False, 
                num_workers=args.num_workers
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
                "esol": 1, 
                "freesolv": 1, 
                "lipophilicity": 1, 
                "qm7": 1,
                "qm8": 12,
            }.get(dataset_name)

            if not args.model_path == "":
                try:
                    encoder = GNN(
                        num_layer=args.num_layers, 
                        emb_dim=args.emb_dim,
                        drop_ratio=args.drop_rate,
                        num_atom_type=args.num_atom_type,
                    )
                    encoder.gnn.load_state_dict(torch.load(args.model_path))
                except:
                    encoder = old_model.GNN(
                        num_layer=args.num_layers, 
                        emb_dim=args.emb_dim,
                        drop_ratio=args.drop_rate,
                        num_atom_type=args.num_atom_type,
                    )
            else:
                encoder = GNN(
                        num_layer=args.num_layers, 
                        emb_dim=args.emb_dim,
                        drop_ratio=args.drop_rate,
                        num_atom_type=args.num_atom_type,
                    )
            
            encoder.to(device)
            encoder.eval()
            
            train_dataset = extract_tsr_dataset(encoder, train_loader, device)
            valid_dataset = extract_tsr_dataset(encoder, vali_loader, device)
            test_dataset = extract_tsr_dataset(encoder, test_loader, device)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers,
            )
            vali_loader = torch.utils.data.DataLoader(
                valid_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers
            )
            
            model = torch.nn.Linear(args.emb_dim, num_tasks).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            
            vali_best_score = -1e8
            test_best_score = -1e8
            for epoch in range(args.num_epochs):                
                train_statistics = train(model, optimizer, train_loader, device)
                for key, val in train_statistics.items():
                    run[f"{dataset_name}/run{runseed}/train/{key}"].log(val)
    
                vali_statistics = evaluate(model, vali_loader, device)
                for key, val in vali_statistics.items():
                    run[f"{dataset_name}/run{runseed}/vali/{key}"].log(val)

                test_statistics = evaluate(model, test_loader, device)
                for key, val in test_statistics.items():
                    run[f"{dataset_name}/run{runseed}/test/{key}"].log(val)

                if vali_statistics["score"] > vali_best_score:
                    vali_best_score = vali_statistics["score"]
                    test_best_score = test_statistics["score"]

                run[f"{dataset_name}/run{runseed}/vali/best_score"].log(vali_best_score)
                run[f"{dataset_name}/run{runseed}/test/best_score"].log(test_best_score)

            dataset2last_vali_score_list[dataset_name].append(vali_statistics["score"])
            dataset2last_test_score_list[dataset_name].append(test_statistics["score"])
            dataset2vali_best_score_list[dataset_name].append(vali_best_score)
            dataset2test_best_score_list[dataset_name].append(test_best_score)
            
            run[f"{dataset_name}/run_avg/test/best_score"] = np.mean(
                dataset2test_best_score_list[dataset_name]
                )
            run[f"{dataset_name}/run_avg/test/best_score_std"] = np.std(
                dataset2test_best_score_list[dataset_name]
                )
            
            run[f"dataset_avg/run_avg/test/best_score"] = np.mean(
                [np.mean(val) for val in dataset2test_best_score_list.values()]
                )

        
if __name__ == "__main__":
    main()
