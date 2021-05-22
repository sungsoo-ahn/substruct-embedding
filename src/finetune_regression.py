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

from model import GNN_graphpred
from data.dataset import MoleculeDataset
from data.splitter import scaffold_split, random_split

criterion = nn.MSELoss(reduction="none")


def train(model, optimizer, loader, device):
    model.train()

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        
        # Loss matrix
        loss = criterion(pred.double(), y).mean()
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
    return {"loss": loss.detach()}


def evaluate(model, loader, device):
    model.eval()
    losses = []
    
    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y = batch.y.view(pred.shape).to(torch.float64)
        loss = criterion(pred.double(), y)
        losses.append(loss.cpu())

    loss = torch.cat(losses, dim=0).mean()
    
    return {"metric": loss}


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=[
        "esol", "freesolv", "lipophilicity", "qm7", "qm8"])
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_scale", type=float, default=1.0)

    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--split_type", type=str, default="scaffold")
    parser.add_argument("--num_runs", type=int, default=5)

    args = parser.parse_args()

    device = torch.device(0)

    run = neptune.init(
        project="sungsahn0215/ssg", name="finetune_regression"
    )
    run["parameters"] = vars(args)

    dataset2vali_best_metric_list = {dataset: [] for dataset in args.datasets}
    dataset2test_best_metric_list = {dataset: [] for dataset in args.datasets}
    dataset2last_vali_metric_list = {dataset: [] for dataset in args.datasets}
    dataset2last_test_metric_list = {dataset: [] for dataset in args.datasets}

    for runseed in range(args.num_runs):
        for dataset_name in args.datasets:
            torch.manual_seed(runseed)
            np.random.seed(runseed)
            torch.cuda.manual_seed_all(runseed)

            dataset = MoleculeDataset("../resource/dataset/" + dataset_name, dataset=dataset_name)
            y_mean = dataset.data.y.mean(dim=0)
            y_std = dataset.data.y.std(dim=0)
            dataset.data.y = (dataset.data.y - y_mean.unsqueeze(0)) / y_std.unsqueeze(0)
            
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
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers
            )
            vali_loader = DataLoader(
                valid_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers
            )

            num_tasks = {
                "esol": 1, 
                "freesolv": 1, 
                "lipophilicity": 1, 
                "qm7": 1, 
                "qm8": 12,
            }.get(dataset_name)

            model = GNN_graphpred(
                num_layer=args.num_layers, 
                emb_dim=args.emb_dim,
                num_tasks=num_tasks,
                drop_ratio=args.drop_rate,
            )
            if not args.model_path == "":
                try:
                    model.gnn.load_state_dict(torch.load(args.model_path))
                except:
                    state_dict = torch.load(args.model_path)
                    new_state_dict = dict()
                    for key in state_dict:
                        if "encoder." in key:
                            new_state_dict[key.replace("encoder.", "")] = state_dict[key]

                    model.gnn.load_state_dict(new_state_dict)

            model.to(device)

            model_param_group = []
            model_param_group.append({"params": model.gnn.parameters()})
            model_param_group.append(
                {"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale}
                )
            optimizer = optim.Adam(model_param_group, lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
            
            vali_best_metric = None
            test_best_metric = None
            for epoch in range(args.num_epochs):                
                train_statistics = train(model, optimizer, train_loader, device)
                for key, val in train_statistics.items():
                    val *= y_std ** 2
                    run[f"{dataset_name}/run{runseed}/train/{key}"].log(val)

                scheduler.step()
    
                vali_statistics = evaluate(model, vali_loader, device)
                for key, val in vali_statistics.items():
                    val *= y_std ** 2
                    run[f"{dataset_name}/run{runseed}/vali/{key}"].log(val)

                test_statistics = evaluate(model, test_loader, device)
                for key, val in test_statistics.items():
                    val *= y_std ** 2
                    run[f"{dataset_name}/run{runseed}/test/{key}"].log(val)

                if vali_best_metric is None or vali_statistics["metric"] < vali_best_metric:
                    vali_best_metric = vali_statistics["metric"]
                    test_best_metric = test_statistics["metric"]

                run[f"{dataset_name}/run{runseed}/vali/best_metric"].log(vali_best_metric)
                run[f"{dataset_name}/run{runseed}/test/best_metric"].log(test_best_metric)

            dataset2last_vali_metric_list[dataset_name].append(vali_statistics["metric"])
            dataset2last_test_metric_list[dataset_name].append(test_statistics["metric"])
            dataset2vali_best_metric_list[dataset_name].append(vali_best_metric)
            dataset2test_best_metric_list[dataset_name].append(test_best_metric)
            
            run[f"{dataset_name}/run_avg/test/last_metric"] = np.mean(
                dataset2last_test_metric_list[dataset_name]
                )
            run[f"{dataset_name}/run_avg/test/best_metric"] = np.mean(
                dataset2test_best_metric_list[dataset_name]
                )

        run[f"dataset_avg/run_avg/test/last_metric"] = np.mean(
            [np.mean(val) for val in dataset2last_test_metric_list.values()]
            )
        run[f"dataset_avg/run_avg/test/best_metric"] = np.mean(
            [np.mean(val) for val in dataset2test_best_metric_list.values()]
            )

        
if __name__ == "__main__":
    main()
