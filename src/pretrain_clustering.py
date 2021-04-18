import os
from collections import defaultdict
import argparse
import numpy as np

import torch
import torch_geometric

from model import NodeEncoder
from data.dataset import MoleculeDataset
from data.splitter import random_split
from data.transform import compose, drop_nodes, mask_nodes
from scheme.graph_clustering import GraphClusteringScheme
from evaluate_knn import get_eval_datasets, evaluate_knn

import neptune.new as neptune

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_warmup_epochs", type=int, default=1)

    parser.add_argument("--scheme", type=str, default="graph_clustering")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cluster_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--log_freq", type=float, default=100)

    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--aug_severity", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=5e-2)
    parser.add_argument("--num_clusters", type=int, default=50000)
    
    parser.add_argument("--neptune_mode", type=str, default="async")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 0
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.scheme == "graph_clustering":
        data_transform = lambda x, edge_index, edg_attr: mask_nodes(
            x, edge_index, edg_attr, aug_severity=args.aug_severity
        )
        scheme = GraphClusteringScheme(
            num_clusters=args.num_clusters, 
            transform=data_transform, 
            temperature=args.temperature
            )
    
    print("Loading model...")
    models = scheme.get_models(
        num_layers=args.num_layers, emb_dim=args.emb_dim, drop_rate=args.drop_rate
    )
    models = models.to(device)
    optim = torch.optim.Adam(models.parameters(), lr=args.lr)

    print("Loading dataset...")
    dataset = MoleculeDataset(
        "../resource/dataset/" + args.dataset, dataset=args.dataset, transform=scheme.transform
    )
    #dataset = torch.utils.data.Subset(dataset, range(10000))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=scheme.collate_fn,
    )

    print("Loading cluster dataset...")
    cluster_dataset = MoleculeDataset("../resource/dataset/" + args.dataset, dataset=args.dataset)
    #cluster_dataset = torch.utils.data.Subset(cluster_dataset, range(10000))
    cluster_loader = torch_geometric.data.DataLoader(
        cluster_dataset,
        batch_size=args.cluster_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    eval_datasets = get_eval_datasets()

    print("Loading neptune...")
    run = neptune.init(
        project="sungsahn0215/substruct-embedding", name="train_embedding", mode=args.neptune_mode
    )
    run["parameters"] = vars(args)
    if args.run_tag == "":
        run_tag = run["sys/id"].fetch()
    else:
        run_tag = args.run_tag
    os.makedirs(f"../resource/result/{run_tag}", exist_ok=True)

    step = 0
    for epoch in range(args.num_epochs):
        run[f"epoch"].log(epoch)

        if (epoch + 1) > args.num_warmup_epochs:
            cluster_statistics = scheme.assign_cluster(cluster_loader, models, device)
            bincount = cluster_statistics.pop("bincount")
            
            if epoch > args.num_warmup_epochs:
                run.pop("cluster/bincount")
            
            for cnt in sorted(bincount.tolist(), reverse=True):
                run["cluster/bincount"].log(cnt)
            
            for key, val in cluster_statistics.items():
                run[f"cluster/{key}"].log(val)

        for batch in loader:
            step += 1

            train_statistics = scheme.train_step(batch, models, optim, device)

            if step % args.log_freq == 0:
                for key, val in train_statistics.items():
                    run[f"train/{key}"].log(val)

        eval_acc = 0.0
        for name in eval_datasets:
            eval_statistics = evaluate_knn(
                models["encoder"],
                eval_datasets[name]["train"],
                eval_datasets[name]["test"],
                device
                )
            for key, val in eval_statistics.items():
                run[f"eval/{name}/{key}"].log(val)

            eval_acc += eval_statistics["acc"] / len(eval_datasets)
            
        run[f"eval/total/acc"].log(eval_acc)
            
        torch.save(
            models["encoder"].state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
        )

    torch.save(models["encoder"].state_dict(), f"../resource/result/{run_tag}/model.pt")

    run.stop()


if __name__ == "__main__":
    main()
