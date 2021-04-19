import os
from collections import defaultdict
import argparse
import numpy as np

import torch
import torch_geometric

from model import NodeEncoder
from data.dataset import MoleculeDataset
from data.splitter import random_split
from data.transform import mask_data_twice
from data.collate import contrastive_collate
from scheme.graph_clustering import GraphClusteringScheme, GraphClusteringModel
from evaluate_knn import get_eval_datasets, evaluate_knn

import neptune.new as neptune

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--cluster_freq", type=int, default=1000)
    parser.add_argument("--log_freq", type=float, default=100)

    parser.add_argument("--scheme", type=str, default="graph_clustering")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cluster_batch_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--num_clusters", type=int, default=50000)
    parser.add_argument("--use_density_rescaling", action="store_true")
    parser.add_argument("--use_euclidean_clustering", action="store_true")
    parser.add_argument("--neptune_mode", type=str, default="async")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 0
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.scheme == "graph_clustering":
        scheme = GraphClusteringScheme(
            num_clusters=args.num_clusters, 
            use_euclidean_clustering=args.use_euclidean_clustering
            )
        model = GraphClusteringModel(use_density_rescaling=args.use_density_rescaling)
        transform = mask_data_twice
        collate_fn = contrastive_collate
    
    print("Loading model...")
    model = model.to(device)
    optim = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=args.lr
        )

    print("Loading dataset...")
    dataset = MoleculeDataset(
        "../resource/dataset/" + args.dataset, 
        dataset=args.dataset, 
        transform=transform,
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print("Loading cluster dataset...")
    cluster_dataset = MoleculeDataset(
        "../resource/dataset/" + args.dataset, 
        dataset=args.dataset,
        )
        
    cluster_loader = torch_geometric.data.DataLoader(
        cluster_dataset,
        batch_size=args.cluster_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
        
    #eval_datasets = get_eval_datasets()

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
        
        for batch in loader:
            step += 1
            train_statistics = scheme.train_step(batch, model, optim, device)
            
            if step % args.log_freq == 0:
                for key, val in train_statistics.items():
                    run[f"train/{key}"].log(val)
            
            if step > args.num_warmup_steps and step % args.cluster_freq == 0:
                cluster_statistics = scheme.assign_cluster(cluster_loader, model, device)
                for key, val in cluster_statistics.items():
                    run[f"cluster/{key}"].log(val)
    
        eval_acc = 0.0
        for name in eval_datasets:
            eval_statistics = evaluate_knn(
                model.compute_ema_features_graph,
                eval_datasets[name]["train"],
                eval_datasets[name]["test"],
                device
                )
            for key, val in eval_statistics.items():
                run[f"eval/{name}/{key}"].log(val)
        
            eval_acc += eval_statistics["acc"] / len(eval_datasets)
            
        run[f"eval/total/acc"].log(eval_acc)
        
        torch.save(
            model.encoder.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
        )

    torch.save(model.encoder.state_dict(), f"../resource/result/{run_tag}/model.pt")

    run.stop()


if __name__ == "__main__":
    main()
