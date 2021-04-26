import os
from collections import defaultdict
import argparse
import numpy as np

import torch
import torch_geometric

from model import NodeEncoder
from data.dataset import MoleculeDataset
from data.splitter import random_split
from data.transform import mask_samepool, mask_diffpool
from data.collate import contrastive_collate
from scheme.node_contrastive import NodeContrastiveScheme, NodeContrastiveModel
from evaluate_knn import get_eval_datasets, evaluate_knn

import neptune.new as neptune
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_warmup_epochs", type=int, default=1)
    parser.add_argument("--log_freq", type=float, default=10)
    parser.add_argument("--cluster_freq", type=float, default=1)

    parser.add_argument("--scheme", type=str, default="node_clustering")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cluster_batch_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--transform_type", type=str, default="once_same")
    parser.add_argument("--mask_rate", type=float, default=0.0)
    parser.add_argument("--pool_rate", type=float, default=0.1)
    parser.add_argument("--use_linear_projector", action="store_true")
    
    parser.add_argument("--use_neptune", action="store_true")
    
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 0
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    scheme = NodeContrastiveScheme()
    model = NodeContrastiveModel(use_linear_projector=args.use_linear_projector)
    if args.transform_type == "once_same":
        transform = lambda data: mask_samepool(data, args.mask_rate, args.pool_rate)
    elif args.transform_type == "once_diff":
        transform = lambda data: mask_diffpool(data, args.mask_rate, args.pool_rate)
        
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

    if args.use_neptune:
        print("Loading neptune...")
        run = neptune.init(project="sungsahn0215/graph-clustering", name="graph_clustering")
        run["parameters"] = vars(args)
        if args.run_tag == "":
            run_tag = run["sys/id"].fetch()
        else:
            run_tag = args.run_tag
        os.makedirs(f"../resource/result/{run_tag}", exist_ok=True)

    step = 0
    for epoch in range(args.num_epochs):
        if args.use_neptune:
            run[f"epoch"].log(epoch)            
        
        for batch in tqdm(loader):
            step += 1
            train_statistics = scheme.train_step(batch, model, optim, device)

            if step % args.log_freq == 0 and args.use_neptune:
                for key, val in train_statistics.items():
                    run[f"train/{key}"].log(val)

        if epoch == 0:
            if args.dataset == "zinc_standard_agent":
                eval_datasets = get_eval_datasets()
            else:
                eval_datasets = get_eval_datasets([args.dataset])

        model.eval()
        eval_acc = 0.0
        for name in eval_datasets:
            eval_statistics = evaluate_knn(
                model.compute_graph_features,
                eval_datasets[name]["train"],
                eval_datasets[name]["test"],
                device
                )
            if args.use_neptune:
                for key, val in eval_statistics.items():
                    run[f"eval/{name}/{key}"].log(val)

            eval_acc += eval_statistics["acc"] / len(eval_datasets)

        if args.use_neptune:
            run[f"eval/total/acc"].log(eval_acc)
        
            torch.save(
                model.encoder.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
            )

    if args.use_neptune:
        torch.save(model.encoder.state_dict(), f"../resource/result/{run_tag}/model.pt")

    run.stop()


if __name__ == "__main__":
    main()
