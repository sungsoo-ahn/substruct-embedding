import os
from collections import defaultdict
import argparse
import numpy as np

import torch

from model import NodeEncoder
from data.dataset import MoleculeDataset
from data.splitter import random_split
from data.transform import compose, drop_nodes, mask_nodes
from scheme.contrastive import ContrastiveScheme
from scheme.optimal_transport_contrastive import OptimalTransportContrastiveScheme

import neptune.new as neptune

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("--scheme", type=str, default="contrast")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--log_freq", type=float, default=100)

    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--aug_severity", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=5e-2)
    parser.add_argument("--num_sinkhorn_iters", type=int, default=5)
    
    parser.add_argument("--neptune_mode", type=str, default="async")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    data_transform = (
        lambda x, edge_index, edg_attr: drop_nodes(x, edge_index, edg_attr, aug_severity=args.aug_severity)
    )

    if args.scheme == "contrast":
        scheme = ContrastiveScheme(transform=data_transform, temperature=args.temperature)
    elif args.scheme == "ot_contrast":
        scheme = OptimalTransportContrastiveScheme(
            transform=data_transform, 
            temperature=args.temperature, 
            num_sinkhorn_iters=args.num_sinkhorn_iters
        )

    # set up encoder
    models = scheme.get_models(
        num_layers=args.num_layers, emb_dim=args.emb_dim, drop_rate=args.drop_rate
    )
    models = models.to(device)
    optim = torch.optim.Adam(models.parameters(), lr=args.lr)

    dataset = MoleculeDataset(
        "../resource/dataset/" + args.dataset, dataset=args.dataset, transform=scheme.transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=scheme.collate_fn,
    )

    run = neptune.init(
       project="sungsahn0215/relation-embedding", name="train_embedding", mode=args.neptune_mode
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

            train_statistics = scheme.train_step(batch, models, optim, device)

            if step % args.log_freq == 0:
               for key, val in train_statistics.items():
                   run[f"train/{key}"].log(val)

        torch.save(
            models["encoder"].state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
        )

    torch.save(models["encoder"].state_dict(), f"../resource/result/{run_tag}/model.pt")

    run.stop()


if __name__ == "__main__":
    main()
