import os
from collections import defaultdict
import argparse
import numpy as np

import torch
import torch_geometric

from data.dataset import MoleculeDataset
from data.splitter import random_split
from data.transform import mask_data, double_mask_data
from data.collate import collate, collate_cat
from scheme.mask_contrast import (
    MaskContrastModel,
    MaskFullContrastModel,
    MaskBalancedContrastModel,
    MaskContrastScheme,
)
from evaluate_knn import get_eval_datasets, evaluate_knn

import neptune.new as neptune

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--log_freq", type=float, default=10)

    parser.add_argument("--scheme", type=str, default="mask_contrast")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--use_neptune", action="store_true")

    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--balance_k", type=int, default=100)
    parser.add_argument("--use_double_mask", action="store_true")


    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.scheme == "mask_contrast":
        scheme = MaskContrastScheme()
        model = MaskContrastModel()
        
    elif args.scheme == "mask_full_contrast":
        scheme = MaskContrastScheme()
        model = MaskFullContrastModel()
        
    elif args.scheme == "mask_balanced_contrast":
        scheme = MaskContrastScheme()
        model = MaskBalancedContrastModel(balance_k=args.balance_k)
                
    if not args.use_double_mask:
        transform = lambda data: mask_data(data, mask_rate=args.mask_rate)
        collate_fn = collate
    else:
        transform = lambda data: double_mask_data(data, mask_rate=args.mask_rate)
        collate_fn = collate_cat
        args.batch_size = args.batch_size // 2


    print("Loading model...")
    model = model.cuda()
    optim = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=args.lr
    )

    print("Loading dataset...")
    dataset = MoleculeDataset(
        "../resource/dataset/" + args.dataset, dataset=args.dataset, transform=transform,
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
        run = neptune.init(project="sungsahn0215/ssg", name="mask_contrast")
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
            train_statistics = scheme.train_step(batch, model, optim)
            if step % args.log_freq == 0:
                for key, val in train_statistics.items():
                    if args.use_neptune:
                        run[f"train/{key}"].log(val)

        if args.use_neptune:
            torch.save(
                model.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
            )

    if args.use_neptune:
        torch.save(model.state_dict(), f"../resource/result/{run_tag}/model.pt")
        run.stop()


if __name__ == "__main__":
    main()
