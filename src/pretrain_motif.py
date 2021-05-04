import os
from collections import defaultdict
import argparse
import numpy as np

import torch
import torch_geometric

from group_dataset import GroupDataset, double_collate
from data.transform import extract_motif_data
from data.splitter import random_split
from scheme.motif_contrast import (
    MotifContrastiveScheme,
    MotifContrastiveModel,
)
from evaluate_knn import get_eval_datasets, evaluate_knn

import neptune.new as neptune

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_group")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--log_freq", type=float, default=10)

    parser.add_argument("--scheme", type=str, default="group_contrast")
    parser.add_argument("--transform", type=str, default="none")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--use_neptune", action="store_true")
    
    parser.add_argument("--drop_scaffold", action="store_true")
    
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    scheme = MotifContrastiveScheme()
    model = MotifContrastiveModel()

    print("Loading model...")
    model = model.cuda()
    optim = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=args.lr
    )

    transform = lambda data: extract_motif_data(data, args.drop_scaffold)
    print("Loading dataset...")
    dataset = GroupDataset(
        "../resource/dataset/" + args.dataset, dataset=args.dataset, transform=transform,
    )
        
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=double_collate,
    )

    if args.use_neptune:
        print("Loading neptune...")
        run = neptune.init(project="sungsahn0215/ssg", name="group_contrast")
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

        for batch0, batch1 in tqdm(loader):
            step += 1
            train_statistics = scheme.train_step(batch0, batch1, model, optim)
            print(train_statistics)
            if step % args.log_freq == 0:
                for key, val in train_statistics.items():
                    if args.use_neptune:
                        run[f"train/{key}"].log(val)

        if args.use_neptune:
            torch.save(
                model.encoder.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
            )

    if args.use_neptune:
        torch.save(model.encoder.state_dict(), f"../resource/result/{run_tag}/model.pt")
        run.stop()


if __name__ == "__main__":
    main()
