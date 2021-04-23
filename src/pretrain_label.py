import os
from collections import defaultdict
import argparse
import numpy as np

import torch
import torch_geometric

from model import NodeEncoder
from data.dataset import MoleculeDataset
from data.splitter import random_split
from data.transform import mask_data_and_node_label, mask_data_and_rw_label
from data.collate import collate
from scheme.masked_rw_pred import MaskedRWPredModel, MaskedRWPredScheme
from scheme.masked_node_pred import MaskedNodePredModel, MaskedNodePredScheme

import neptune.new as neptune
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--log_freq", type=float, default=10)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--scheme", type=str, default="masked_rw_pred")

    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--walk_length", type=int, default=6)
    parser.add_argument("--use_neptune", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.scheme == "masked_node_pred":
        scheme = MaskedNodePredScheme()
        model = MaskedNodePredModel()
        transform = lambda data: mask_data_and_node_label(data, mask_rate=args.mask_rate)

    elif args.scheme == "masked_rw_pred":
        scheme = MaskedRWPredScheme()
        model = MaskedRWPredModel()
        transform = lambda data: mask_data_and_rw_label(
            data, walk_length=args.walk_length, mask_rate=args.mask_rate
            )

    print("Loading model...")
    model = model.cuda()
    optim = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=1e-3
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
        collate_fn=collate,
        drop_last=True
    )

    if args.use_neptune:
        print("Loading neptune...")
        run = neptune.init(project="sungsahn0215/pretrain-label", name=args.scheme)
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

        for batch in loader:
            step += 1
            train_statistics = scheme.train_step(batch, model, optim)
            if step % args.log_freq == 0 and args.use_neptune:
                for key, val in train_statistics.items():
                    run[f"train/{key}"].log(val)

        torch.save(
            model.encoder.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
        )

    torch.save(model.encoder.state_dict(), f"../resource/result/{run_tag}/model.pt")

    if args.use_neptune:
        run.stop()


if __name__ == "__main__":
    main()
