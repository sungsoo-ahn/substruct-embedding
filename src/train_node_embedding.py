import os
from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

from model import NodeEncoderWithHead
from data.dataset import MoleculePairDataset, MoleculeDataset
from data.dataloader import MatchedPairDataLoader
from data.splitter import random_split

import neptune.new as neptune

def compute_distance(center0, center1):
    return torch.linalg.norm(center0 - center1, dim=1).unsqueeze(1)


def compute_energy(emb0, emb1):
    center0, log_radius0 = torch.split(emb0, [emb0.size(1) - 1, 1], dim=1)
    center1, log_radius1 = torch.split(emb1, [emb1.size(1) - 1, 1], dim=1)
    protrusion = compute_distance(center0, center1) - (
        torch.exp(log_radius0) - torch.exp(log_radius1)
    )
    energy = protrusion
    return energy


def train(model, batch, optim, margin, device):
    model.train()

    batch = batch.to(device)

    disk_emb0 = model(batch.x0, batch.edge_index0, batch.edge_attr0)
    graph_center, graph_radius = torch.chunk(disk_emb0, chunks=2, dim=1)

    pos_disk_emb1 = model(batch.x1, batch.edge_index1, batch.edge_attr1)
    pos_disk_emb1 = pos_disk_emb1[batch.mask > 0.5]
    shifts = random.randint(1, pos_disk_emb1.size(0)-1)
    neg_disk_emb1 = torch.roll(pos_disk_emb1, shifts=shifts, dims=0)

    pos_energy = compute_energy(disk_emb0, pos_disk_emb1)
    pos_elem_loss = torch.clamp(pos_energy, min=0.0)
    pos_loss = pos_elem_loss.mean()

    neg_energy = compute_energy(disk_emb0, neg_disk_emb1)
    neg_elem_loss = torch.clamp(margin - neg_energy, min=0.0)
    neg_loss = neg_elem_loss.mean()
    loss = pos_loss + neg_loss

    optim.zero_grad()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    loss.backward()
    optim.step()

    loss = loss.detach()
    acc = 0.5 * (
        (pos_energy < margin * 0.5).float().mean()
        + (neg_energy > margin * 0.5).float().mean()
    )

    statistics = {"loss": loss, "pos_loss": pos_loss, "neg_loss": neg_loss, "acc": acc}

    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_scaffold_network")
    parser.add_argument("--num_epochs", type=float, default=200)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--margin", type=float, default=1.0)

    parser.add_argument("--log_freq", type=float, default=100)

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculePairDataset("../resource/dataset/" + args.dataset, dataset=args.dataset)
    loader = MatchedPairDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # set up encoder
    model = NodeEncoderWithHead(
        num_head_layers=2,
        head_dim=args.emb_dim+1,
        num_encoder_layers=args.num_layers,
        emb_dim=args.emb_dim,
        drop_rate=args.drop_rate
        )
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    run = neptune.init(project="sungsahn0215/relation-embedding", name="train_embedding")
    run["parameters"] = vars(args)
    run_id = run["sys/id"].fetch()
    os.makedirs(f"../resource/result/{run_id}")

    step = 0
    for epoch in range(args.num_epochs):
        for batch in tqdm(loader):
            step += 1

            train_statistics = train(model, batch, optim, args.margin, device)

            if step % args.log_freq == 0:
                for key, val in train_statistics.items():
                    run[f"train/{key}"].log(val)

        torch.save(model.encoder.state_dict(), f"../resource/result/{run_id}/model.pt")

    run.stop()

if __name__ == "__main__":
    main()
