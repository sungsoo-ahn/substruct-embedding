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
from data.dataset import MoleculeDataset
from data.dataloader import PairDataLoader
from data.splitter import random_split
from data.transform import CutLinkerBond, CutLinkerAndMaskAtom

import neptune.new as neptune

def compute_distance(center0, center1):
    return torch.linalg.norm(center0 - center1, dim=1).unsqueeze(1)


def compute_energy(emb0, emb1):
    center0, log_radius0 = torch.split(emb0, [emb0.size(1) - 1, 1], dim=1)
    center1, log_radius1 = torch.split(emb1, [emb1.size(1) - 1, 1], dim=1)
    protrusion = compute_distance(center0, center1) - (
        torch.exp(log_radius1) - torch.exp(log_radius0)
    )
    energy = protrusion
    return energy

def compute_disk_loss(emb0, pos_emb1, neg_emb1, margin):
    pos_energy = compute_energy(emb0, pos_emb1)
    pos_elem_loss = torch.clamp(pos_energy, min=0.0)
    pos_loss = pos_elem_loss.mean()

    neg_energy = compute_energy(emb0, neg_emb1)
    neg_elem_loss = torch.clamp(margin - neg_energy, min=0.0)
    neg_loss = neg_elem_loss.mean()
    
    loss = pos_loss + neg_loss
    acc = 0.5 * (
        (pos_energy < margin * 0.5).float().mean()
        + (neg_energy > margin * 0.5).float().mean()
    )
        
    statistics = {
        "loss": loss.detach(), 
        "pos_loss": pos_loss.detach(), 
        "neg_loss": neg_loss.detach(), 
        "acc": acc
        }

    return loss, statistics


def compute_triplet_loss(emb0, pos_emb1, neg_emb1, margin):
    pos_distance = compute_distance(emb0, pos_emb1)
    neg_distance = compute_distance(emb0, neg_emb1)
    loss = torch.clamp(pos_distance - neg_distance + margin, min=0).mean()
    statistics = {"loss": loss.detach()}
    
    return loss, statistics


def train(model, batch, optim, margin, loss_type, device):
    model.train()

    batch = batch.to(device)

    emb0 = model(batch.x0, batch.edge_index0, batch.edge_attr0)
    emb0 = emb0[batch.mask > 0.5]
    graph_center, graph_radius = torch.chunk(emb0, chunks=2, dim=1)

    pos_emb1 = model(batch.x1, batch.edge_index1, batch.edge_attr1)
    shifts = random.randint(1, pos_emb1.size(0)-1)
    neg_emb1 = torch.roll(pos_emb1, shifts=shifts, dims=0)

    if loss_type == "disk":
        loss, statistics = compute_disk_loss(emb0, pos_emb1, neg_emb1, margin)
    elif loss_type == "triplet":
        loss, statistics = compute_triplet_loss(emb0, pos_emb1, neg_emb1, margin)
    else:
        raise NotImplementedError
        
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    statistics["batch_size"] = batch.batch_size
    
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--num_epochs", type=float, default=50)
    
    parser.add_argument("--loss_type", type=str, default="disk")
    parser.add_argument("--transform_type", type=str, default="cut_linker_bond")
    parser.add_argument("--num_head_layers", type=int, default=1)
    
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--margin", type=float, default=1.0)

    parser.add_argument("--log_freq", type=float, default=100)

    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--disable_neptune", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.transform_type == "cut_linker_bond":
        transform = CutLinkerBond() 
    elif args.transform_type == "cut_linker_bond_and_mask_atom":
        transform = CutLinkerAndMaskAtom()
    else:
        raise NotImplementedError

    dataset = MoleculeDataset(
        "../resource/dataset/" + args.dataset, 
        dataset=args.dataset,
        transform=transform
        )

    loader = PairDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # set up encoder
    model = NodeEncoderWithHead(
        num_head_layers=args.num_head_layers,
        head_dim=args.emb_dim+1,
        num_encoder_layers=args.num_layers,
        emb_dim=args.emb_dim,
        drop_rate=args.drop_rate
        )
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    run = neptune.init(project="sungsahn0215/relation-embedding", name="train_embedding")
    run["parameters"] = vars(args)
    if args.run_tag == "":
        run_tag = run["sys/id"].fetch()
    else:
        run_tag = args.run_tag
        
    os.makedirs(f"../resource/result/{run_tag}")

    step = 0
    for epoch in range(args.num_epochs):
        for batch in loader:
            step += 1

            train_statistics = train(model, batch, optim, args.margin, args.loss_type, device)

                for key, val in train_statistics.items():
                    run[f"train/{key}"].log(val)

        torch.save(model.encoder.state_dict(), f"../resource/result/{run_tag}/model.pt")

    run.stop()

if __name__ == "__main__":
    main()
