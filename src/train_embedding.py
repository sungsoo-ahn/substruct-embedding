from collections import defaultdict
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

from model import GraphEncoderWithHead
from data.dataset import MoleculePairDataset, MoleculeDataset
from data.dataloader import PairDataLoader
from data.splitter import random_split

import neptune.new as neptune
#import neptune

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


def train(encoder, batch, encoder_optim, margin, device):
    encoder.train()

    batch = batch.to(device)

    disk_emb0 = encoder(batch.x0, batch.edge_index0, batch.edge_attr0, batch.batch0)
    graph_center, graph_radius = torch.chunk(disk_emb0, chunks=2, dim=1)

    pos_disk_emb1 = encoder(batch.x1, batch.edge_index1, batch.edge_attr1, batch.batch1)
    neg_disk_emb1 = torch.roll(pos_disk_emb1, shifts=1, dims=0)

    pos_energy = compute_energy(disk_emb0, pos_disk_emb1)
    pos_elem_loss = torch.clamp(pos_energy, min=0.0)
    pos_loss = pos_elem_loss.mean()

    neg_energy = compute_energy(disk_emb0, neg_disk_emb1)
    neg_elem_loss = torch.clamp(margin - neg_energy, min=0.0)
    neg_loss = neg_elem_loss.mean()
    loss = pos_loss + neg_loss

    encoder_optim.zero_grad()
    loss.backward()
    encoder_optim.step()

    loss = loss.detach()
    acc = 0.5 * (
        (pos_energy < margin * 0.5).float().mean()
        + (neg_energy > margin * 0.5).float().mean()
    )

    statistics = {"loss": loss, "pos_loss": pos_loss, "neg_loss": neg_loss, "acc": acc}

    return statistics


def main():
    TRAIN_BATCH_SIZE = 256
    EVAL_BATCH_SIZE = 32
    NUM_WORKERS = 8
    LR = 1e-3
    TRAIN_LOG_FREQ = 10
    DATASET_DIR = "../resource/dataset/"
    RESULT_DIR = "../resource/result/"
    DATASET = "zinc_scaffold_network"
    NUM_EPOCHS = 200
    DISK_DIM = 256

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_walk_length", type=int, default=10)
    parser.add_argument("--max_walk_length", type=int, default=40)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset and transform function.
    train_dataset = MoleculePairDataset(DATASET_DIR + DATASET, dataset=DATASET)


    train_loader = PairDataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        compute_true_target=False,
        num_workers=NUM_WORKERS,
    )

    # set up encoder
    encoder = GraphEncoderWithHead(head_dim=DISK_DIM + 1).to(device)

    # set up optimizer
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=LR)

    run = neptune.init(project="sungsahn0215/relation-embedding")
    run["parameters"] = vars(args)
    # neptune.create_experiment(name="graph-order", params=vars(args))

    step = 0
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_loader):
            step += 1

            train_statistics = train(encoder, batch, encoder_optim, args.margin, device)

            if step % TRAIN_LOG_FREQ == 0:
                for key, val in train_statistics.items():
                    run[f"train/{key}"].log(val)

        torch.save(GraphEncoderWithHead.encoder.state_dict(), "../resource/result/encoder.pt")


if __name__ == "__main__":
    main()
