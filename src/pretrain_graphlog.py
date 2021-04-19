import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import os, sys
import pdb
import copy
import random

from model import NodeEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from data.dataset import MoleculeDataset
from evaluate_knn import get_eval_datasets, evaluate_knn
import neptune.new as neptune

class ProjectNet(torch.nn.Module):
    def __init__(self, rep_dim):
        super(ProjectNet, self).__init__()
        self.rep_dim = rep_dim
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.rep_dim, self.rep_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.rep_dim, self.rep_dim),
        )

    def forward(self, x):
        x_proj = self.proj(x)

        return x_proj


# Graph pooling functions
def pool_func(x, batch, mode="mean"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)


# Mask some nodes in a graph
def mask_nodes(batch, args, num_atom_type=119):
    masked_node_indices = list()

    # select indices of masked nodes
    for i in range(batch.batch[-1] + 1):
        idx = torch.nonzero((batch.batch == i).float()).squeeze(-1)
        num_node = idx.shape[0]
        if args.mask_num == 0:
            sample_size = int(num_node * args.mask_rate + 1)
        else:
            sample_size = min(args.mask_num, int(num_node * 0.5))
        masked_node_idx = random.sample(idx.tolist(), sample_size)
        masked_node_idx.sort()
        masked_node_indices += masked_node_idx

    batch.masked_node_indices = torch.tensor(masked_node_indices)

    # mask nodes' features
    for node_idx in masked_node_indices:
        batch.x[node_idx] = torch.tensor([num_atom_type, 0])

    return batch


# InfoNCE loss within a graph
def intra_NCE_loss(node_reps, node_mask_reps, batch, tau=0.1, epsilon=1e-6):
    node_reps_norm = torch.norm(node_reps, dim=1).unsqueeze(-1)
    node_mask_reps_norm = torch.norm(node_mask_reps, dim=1).unsqueeze(-1)
    sim = torch.mm(node_reps, node_mask_reps.t()) / (
        torch.mm(node_reps_norm, node_mask_reps_norm.t()) + epsilon
    )
    exp_sim = torch.exp(sim / tau)

    mask = torch.stack([(batch.batch == i).float() for i in batch.batch.tolist()], dim=1)
    exp_sim_mask = exp_sim * mask
    exp_sim_all = torch.index_select(exp_sim_mask, 1, batch.masked_node_indices)
    exp_sim_positive = torch.index_select(exp_sim_all, 0, batch.masked_node_indices)
    positive_ratio = exp_sim_positive.sum(0) / (exp_sim_all.sum(0) + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / batch.masked_node_indices.shape[0]
    mask_select = torch.index_select(mask, 1, batch.masked_node_indices)
    thr = 1.0 / mask_select.sum(0)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt


# InfoNCE loss across different graphs
def inter_NCE_loss(graph_reps, graph_mask_reps, device, tau=0.1, epsilon=1e-6):
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    graph_mask_reps_norm = torch.norm(graph_mask_reps, dim=1).unsqueeze(-1)
    sim = torch.mm(graph_reps, graph_mask_reps.t()) / (
        torch.mm(graph_reps_norm, graph_mask_reps_norm.t()) + epsilon
    )
    exp_sim = torch.exp(sim / tau)

    mask = torch.eye(graph_reps.shape[0]).to(device)
    positive = (exp_sim * mask).sum(0)
    negative = (exp_sim * (1 - mask)).sum(0)
    positive_ratio = positive / (positive + negative + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / graph_reps.shape[0]
    thr = 1.0 / ((1 - mask).sum(0) + 1.0)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt


# InfoNCE loss for global-local mutual information maximization
def gl_NCE_loss(node_reps, graph_reps, batch, tau=0.1, epsilon=1e-6):
    node_reps_norm = torch.norm(node_reps, dim=1).unsqueeze(-1)
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    sim = torch.mm(node_reps, graph_reps.t()) / (
        torch.mm(node_reps_norm, graph_reps_norm.t()) + epsilon
    )
    exp_sim = torch.exp(sim / tau)

    mask = torch.stack([(batch == i).float() for i in range(graph_reps.shape[0])], dim=1)
    positive = exp_sim * mask
    negative = exp_sim * (1 - mask)
    positive_ratio = positive / (positive + negative.sum(0).unsqueeze(0) + epsilon)

    NCE_loss = -torch.log(positive_ratio + (1 - mask)).sum() / node_reps.shape[0]
    thr = 1.0 / ((1 - mask).sum(0) + 1.0).unsqueeze(0)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt


# InfoNCE loss between graphs and prototypes
def proto_NCE_loss(
    graph_reps,
    graph_mask_reps,
    proto_list,
    proto_connection,
    tau=0.1,
    decay_ratio=0.7,
    epsilon=1e-6,
):
    # similarity for original and masked graphs
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    graph_mask_reps_norm = torch.norm(graph_mask_reps, dim=1).unsqueeze(-1)
    exp_sim_list = []
    exp_sim_list_ = []
    mask_list = []

    for i in range(len(proto_list) - 1, -1, -1):
        proto = proto_list[i]
        proto_norm = torch.norm(proto, dim=1).unsqueeze(-1)

        sim = torch.mm(graph_reps, proto.t()) / (
            torch.mm(graph_reps_norm, proto_norm.t()) + epsilon
        )
        exp_sim = torch.exp(sim / tau)
        sim_mask = torch.mm(graph_mask_reps, proto.t()) / (
            torch.mm(graph_mask_reps_norm, proto_norm.t()) + epsilon
        )
        exp_sim_mask = torch.exp(sim_mask / tau)
        if i != (len(proto_list) - 1):
            exp_sim_last = exp_sim_list[-1]
            idx_last = torch.argmax(exp_sim_last, dim=1).unsqueeze(-1)
            connection = proto_connection[i]
            connection_mask = (connection.unsqueeze(0) == idx_last.float()).float()
            exp_sim = exp_sim * connection_mask
            exp_sim_mask = exp_sim_mask * connection_mask

        mask = (exp_sim == exp_sim.max(1)[0].unsqueeze(-1)).float()

        exp_sim_list.append(exp_sim)
        exp_sim_list_.append(exp_sim_mask)
        mask_list.append(mask)

    # define InfoNCE loss
    NCE_loss = 0

    for i in range(len(proto_list)):
        exp_sim_mask = exp_sim_list_[i]
        mask = mask_list[i]

        positive = exp_sim_mask * mask
        negative = exp_sim_mask * (1 - mask)
        positive_ratio = positive.sum(1) / (positive.sum(1) + negative.sum(1) + epsilon)
        NCE_loss += -torch.log(positive_ratio).mean()

    # update prototypes
    mask_last = mask_list[-1]
    cnt = mask_last.sum(0)
    batch_cnt = mask_last.t() / (cnt.unsqueeze(-1) + epsilon)
    batch_mean = torch.mm(batch_cnt, graph_reps)
    proto_list[0].data = (
        proto_list[0].data * (cnt == 0).float().unsqueeze(-1).data
        + (proto_list[0].data * decay_ratio + batch_mean.data * (1 - decay_ratio))
        * (cnt != 0).float().unsqueeze(-1).data
    )

    for i in range(1, len(proto_list)):
        proto = proto_list[i]
        proto_last = proto_list[i - 1]
        connection = proto_connection[i - 1]
        connection_mask = torch.stack(
            [(connection == j).float() for j in range(proto.shape[0])], dim=0
        )
        connection_cnt = connection_mask.sum(1).unsqueeze(-1)
        connection_weight = connection_mask / (connection_cnt + epsilon)
        proto_list[i].data = torch.mm(connection_weight, proto_last).data

    return NCE_loss, proto_list


# Update prototypes with batch information
def update_proto_lowest(graph_reps, proto, proto_state, decay_ratio=0.7, epsilon=1e-6):
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    proto_norm = torch.norm(proto, dim=1).unsqueeze(-1)
    sim = torch.mm(graph_reps, proto.t()) / (torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)

    # update states of prototypes
    mask = (sim == sim.max(1)[0].unsqueeze(-1)).float()
    cnt = mask.sum(0)
    proto_state.data = (
        proto_state.data + (cnt != 0).float().data - proto_state.data * (cnt != 0).float().data
    )

    # update prototypes
    batch_cnt = mask.t() / (cnt.unsqueeze(-1) + epsilon)
    batch_mean = torch.mm(batch_cnt, graph_reps)
    proto.data = (
        proto.data * (cnt == 0).float().unsqueeze(-1).data
        + (proto.data * decay_ratio + batch_mean.data * (1 - decay_ratio))
        * (cnt != 0).float().unsqueeze(-1).data
    )

    return proto, proto_state


# Initialze prototypes and their state
def init_proto_lowest(args, model, proj, loader, device, proto, proto_state):
    model.eval()
    proj.eval()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        # get node and graph representations
        node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
        graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)

        # feature projection
        graph_reps_proj = proj(graph_reps)

        # update prototypes
        proto, proto_state = update_proto_lowest(
            graph_reps_proj, proto, proto_state, decay_ratio=args.decay_ratio
        )

    idx = torch.nonzero(proto_state).squeeze(-1)
    proto_selected = torch.index_select(proto, 0, idx)

    return proto_selected, proto_state


# Initialze prototypes and their state
def init_proto(args, proto_, proto, proto_state, device, num_iter=20):
    proto_connection = torch.zeros(proto_.shape[0]).to(device)

    for iter in range(num_iter):
        for i in range(proto_.shape[0]):
            # update winner
            sim = torch.mm(proto, proto_[i, :].unsqueeze(-1)).squeeze(-1)
            idx = torch.argmax(sim)
            if iter == (num_iter - 1):
                proto_state[idx] = 1
            proto_connection[i] = idx
            proto.data[idx, :] = proto.data[idx, :] * args.decay_ratio + proto_.data[i, :] * (
                1 - args.decay_ratio
            )

            # penalize rival
            sim[idx] = 0
            rival_idx = torch.argmax(sim)
            proto.data[rival_idx, :] = proto.data[rival_idx, :] * (
                2 - args.decay_ratio
            ) - proto_.data[i, :] * (1 - args.decay_ratio)

    indices = torch.nonzero(proto_state).squeeze(-1)
    proto_selected = torch.index_select(proto, 0, indices)
    for i in range(indices.shape[0]):
        idx = indices[i]
        idx_connection = torch.nonzero((proto_connection == idx.float()).float()).squeeze(-1)
        proto_connection[idx_connection] = i

    return proto_selected, proto_state, proto_connection


# For one epoch pretraining
def pretrain(args, model, proj, loader, optimizer, device, run):
    model.train()
    proj.train()

    NCE_loss_intra_cnt = 0
    NCE_loss_inter_cnt = 0
    correct_intra_cnt = 0
    correct_inter_cnt = 0
    total_intra_cnt = 0
    total_inter_cnt = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_mask = copy.deepcopy(batch)
        batch_mask = mask_nodes(batch_mask, args)
        batch, batch_mask = batch.to(device), batch_mask.to(device)

        # get node and graph representations
        node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
        node_mask_reps = model(batch_mask.x, batch_mask.edge_index, batch_mask.edge_attr)
        graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)
        graph_mask_reps = pool_func(node_mask_reps, batch_mask.batch, mode=args.graph_pooling)

        # feature projection
        node_reps_proj = proj(node_reps)
        node_mask_reps_proj = proj(node_mask_reps)
        graph_reps_proj = proj(graph_reps)
        graph_mask_reps_proj = proj(graph_mask_reps)

        # InfoNCE loss
        NCE_loss_intra, correct_intra = intra_NCE_loss(
            node_reps_proj, node_mask_reps_proj, batch_mask, tau=args.tau
        )
        NCE_loss_inter, correct_inter = inter_NCE_loss(
            graph_reps_proj, graph_mask_reps_proj, device, tau=args.tau
        )

        NCE_loss_intra_cnt += NCE_loss_intra.item()
        NCE_loss_inter_cnt += NCE_loss_inter.item()
        correct_intra_cnt += correct_intra
        correct_inter_cnt += correct_inter
        total_intra_cnt += batch_mask.masked_node_indices.shape[0]
        total_inter_cnt += graph_reps.shape[0]

        # optimization
        optimizer.zero_grad()
        NCE_loss = args.alpha * NCE_loss_intra + args.beta * NCE_loss_inter
        NCE_loss.backward()
        optimizer.step()

        
        if (step + 1) % args.disp_interval == 0:
            run["step/loss/intraNCE"].log(NCE_loss_intra.item())
            run["step/acc/intraNCE"].log(float(correct_intra_cnt) / float(total_intra_cnt))
            run["step/loss/interNCE"].log(NCE_loss_inter.item())
            run["step/acc/interNCE"].log(float(correct_inter_cnt) / float(total_inter_cnt))

    return (
        NCE_loss_intra_cnt / step,
        float(correct_intra_cnt) / float(total_intra_cnt),
        NCE_loss_inter_cnt / step,
        float(correct_inter_cnt) / float(total_inter_cnt),
    )


# For every epoch training
def train(args, model, proj, loader, optimizer, device, proto, proto_connection, run):
    model.train()
    proj.train()

    NCE_loss_intra_cnt = 0
    NCE_loss_inter_cnt = 0
    NCE_loss_proto_cnt = 0
    correct_intra_cnt = 0
    correct_inter_cnt = 0
    total_intra_cnt = 0
    total_inter_cnt = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_mask = copy.deepcopy(batch)
        batch_mask = mask_nodes(batch_mask, args)
        batch, batch_mask = batch.to(device), batch_mask.to(device)

        # get node and graph representations
        node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
        node_mask_reps = model(batch_mask.x, batch_mask.edge_index, batch_mask.edge_attr)
        graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)
        graph_mask_reps = pool_func(node_mask_reps, batch_mask.batch, mode=args.graph_pooling)

        # feature projection
        node_reps_proj = proj(node_reps)
        node_mask_reps_proj = proj(node_mask_reps)
        graph_reps_proj = proj(graph_reps)
        graph_mask_reps_proj = proj(graph_mask_reps)

        # InfoNCE loss
        NCE_loss_intra, correct_intra = intra_NCE_loss(
            node_reps_proj, node_mask_reps_proj, batch_mask, tau=args.tau
        )
        NCE_loss_inter, correct_inter = inter_NCE_loss(
            graph_reps_proj, graph_mask_reps_proj, device, tau=args.tau
        )
        NCE_loss_proto, proto = proto_NCE_loss(
            graph_reps_proj,
            graph_mask_reps_proj,
            proto,
            proto_connection,
            tau=args.tau,
            decay_ratio=args.decay_ratio,
        )

        NCE_loss_intra_cnt += NCE_loss_intra.item()
        NCE_loss_inter_cnt += NCE_loss_inter.item()
        NCE_loss_proto_cnt += NCE_loss_proto.item()
        correct_intra_cnt += correct_intra
        correct_inter_cnt += correct_inter
        total_intra_cnt += batch_mask.masked_node_indices.shape[0]
        total_inter_cnt += graph_reps.shape[0]

        # optimization
        optimizer.zero_grad()
        NCE_loss = (
            args.alpha * NCE_loss_intra + args.beta * NCE_loss_inter + args.gamma * NCE_loss_proto
        )
        NCE_loss.backward()
        optimizer.step()

        if (step + 1) % args.disp_interval == 0:
            run["step/loss/intraNCE"].log(NCE_loss_intra.item())
            run["step/acc/intraNCE"].log(float(correct_intra_cnt) / float(total_intra_cnt))
            run["step/loss/interNCE"].log(NCE_loss_inter.item())
            run["step/acc/interNCE"].log(float(correct_inter_cnt) / float(total_inter_cnt))
            run["step/loss/protoNCE"].log(NCE_loss_proto.item())
            
    return (
        NCE_loss_intra_cnt / step,
        float(correct_intra_cnt) / float(total_intra_cnt),
        NCE_loss_inter_cnt / step,
        float(correct_inter_cnt) / float(total_inter_cnt),
        NCE_loss_proto_cnt / step,
        proto,
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="GraphLoG for GNN pre-training")
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="input batch size for training (default: 512)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train (default: 100)"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--decay", type=float, default=0, help="weight decay (default: 0)")
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5).",
    )
    parser.add_argument(
        "--emb_dim", type=int, default=300, help="embedding dimensions (default: 300)"
    )
    parser.add_argument("--dropout_ratio", type=float, default=0, help="dropout ratio (default: 0)")
    parser.add_argument(
        "--mask_rate", type=float, default=0.3, help="dropout ratio (default: 0.15)"
    )
    parser.add_argument(
        "--mask_num", type=int, default=0, help="the number of masked nodes (default: 0)"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features are combined across layers. last, sum, max or concat",
    )
    parser.add_argument(
        "--graph_pooling", type=str, default="mean", help="graph level pooling (sum, mean, max)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="zinc_standard_agent",
        help="root directory of dataset for pretraining",
    )
    parser.add_argument(
        "--output_model_file", type=str, default="", help="filename to output the model"
    )
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--seed", type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of workers for dataset loading"
    )
    parser.add_argument(
        "--tau", type=float, default=0.04, help="the temperature parameter for softmax"
    )
    parser.add_argument(
        "--decay_ratio", type=float, default=0.95, help="the decay ratio for moving average"
    )
    parser.add_argument(
        "--num_proto", type=int, default=100, help="the number of initial prototypes"
    )
    parser.add_argument("--hierarchy", type=int, default=3, help="the number of hierarchy")
    parser.add_argument(
        "--alpha", type=float, default=1, help="the weight of intra-graph InfoNCE loss"
    )
    parser.add_argument(
        "--beta", type=float, default=1, help="the weight of inter-graph InfoNCE loss"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="the weight of prototype InfoNCE loss"
    )
    parser.add_argument("--disp_interval", type=int, default=10, help="the display interval")
    parser.add_argument("--run_tag", type=str, default="graph_log")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("num GNN layer: %d" % (args.num_layer))

    # set up dataset and transform function.
    dataset = MoleculeDataset("../resource/dataset/" + args.dataset, dataset=args.dataset)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # set up pretraining models and feature projector
    model = NodeEncoder(args.num_layer, args.emb_dim, args.dropout_ratio).to(device)
    if args.JK == "concat":
        proj = ProjectNet((args.num_layer + 1) * args.emb_dim).to(device)
    else:
        proj = ProjectNet(args.emb_dim).to(device)

    def featurizer(x, edge_index, edge_attr, batch):
        node_reps = model(x, edge_index, edge_attr)
        graph_reps = pool_func(node_reps, batch, mode=args.graph_pooling)
        graph_reps_proj = proj(graph_reps)
        return graph_reps_proj

    eval_datasets = get_eval_datasets()


    # set up optimizer
    model_param_group = [
        {"params": model.parameters(), "lr": args.lr},
        {"params": proj.parameters(), "lr": args.lr},
    ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    # initialize prototypes and their state
    if args.JK == "concat":
        proto = [
            torch.rand((args.num_proto, (args.num_layer + 1) * args.emb_dim)).to(device)
            for i in range(args.hierarchy)
        ]
    else:
        proto = [
            torch.rand((args.num_proto, args.emb_dim)).to(device) for i in range(args.hierarchy)
        ]
    proto_state = [torch.zeros(args.num_proto).to(device) for i in range(args.hierarchy)]
    proto_connection = []

    print("Loading neptune...")
    run = neptune.init(
        project="sungsahn0215/substruct-embedding", name="train_embedding"
    )
    run["parameters"] = vars(args)
    run_tag = "graph_log"
    os.makedirs(f"../resource/result/{run_tag}", exist_ok=True)

    # pre-training with only intra and inter InfoNCE losses
    train_intra_loss, train_intra_acc, train_inter_loss, train_inter_acc = pretrain(
        args, model, proj, loader, optimizer, device, run
    )
    
    run["epoch/loss/intraNCE"].log(train_intra_loss)
    run["epoch/acc/intraNCE"].log(train_intra_acc)
    run["epoch/loss/interNCE"].log(train_inter_loss)
    run["epoch/acc/interNCE"].log(train_inter_acc)
    
    model.eval()
    proj.eval()
    eval_acc = 0.0
    for name in eval_datasets:
        eval_statistics = evaluate_knn(
            featurizer,
            eval_datasets[name]["train"],
            eval_datasets[name]["test"],
            device
            )
        for key, val in eval_statistics.items():
            run[f"eval/{name}/{key}"].log(val)
    
        eval_acc += eval_statistics["acc"] / len(eval_datasets)

    run[f"eval/total/acc"].log(eval_acc)

    model.train()
    proj.train()
    
    # initialize prototypes and their state according to pretrained representations
    print("Initalize prototypes: layer 1")
    tmp_proto, tmp_proto_state = init_proto_lowest(
        args, model, proj, loader, device, proto[0], proto_state[0]
    )
    proto[0] = tmp_proto
    proto_state[0] = tmp_proto_state

    for i in range(1, args.hierarchy):
        print("Initialize prototypes: layer ", i + 1)
        tmp_proto, tmp_proto_state, tmp_proto_connection = init_proto(
            args, proto[i - 1], proto[i], proto_state[i], device
        )
        proto[i] = tmp_proto
        proto_state[i] = tmp_proto_state
        proto_connection.append(tmp_proto_connection)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        (
            train_intra_loss,
            train_intra_acc,
            train_inter_loss,
            train_inter_acc,
            train_proto_loss,
            proto,
        ) = train(args, model, proj, loader, optimizer, device, proto, proto_connection, run)
        
        run["epoch/loss/intraNCE"].log(train_intra_loss)
        run["epoch/acc/intraNCE"].log(train_intra_acc)
        run["epoch/loss/interNCE"].log(train_inter_loss)
        run["epoch/acc/interNCE"].log(train_inter_acc)
        run["epoch/loss/protoNCE"].log(train_proto_loss)
        
        model.eval()
        proj.eval()
        eval_acc = 0.0
        for name in eval_datasets:
            eval_statistics = evaluate_knn(
                featurizer,
                eval_datasets[name]["train"],
                eval_datasets[name]["test"],
                device
                )
            for key, val in eval_statistics.items():
                run[f"eval/{name}/{key}"].log(val)
        
            eval_acc += eval_statistics["acc"] / len(eval_datasets)
    
        run[f"eval/total/acc"].log(eval_acc)

        model.train()
        proj.train()

        torch.save(
            model.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
        )

    torch.save(model.state_dict(), f"../resource/result/{run_tag}/model.pt")

if __name__ == "__main__":
    main()
