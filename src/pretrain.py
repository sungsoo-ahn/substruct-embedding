import os
from collections import defaultdict
import argparse
import numpy as np

import torch
import torch_geometric

from frag_dataset import FragDataset, multiple_collate
from scheme import sample, partition
from data.transform import sample_data, partition_data
import neptune.new as neptune

from tqdm import tqdm

def compute_accuracy(pred, target):
    acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
    return acc

def train_step(batchs, model, optim):
    model.train()

    logits_and_labels = model.compute_logits_and_labels(batchs)

    cum_loss = 0.0
    statistics = dict()
    for key in logits_and_labels:
        logits, labels = logits_and_labels[key]    
        loss = model.criterion(logits, labels)
        acc = compute_accuracy(logits, labels)
        
        statistics[f"{key}/loss"] = loss.detach()
        statistics[f"{key}/acc"] = acc
        
        cum_loss += loss            

    optim.zero_grad()
    cum_loss.backward()
    optim.step()

    return statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_brics")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--log_freq", type=float, default=10)

    parser.add_argument("--scheme", type=str, default="frag_node_contrast")
    parser.add_argument("--transform", type=str, default="none")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--use_neptune", action="store_true")
    
    parser.add_argument("--sample_p", type=float, default=0.5)
    parser.add_argument("--use_dangling_node_features", action="store_true")
    parser.add_argument("--use_mlp_predict", action="store_true")
    
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.scheme == "sample":
        model = sample.Model()
        transform = lambda data: sample_data(data, args.sample_p)
    
    elif args.scheme == "partition":
        model = partition.Model(args.use_dangling_node_features, args.use_mlp_predict)
        transform = partition_data
    
    
    print("Loading model...")
    model = model.cuda()
    optim = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=args.lr
    )

    print("Loading dataset...")
    dataset = FragDataset(
        "../resource/dataset/" + args.dataset, dataset=args.dataset, transform=transform,
    )
        
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multiple_collate,
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

        for batchs in tqdm(loader):
            step += 1
            train_statistics = train_step(batchs, model, optim)
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
