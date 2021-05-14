import os
from collections import defaultdict
import argparse
import numpy as np

import torch

from frag_dataset import FragDataset
from scheme import maskpred, contrastive, junction_maskpred, junction_contrastive
from data.transform import contract_both, contract_once
from data.collate import double_collate
import neptune.new as neptune

from tqdm import tqdm
from time import asctime

def compute_accuracy(pred, target):
    acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
    return acc

def train_step(batch, model, optim):
    model.train()

    statistics = dict()
    logits, labels = model.compute_logits_and_labels(batch)
    loss = model.criterion(logits, labels)
    acc = compute_accuracy(logits, labels)
    
    statistics["loss"] = loss.detach()
    statistics["acc"] = acc

    optim.zero_grad()
    loss.backward()
    optim.step()

    return statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_brics")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--log_freq", type=float, default=100)

    parser.add_argument("--scheme", type=str, default="contrastive")
    parser.add_argument("--transform", type=str, default="none")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--use_neptune", action="store_true")
    
    parser.add_argument("--aggr", type=str, default="max")
    parser.add_argument("--use_relation", action="store_true")  
    parser.add_argument("--mask_p", type=float, default=0.15)
    parser.add_argument("--contract_p", type=float, default=0.8)
    parser.add_argument("--contract_type", type=str, default="once")
    parser.add_argument("--drop_junction", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    model = contrastive.Model(args.aggr, args.use_relation)
    if args.contract_type == "once":
        transform = lambda data: contract_once(
            data, contract_p=args.contract_p, mask_p=args.mask_p, drop_junction=args.drop_junction
            )
    elif args.contract_type == "both":
        transform = lambda data: contract_both(
            data, contract_p=args.contract_p, mask_p=args.mask_p, drop_junction=args.drop_junction
            )
        
    collate = double_collate
        
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
        collate_fn=collate,
    )

    if args.use_neptune:
        print("Loading neptune...")
        run = neptune.init(
            project="sungsahn0215/ssg", 
            name="group_contrast", 
            source_files=["*.py", "**/*.py"], 
            )
        run["parameters"] = vars(args)
        if args.run_tag == "":
            run_tag = run["sys/id"].fetch()
        else:
            run_tag = args.run_tag
        os.makedirs(f"../resource/result/{run_tag}", exist_ok=True)

    step = 0
    cum_train_statistics = defaultdict(float)
    for epoch in range(args.num_epochs):
        print(f"[{asctime()}] epoch: {epoch}")
        if args.use_neptune:
            run[f"epoch"].log(epoch)

        for batch in (loader):
            step += 1
            train_statistics = train_step(batch, model, optim)
            for key, val in train_statistics.items():
                cum_train_statistics[key] += val / args.log_freq
                
            if step % args.log_freq == 0:
                prompt = ""
                for key, val in cum_train_statistics.items():
                    prompt += f"train/{key}: {val:.2f} "
                    if args.use_neptune:
                        run[f"train/{key}"].log(val)
                        
                cum_train_statistics = defaultdict(float)
                
                print(f"[{asctime()}] {prompt}")

        if args.use_neptune:
            torch.save(
                model.encoder.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
            )

    if args.use_neptune:
        torch.save(model.encoder.state_dict(), f"../resource/result/{run_tag}/model.pt")
        run.stop()


if __name__ == "__main__":
    main()
