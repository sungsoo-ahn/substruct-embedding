import os
from collections import defaultdict
import argparse
import numpy as np
import random

import torch

from frag_dataset import FragDataset
from scheme import edge_predictive, edge_contrastive
from data.transform import multi_fragment
from data.collate import multifrag_collate
import neptune.new as neptune

from tqdm import tqdm
from time import asctime

def train_step(batchs, model, optim):
    model.train()

    statistics = dict()
    logits, labels = model.compute_logits_and_labels(batchs)
    loss = model.criterion(logits, labels)
    acc = model.compute_accuracy(logits, labels)

    statistics["loss"] = loss.detach()
    statistics["acc"] = acc

    optim.zero_grad()
    loss.backward()
    optim.step()

    return statistics

def valid_step(batchs, model):
    model.train()

    statistics = dict()
    logits, labels = model.compute_logits_and_labels(batchs)
    loss = model.criterion(logits, labels)
    acc = model.compute_accuracy(logits, labels)

    statistics["loss"] = loss.detach()
    statistics["acc"] = acc
    
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_brics")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--log_freq", type=float, default=100)
    parser.add_argument("--resume_path", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--use_neptune", action="store_true")

    parser.add_argument("--use_valid", action="store_true")
    
    parser.add_argument("--scheme", type=str, default="predictive")
    parser.add_argument("--drop_p", type=float, default=0.5)
    parser.add_argument("--min_num_nodes", type=int, default=0)
    parser.add_argument("--proj_type", type=int, default=0)
    parser.add_argument("--aug_x", action="store_true")
    parser.add_argument("--x_mask_rate", type=float, default=0.0)
    
    parser.add_argument("--input_model_path", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    if args.scheme == "predictive":
        model = edge_predictive.Model()
    elif args.scheme == "contrastive":
        model = edge_contrastive.Model()
    
    transform = lambda data: multi_fragment(data, args.drop_p, args.aug_x, args.x_mask_rate)
    
    print("Loading model...")
    model = model.cuda()
    optim = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=args.lr
    )
    
    if args.input_model_path != "":
        state_dict = torch.load(args.input_model_path)
        model.encoder.load_state_dict(state_dict)

    
    if args.resume_path != "":
        print("Loading checkpoint...")
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        run_id = checkpoint["run_id"]
    else:
        start_epoch = 0
        run_id = None

    print("Loading dataset...")
    dataset = FragDataset(
        "../resource/dataset/" + args.dataset, dataset=args.dataset, transform=transform,
    )
    
    if args.use_valid:
        print("Splitting dataset...")
        perm = list(range(len(dataset)))
        random.shuffle(perm)
        valid_dataset = dataset[torch.tensor(perm[:100000])]
        dataset = dataset[torch.tensor(perm[100000:])]
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multifrag_collate,
    )

    if args.use_valid:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )


    if args.use_neptune:
        print("Loading neptune...")
        run = neptune.init(
            project="sungsahn0215/ssg",
            name="group_contrast",
            source_files=["*.py", "**/*.py"],
            run=run_id
            )
        run["parameters"] = vars(args)
        run_id = run["sys/id"].fetch()
        if args.run_tag == "":
            run_tag = run_id
        else:
            run_tag = args.run_tag
        os.makedirs(f"../resource/result/{run_tag}", exist_ok=True)

    step = 0
    cum_train_statistics = defaultdict(float)
    for epoch in range(start_epoch+1, args.num_epochs):
        print(f"[{asctime()}] epoch: {epoch}")
        if args.use_neptune:
            run[f"epoch"].log(epoch)

        for batchs in tqdm(loader):
            step += 1
            train_statistics = train_step(batchs, model, optim)
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
                    
        if args.use_valid:
            cum_valid_statistics = defaultdict(float)
            for batchs in (valid_loader):
                with torch.no_grad():
                    valid_statistics = valid_step(batchs, model)
                
                for key, val in valid_statistics.items():
                    cum_valid_statistics[key] += val / len(valid_loader)

            prompt = ""
            for key, val in cum_valid_statistics.items():
                prompt += f"valid/{key}: {val:.2f} "
                if args.use_neptune:
                    run[f"valid/{key}"].log(val)

            print(f"[{asctime()}] {prompt}")

        
        if args.use_neptune:
            checkpoint = {
                "epoch": epoch,
                "optim": optim.state_dict(),
                "model": model.state_dict(),
                "run_id": run_id,
            }
            torch.save(checkpoint, f"../resource/result/{run_tag}/checkpoint.pt")
            torch.save(
                model.encoder.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
            )

    if args.use_neptune:
        torch.save(model.encoder.state_dict(), f"../resource/result/{run_tag}/model.pt")
        run.stop()


if __name__ == "__main__":
    main()
