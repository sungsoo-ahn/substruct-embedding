import os
from collections import defaultdict
import argparse
import numpy as np

import torch
import torch_geometric

from model import NodeEncoder
from data.dataset import MoleculeDataset
from data.splitter import random_split
from data.transform import mask_data_twice
from data.collate import collate_twice
from scheme.moco import MoCoScheme, MoCoModel
from evaluate_knn import get_eval_datasets, evaluate_knn

import neptune.new as neptune

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--log_freq", type=float, default=10)
    
    parser.add_argument("--scheme", type=str, default="moco")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--run_tag", type=str, default="")
    
    parser.add_argument("--use_neptune", action="store_true")
    
    parser.add_argument("--transform_type", type=str, default="mask")
    parser.add_argument("--pool_type", type=str, default="mask")
    parser.add_argument("--mask_rate", type=float, default=0.0)
    parser.add_argument("--pool_rate", type=float, default=0.1)
    
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    scheme = MoCoScheme()
    model = MoCoModel(pool_type=args.pool_type)
    
    if args.transform_type == "mask":
        transform = lambda data: mask_data_twice(data, args.mask_rate, args.pool_rate)
        
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
        collate_fn=collate_twice,
    )

    if args.use_neptune:
        print("Loading neptune...")
        run = neptune.init(project="sungsahn0215/graph-contrastive", name="train_embedding")
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

            if step % args.log_freq == 0:
                for key, val in train_statistics.items():
                    if args.use_neptune:
                        run[f"train/{key}"].log(val)

        if epoch == 0:
            if args.dataset == "zinc_standard_agent":
                eval_datasets = get_eval_datasets()
            else:
                eval_datasets = get_eval_datasets([args.dataset])

        model.eval()
        eval_acc = 0.0
        for idx, name in enumerate(eval_datasets):
            eval_statistics = evaluate_knn(
                model.compute_graph_features,
                eval_datasets[name]["train"],
                eval_datasets[name]["test"],
                )
            for key, val in eval_statistics.items():
                if args.use_neptune:
                    run[f"eval/{name}/{key}"].log(val)

            eval_acc += eval_statistics["acc"] / len(eval_datasets)

        if args.use_neptune:
            run[f"eval/total/acc"].log(eval_acc)
        
            torch.save(
                model.ema_encoder.state_dict(), f"../resource/result/{run_tag}/model_{epoch:02d}.pt"
            )

    if args.use_neptune:
        torch.save(model.ema_encoder.state_dict(), f"../resource/result/{run_tag}/model.pt")

    run.stop()


if __name__ == "__main__":
    main()
