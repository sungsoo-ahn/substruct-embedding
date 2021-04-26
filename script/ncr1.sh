#!/bin/bash

RUN_TAG="ncr1"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

python pretrain.py \
--scheme node_clustering_realmask \
--num_warmup_epochs 100 \
--run_tag $RUN_TAG

python finetune.py \
--datasets "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG