#!/bin/bash

RUN_TAG="graph_clustering"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--num_cluster 9000 \
--log_freq 1 \
--cluster_freq 2 \
--num_epochs 20 \
--scheme graph_clustering \
--dataset muv \
--run_tag $RUN_TAG

python finetune.py \
--dataset muv \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
