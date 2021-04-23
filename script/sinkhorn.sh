#!/bin/bash

NUM_CENTROIDS=$1

RUN_TAG="sinkhorn_${NUM_CENTROIDS}"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain_sinkhorn.py \
--num_epochs 20 \
--run_tag $RUN_TAG \
--num_centroids $NUM_CENTROIDS \
--use_neptune

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox"
do
	python finetune.py \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
