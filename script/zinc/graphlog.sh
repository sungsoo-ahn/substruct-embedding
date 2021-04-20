#!/bin/bash

NEPTUNE_MODE=$1
RUN_TAG="graph_clustering"
MODEL_PATH="../resource/result/graphlog/model.pt"

echo $NEPTUNE_MODE

python pretrain_graphlog.py

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" "hiv"
do
	python finetune.py \
	--neptune_mode $NEPTUNE_MODE \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
