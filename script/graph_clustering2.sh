#!/bin/bash

NEPTUNE_MODE=$1
RUN_TAG="graph_clustering2"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $NEPTUNE_MODE
echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--neptune_mode $NEPTUNE_MODE \
--scheme graph_clustering \
--ema_rate 0.995 \
--num_epochs 10 \
--run_tag $RUN_TAG

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" "hiv"
do
	python finetune.py \
	--neptune_mode $NEPTUNE_MODE \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
