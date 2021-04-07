#!/bin/bash

NEPTUNE_MODE=$1
RUN_TAG="nm"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $NEPTUNE_MODE
echo $WALK_LENGTH_RATE
echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--neptune_mode $NEPTUNE_MODE \
--scheme node_mask --run_tag $RUN_TAG

for DATASET in "tox21" "bace" "bbbp" "toxcast" #"sider" "clintox" "hiv" "muv"
do
	python finetune.py \
	--neptune_mode $NEPTUNE_MODE \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
