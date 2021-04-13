#!/bin/bash

NEPTUNE_MODE=$1
EDGE_MASK_RATIO=$2
RUN_TAG="emam_${EDGE_MASK_RATIO}"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $NEPTUNE_MODE
echo $WALK_LENGTH_RATE
echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--neptune_mode $NEPTUNE_MODE \
--scheme edge_mask \
--edge_mask_rate $EDGE_MASK_RATIO \
--edge_attr_mask \
--run_tag $RUN_TAG

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" "hiv" "muv"
do
	python finetune.py \
	--neptune_mode $NEPTUNE_MODE \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
