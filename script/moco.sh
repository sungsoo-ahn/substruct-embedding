#!/bin/bash

MASK_RATE=$1
POOL_RATE=$2

RUN_TAG="moco_${MASK_RATE}_${POOL_RATE}"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--scheme moco \
--transform_type mask \
--pool_type mask \
--num_epochs 20 \
--mask_rate $MASK_RATE \
--pool_rate $POOL_RATE \
--run_tag $RUN_TAG

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox"
do
	python finetune.py \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
