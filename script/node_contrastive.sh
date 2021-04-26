#!/bin/bash

MASK_RATE=$1
POOL_RATE=$2

RUN_TAG="nc_${MASK_RATE}_${POOL_RATE}"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--num_epochs 5 \
--mask_rate $MASK_RATE \
--pool_rate $POOL_RATE \
--use_neptune \
--run_tag $RUN_TAG

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox"
do
	python finetune.py \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG \
    --num_runs 1
done