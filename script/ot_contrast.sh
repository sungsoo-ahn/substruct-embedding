#!/bin/bash

NEPTUNE_MODE=$1
SINKHORN_ITER=$2
RUN_TAG="contrast_${SINKHORN_ITER}"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $NEPTUNE_MODE
echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--neptune_mode $NEPTUNE_MODE \
--scheme ot_contrast \
--num_epochs 5 \
--num_sinkhorn_iters $SINKHORN_ITER \
--run_tag $RUN_TAG

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" "hiv" "muv"
do
	python finetune.py \
	--neptune_mode $NEPTUNE_MODE \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
