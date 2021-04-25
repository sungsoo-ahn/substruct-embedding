#!/bin/bash

RW_LEN=$1

RUN_TAG="masked_rw_pred_${RW_LEN}"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain_label.py \
--scheme masked_rw_pred \
--num_epochs 10 \
--walk_length $RW_LEN \
--run_tag $RUN_TAG \
--use_neptune

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox"
do
	python finetune.py \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
	--num_runs 1
done
