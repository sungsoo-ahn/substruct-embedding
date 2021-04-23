#!/bin/bash

RUN_TAG="masked_node_pred"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain_label.py \
--scheme masked_node_pred \
--num_epochs 100 \
--run_tag $RUN_TAG \
--use_neptune

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox"
do
	python finetune.py \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
