#!/bin/bash

RUN_TAG="node_clustering"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--scheme node_clustering \
--run_tag $RUN_TAG

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" "hiv"
do
	python finetune.py \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
