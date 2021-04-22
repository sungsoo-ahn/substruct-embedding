#!/bin/bash

RUN_TAG="node_clustering_noaug_linear"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain_clustering.py \
--scheme node_clustering_noaug \
--use_linear_projection \
--num_epochs 20 \
--run_tag $RUN_TAG

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" "hiv"
do
	python finetune.py \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
