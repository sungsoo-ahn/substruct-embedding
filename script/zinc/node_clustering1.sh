#!/bin/bash

RUN_TAG="node_clustering_graph"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--scheme node_clustering \
--num_epochs 20 \
--contrastive_type graph \
--run_tag $RUN_TAG \
--use_neptune

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox"
do
	python finetune.py \
	--dataset $DATASET \
	--model_path $MODEL_PATH \
	--run_tag $RUN_TAG
done
