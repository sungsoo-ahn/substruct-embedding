#!/bin/bash

RUN_TAG="edge_contrast0"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--use_neptune \
--scheme edge_contrast \
--transform edge_mask \
--mask_rate 0.1 \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "toxcast" "sider" "clintox" "tox21" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
