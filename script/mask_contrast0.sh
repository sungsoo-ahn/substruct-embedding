#!/bin/bash

RUN_TAG="mask_contrast0"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--run_tag $RUN_TAG

python finetune.py \
--datasets "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
