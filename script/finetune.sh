#!/bin/bash

RUN_TAG=$1
MODEL_PATH="../resource/model/${RUN_TAG}.pth"

echo $RUN_TAG
echo $MODEL_PATH

python finetune.py \
--datasets "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
