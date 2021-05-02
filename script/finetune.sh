#!/bin/bash

RUN_TAG=$1
MODEL_PATH="../resource/result/mask_safe_contrast/model_15.pt"

echo $RUN_TAG
#echo $MODEL_PATH

#--model_path $MODEL_PATH \

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG