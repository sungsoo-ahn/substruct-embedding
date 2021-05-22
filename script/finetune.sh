#!/bin/bash

RUN_TAG=$1
MODEL_PATH="../resource/model/${RUN_TAG}.pth"

echo $RUN_TAG
echo $MODEL_PATH

#"bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" 

python finetune.py \
--datasets "freesolv" "esol" "lipophilicity" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
