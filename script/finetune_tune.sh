#!/bin/bash

SCHEME=$1
LR=$2
LR_SCALE=$3
RUN_TAG="${SCHEME}_${LR}_${LR_SCALE}"
MODEL_PATH="../resource/model/${SCHEME}.pth"

echo $RUN_TAG
echo $MODEL_PATH

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" \
--model_path $MODEL_PATH \
--lr $LR \
--lr_scale $LR_SCALE \
--run_tag $RUN_TAG
