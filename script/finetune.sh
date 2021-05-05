#!/bin/bash

RUN_TAG=$1
MODEL_PATH="../resource/model/${RUN_TAG}.pth"

echo $RUN_TAG
#echo $MODEL_PATH

#--model_path $MODEL_PATH \

python finetune.py \
--datasets "hiv" "muv" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
