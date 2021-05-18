#!/bin/bash

RUN_TAG="contrast0_supervised"
MODEL_PATH="../resource/model/fragcontrast.pth"
RESUME_PATH="../resource/result/contrast3/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/model/fragcontrast_supervised.pth"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

#python supervised.py \
#--input_model_path $MODEL_PATH \
#--output_model_path $SUPERVISED_MODEL_PATH \
#--use_neptune \
#--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $SUPERVISED_MODEL_PATH \
--run_tag $RUN_TAG
