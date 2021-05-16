#!/bin/bash

RUN_TAG="contrast3"
MODEL_PATH="../resource/result/${RUN_TAG}/model_63.pt"
RESUME_PATH="../resource/result/${RUN_TAG}/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

#python pretrain.py \
#--scheme contrastive \
#--transform_type pair \
#--use_double_projector \
#--use_dangling_mask \
#--use_valid \
#--use_neptune \
#--resume_path $RESUME_PATH \
#--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG