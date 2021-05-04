#!/bin/bash

RUN_TAG="group_contrast5"
MODEL_PATH="../resource/result/${RUN_TAG}/model_00.pt"

echo $RUN_TAG
echo $MODEL_PATH

"""
python pretrain_group.py \
--self_contrast \
--use_neptune \
--run_tag $RUN_TAG
"""

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
