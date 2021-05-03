#!/bin/bash

RUN_TAG="group_contrast1"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain_group.py \
--use_neptune \
--reweight \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
