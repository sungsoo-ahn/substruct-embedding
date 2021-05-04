#!/bin/bash

RUN_TAG="group_contrast4"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain_group.py \
--group_contrast \
--atom_contrast \
--self_contrast \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
