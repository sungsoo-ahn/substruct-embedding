#!/bin/bash

RUN_TAG="mask_contrast1"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--use_neptune \
--scheme robust_mask_contrast \
--gce_coef 0.3 \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast"\
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
