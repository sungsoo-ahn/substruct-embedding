#!/bin/bash

RUN_TAG="mask_contrast_full"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--use_neptune \
--scheme mask_contrast \
--transform mask \
--num_epochs 100 \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "toxcast" "sider" "clintox" "tox21" "hiv" "muv" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
