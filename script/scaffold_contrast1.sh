#!/bin/bash

RUN_TAG="scaffold_contrast1"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain_scaffold.py \
--use_neptune \
--scheme graph_contrast \
--mask_scaffold_features \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
