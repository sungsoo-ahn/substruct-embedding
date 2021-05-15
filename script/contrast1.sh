#!/bin/bash

RUN_TAG="contrast1"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

python pretrain.py \
--contract_type once \
--contract_p 0.9 \
--mask_p 0.0 \
--use_double_encoder \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG