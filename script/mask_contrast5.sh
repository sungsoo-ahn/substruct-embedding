#!/bin/bash

RUN_TAG="mask_contrast5"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH
"""

"""
python pretrain.py \
--use_neptune \
--scheme mask_balanced_contrast \
--mask_rate 0.15 \
--run_tag $RUN_TAG
"""

python finetune.py \
--datasets "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
