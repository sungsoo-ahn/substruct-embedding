#!/bin/bash

RUN_TAG="mask_contrast2"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH
"""

python pretrain.py \
--use_neptune \
--scheme mask_contrast \
--mask_rate 0.45 \
--run_tag $RUN_TAG
"""

python finetune.py \
--datasets "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
