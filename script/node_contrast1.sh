#!/bin/bash

RUN_TAG="node_contrast1"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py \
--use_neptune \
--logit_sample_ratio 1.0 \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG
