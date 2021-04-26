#!/bin/bash

RUN_TAG="graphlog0"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

python pretrain_graphlog.py \
--alpha 1.0 \
--beta 0.0 \
--gamma 0.0 \
--run_tag $RUN_TAG

python finetune.py \
--datasets "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG