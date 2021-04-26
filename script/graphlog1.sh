#!/bin/bash

RUN_TAG="graphlog1"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

python pretrain_graphlog.py \
--alpha 1.0 \
--beta 0.1 \
--gamma 0.1 \
--run_tag $RUN_TAG

python finetune.py \
--datasets "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG