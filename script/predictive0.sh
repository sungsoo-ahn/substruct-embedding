#!/bin/bash

RUN_TAG="predictive0"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
RESUME_PATH="../resource/result/${RUN_TAG}/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

python pretrain.py \
--scheme predictive \
--num_epochs 20 \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "freesolv" "esol" "sider" "bace" "bbbp" "clintox" "lipophilicity" "tox21" "qm7" "toxcast" "qm8" "hiv" "muv" \
--model_path $MODEL_PATH \
--num_runs 5 \
--run_tag $RUN_TAG