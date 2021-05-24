#!/bin/bash

RUN_TAG="predictive9"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
RESUME_PATH="../resource/result/${RUN_TAG}/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

python pretrain.py \
--scheme predictive \
--version 1 \
--drop_p 0.5 \
--add_fake \
--num_epochs 100 \
--use_neptune \
--resume_path $RESUME_PATH \
--run_tag $RUN_TAG

python supervised.py \
--input_model_path $MODEL_PATH \
--output_model_path $SUPERVISED_MODEL_PATH \
--epochs 100 \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "freesolv" "esol" "sider" "bace" "bbbp" "clintox" "lipophilicity" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--num_runs 5 \
--run_tag $RUN_TAG