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
--num_epochs 20 \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "freesolv" "esol" "sider" "bace" "bbbp" "clintox" "lipophilicity" "tox21" "toxcast" "hiv" "muv" \
--num_atom_type 121 \
--model_path $MODEL_PATH \
--num_runs 5 \
--run_tag $RUN_TAG