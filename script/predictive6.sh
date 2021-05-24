#!/bin/bash

RUN_TAG="predictive6"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
RESUME_PATH="../resource/result/${RUN_TAG}/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

python pretrain.py \
--scheme predictive \
--version 0 \
--drop_p 0.5 \
--add_fake \
--x_mask_rate 0.15 \
--num_epochs 100 \
--use_neptune \
--resume_path $RESUME_PATH \
--run_tag $RUN_TAG


python supervised.py \
--input_model_path $MODEL_PATH \
--output_model_path $SUPERVISED_MODEL_PATH \
--num_atom_type 121 \
--epochs 100 \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "freesolv" "esol" "sider" "bace" "bbbp" "clintox" "lipophilicity" "tox21" "toxcast" "hiv" "muv" \
--num_atom_type 121 \
--model_path $MODEL_PATH \
--num_runs 5 \
--run_tag $RUN_TAG