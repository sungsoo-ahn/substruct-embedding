#!/bin/bash

RUN_TAG="predictive7"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
RESUME_PATH="../resource/result/${RUN_TAG}/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

#python pretrain.py \
#--scheme predictive \
#--version 2 \
#--add_fake \
#--drop_p 0.5 \
#--num_epochs 20 \
#--x_mask_rate 0.15 \
#--run_tag $RUN_TAG
#--use_neptune \

python finetune.py \
--datasets "freesolv" "esol" "sider" "bace" "bbbp" "clintox" "lipophilicity" "tox21" "toxcast" "hiv" "muv" \
--num_atom_type 121 \
--model_path $MODEL_PATH \
--num_runs 5 \
--run_tag $RUN_TAG