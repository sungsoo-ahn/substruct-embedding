#!/bin/bash

RUN_TAG="edge_pred0"
MODEL_PATH="../resource/result/${RUN_TAG}/model_68.pt"
RESUME_PATH="../resource/result/${RUN_TAG}/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

#python pretrain_multifrag.py \
#--scheme edge_predictive \
#--num_epochs 100 \
#--use_neptune \
#--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--num_runs 5 \
--run_tag $RUN_TAG