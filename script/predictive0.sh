#!/bin/bash

RUN_TAG="edge_pred0"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
RESUME_PATH="../resource/result/${RUN_TAG}/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

python pretrain_multifrag.py \
--scheme predictive \
--drop_p 0.5 \
--num_epochs 20 \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "freesolv" "esol" "sider" "bace" "bbbp" "clintox" "lipophilicity" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--num_runs 3 \
--run_tag $RUN_TAG