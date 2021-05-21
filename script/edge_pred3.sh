#!/bin/bash

RUN_TAG="edge_pred3"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
RESUME_PATH="../resource/result/${RUN_TAG}/checkpoint.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

python pretrain_multifrag.py \
--scheme edge_predictive \
--drop_p 0.3 \
--x_mask_rate 0.15 \
--resume_path $RESUME_PATH \
--num_epochs 50 \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--num_runs 1 \
--run_tag $RUN_TAG