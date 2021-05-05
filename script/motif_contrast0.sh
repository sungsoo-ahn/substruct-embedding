#!/bin/bash

RUN_TAG="motif_contrast0"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

python pretrain_motif.py \
--drop_scaffold \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv"\
--model_path $MODEL_PATH \
--run_tag $RUN_TAG

python supervised.py \
--input_model_path $MODEL_PATH \
--output_model_path $SUPERVISED_MODEL_PATH \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv"\
--model_path $SUPERVISED_MODEL_PATH \
--run_tag $RUN_TAG
