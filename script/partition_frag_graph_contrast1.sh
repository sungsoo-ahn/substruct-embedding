#!/bin/bash

RUN_TAG="partition_frag_graph_contrast1"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
SUPERVISED_MODEL_PATH="../resource/result/${RUN_TAG}/model_supervised.pt"

echo $RUN_TAG
echo $MODEL_PATH
echo $SUPERVISED_MODEL_PATH

python pretrain.py \
--scheme partition_frag_graph_contrast \
--batch_size 1024 \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--run_tag $RUN_TAG

python supervised.py \
--input_model_path $MODEL_PATH \
--output_model_path $SUPERVISED_MODEL_PATH \
--use_neptune \
--run_tag $RUN_TAG

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $SUPERVISED_MODEL_PATH \
--run_tag $RUN_TAG
