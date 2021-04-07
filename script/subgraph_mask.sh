#!/bin/bash

WALK_LENGTH_RATE=$1
RUN_TAG="wlr_${WALK_LENGTH_RATE}"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $WALK_LENGTH_RATE
echo $RUN_TAG
echo $MODEL_PATH

#python pretrain.py --scheme subgraph_masking --walk_length_rate $WALK_LENGTH_RATE --run_tag $RUN_TAG

for RUNSEED in 0 1 2 3 4 5 6 7 8 9
do
	for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox"
	do
		python finetune.py --runseed $RUNSEED --dataset $DATASET --model_path $MODEL_PATH
	done	
done
