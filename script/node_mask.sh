#!/bin/bash

RUN_TAG="nm"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $WALK_LENGTH_RATE
echo $RUN_TAG
echo $MODEL_PATH

python pretrain.py --scheme node_masking --node_mask_rate 0.3 --run_tag $RUN_TAG

for RUNSEED in 0 1 2 3 4 5 6 7 8 9
do
	for DATASET in "tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox"
	do
		python finetune.py --runseed $RUNSEED --dataset $DATASET --model_path $MODEL_PATH
	done	
done