#!/bin/bash

LOSS_TYPE=$1
TRANSFORM_TYPE=$2
RUN_TAG="${LOSS_TYPE}_${TRANSFORM_TYPE}"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"

echo $LOSS_TYPE
echo $TRANSFORM_TYPE
echo $RUN_TAG
echo $MODEL_PATH

python train_embedding.py --loss_type $LOSS_TYPE --transform_type $TRANSFORM_TYPE --run_tag $RUN_TAG

for RUNSEED in 0 1 2 3 4 5 6 7 8 9
do
	for DATASET in "tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox"
	do
		python finetune.py --runseed $RUNSEED --dataset $DATASET --model_path $MODEL_PATH
	done	
done