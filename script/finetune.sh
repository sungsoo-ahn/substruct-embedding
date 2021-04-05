#!/bin/bash
MODEL_PATH=$1
DATASET=$2

for RUNSEED in 0 1 2 3 4 5 6 7 8 9
do
	python finetune.py --runseed $RUNSEED --model_path $MODEL_PATH --dataset $DATASET	
done