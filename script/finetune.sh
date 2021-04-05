#!/bin/bash
MODEL_PATH=$0
DATASET=$1

for RUNSEED in 0 1 2 3 4 5 6 7 8 9
do
	python finetune.py --runseed $RUNSEED --model_path $MODEL_PATH --dataset $DATASET	
done