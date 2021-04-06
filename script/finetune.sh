#!/bin/bash

for RUNSEED in 0 1 2 3 4 5 6 7 8 9
do
	for DATASET in "tox21" "hiv" "pcba" "muv" "bace" "bbbp" "toxcast" "sider" "clintox"
	do
		python finetune.py --runseed $RUNSEED --dataset $DATASET
	done	
done