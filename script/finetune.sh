#!/bin/bash

NEPTUNE_MODE=$1

echo $NEPTUNE_MODE

for DATASET in "tox21" "bace" "bbbp" "toxcast" "sider" "clintox" "hiv" "muv"
do
	python finetune.py \
	--neptune_mode $NEPTUNE_MODE \
	--dataset $DATASET \
	--run_tag "nopretrain"
done	
