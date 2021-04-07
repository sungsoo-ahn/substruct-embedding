#!/bin/bash

NEPTUNE_MODE=$1

echo $NEPTUNE_MODE

for DATASET in "tox21" "hiv" "pcba" "muv" "bace" "bbbp" "toxcast" "sider" "clintox"
do
	python finetune.py \
	--neptune_mode $NEPTUNE_MODE \
	--runseed $RUNSEED \
	--dataset $DATASET \
	--run_tag "nopretrain"
done	
