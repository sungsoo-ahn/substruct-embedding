#!/bin/bash

RUN_TAG=$1
INPUT_MODEL_PATH="../resource/model/${RUN_TAG}.pth"
OUTPUT_MODEL_PATH="../resource/model/${RUN_TAG}_supervised.pth"

echo $RUN_TAG

python supervised.py \
--input_model_path $INPUT_MODEL_PATH \
--output_model_path $OUTPUT_MODEL_PATH \
--use_neptune \
--run_tag $RUN_TAG
