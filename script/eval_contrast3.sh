RUN_TAG="contrast3"
MODEL_PATH="../resource/result/${RUN_TAG}/model.pt"
RESUME_PATH="../resource/result/contrast3/checkpoint.pt"

echo $RUN_TAG
echo $MODEL_PATH

python finetune.py \
--datasets "bace" "bbbp" "sider" "clintox" "tox21" "toxcast" "hiv" "muv" \
--model_path $MODEL_PATH \
--num_runs 5 \
--run_tag $RUN_TAG
