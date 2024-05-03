#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
ROOT=$HOME/work/llama3.gsm8k/src/natuan/llama-recipes/src/llama_recipes

MODEL_ROOT=$HOME/models02/llama3/gsm8k/llama-recipes/dense_finetuned

models=($(find ${MODEL_ROOT} -maxdepth 1 -mindepth 1 -type d -exec basename {} \;))

for MODEL_NAME in "${models[@]}"
do
    DST_MODEL_FOLDER=$MODEL_ROOT/$MODEL_NAME/hf
    if test -d ${DST_MODEL_FOLDER}; then
        echo "Skip ${MODEL_NAME}."
        rm -rf $MODEL_ROOT/$MODEL_NAME/home
        continue
    fi

    echo "Converting $MODEL_NAME"
  
    CKPT_FOLDER=$MODEL_ROOT/$MODEL_NAME/home/tuan/models02/llama3/Meta-Llama-3-8B
    SRC_MODEL_FOLDER=$HOME/models02/llama3/Meta-Llama-3-8B
    
    python $ROOT/inference/checkpoint_converter_fsdp_hf.py \
	   --fsdp_checkpoint_path $CKPT_FOLDER \
	   --consolidated_model_path $DST_MODEL_FOLDER \
	   --HF_model_path_or_name $SRC_MODEL_FOLDER

    if test -d ${DST_MODEL_FOLDER}; then
        rm -rf $MODEL_ROOT/$MODEL_NAME/home
    fi
done
