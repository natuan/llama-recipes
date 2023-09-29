#!/bin/bash

MODEL_ROOT=$HOME/models/base_finetuned
CKPT_FOLDER=$MODEL_ROOT/Llama-2-7b-hf@gsm8k@lr1e-5@B32@wd0.0@ep2
SRC_MODEL_FOLDER=$MODEL_ROOT/Llama-2-7b-hf
DST_MODEL_FOLDER=$CKPT_FOLDER/hf

ROOT=/home/tuan/src/natuan/llama-recipes/src/llama_recipes

python $ROOT/inference/checkpoint_converter_fsdp_hf.py \
       --fsdp_checkpoint_path $CKPT_FOLDER \
       --consolidated_model_path $DST_MODEL_FOLDER \
       --HF_model_path_or_name $SRC_MODEL_FOLDER
