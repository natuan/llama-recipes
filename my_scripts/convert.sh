#!/bin/bash

MODEL_ROOT=$HOME/models/llama/base_finetuned
CKPT_FOLDER=$MODEL_ROOT/Llama-2-7b-hf@gsm8k@lr3e-5@B16@GrAcc1@W0.1@ep2@GPUs8@CL1@RLW0.4@WD0.0
SRC_MODEL_FOLDER=$MODEL_ROOT/Llama-2-7b-hf
DST_MODEL_FOLDER=$CKPT_FOLDER/hf

ROOT=$HOME/src/natuan/llama-recipes/src/llama_recipes

python $ROOT/inference/checkpoint_converter_fsdp_hf.py \
       --fsdp_checkpoint_path $CKPT_FOLDER \
       --consolidated_model_path $DST_MODEL_FOLDER \
       --HF_model_path_or_name $SRC_MODEL_FOLDER
