#!/bin/bash

ROOT=$HOME/src/natuan/llama-recipes/src/llama_recipes

MODEL_ROOT=/network/tuan/models/llama/GSM8K/clip_softmax

for MODEL_NAME in Llama-2-7b-hf@gsm8k@lr3e-5@B16@GrAcc1@W0.1@ep2@GPUs4@ClipSM_OFF@ID18250
do
    echo "Converting $MODEL_NAME"
    CKPT_FOLDER=$MODEL_ROOT/$MODEL_NAME/home/tuan/models/llama/Llama-2-7b-hf

    #rsync -avhW --progress $CKPT_FOLDER/home/tuan/models/llama/Llama-2-7b-hf/*.* $CKPT_FOLDER/

    SRC_MODEL_FOLDER=$HOME/models/llama/base_finetuned/Llama-2-7b-hf
    DST_MODEL_FOLDER=$MODEL_ROOT/$MODEL_NAME/hf

    python $ROOT/inference/checkpoint_converter_fsdp_hf.py \
	   --fsdp_checkpoint_path $CKPT_FOLDER \
	   --consolidated_model_path $DST_MODEL_FOLDER \
	   --HF_model_path_or_name $SRC_MODEL_FOLDER
done
