#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
ROOT=$HOME/work/llama3.gsm8k/src/huggingface/transformers

MODEL_ROOT=$HOME/models02/llama3/gsm8k/llama-recipes/dense_finetuned

for MODEL_NAME in linear@lr3e-4@B@GrAcc1@GrClip0@GrThr2.0@W0.1@ep2@GPUs8@WD0.0@ID10118-
do
    echo "Converting $MODEL_NAME"
  
    CKPT_FOLDER=$MODEL_ROOT/$MODEL_NAME/home/tuan/models02/llama3/Meta-Llama-3-8B
    SRC_MODEL_FOLDER=$HOME/models02/llama3/Meta-Llama-3-8B
    DST_MODEL_FOLDER=$MODEL_ROOT/$MODEL_NAME/hf
    
    #python $ROOT/inference/checkpoint_converter_fsdp_hf.py \
	#   --fsdp_checkpoint_path $CKPT_FOLDER \
	#   --consolidated_model_path $DST_MODEL_FOLDER \
	#   --HF_model_path_or_name $SRC_MODEL_FOLDER
    
    python $ROOT/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir $CKPT_FOLDER --model_size 8B --output_dir $DST_MODEL_FOLDER
done
