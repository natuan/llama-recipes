#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

SRC_MODEL_DIR=/home/abhinav/models
MODEL_NAME=Llama-2-7b-hf
SRC_MODEL=$SRC_MODEL_DIR/$MODEL_NAME

DIST_MODEL_ROOT=$HOME/models/$MODEL_NAME/my_checkpoints
DIST_MODEL_FT=$HOME/models/$MODEL_NAME/my_finetuned

export WANDB_RUN_GROUP=foo
export WANDB_NAME=$DIST_MODEL_FT

torchrun --nnodes 1 --nproc_per_node 1  examples/finetuning.py \
	 --dataset gsm8k_dataset \
	 --enable_fsdp \
	 --pure_bf16 \
	 --model_name $SRC_MODEL \
	 --dist_checkpoint_root_folder $DIST_MODEL_ROOT \
	 --dist_checkpoint_folder $DIST_MODEL_FT


# For reference, do not remove/change
# Suggested command for full-parameter finetuning
# torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --use_fast_kernels
