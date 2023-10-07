#!/bin/bash

export CUDA_VISIBLE_DEVICES=3


source $HOME/src/natuan/llama-recipes/my_scripts/start_here.sh

MODEL_DIR=/data/models/tuan/llama

SRC_MODEL_ORG_OR_DIR=$MODEL_DIR/PY007
MODEL_NAME=TinyLlama-1.1B-step-50K-105b
SRC_MODEL=$SRC_MODEL_ORG_OR_DIR/$MODEL_NAME

DIST_MODEL_ROOT=$MODEL_DIR/debug

RESULT_LOSS_WEIGHT=0.5

for EPOCHS in 2
do
    export WANDB_RUN_GROUP="Full training set - Epochs ${EPOCHS}"
    for LR in 1e-5
    do
	for BS in 2
	do
	    for WD in 0.0
	    do
		DIST_MODEL_FT=$MODEL_NAME@gsm8k@lr$LR@B$BS@wd$WD@ep$EPOCHS
		export WANDB_NAME=$DIST_MODEL_FT
		torchrun --nnodes 1 --nproc_per_node 1 examples/finetuning.py \
			 --dataset gsm8k_dataset \
			 --num_epochs $EPOCHS \
			 --lr $LR \
			 --use_custom_loss \
			 --result_loss_weight $RESULT_LOSS_WEIGHT \
			 --batch_size_training $BS \
			 --weight_decay $WD \
			 --enable_fsdp \
			 --pure_bf16 \
			 --model_name $SRC_MODEL \
			 --dist_checkpoint_root_folder $DIST_MODEL_ROOT \
			 --dist_checkpoint_folder $DIST_MODEL_FT
		cp "$0" $DIST_MODEL_FT/train_command.sh
	    done
	done
    done
done

# 			 
# For reference, do not remove/change
# Suggested command for full-parameter finetuning
# torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --use_fast_kernels
