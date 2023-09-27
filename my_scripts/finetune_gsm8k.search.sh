#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

source $HOME/src/facebookreseach/llama-recipes/my_scripts/start_here.sh

SRC_MODEL_DIR=$HOME/models
MODEL_NAME=Llama-2-7b-hf
SRC_MODEL=$SRC_MODEL_DIR/$MODEL_NAME

DIST_MODEL_ROOT=$HOME/models/base_finetuned_search

for EPOCHS in 3 5
do
    export WANDB_RUN_GROUP="Hyperparam Search (Test as Dev) - Epochs ${EPOCHS}"
    for LR in 1e-5 5e-6 1e-4
    do
	for BS in 16
	do
	    for WD in 0.0
	    do
		DIST_MODEL_FT=$MODEL_NAME@gsm8k@lr$LR@B$BS@wd$WD@ep$EPOCHS
		export WANDB_NAME=$DIST_MODEL_FT
		torchrun --nnodes 1 --nproc_per_node 4 examples/finetuning.py \
			 --dataset gsm8k_dataset \
			 --num_epochs $EPOCHS \
			 --lr $LR \
			 --batch_size_training $BS \
			 --weight_decay $WD \
			 --enable_fsdp \
			 --pure_bf16 \
			 --model_name $SRC_MODEL \
			 --dist_checkpoint_root_folder $DIST_MODEL_ROOT \
			 --dist_checkpoint_folder $DIST_MODEL_FT
		rm -r $DIST_MODEL_ROOT/*
	    done
	done
    done
done


# For reference, do not remove/change
# Suggested command for full-parameter finetuning
# torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --use_fast_kernels
