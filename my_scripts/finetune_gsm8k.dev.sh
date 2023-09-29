#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
NPROC=1

source $HOME/src/natuan/llama-recipes/my_scripts/start_here.sh

SRC_MODEL_DIR=$HOME/models
MODEL_NAME=Llama-2-7b-hf
SRC_MODEL=$SRC_MODEL_DIR/$MODEL_NAME

DIST_MODEL_ROOT=$HOME/models/base_finetuned_search

LR_SCHED=linear
WARM=0.1

for EPOCHS in 1
do
    for LR in 1e-5
    do
	for BS in 8
	do
	    export WANDB_RUN_GROUP="Hyperparam Search (Dev from Train) - Epochs $EPOCHS, LR $LS, Batch $BS"
	    for SEED in 53 241
	    do
		DIST_MODEL_FT=$MODEL_NAME@gsm8k@lr$LR@B$BS@W$WARM@ep$EPOCHS@GPUs$NPROC@SEED$SEED
		export WANDB_NAME=$DIST_MODEL_FT
		torchrun --nnodes 1 --nproc_per_node $NPROC examples/finetuning.py \
			 --dataset gsm8k_dataset \
			 --num_epochs $EPOCHS \
			 --test_as_dev 0 \
			 --dev_set_seed $SEED \
			 --lr $LR \
			 --lr_scheduler $LR_SCHED \
			 --warmup_ratio $WARM \
			 --batch_size_training $BS \
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
