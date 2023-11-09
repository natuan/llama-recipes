#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l)+1))

source $HOME/src/natuan/llama-recipes/my_scripts/start_here.sh

MODEL_DIR=$HOME/models/llama
MODEL_NAME=Llama-2-7b-hf
SRC_MODEL=$MODEL_DIR/$MODEL_NAME

DIST_MODEL_ROOT=/network/tuan/models/llama/GSM8K/clip_softmax_V2

LR_SCHED=linear
WARM=0.1

GRAD_CLIP=1
GRAD_THRES=1.0

BS=16
GRAD_ACC=1

WD=0.0

for EPOCHS in 2
do
    for LR in 3e-5
    do
	    export WANDB_RUN_GROUP=" Training w/o Clipped Softmax - Epochs $EPOCHS, LR $LR, Batch $BS, GradAccum $GRAD_ACC"
	    ID=$RANDOM
	    DIST_MODEL_FT=$MODEL_NAME@gsm8k@lr$LR@B$BS@GrAcc$GRAD_ACC@W$WARM@ep$EPOCHS@GPUs$NPROC@ClipSM_OFF@ID$ID
	    export WANDB_NAME=$DIST_MODEL_FT
	    torchrun --nnodes 1 --nproc_per_node $NPROC examples/finetuning.py \
		     --dataset gsm8k_dataset \
		     --num_epochs $EPOCHS \
		     --use_custom_loss 0 \
		     --test_as_dev 1 \
		     --use_gradient_clipping $GRAD_CLIP \
		     --gradient_clipping_thresh $GRAD_THRES \
		     --weight_decay $WD \
		     --lr $LR \
		     --lr_scheduler $LR_SCHED \
		     --gradient_accumulation_steps $GRAD_ACC \
		     --warmup_ratio $WARM \
		     --batch_size_training $BS \
		     --enable_fsdp \
		     --pure_bf16 \
		     --model_name $SRC_MODEL \
		     --dist_checkpoint_root_folder $DIST_MODEL_ROOT \
		     --dist_checkpoint_folder $DIST_MODEL_FT

    done
done


# For reference, do not remove/change
# Suggested command for full-parameter finetuning
# torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --use_fast_kernels
