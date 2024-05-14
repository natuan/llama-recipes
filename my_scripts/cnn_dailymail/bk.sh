#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l)+1))

ROOT=$HOME/work/llama2.cnn_dailymail.llama-recipes/src/neuralmagic/llama-recipes
source $ROOT/my_scripts/start_here.sh

SRC_MODEL_NAME=Llama-2-7b-hf
SRC_MODEL=$HOME/models/llama2/$SRC_MODEL_NAME

#DST_MODEL_ROOT=$HOME/models/llama2/cnn_dailymail/llama-recipes/dense_finetuned
DST_MODEL_ROOT=$HOME/models02/llama2/train_benchmark/cnn_daily/llama-recipes/dense_finetuned

LR_SCHED=linear
WARM=0.1

GRAD_THRES=1.0

BS=16
GRAD_CLIP=0

WD=0.0

EPOCHS=1

for GRAD_ACC in 4
do
   for LR in 8e-5
   do
	    export WANDB_RUN_GROUP="Epochs $EPOCHS, LR $LR, Batch $BS, GradAccum $GRAD_ACC"

	    ID=$RANDOM
		SEED=$ID
	    DST_MODEL_FT=$SRC_MODEL_NAME@$LR_SCHED@lr$LR@B$BS@GrAcc$GRAD_ACC@W$WARM@ep$EPOCHS@GPUs$NPROC@WD$WD@ID$ID
	    export WANDB_NAME=$DST_MODEL_FT
	    torchrun --nnodes 1 --nproc_per_node $NPROC examples/finetuning_sparse.py \
		     --save_model 1 \
			 --seed $SEED \
		     --wandb_log_every_optimizer_steps 50 \
		     --dataset cnn_dailymail_dataset \
		     --num_epochs $EPOCHS \
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
		     --eval_every_steps 300 \
		     --model_name $SRC_MODEL \
		     --dist_checkpoint_root_folder $DST_MODEL_ROOT \
		     --dist_checkpoint_folder $DST_MODEL_FT
    done
done
