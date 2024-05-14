#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l)+1))

ROOT=$HOME/work/llama3.gsm8k/src/natuan/llama-recipes
source $ROOT/my_scripts/cnn_dailymail/start_here.sh

CLEARML_PROJECT="tuan-llama3-cnn_dailymail-dense_ft"

DATASET=cnn_dailymail_dataset

SRC_MODEL_NAME=Meta-Llama-3-8B
SRC_MODEL=$HOME/models02/llama3/$SRC_MODEL_NAME

DST_MODEL_ROOT=$HOME/models02/llama3/cnn_dailymail/llama-recipes/dense_finetuned/ongoing

LR_SCHED=cosine
WARM=0.1

GRAD_CLIP=0
GRAD_THRES=2.0

TRAIN_BS=4
VAL_BS=1

WD=0.0

GRAD_ACC=16
for EPOCHS in 1
do
   for LR in 8e-5
   do
	    ID=$RANDOM
	    DST_MODEL_FT=$LR_SCHED@lr$LR@B$BS@GrAcc$GRAD_ACC@GrClip$GRAD_CLIP@GrThr$GRAD_THRES@W$WARM@ep$EPOCHS@GPUs$NPROC@WD$WD@ID$ID
	    torchrun --nnodes 1 --nproc_per_node $NPROC recipes/finetuning/finetuning.py \
		    --save_model 1 \
		    --dataset $DATASET \
			--clearml_project $CLEARML_PROJECT \
			--clearml_log_every_steps 50 \
			--eval_every_steps 300 \
			--num_epochs $EPOCHS \
		    --gradient_clipping $GRAD_CLIP \
		    --gradient_clipping_threshold $GRAD_THRES \
		    --weight_decay $WD \
		    --lr $LR \
		    --lr_scheduler $LR_SCHED \
		    --gradient_accumulation_steps $GRAD_ACC \
		    --warmup_ratio $WARM \
		    --batch_size_training $TRAIN_BS \
			--val_batch_size $VAL_BS \
		    --enable_fsdp \
		    --pure_bf16 \
		    --model_name $SRC_MODEL \
		    --dist_checkpoint_root_folder $DST_MODEL_ROOT \
		    --dist_checkpoint_folder $DST_MODEL_FT
    done
done

#--eval_every_steps 100 \