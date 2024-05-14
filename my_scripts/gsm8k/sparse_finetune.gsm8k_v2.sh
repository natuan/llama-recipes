#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l)+1))

ROOT=$HOME/work/llama3.gsm8k/src/natuan/llama-recipes
source $ROOT/my_scripts/gsm8k/start_here.sh

CLEARML_PROJECT="tuan-llama3-gsm8k_v2-sparse_ft--sp50_mask24"

DATASET=gsm8k_v2_dataset

TEACHER_MODEL_DIR=$HOME/models02/llama3/gsm8k_v2/llama-recipes/potential/dense_finetuned
TEACHER_ID=26639
TEACHER_NAME=linear@lr3e-5@B@GrAcc1@GrClip0@GrThr2.0@W0.1@ep4@GPUs8@WD0.0@ID$TEACHER_ID
TEACHER=$TEACHER_MODEL_DIR/$TEACHER_NAME/hf

#SRC_ID=14891
#SRC_MODEL_DIR=$HOME/models02/llama3/gsm8k_v2/sparsegpt/uniform
#SRC_MODEL_NAME=Src26639@uni.sp50.unstr@Sp50@N2048@ID$SRC_ID

SRC_ID=11119
SRC_MODEL_DIR=$HOME/models02/llama3/gsm8k_v2/sparsegpt/uniform
SRC_MODEL_NAME=Src26639@uni.sp50.mask24@Sp50@N2048@ID${SRC_ID}
SRC_MODEL=$SRC_MODEL_DIR/$SRC_MODEL_NAME

DST_MODEL_ROOT=$HOME/models02/llama3/gsm8k_v2/llama-recipes/sparse_finetuned/ongoing

LR_SCHED=linear
WARM=0.1

GRAD_CLIP=1

TRAIN_BS=2
VAL_BS=1

WD=0.0

EPOCHS=8
for GRAD_ACC in 16 32; do
for GRAD_THRES in 1.0 2.0 4.0 8.0
do
   for LR in 3e-5 5e-5 8e-5
   do
	    ID=$RANDOM
	    DST_MODEL_FT=Src${SRC_ID}@$LR_SCHED@lr$LR@B${TRAIN_BS}@GrAcc$GRAD_ACC@GradCL${GRAD_CLIP}@GradThr${GRAD_THRES}@ep$EPOCHS@GPUs$NPROC@ID$ID
	    torchrun --nnodes 1 --nproc_per_node $NPROC recipes/finetuning/finetuning_sparse.py \
		    --sparse \
		    --save_model 1 \
		    --dataset $DATASET \
			--clearml_project $CLEARML_PROJECT \
			--clearml_log_every_steps 2 \
			--eval_every_steps 5 \
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
		    --dist_checkpoint_folder $DST_MODEL_FT \
			--kd_config.enabled \
			--kd_config.output \
			--kd_config.layerwise \
			--kd_config.teacher_model_path $TEACHER \
			--kd_config.hardness_ce 1.0 \
			--kd_config.hardness_kd_output 0.0 \
			--kd_config.hardness_kd_layerwise 1.0
    done
done
done
#--eval_every_steps 100 \

