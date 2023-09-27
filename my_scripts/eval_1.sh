#/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
ROOT=$HOME/src/EleutherAI/lm-evaluation-harness

MODEL_ORG_OR_DIR=$HOME/models
MODEL_NAME=Llama-2-7b-hf@gsm8k@lr1e-4@wd0.0@ep3@ID23165/hf

#MODEL_ORG_OR_DIR=mosaicml
#MODEL_NAME=mpt-7b

MODEL=$MODEL_ORG_OR_DIR/$MODEL_NAME

#MODEL=facebook/opt-1.3b

TASK=arc_challenge
TASK=gsm8k

SHOTS=0

BATCH=2

python $ROOT/main.py \
       --model hf-causal \
       --model_args pretrained=$MODEL \
       --tasks $TASK \
       --num_fewshot=$SHOTS \
       --batch_size=$BATCH \
       --write_out \
       --output_base_path $ROOT/my_outputs/$MODEL_NAME/$TASK \
       --device cuda:1
