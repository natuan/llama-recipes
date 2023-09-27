#/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
ROOT=$HOME/src/EleutherAI/lm-evaluation-harness

MODEL_ORG_OR_DIR=$HOME/models/base_finetuned
MODEL_NAME=Llama-2-7b-hf@gsm8k@lr1e-5@B4@wd0.0@ep8/hf

MODEL_ORG_OR_DIR=$HOME/models
MODEL_NAME=Llama-2-7b-hf

MODEL=$MODEL_ORG_OR_DIR/$MODEL_NAME

#MODEL=facebook/opt-1.3b

TASK=arc_challenge
TASK=gsm8k

SHOTS=8

BATCH=16

python $ROOT/main.py \
       --model hf-causal-experimental \
       --model_args pretrained=$MODEL,use_accelerate=True,dtype=bfloat16 \
       --tasks $TASK \
       --num_fewshot=$SHOTS \
       --batch_size=$BATCH \
       --write_out \
       --output_base_path $HOME/models/test_accuracy_$MODEL_$TASK \
       --device cuda
