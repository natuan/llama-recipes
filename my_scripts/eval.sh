#/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ROOT=$HOME/src/EleutherAI/lm-evaluation-harness

MODEL_ORG_OR_DIR=$HOME/models/llama/base_finetuned
MODEL_NAME=Llama-2-7b-hf@gsm8k@lr3e-5@B16@GrAcc1@W0.1@ep2@GPUs8@CL1@RLW0.3@WD0.0/hf

# MODEL_ORG_OR_DIR=$HOME/models
# MODEL_NAME=Llama-2-7b-hf

MODEL=$MODEL_ORG_OR_DIR/$MODEL_NAME

TASK=gsm8k

SHOTS=0

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
