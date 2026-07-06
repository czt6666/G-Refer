#!/bin/bash
# Phase 3 fair A/B, baseline arm: TinyLlama-1.1B, LoRA, NO soft-prompt.
# Trained on raft_data/yelp_subgraph5k/ (4300 examples, exactly the subset
# that has real subgraph embeddings -- same data as the soft-prompt arm, only
# difference is --subgraph_embed_path being absent).
OUTPUT=$1
ZERO_STAGE=$2

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/opt/miniconda/envs/g-refer/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/workspace/python/G-Refer/models/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0

if [ "$OUTPUT" == "" ]; then
    OUTPUT=../../ckpts/yelp_ab_baseline
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed --include localhost:2 --master_port=29504 main.py  \
   --data_path  local/jsonfile--yelp_subgraph5k \
   --data_split 10,0,0 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 256 \
   --learning_rate 2e-5  \
   --weight_decay 0. \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --only_optimize_lora \
   --lora_dim 8 \
   --lora_module_name "layers." \
   --num_warmup_steps 50 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
