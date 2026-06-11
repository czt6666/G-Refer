#!/bin/bash
# G-Refer RAFT (Step 7) — small-model verification run.
#
# The paper uses Meta-Llama-3-8B, which does NOT fit the ~2.8GB free per GPU on
# this shared host. This script swaps in TinyLlama-1.1B (same Llama architecture,
# SentencePiece tokenizer so the code's LlamaTokenizer works) to verify the RAFT
# pipeline launches, loads raft_data, and trains/logs correctly.
#
# Deviations from run_yelp.sh (all to fit memory; noted for the record):
#   - model: TinyLlama-1.1B instead of Llama-3-8B
#   - GPUs: localhost:0,1,2,3 (host has 4) instead of 0..7
#   - max_seq_len 1024 (was 2048), num_train_epochs 1 (was 2)
#   - ZeRO-3 + CPU offload (--offload) enabled
#   - --data_path local/jsonfile--yelp (upstream run_yelp.sh dropped the "yelp")
# DeepSpeed Team / Microsoft (Apache-2.0) base.
OUTPUT=$1
ZERO_STAGE=$2

# CUDA toolkit for DeepSpeed JIT op compilation (CPUAdam / fused kernels).
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# GPUs are shared with other tenants; reduce fragmentation so tiny allocations
# don't fail when only a few MB are free.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/workspace/python/G-Refer/models/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0

if [ "$OUTPUT" == "" ]; then
    OUTPUT=../../ckpts/yelp_grefer_tinyllama
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2 --master_port=29501 main.py  \
   --data_path  local/jsonfile--yelp \
   --data_split 10,0,0 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 256 \
   --learning_rate 2e-5  \
   --weight_decay 0. \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --only_optimize_lora \
   --lora_dim 8 \
   --lora_module_name "layers." \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --offload \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
