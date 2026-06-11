#!/bin/bash
# G-Refer RAFT (Step 7) — PAPER model (Meta-Llama-3-8B), ready-to-run.
#
# Precondition: the GPUs listed in --include must be FREE (this shared host
# normally has other tenants holding ~21GB/card; free them first).
#
# Default = 4-GPU ZeRO-3, no CPU offload (fast). For a single dedicated GPU,
# pass GPUS=0 OFFLOAD=1 (see bottom) — slower, needs param offload to fit 24GB.
OUTPUT=$1
ZERO_STAGE=$2

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

MODEL_PATH=/root/workspace/python/G-Refer/models/LLM-Research/Meta-Llama-3-8B

# --- knobs (override via env) ---
GPUS=${GPUS:-0,1,2,3}        # set to e.g. "0" for single-GPU
OFFLOAD_FLAG=${OFFLOAD:+--offload}   # OFFLOAD=1 -> CPU offload (needed on 1 GPU)
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}     # paper uses 2048; 1024 is safer on 3090

if [ "$OUTPUT" == "" ]; then
    OUTPUT=../../ckpts/yelp_grefer_llama3
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --include localhost:$GPUS --master_port=29501 main.py  \
   --data_path  local/jsonfile--yelp \
   --data_split 10,0,0 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len $MAX_SEQ_LEN \
   --learning_rate 2e-5  \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --only_optimize_lora \
   --lora_dim 8 \
   --lora_module_name "layers." \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   $OFFLOAD_FLAG \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

# Single-GPU example:
#   GPUS=0 OFFLOAD=1 MAX_SEQ_LEN=1024 bash run_yelp_llama3.sh
