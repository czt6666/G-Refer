#!/bin/bash
# G-Refer RAFT (Step 7) — Phase 4 GraphLoRA smoke test.
#
# Same as run_yelp_subgraph_smoke.sh (Phase 3 soft-prompt) plus --use_graph_lora,
# which additionally gates every LoRA layer's low-rank update per-example by a
# projection of the same subgraph embedding (see ds_training/utils/module/lora.py
# GraphLoRALinear/GraphLoRAGate).
OUTPUT=$1
ZERO_STAGE=$2

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/opt/miniconda/envs/g-refer/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/workspace/python/G-Refer/models/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0
SUBGRAPH_EMBEDS=/root/workspace/python/G-Refer/subgraph_retriever/embeds/yelp_trn_subgraph_embeds.pt

if [ "$OUTPUT" == "" ]; then
    OUTPUT=../../ckpts/yelp_grefer_tinyllama_graphlora
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed --include localhost:1 --master_port=29503 main.py  \
   --data_path  local/jsonfile--yelp \
   --data_split 10,0,0 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 256 \
   --learning_rate 2e-5  \
   --weight_decay 0. \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --only_optimize_lora \
   --lora_dim 8 \
   --lora_module_name "layers." \
   --num_warmup_steps 10 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --subgraph_embed_path $SUBGRAPH_EMBEDS \
   --subgraph_dim 256 \
   --use_graph_lora \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
