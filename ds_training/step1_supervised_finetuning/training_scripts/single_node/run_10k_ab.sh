#!/bin/bash
# Phase 6: Power-Link + GraphLoRA (combined) vs baseline, at 10k-pair scale,
# across all 3 datasets (yelp/amazon/google). Both arms train on the exact
# same (uid,iid) pairs (raft_data/<dataset>_10k/, 8500/500/1000 train/eval/
# test -- the pairs with real subgraph embeddings) and the same
# hyperparameters; only two things differ in the treatment arm:
#   1. prompts come from raft_data/<dataset>_10k_powerlink/ (path segment
#      regenerated with Power-Link instead of the original real Dijkstra
#      text)
#   2. --use_graph_lora --subgraph_embed_path is set (GraphLoRA gate,
#      requires the subgraph soft-prompt mechanism as in Phase 3/4)
#
# max_seq_len=2048 (not 256, unlike Phase 3/4's smaller yelp_subgraph5k runs):
# measured token lengths of prompt+chosen at this 10k scale run up to 1815
# tokens (amazon baseline), with p50 already 580-1050 across datasets/variants.
# tokenizer(..., max_length=256, truncation=True) truncates from the right,
# so at 256 the ~40-token "chosen" explanation -- appended after the prompt --
# was being cut off entirely in 85-96% of training examples, corrupting the
# training signal (first attempt at this experiment produced catastrophic
# quality collapse, e.g. BERT-F1 0.14, before this was caught and fixed).
#
# Usage:
#   bash run_10k_ab.sh <dataset:yelp|amazon|google> <variant:baseline|graphlora> <gpu_id> <port> [output_dir]
DATASET=$1
VARIANT=$2
GPU=$3
PORT=$4
OUTPUT=$5

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/opt/miniconda/envs/g-refer/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/workspace/python/G-Refer/models/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0
SUBGRAPH_EMBEDS=/root/workspace/python/G-Refer/subgraph_retriever/embeds/${DATASET}_trn_subgraph_embeds.pt

if [ "$OUTPUT" == "" ]; then
    OUTPUT=../../ckpts/${DATASET}_10k_${VARIANT}
fi
mkdir -p $OUTPUT

if [ "$VARIANT" == "baseline" ]; then
    DATA_PATH="local/jsonfile--${DATASET}_10k"
    EXTRA_ARGS=""
elif [ "$VARIANT" == "graphlora" ]; then
    DATA_PATH="local/jsonfile--${DATASET}_10k_powerlink"
    EXTRA_ARGS="--subgraph_embed_path $SUBGRAPH_EMBEDS --subgraph_dim 256 --use_graph_lora"
else
    echo "unknown variant: $VARIANT (expected baseline|graphlora)"; exit 1
fi

deepspeed --include localhost:$GPU --master_port=$PORT main.py  \
   --data_path $DATA_PATH \
   --data_split 10,0,0 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 2048 \
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
   --zero_stage 2 \
   $EXTRA_ARGS \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
