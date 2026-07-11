#!/bin/bash
# Full-scale Amazon-books run: baseline / power-link / graphlora, three methods,
# on the FULL real dataset (raft_data/amazon_full/: 94841 train / 1000 eval /
# 3000 test -- carved out of the paper's real raft_data/amazon/train.json,
# not the 10k subset used in the earlier phase6 experiment).
#
# variant=baseline:   original real prompts, vanilla LoRA. No dependency, runs
#                     immediately.
# variant=graphlora:  SAME original real prompts (not power-link-modified) +
#                     subgraph soft-prompt + GraphLoRA gate, using the existing
#                     subgraph_retriever/embeds/amazon_trn_subgraph_embeds.pt
#                     (only covers 10000/94841 pairs; the rest fall back to a
#                     zero vector per data_utils.py's existing missing-embedding
#                     path). This isolates the GraphLoRA fine-tuning mechanism
#                     alone (matches Phase 4's controlled ablation design),
#                     deliberately NOT bundled with Power-Link this time.
# variant=powerlink:  prompts from raft_data/amazon_full_powerlink/ (path
#                     segment regenerated with Power-Link on the synthetic
#                     graph, full data -- must be generated first via
#                     path_retriever/generate_powerlink_prompts.py before this
#                     variant can run), vanilla LoRA (no graphlora gate). This
#                     isolates the retrieval-method change alone.
#
# max_seq_len=2304 (not 2048): scanned the true max prompt+chosen token length
# across all 94841 real training rows (not just a 300-row sample) and found
# 2265 tokens at row 71248 -- 2048 would still truncate that outlier's chosen
# explanation. 2304 gives a safety margin above the confirmed max with no
# truncation anywhere in the dataset.
#
# Usage:
#   bash run_amazon_full_ab.sh <variant:baseline|graphlora|powerlink> <gpu_id> <port> [output_dir]
VARIANT=$1
GPU=$2
PORT=$3
OUTPUT=$4

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/opt/miniconda/envs/g-refer/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/root/workspace/python/G-Refer/models/AI-ModelScope/TinyLlama-1___1B-Chat-v1___0
SUBGRAPH_EMBEDS=/root/workspace/python/G-Refer/subgraph_retriever/embeds/amazon_trn_subgraph_embeds.pt

if [ "$OUTPUT" == "" ]; then
    OUTPUT=../../ckpts/amazon_full_${VARIANT}
fi
mkdir -p $OUTPUT

if [ "$VARIANT" == "baseline" ]; then
    DATA_PATH="local/jsonfile--amazon_full"
    EXTRA_ARGS=""
elif [ "$VARIANT" == "graphlora" ]; then
    DATA_PATH="local/jsonfile--amazon_full"
    EXTRA_ARGS="--subgraph_embed_path $SUBGRAPH_EMBEDS --subgraph_dim 256 --use_graph_lora"
elif [ "$VARIANT" == "powerlink" ]; then
    DATA_PATH="local/jsonfile--amazon_full_powerlink"
    EXTRA_ARGS=""
else
    echo "unknown variant: $VARIANT (expected baseline|graphlora|powerlink)"; exit 1
fi

deepspeed --include localhost:$GPU --master_port=$PORT main.py  \
   --data_path $DATA_PATH \
   --data_split 10,0,0 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 2304 \
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
