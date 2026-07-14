#!/usr/bin/env bash
# 最简单：先 power，再 dijkstra。每步打印开始/结束时间，中间文件全留下。
# GPU: 0,1,2,3
#
# 用法：
#   cd /root/workspace/python/G-Refer
#   export MODEL_PATH=ckpts/yelp_grefer_llama3
#   bash experiments/phase1/eval_test3000_ab.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT=convert_files/phase1_test3000
MODEL_PATH="${MODEL_PATH:-ckpts/yelp_grefer_llama3}"
mkdir -p "$OUT" "$OUT/power" "$OUT/dijkstra" "$OUT/logs"

log() { echo "[$(date '+%F %T')] $*"; }

########################################
# 1) POWER：计时检索
########################################
log "==== POWER 检索 START ===="
cd path_retriever
python phase1_quality_compare.py \
  --method power \
  --n_samples 3000 \
  --output_dir "../$OUT" \
  --resume \
  2>&1 | tee "../$OUT/logs/1_power_retrieve.log"
cd ..
log "==== POWER 检索 END ===="

cp "$OUT/test_power.json" "$OUT/power/test.json"

########################################
# 2) POWER：推理（GPU 0123）
########################################
log "==== POWER 推理 START ===="
cd ds_inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer.py \
  --model_path "../$MODEL_PATH" \
  --streategy Parallel \
  --batch_size 8 \
  --save_dir "../$OUT/power" \
  2>&1 | tee "../$OUT/logs/2_power_infer.log"
cd ..
log "==== POWER 推理 END ===="

########################################
# 3) POWER：打分
########################################
log "==== POWER 评测 START ===="
python evaluation/eval_lite.py \
  --gen_path "$OUT/power/gen_datas.jsonl" \
  --dataset power_test3000 \
  2>&1 | tee "$OUT/logs/3_power_eval.log"
log "==== POWER 评测 END ===="

########################################
# 4) BASELINE(dijkstra)：计时检索
########################################
log "==== DIJKSTRA 检索 START ===="
cd path_retriever
python phase1_quality_compare.py \
  --method dijkstra \
  --n_samples 3000 \
  --output_dir "../$OUT" \
  --resume \
  2>&1 | tee "../$OUT/logs/4_dijkstra_retrieve.log"
cd ..
log "==== DIJKSTRA 检索 END ===="

cp "$OUT/test_dijkstra.json" "$OUT/dijkstra/test.json"

########################################
# 5) BASELINE：推理（GPU 0123）
########################################
log "==== DIJKSTRA 推理 START ===="
cd ds_inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer.py \
  --model_path "../$MODEL_PATH" \
  --streategy Parallel \
  --batch_size 8 \
  --save_dir "../$OUT/dijkstra" \
  2>&1 | tee "../$OUT/logs/5_dijkstra_infer.log"
cd ..
log "==== DIJKSTRA 推理 END ===="

########################################
# 6) BASELINE：打分
########################################
log "==== DIJKSTRA 评测 START ===="
python evaluation/eval_lite.py \
  --gen_path "$OUT/dijkstra/gen_datas.jsonl" \
  --dataset dijkstra_test3000 \
  2>&1 | tee "$OUT/logs/6_dijkstra_eval.log"
log "==== DIJKSTRA 评测 END ===="

log "全部完成。结果在 $OUT/"
ls -la "$OUT"/timing_*_summary.json "$OUT"/power/gen_datas.jsonl "$OUT"/dijkstra/gen_datas.jsonl "$OUT"/logs/
