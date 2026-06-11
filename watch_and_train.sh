#!/bin/bash
# Auto-launch the G-Refer paper-model RAFT run (Meta-Llama-3-8B) as soon as
# enough GPU memory is genuinely free. The shared host normally has other
# tenants holding ~21GB/card (only ~2.9GB free) which is too little for an 8B
# model. This watcher polls and fires the run the moment cards are freed.
#
# Policy:
#   - GPU counts as "free" if it has >= FREE_MIN MiB free (default 20000).
#   - >=2 free GPUs -> 2-card ZeRO-2, no offload (fastest, recommended).
#   - exactly 1 free GPU -> single-card ZeRO-3 + CPU offload.
#   - gives up after MAX_WAIT_MIN minutes.
cd /root/workspace/python/G-Refer/ds_training/step1_supervised_finetuning
source /opt/miniconda/etc/profile.d/conda.sh
conda activate g-refer

FREE_MIN=${FREE_MIN:-20000}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-240}
SCRIPT=training_scripts/single_node/run_yelp_llama3.sh
iters=$(( MAX_WAIT_MIN * 2 ))   # 30s per poll

echo "[watch] waiting for >=1 GPU with >= ${FREE_MIN} MiB free (max ${MAX_WAIT_MIN} min)..."
for i in $(seq 1 $iters); do
  # list of gpu indices that have >= FREE_MIN free
  mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                      | awk -v m=$FREE_MIN -F', ' '$2+0 >= m {print $1}')
  n=${#FREE[@]}
  if [ "$n" -ge 2 ]; then
    G="${FREE[0]},${FREE[1]}"
    echo "[watch] $(date) -> $n free GPUs ($G). Launching 2-card ZeRO-2 run."
    GPUS=$G MAX_SEQ_LEN=1024 bash $SCRIPT "" 2
    echo "[watch] training process exited (rc=$?). See ckpts/yelp_grefer_llama3/training.log"
    exit 0
  elif [ "$n" -ge 1 ]; then
    G="${FREE[0]}"
    echo "[watch] $(date) -> 1 free GPU ($G). Launching single-card ZeRO-3 + offload."
    GPUS=$G OFFLOAD=1 MAX_SEQ_LEN=1024 bash $SCRIPT "" 3
    echo "[watch] training process exited (rc=$?). See ckpts/yelp_grefer_llama3/training.log"
    exit 0
  fi
  sleep 30
done
echo "[watch] gave up after ${MAX_WAIT_MIN} min; no GPU freed."
exit 1
