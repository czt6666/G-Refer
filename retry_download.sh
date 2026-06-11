#!/bin/bash
# Keep retrying the Google Drive download until all files arrive (Drive is
# throttling intermittently from this network). Stops when download_data.py
# exits 0 (all files present) or after MAX attempts.
cd /root/workspace/python/G-Refer
MAX=60
for i in $(seq 1 $MAX); do
  echo "===== retry attempt $i/$MAX $(date) ====="
  /opt/miniconda/bin/python -u download_data.py && { echo "ALL DONE"; exit 0; }
  echo "--- progress: raft=$(find raft_data -type f 2>/dev/null|wc -l)/9 expl=$(ls path_retriever/saved_explanations 2>/dev/null|wc -l)/12 models=$(ls path_retriever/saved_models 2>/dev/null|wc -l)/6 data=$(find data -type f 2>/dev/null|wc -l)/24 ---"
  sleep 30
done
echo "GAVE UP after $MAX attempts"
exit 1
