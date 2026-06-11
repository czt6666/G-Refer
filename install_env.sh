#!/bin/bash
# Staged environment install for G-Refer (conda env: g-refer).
set -e
source /opt/miniconda/etc/profile.d/conda.sh
conda activate g-refer

PIP="pip install --no-input"

echo "=== [1/3] torch (CUDA build from default mirror, good for RTX 3090) ==="
# pytorch.org is unreachable from here; the configured aliyun PyPI mirror serves
# the CUDA-bundled linux wheels. torch 2.4.x keeps dgl/deepspeed compatibility.
$PIP torch==2.4.1 torchvision==0.19.1

echo "=== [2/4] core deps ==="
$PIP \
  "transformers>=4.31.0,!=4.33.2" \
  "deepspeed>=0.9.0" \
  "datasets>=2.8.0" \
  "sentencepiece>=0.1.97" \
  "accelerate>=0.15.0" \
  "protobuf==3.20.3" \
  torch_geometric \
  scikit-learn tqdm pandas numpy scipy tensorboard openai matplotlib networkx pytz pyyaml evaluate

echo "=== [3/3] dgl (for path_retriever steps 1-6; optional for RAFT) ==="
$PIP dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html || echo "WARN: dgl install failed (only needed for graph-retrieval steps 1-6)"

# NOTE: eval-only extras (bleurt, bart_score) are installed separately when Step 9 is reached.

echo "=== DONE ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
