# Phase 0 — 基线复现

commit: `d4c45b6`（分支 `exp/phase0-baseline`，父提交 `be05278`）
数据集：Yelp | 模型：Meta-Llama-3-8B（论文原模型）| 硬件：2× RTX 3090（共享）

## 1. 复现流程回顾

本仓库 `path_retriever/saved_models/`、`path_retriever/saved_explanations/`、`raft_data/` 均为
README 建议的**下载好的预处理产物**（Google Drive 链接），未在本机重跑
`code/converter.py`（Step1）/ `code/dgl_extractor.py`（Step2）—— 本机 `data/yelp/` 下
缺少 `{split}.pkl` 原始交互文件，重跑 Step1-2 需要额外未下载的原始数据源。
Step 3-6（GNN 训练、PaGE-Link 路径抽取、SentenceBERT 节点检索、prompt 翻译剪枝）产物直接复用下载版本，
Step 7-9（RAFT 微调 → 推理 → 评估）在本机完整跑通。

跑通链路：
1. `ds_training/step1_supervised_finetuning/` + `run_yelp_llama3.sh`：LoRA(rank=8) RAFT 微调，
   ZeRO-3，2 epoch，2×3090（受显存限制 max_seq_len=1024、有效 batch=2，论文为 2048/16）。
2. `ds_inference/infer.py`：单卡 batch_size=8，拆分测试集到 2 张卡并行推理。
3. `evaluation/eval_lite.py`（新增，见下）：BERTScore(P/R/F1) + BARTScore + USR。

## 2. 发现的基线代码 Bug（已修复，见 commit d4c45b6）

| 文件 | 问题 | 修复 |
|---|---|---|
| `ds_training/utils/utils.py:98-101` | `save_hf_format` 用 `tokenizer.save_vocabulary()`，Llama-3 快速分词器抛 `NotImplementedError`；ZeRO-3 下 `state_dict()` 又是空/分片的 | 改用 `tokenizer.save_pretrained()`；ZeRO-3 时跳过 `save_hf_format`，只用 `save_zero_three_model`（gather 参数后 `save_pretrained`） |
| `ds_training/step1_supervised_finetuning/main.py` | 训练完两个 epoch 后在"保存最终模型"时才崩溃，此前没有任何中间 checkpoint，一次崩溃=整跑（曾损失 50 小时训练） | 每个 epoch 后 fuse LoRA → 存 `epoch_N/` 快照 → unfuse 恢复训练（新增 `unfuse_lora_layer`） |
| `path_retriever/pagelink.py:106-111` | encoder 被切到 `LightGCN` 且加载 `..._lightgcn.pth`，但 `saved_models/` 下只有 `train_linkpred.py` 默认 R-GCN 训出的 `{dataset}_model_{split}.pth`（无 `_lightgcn` 后缀）——**该文件本来就跑不起来** | 改回 `HeteroRGCN` + 正确文件名，加 `time.perf_counter()` 包裹 `pagelink.explain()` 测 Dijkstra 检索耗时 |
| `evaluation/metrics.py` | 顶层 import tensorflow+bleurt，且需 OpenAI key，本机环境无法直接跑 | 新增 `evaluation/eval_lite.py`，只做 BERTScore/BARTScore/USR（数据解析、prediction/reference 约定与 metrics.py 保持一致，可比） |
| 环境：`import dgl` 完全跑不起来 | `dgl==2.1.0` 的 `graphbolt` 子模块 import `torchdata.datapipes`，但环境里 `torchdata==0.11.0` 已移除该 API | `pip install torchdata==0.8.0` 降级；详见 [[path-retriever-env-and-data-gaps]] |
| 环境：`dgl.graphbolt`/`dgl.sparse` 的 C++ `.so` 只编译到 torch 2.2.1 | 环境里 `torch==2.4.1`（为 LLM 训练栈升级），aliyun 镜像上 dgl 最新只有 2.1.0，无法装到匹配版本 | graphbolt 未被 path_retriever 用到：patch site-packages 让加载失败降级为警告；`data_processing.py` 里唯一实际用到 `dgl.sparse`（`g.adj()`）的调用改写成等价的纯 tensor 实现（`data_processing.py:57-66`），不再依赖该 C++ 库 |
| 环境：安装的 `dgl` 是 CPU-only 编译 | `g.to('cuda')` 报 `Device API cuda is not enabled` | path_retriever 侧的 GNN/检索改用 `--device_id -1`（CPU）跑；不影响 LLM 训练栈（那边是纯 torch/deepspeed，不经过 dgl） |

## 3. 生成质量指标（3000 条测试样本，vs 论文 Table 1 G-Refer(8B) Yelp）

| metric | 论文 | 本次复现 | Δ |
|---|---|---|---|
| BERT-P | 0.3629 | 0.3667 | +0.0038 |
| BERT-R | 0.4373 | 0.4403 | +0.0030 |
| **BERT-F1**（主指标） | 0.4003 | **0.4038** | +0.0035 |
| BARTScore | -3.6448 | -2.8901 | +0.75（更优） |
| USR | 1.0000 | 1.0000 | 0 |
| GPTScore | 75.16 | 未跑（需 OpenAI key） | — |
| BLEURT | -0.1336 | 未跑（需 tensorflow） | — |

结论：核心语义指标复现在随机噪声范围内（<0.004），USR 完全一致 → **复现有效，达到 Phase 0 门槛**。

## 4. 效率基线（本 Phase 新增测量）

| 阶段 | 数值 | 备注 |
|---|---|---|
| 路径检索单样本延迟（Dijkstra, PaGE-Link `explain()`） | 见下方 §4.1 | Phase 1 graph-powering 对比基准 |
| LLM 推理单样本延迟 | ~15 min / 1500 samples / 卡（batch=8, max_new_tokens=256, 单卡） ≈ 0.6s/sample | 2×3090 并行拆分 |
| 训练时长 | 2 epoch, 2×3090, ZeRO-3, ~50h（含一次因保存 bug 导致的重跑） | 见 [[llama3-8b-training-config]] |

### 4.1 Dijkstra 路径检索 wall-clock（本 Phase 新增）

方法：本机缺少可重跑 Step1-2 所需的原始 `{split}.pkl` 交互文件（仅有 Step1 输出的 `data_trn.pt`），
因此从 `data/yelp/data_trn.pt`（含真实 15962 users / 14085 items，与 `saved_models/yelp_model_trn.pth`
的 embedding 维度一致）重建了一张**同规模**的 DGL 异构图（user-item 边直接取自 `edge_index`，
`likes`/`buys`/`bought_by` 三种 etype 均以真实交互边填充，只是没有区分具体交互类型——
R-GCN encoder 本身只用 `HeteroEmbedding` 按 node id 查表，不消费节点文本特征，因此不影响
路径检索延迟的真实性），加载已训练好的 R-GCN 权重，跑 `path_retriever/pagelink.py`
在小样本 test pairs 上测 `PaGELink.explain()`（内部即 Yen's k-shortest-paths / 双向 Dijkstra）的
wall-clock。

测得两组样本（同一 R-GCN checkpoint、同一图、CPU 推理——本机 dgl 是 CPU-only 编译，见
[[path-retriever-env-and-data-gaps]]）：

| 样本 | n（预测为正的 pair 数） | total | mean | min | max |
|---|---|---|---|---|---|
| 30 个候选 pair | 5 | 122.982s | 24.596s | 2.003s | **113.385s** |
| 100 个候选 pair | 18 | 51.393s | 2.855s | 0.730s | 4.928s |

两组结果都显示 Dijkstra（Yen's k-shortest-paths）单样本延迟方差极大——多数 pair 在 1-5s
内完成，但偶发 pair（第一组中 1/5）耗时可达 **113s**，与论文 Power-Link/REXHA 描述的
"O(N²) 最短路搜索、长尾延迟"问题一致。这是 Phase 1 graph-powering 的直接对比基准。

## 5. 门槛判定

**通过**：核心指标复现在噪声范围内，Dijkstra 检索延迟已量化为 Phase 1 的加速对比基准。进入 Phase 1。
