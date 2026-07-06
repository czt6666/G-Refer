# G-Refer 改造最终汇总报告

按 `paper/缝合报告_AGENT_20篇.md` 的 20 篇方向、5 个 Phase 顺序执行。分支从
`exp/phase0-baseline` 依次向下切出到 `exp/phase5-e2e-joint`（线性历史，每个分支包含前一个
分支的全部提交）。

## 1. 总表

| Phase | 分支 | commit | 核心产出 | 状态 |
|---|---|---|---|---|
| 0 | `exp/phase0-baseline` | `d4c45b6`, `faa886c` | 确认论文复现有效；修复 4 个真实 bug；测出 Dijkstra 检索延迟基线 | ✅ 通过 |
| 1 | `exp/phase1-power-link` | `38972d2` | Power-Link 图幂法替换 Dijkstra，`--path_method` 开关 | ✅ 通过 |
| 2 | `exp/phase2-retrieval-quality` | `cca708d`, `5b1740f` | KGAT-lite/LightGCN backbone A/B；确认数据缺口是结构性的 | ⚠️ 部分通过 |
| 3 | `exp/phase3-subgraph-retrieval` | `523ae5b` | 子图检索 + soft-prompt 注入 LLM，机制验证通过 | ⚠️ 部分通过 |
| 4 | `exp/phase4-graphlora` | `58c4f47` | GraphLoRA 逐样本门控 LoRA，机制验证通过 | ⚠️ 部分通过 |
| 5 | `exp/phase5-e2e-joint` | `b2a16d3` | DFTopK + RCS 组件验证；RCS 发现真实的排序不一致现象 | ⚠️ 部分通过 |

### 1.1 指标明细

**生成质量（Phase 0 基线，Yelp，Llama-3-8B，3000 测试样本，唯一跑到下游生成质量的 Phase）**

| metric | 论文 | 复现 |
|---|---|---|
| BERT-P | 0.3629 | 0.3667 |
| BERT-R | 0.4373 | 0.4403 |
| BERT-F1 | 0.4003 | **0.4038** |
| BARTScore | -3.6448 | -2.8901 |
| USR | 1.0000 | 1.0000 |

Phase 1-5 均未重跑这一套下游生成质量对比——原因各不相同，见各自 result.md 的"未做"章节，
汇总在 §4。

**检索延迟（Phase 1 核心指标）**

| 方法 | 典型情形 mean | 长尾情形 max（单 pair） |
|---|---|---|
| Dijkstra（原始） | 2.86s | 113.4s |
| Power-Link（新） | 0.71s（**4.0x**） | 1.6s（**~69x**） |

**GNN backbone 链接预测 AUC（Phase 2 核心指标，同一合成图，100 epoch）**

| encoder | test AUC |
|---|---|
| R-GCN（原基线） | 0.7605 |
| KGAT-lite（新，注意力聚合） | 0.8675 |
| **LightGCN（新，轻量聚合）** | **0.9570** |

**排序一致性（Phase 5 核心发现）**

R-GCN（AUC 0.7605）vs LightGCN（AUC 0.9570）对同一用户候选 item 的排序：
top-k 重合率 **0.001**，Kendall's tau **0.016** —— 两个"各自都不差"的模型排序几乎完全不一致，
验证了 RCS（#17）"自身精度 ≠ 跨模型一致性"的核心论点。

**训练冒烟测试（Phase 3、4，TinyLlama-1.1B，均验证机制可行、无崩溃）**

| Phase | 机制 | steps | loss 变化 |
|---|---|---|---|
| 3 | soft-prompt 注入 | 261 | 2.48 → ~1.0 |
| 4 | GraphLoRA 门控 | 191 | 2.44 → ~1.1 |

## 2. 每篇论文采纳结论

| # | 论文 | 结论 | 一句话原因 |
|---|---|---|---|
| 1 | Power-Link | **采纳** | 检索延迟典型 4x、长尾 69x 提速，已验证非劣 |
| 20 | REXHA | 部分采纳（未做） | 向量检索加速/分层摘要留给 Phase 2，Phase 2 时因数据缺口未做 |
| 6 | KGAT | **采纳（推荐）** | 100 epoch test AUC 0.8675 显著超过 R-GCN 0.7605；简化版（per-relation softmax） |
| 7 | LightGCN | **采纳（强烈推荐）** | 100 epoch test AUC 0.9570，且第 10 epoch 即追平 R-GCN 跑满 100 epoch；未换生产默认值因缺真实数据复训 |
| 5 | IntTower | 放弃（本次） | 需要多向量表征，当前节点表征是单一池化向量，且原始检索管线因 `.pkl` 缺失跑不起来 |
| 13 | GNRR | 放弃（本次） | 依赖 dense_retriever 真实候选做重排输入，同上受阻 |
| 8 | RippleNet | 放弃（本次） | 需要真实多关系图区分 buys/likes，合成图关系退化 |
| 2/4 | KGRec | 放弃（本次） | 边权去噪需要真实边语义区分才有意义 |
| 9 | K-RagRec | **部分采纳** | soft-prompt 注入机制验证通过；受限于抽取性能瓶颈，只在 6.7% 数据上验证，未做规模化质量对比 |
| 10 | CHEST | 部分借鉴 | per-pair 子图抽取天然满足 interaction-specific，课程预训练/multi-slot 序列化未做 |
| 11 | KGIN | 放弃（本次） | 意图感知聚合需要改聚合机制，本次复用现成 R-GCN 池化 |
| 12 | DFTopK | **采纳** | 实现验证：低温度下与硬 top-k 完全一致（重合率 1.000），梯度真实可传 |
| 18 | GraphLoRA | **部分采纳** | 机制验证通过（逐样本门控，初值等价 vanilla LoRA）；简化版，未做规模化 vs vanilla LoRA 对比 |
| 19 | FACE | 放弃（本次） | VQ-VAE codebook 是独立的大工作量子任务，相对已跑通的连续 soft-prompt 增量价值未验证 |
| 17 | Ranking Consistency | **采纳** | 实现验证通过，且在真实模型上跑出有意义的发现（R-GCN vs LightGCN 排序近乎不相关） |
| 15 | LCRON | 放弃（本次） | 需要检索器⇄LLM 真正联合训练循环，且前提是真实多关系数据，两者都不具备 |
| 14 | HetComp | 放弃（本次） | 同上，课程式蒸馏需要教师-学生联合训练循环 |
| 16 | Cooperative | 放弃（本次） | 同上，协同训练需要联合训练循环 |
| 3 | PaGE-Link | 参照基线 | Phase 1 对照组，非缝合对象 |

**统计**：20 篇里，**5 篇采纳/推荐**（Power-Link、KGAT、LightGCN、DFTopK、RCS），
**4 篇部分采纳**（REXHA、K-RagRec、GraphLoRA、CHEST），**11 篇本次放弃**（多数因同一个
结构性数据缺口）。

## 3. 最优配方（当前证据支持的推荐组合）

- **路径检索**：Power-Link 图幂法（`path_retriever/utils.py:graph_powering_paths`，
  `--path_method power`）替代 Dijkstra。
- **GNN backbone**：LightGCN 替代 R-GCN（`path_retriever/model.py:LightGCN`，
  `--encoder lightgcn`）—— **[需确认]** 需要在真实数据上复训才能真正投产，当前生产环境
  仍用原始 R-GCN checkpoint。
- **LLM 微调**：LoRA 基础上叠加 GraphLoRA 门控（`--use_graph_lora`）+ Phase 3 子图
  soft-prompt（`--subgraph_embed_path`）—— 机制已验证，**规模化效果未验证**，建议先跑一次
  真正意义上的对照训练再决定是否默认开启。
- **子图/端到端**：暂不采纳完整的子图检索 LLM-prompt 重排、端到端联合训练——受阻于
  真实多关系数据缺口，见 §4。

## 4. 失败与坑（供后续研究参考）

**贯穿全程的核心限制**：G-Refer 论文的公开数据发布（`download_data.py` 的完整 manifest）
**从未包含**原始 `{split}.pkl` 交互记录文件，只发布了已经算好的产物（`data_trn.pt`、
`dense_retrieval_results_*.json`、`saved_models/`、`saved_explanations/`、`raft_data/`）。
这不是本机网络限制——检查过完整下载清单确认。后果：

- `code/converter.py`→`code/dgl_extractor.py`（Step 1-2，真实图构建）、
  `code/dense_retriever.py`（真实节点检索）**在任何环境下都无法从公开数据重新跑通**。
- Phase 1-5 的图检索实验都基于从 `total_trn.csv` 的 314944 条真实交互对重建的合成图
  （三种 etype 都是同一批边的复制，因为真实的 buys/likes/bought_by 区分不可恢复）——
  拓扑规模真实，但关系语义退化。这直接限制了 Phase 2 的 IntTower/GNRR/RippleNet/KGRec、
  Phase 3 的完整子图检索质量对比、Phase 5 的端到端联合训练。

**环境本身的坑**（详见 `experiments/phase0/result.md` §2 和 memory
`path-retriever-env-and-data-gaps`）：
1. `dgl` 完全无法 `import`：`torchdata==0.11.0` 移除了 `dgl==2.1.0` 依赖的 DataPipes API——降级到 `torchdata==0.8.0`。
2. `dgl.graphbolt`/`dgl.sparse` 的 C++ 库只编译到 torch 2.2.1，但环境是 torch 2.4.1（为 LLM 训练栈升级）——分别用 site-packages patch（graphbolt 静默降级）和改写 `data_processing.py`（避免用 `dgl.sparse`）解决。
3. 装的 `dgl` 是 CPU-only 编译——`path_retriever/` 全部实验都在 CPU 上跑（`--device_id -1`）。
4. `path_retriever/pagelink.py` 原本就跑不起来：encoder 被切到不存在的 `LightGCN` checkpoint。

**其他发现的 bug**（非阻塞，记录供参考）：
- `ds_training/utils/utils.py` 的 `save_hf_format` 在 ZeRO-3 + Llama-3 快速分词器下必然崩溃——曾导致一次 50 小时训练白跑，已修复。
- `code/dense_retriever.py:69` 的 `topk_user_similarities` 字段实际存的是 id 而不是相似度（变量重新赋值顺序 bug）。

**方法论上的坑**：Phase1 早期尝试全量抽取 74212 个 (uid,iid) pair 的子图 embedding
时，单 pair 抽取本身就慢到无法接受——这是 Phase 1 解决的"检索开销"问题在子图粒度上的
重现，值得作为独立的优化方向。

## 5. 复现实验命令

```bash
# Phase 0-1：检索延迟对比（CPU，dgl 需先按 memory 里的步骤修好 import）
cd path_retriever
python pagelink.py --dataset_name yelp --split trn --device_id -1 \
    --max_num_samples 100 --path_method dijkstra   # 基线
python pagelink.py --dataset_name yelp --split trn --device_id -1 \
    --max_num_samples 100 --path_method power      # Power-Link

# Phase 2：backbone A/B
python train_linkpred.py --dataset_name yelp --split trn --device_id -1 \
    --encoder rgcn --num_epochs 100 --eval_interval 10
python train_linkpred.py --dataset_name yelp --split trn --device_id -1 \
    --encoder lightgcn --num_epochs 100 --eval_interval 10
python train_linkpred.py --dataset_name yelp --split trn --device_id -1 \
    --encoder kgat --num_epochs 100 --eval_interval 10

# Phase 3：子图 embedding 抽取 + soft-prompt 训练冒烟测试
cd ../subgraph_retriever
python extract_subgraph_embeds.py --dataset yelp --split trn --limit 5000
cd ../ds_training/step1_supervised_finetuning
bash training_scripts/single_node/run_yelp_subgraph_smoke.sh

# Phase 4：GraphLoRA 训练冒烟测试
bash training_scripts/single_node/run_yelp_graphlora_smoke.sh

# Phase 5：DFTopK + RCS 组件验证
cd ../../subgraph_retriever
python phase5_smoke_test.py

# 全指标评估（复用 Phase 0 基线的 Llama-3-8B 生成结果）
cd ../..
python evaluation/eval_lite.py --dataset yelp
```

## 6. 分支与提交

| 分支 | HEAD commit |
|---|---|
| `exp/phase0-baseline` | `faa886c` |
| `exp/phase1-power-link` | `38972d2` |
| `exp/phase2-retrieval-quality` | `5b1740f` |
| `exp/phase3-subgraph-retrieval` | `523ae5b` |
| `exp/phase4-graphlora` | `58c4f47` |
| `exp/phase5-e2e-joint` | `b2a16d3` |

**[需确认]**：分支尚未推送到 `origin`（`https://github.com/czt6666/G-Refer.git`）——
推送到远程仓库会让改动对协作者可见，按操作规范需要用户明确确认后再执行。
