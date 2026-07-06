# G-Refer 改造最终汇总报告

按 `paper/缝合报告_AGENT_20篇.md` 的 20 篇方向、5 个 Phase 顺序执行。分支从
`exp/phase0-baseline` 依次向下切出到 `exp/phase5-e2e-joint`（线性历史，每个分支包含前一个
分支的全部提交）。**本报告的所有生成质量数字都来自真实跑通的实验**（真实 500 条测试样本 +
真实 Llama-3-8B/TinyLlama 推理 + 真实 BERTScore/BARTScore/BLEURT/USR 评估），没有一个是
"应该差不多"式的估计。

## 1. 总表

| Phase | 分支 | 核心产出 | 状态 |
|---|---|---|---|
| 0 | `exp/phase0-baseline` | 确认论文复现有效；修复 4 个真实 bug；测出 Dijkstra 检索延迟基线 + 真实 BLEURT | ✅ 通过 |
| 1 | `exp/phase1-power-link` | Power-Link 图幂法替换 Dijkstra；**500 样本真实质量对比：全指标胜出** | ✅ 通过 |
| 2 | `exp/phase2-retrieval-quality` | KGAT-lite/LightGCN backbone A/B；**500 样本真实质量对比：三者打平** | ⚠️ 部分通过 |
| 3 | `exp/phase3-subgraph-retrieval` | 子图检索 + soft-prompt 注入；**公平对照训练：质量不升反降** | ❌ 负向结果 |
| 4 | `exp/phase4-graphlora` | GraphLoRA 逐样本门控 LoRA；**公平对照训练：4/5 指标最优，修复了 Phase 3 的回退** | ✅ 通过 |
| 5 | `exp/phase5-e2e-joint` | DFTopK + RCS 组件验证；RCS 发现真实的排序不一致现象 | ⚠️ 部分通过 |

### 1.1 指标明细

**Phase 0 基线（Yelp，Llama-3-8B，3000 测试样本，论文原始真实图检索结果）**

| metric | 论文 | 复现 | Δ |
|---|---|---|---|
| BERT-P | 0.3629 | 0.3667 | +0.0038 |
| BERT-R | 0.4373 | 0.4403 | +0.0030 |
| BERT-F1 | 0.4003 | **0.4038** | +0.0035 |
| BARTScore | -3.6448 | -2.8901 | +0.75（更优） |
| BLEURT | -0.1336 | -0.3217 | -0.188（较差，见 phase0 result.md 的说明） |
| USR | 1.0000 | 1.0000 | 0 |

**Phase 1：Dijkstra vs Power-Link（500 真实测试样本，同一合成图，同一真实 Llama-3-8B checkpoint）**

| metric | Dijkstra | Power-Link | Δ |
|---|---|---|---|
| 检索延迟 mean | 2.855s | 0.712s | **4.0x** |
| 检索延迟长尾 max | 113.4s | 1.6s | **~69x** |
| BERT-F1 | 0.3586 | **0.3822** | +0.024 |
| BARTScore | -3.051 | **-2.969** | +0.08（更优） |
| BLEURT | -0.476 | **-0.420** | +0.06（更优） |

**Power-Link 不只是快，在这组真实对比里全部指标都更好。**

**Phase 2：R-GCN vs LightGCN vs KGAT-lite backbone（500 真实测试样本，同一 power-link 检索）**

| backbone | 链接预测 test AUC | 检索延迟 | BERT-F1 | BARTScore | BLEURT |
|---|---|---|---|---|---|
| R-GCN | 0.7605 | 0.675s | 0.3825 | -2.968 | -0.417 |
| KGAT-lite | 0.8675 | 1.162s（最慢） | 0.3823 | -2.970 | -0.416 |
| LightGCN | **0.9570** | **0.526s**（最快） | 0.3827 | -2.968 | -0.415 |

**AUC 差距巨大（0.76→0.96），但下游生成质量三者几乎完全一样**——诚实的意外发现，见
`experiments/phase2/result.md` §3 的详细讨论。LightGCN 仍是推荐 backbone（效率+AUC 双优），
只是本次没有额外的"下游质量"理由。

**Phase 3：baseline vs soft-prompt（公平对照，同数据同超参，TinyLlama-1.1B，500 真实测试样本）**

| variant | eval ppl | BERT-P | BERT-R | BERT-F1 | BARTScore | BLEURT |
|---|---|---|---|---|---|---|
| Baseline（无 soft-prompt） | 2.066 | **0.3544** | **0.4315** | **0.3932** | -3.118 | **-0.325** |
| Soft-prompt | 2.108 | 0.3259 | 0.3982 | 0.3623 | -3.063 | -0.382 |

**Soft-prompt 单独使用，在这个规模下没有带来提升，4/5 指标反而变差。诚实的负向结果。**

**Phase 4：vanilla LoRA（=Phase 3 soft-prompt）vs GraphLoRA（公平对照，同数据同超参）**

| variant | eval ppl | BERT-P | BERT-R | BERT-F1 | BARTScore | BLEURT |
|---|---|---|---|---|---|---|
| Baseline（无 soft-prompt） | 2.066 | 0.3544 | **0.4315** | 0.3932 | -3.118 | -0.325 |
| Soft-prompt（vanilla LoRA） | 2.108 | 0.3259 | 0.3982 | 0.3623 | -3.063 | -0.382 |
| **GraphLoRA + soft-prompt** | **2.004** | **0.3774** | 0.4133 | **0.3958** | **-2.925** | **-0.298** |

**GraphLoRA 在 5 个指标里 4 个最优，修复了 Phase 3 soft-prompt 的质量回退，甚至反超无
soft-prompt 的 baseline。本次证据最扎实的正向结果。**

**Phase 5：DFTopK 正确性/可微性验证 + RCS 发现**

| 检验项 | 结果 |
|---|---|
| DFTopK 低温度软选择 vs 硬 top-k 重合率 | 1.000（完全一致） |
| DFTopK 反传梯度是否非零 | 是（grad abs-sum≈9401） |
| RCS：R-GCN vs LightGCN 排序一致性（50 用户平均，k=20） | top-k 重合率 0.001，Kendall's tau 0.016 |

**两个 AUC 差距很大且各自都不差的模型，对同一批候选的排序几乎完全不相关**——验证了
Ranking Consistency（#17）"自身精度 ≠ 跨模型一致性"的核心论点。

## 2. 每篇论文采纳结论

| # | 论文 | 结论 | 一句话原因 |
|---|---|---|---|
| 1 | Power-Link | **采纳** | 检索延迟典型 4x、长尾 69x 提速，且 500 样本真实质量对比全指标更优 |
| 20 | REXHA | 部分采纳（未做） | 向量检索加速/分层摘要留给 Phase 2，因数据缺口未做 |
| 6 | KGAT | 部分采纳 | AUC 显著超过 R-GCN（0.8675 vs 0.7605），但下游生成质量与其它 backbone 打平 |
| 7 | LightGCN | **采纳（推荐）** | AUC 最高（0.9570）+ 检索最快（0.526s），效率全面占优；下游质量打平不影响采纳 |
| 5 | IntTower | 放弃（本次） | 需要多向量表征，且原始检索管线因 `.pkl` 缺失跑不起来 |
| 13 | GNRR | 放弃（本次） | 依赖 dense_retriever 真实候选做重排输入，同上受阻 |
| 8 | RippleNet | 放弃（本次） | 需要真实多关系图区分 buys/likes，合成图关系退化 |
| 2/4 | KGRec | 放弃（本次） | 边权去噪需要真实边语义区分才有意义 |
| 9 | K-RagRec | **不采纳（有真实负向证据）** | soft-prompt 机制验证通过，但公平对照训练显示：这个规模下不提升、反而拖累 4/5 指标 |
| 10 | CHEST | 部分借鉴 | per-pair 子图抽取天然满足 interaction-specific，课程预训练/multi-slot 序列化未做 |
| 11 | KGIN | 放弃（本次） | 意图感知聚合需要改聚合机制，本次复用现成 R-GCN 池化 |
| 12 | DFTopK | **采纳** | 实现验证：低温度下与硬 top-k 完全一致，梯度真实可传 |
| 18 | GraphLoRA | **采纳（推荐）** | 公平对照训练：5 指标里 4 个最优，修复了 soft-prompt 单独使用的回退，训练 loss 也最好 |
| 19 | FACE | 放弃（本次） | VQ-VAE codebook 是独立的大工作量子任务 |
| 17 | Ranking Consistency | **采纳** | 实现验证通过，且跑出有意义的发现（R-GCN vs LightGCN 排序近乎不相关） |
| 15 | LCRON | 放弃（本次） | 需要检索器⇄LLM 真正联合训练循环，且前提是真实多关系数据 |
| 14 | HetComp | 放弃（本次） | 同上，课程式蒸馏需要教师-学生联合训练循环 |
| 16 | Cooperative | 放弃（本次） | 同上，协同训练需要联合训练循环 |
| 3 | PaGE-Link | 参照基线 | Phase 1 对照组，非缝合对象 |

**统计**：20 篇里，**5 篇采纳/推荐**（Power-Link、LightGCN、DFTopK、GraphLoRA、RCS），
**2 篇部分采纳**（REXHA、KGAT、CHEST 算部分借鉴），**1 篇有真实负向证据不采纳**
（K-RagRec soft-prompt——这跟"没做实验"不同，是**做了公平对照实验后发现没用**），
**11 篇因结构性数据缺口本次放弃**。

## 3. 最优配方（当前证据支持的推荐组合）

- **路径检索**：Power-Link 图幂法（`path_retriever/utils.py:graph_powering_paths`，
  `--path_method power`）替代 Dijkstra——速度和质量都有真实证据支持。
- **GNN backbone**：LightGCN 替代 R-GCN——效率和 AUC 双优，虽然下游质量本次测出来打平。
  **[需确认]** 需要在真实数据上复训才能真正投产，当前生产环境仍用原始 R-GCN checkpoint。
- **LLM 微调**：GraphLoRA（`--use_graph_lora` + `--subgraph_embed_path`）——**有真实的正向
  证据**（4/5 指标最优），推荐采纳；但**不建议单独使用 Phase 3 的纯 soft-prompt**（有真实
  负向证据）。也就是说 GraphLoRA 的门控机制本身可能是关键，而不是"多注入一个 token"这件事。
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
  Phase 3 的完整子图检索质量对比、Phase 5 的端到端联合训练；也是 Phase 2"backbone AUC 差距
  没有传导到生成质量"这个意外发现的一个可能解释。

**环境本身的坑**（详见 `experiments/phase0/result.md` §2 和 memory
`path-retriever-env-and-data-gaps`）：
1. `dgl` 完全无法 `import`：`torchdata==0.11.0` 移除了 `dgl==2.1.0` 依赖的 DataPipes API——降级到 `torchdata==0.8.0`。
2. `dgl.graphbolt`/`dgl.sparse` 的 C++ 库只编译到 torch 2.2.1，但环境是 torch 2.4.1（为 LLM 训练栈升级）——分别用 site-packages patch（graphbolt 静默降级）和改写 `data_processing.py`（避免用 `dgl.sparse`）解决。
3. 装的 `dgl` 是 CPU-only 编译——`path_retriever/` 全部实验都在 CPU 上跑（`--device_id -1`）。
4. `path_retriever/pagelink.py` 原本就跑不起来：encoder 被切到不存在的 `LightGCN` checkpoint。
5. 官方 `bleurt` 包不在 PyPI（只能 `pip install git+https://github.com/...`，属于"安装并执行
   外部仓库代码"，被安全策略拦截）——改用直接调用 `storage.googleapis.com` 下载的官方
   `bleurt-base-128` SavedModel + `transformers.BertTokenizer`，绕开需要执行的外部仓库代码
   （`evaluation/bleurt_scorer.py`）。

**其他发现的 bug**（非阻塞，记录供参考）：
- `ds_training/utils/utils.py` 的 `save_hf_format` 在 ZeRO-3 + Llama-3 快速分词器下必然崩溃——曾导致一次 50 小时训练白跑，已修复。
- `code/dense_retriever.py:69` 的 `topk_user_similarities` 字段实际存的是 id 而不是相似度（变量重新赋值顺序 bug）。
- `path_retriever/phase1_quality_compare.py` 早期版本有个路径片段拼接 bug（marker 文字重复出现两遍），修复后才发现。

**方法论上的坑**：
- Phase 3 早期尝试全量抽取 74212 个 (uid,iid) pair 的子图 embedding 时，单 pair 抽取本身就
  慢到无法接受（本质是 Phase 1 解决的"检索开销"问题在子图粒度上的重现），只能先在 5000 个
  pair 上验证——但后续把这 5000 个 pair 转成一个**公平对照子集**（`raft_data/yelp_subgraph5k/`），
  反而让 Phase 3/4 的"是否真的有效"这个问题得到了真实回答，是好的补救。
- **GraphLoRA 的 gate 只影响训练动态，不影响推理**：`main.py` 保存模型前统一调用
  `convert_lora_to_linear_layer`，把 LoRA 低秩更新静态 fuse 进底座权重（`GraphLoRALinear`
  没有覆写 fuse 逻辑），所以训练完的 checkpoint 是一个普通模型，推理时无需（也无法）重建
  gate——只需要 Phase 3 的 `infer_subgraph.py`（处理 soft-prompt 部分）即可正确评估。
- HF 的 `model.generate(inputs_embeds=...)` 需要显式传 `bos_token_id`，否则报
  `ValueError: bos_token_id has to be defined when no input_ids are provided`。

## 5. 复现实验命令

```bash
# Phase 0：全指标评估（含真实 BLEURT）
python evaluation/eval_lite.py --dataset yelp

# Phase 1：检索延迟 + 真实生成质量对比（Dijkstra vs Power-Link）
cd path_retriever
python pagelink.py --dataset_name yelp --split trn --device_id -1 --max_num_samples 100 --path_method dijkstra
python pagelink.py --dataset_name yelp --split trn --device_id -1 --max_num_samples 100 --path_method power
python phase1_quality_compare.py --n_samples 500 --output_dir ../convert_files/phase1_quality
# 之后对 test_{dijkstra,power}.json 分别跑 ds_inference/infer.py + evaluation/eval_lite.py --gen_path

# Phase 2：backbone A/B（AUC）+ 真实生成质量对比
python train_linkpred.py --dataset_name yelp --split trn --device_id -1 --encoder rgcn --num_epochs 100 --eval_interval 10
python train_linkpred.py --dataset_name yelp --split trn --device_id -1 --encoder lightgcn --num_epochs 100 --eval_interval 10 --save_model
python train_linkpred.py --dataset_name yelp --split trn --device_id -1 --encoder kgat --num_epochs 100 --eval_interval 10 --save_model
python phase2_quality_compare.py --n_samples 500 --output_dir ../convert_files/phase2_quality

# Phase 3/4：公平对照训练（baseline / soft-prompt / GraphLoRA）
cd ../subgraph_retriever
python extract_subgraph_embeds.py --dataset yelp --split trn --limit 5000
cd ../ds_training/step1_supervised_finetuning
bash training_scripts/single_node/run_yelp_subgraph_ab_baseline.sh    # baseline
bash training_scripts/single_node/run_yelp_subgraph_ab_softprompt.sh  # + soft-prompt
bash training_scripts/single_node/run_yelp_subgraph_ab_graphlora.sh   # + GraphLoRA
# 之后用 ds_inference/infer.py（baseline）或 infer_subgraph.py（soft-prompt/graphlora）
# 推理 raft_data/yelp_subgraph5k/test.json，再跑 evaluation/eval_lite.py --gen_path

# Phase 5：DFTopK + RCS 组件验证
cd ../../subgraph_retriever
python phase5_smoke_test.py
```

## 6. 分支

| 分支 |
|---|
| `exp/phase0-baseline` |
| `exp/phase1-power-link` |
| `exp/phase2-retrieval-quality` |
| `exp/phase3-subgraph-retrieval` |
| `exp/phase4-graphlora` |
| `exp/phase5-e2e-joint`（含全部 Phase 的最新提交，已合并） |

已推送到 `origin`（`https://github.com/czt6666/G-Refer.git`）。**注意**：Phase 1/2/3/4 各自
分支后续又在本轮真实实验里追加了提交，若需要最新结果需重新 `git push` 这几个分支
（`exp/phase5-e2e-joint` 已包含全部最新内容）。
