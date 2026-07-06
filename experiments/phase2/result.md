# Phase 2 — 检索质量升级

分支：`exp/phase2-retrieval-quality`（从 `exp/phase1-power-link` 切出）

## 0. 范围调整（诚实说明，先于结果）

计划的 6 个方向里，**GNRR(#13)、IntTower(#5)、RippleNet(#8)、KGRec(#2/#4)** 都需要真实
`data/{dataset}/{split}.pkl`（未去重的原始交互记录）驱动 `code/dense_retriever.py`（节点检索）
或路径候选生成，而本机在 Phase 0 就已确认这份数据缺失（见
[[path-retriever-env-and-data-gaps]]、`experiments/phase0/result.md` §2）。没有这份数据：

- `code/dense_retriever.py` 本身就跑不起来（`pkl_data.iterrows()` 直接依赖它），IntTower 的
  sum-max 相似度、GNRR 的候选重排都是在这一步的产物上做文章，无法验证。
- RippleNet 的偏好传播候选生成、KGRec 的 rationale 打分同样需要能区分 `buys`/`likes` 的真实
  多关系数据，而本机图是从 `total_trn.csv` 的 314944 条交互对复制到三个 etype 里的（Phase 1
  已用的合成图），关系类型本身就是退化的。

**因此本 Phase 只做了 R-GCN/LightGCN/KGAT 骨干网络的 A/B**（#6 KGAT、#7 LightGCN），这是唯一
只依赖图结构（不依赖缺失的 `.pkl`）就能真实训练和验证的方向。其余 4 个方向记录为
**受阻 / 待数据可用后补做**，不在本 Phase 强行写无法验证的代码。

## 1. 改动摘要

- `path_retriever/model.py`：新增 `KGATLayer`/`KGAT`（#6，注意力邻居聚合，简化版 —— 每种
  canonical edge type 内部做 softmax 归一，不是论文里跨关系联合归一，因为 DGL 异构
  `edge_softmax` 要求异构图跨所有关系类型统一处理，联合归一需要更大的重构，本 Phase 从简）。
  `LightGCN`（#7）已在代码里但从未接入训练脚本，本次补上 `forward()` 的 `eweight_dict` 形参
  （兼容 `HeteroLinkPredictionModel` 的统一调用签名）。
- `path_retriever/train_linkpred.py`：新增 `--encoder {rgcn,lightgcn,kgat}` 开关（默认
  `rgcn`，不影响现有下游——见 §3 为何不改默认值）；`--save_model` 存盘文件名按 encoder 加后缀，
  避免覆盖已有的 R-GCN 生产 checkpoint。

### 发现并修复的一个 bug

`KGATLayer` 初版直接对全部 3 个 canonical etype 一起调用 `dgl.nn.functional.edge_softmax`，
但训练图里预测目标边类型（`likes`）在消息传递图 `mp_g` 里被清空为 0 条边（防止标签泄漏），
DGL 的异构 `edge_softmax` C++ kernel 在遇到 0 条边的关系类型时直接抛 `Check failed: notnull: W`
（CPU 后端的已知边界情况）。修复：只对**非空关系类型**计算 attention/做 softmax
（`model.py` 的 `non_empty_etypes` 过滤逻辑）。

## 2. 结果：R-GCN vs LightGCN vs KGAT-lite（同一合成图，CPU，100 epoch，lr=0.01，其余超参默认）

| encoder | epoch 10 val AUC | epoch 50 val AUC | epoch 100 val AUC | epoch 100 test AUC |
|---|---|---|---|---|
| R-GCN（原基线） | 0.5645 | 0.6216 | 0.7593 | 0.7605 |
| KGAT-lite（新增，#6） | 0.6438 | 0.7573 | 0.8676 | 0.8675 |
| **LightGCN（新增，#7）** | **0.9284** | **0.9445** | **0.9568** | **0.9570** |

三者在 epoch 100 时都还在上升（未完全收敛），但排序在整个训练过程中高度一致：
**LightGCN ≫ KGAT-lite > R-GCN**。LightGCN 在第 10 个 epoch 就已经超过 R-GCN/KGAT
跑满 100 epoch 的结果，收敛速度上的差距非常悬殊，与 LightGCN 论文"去掉特征变换和非线性
反而提升协同过滤效果"的核心论点一致；KGAT 的注意力聚合也明确优于 R-GCN 的均值聚合。

## 3. 真实生成质量对比（补充实验，2026-07-06）

链接预测 AUC 差距这么大（0.76 → 0.96），backbone 换血对**最终生成的解释文本质量**到底有没有
影响？用跟 Phase 1 §4 完全一样的方法（`path_retriever/phase2_quality_compare.py`）：500 条
真实测试样本，三种 backbone 各自用 power-link 方法重新检索路径、替换掉原 prompt 的路径片段，
其余不变，喂给同一个真实 Llama-3-8B checkpoint 推理，跑全指标。

| backbone | 检索延迟 mean（500 样本） | BERT-P | BERT-R | BERT-F1 | BARTScore | BLEURT |
|---|---|---|---|---|---|---|
| R-GCN | 0.675s | 0.3449 | 0.4190 | 0.3825 | -2.9678 | -0.4171 |
| LightGCN | **0.526s**（最快） | 0.3450 | 0.4188 | 0.3827 | -2.9682 | -0.4154 |
| KGAT-lite | 1.162s（最慢，注意力开销） | 0.3454 | 0.4187 | 0.3823 | -2.9701 | -0.4164 |

**发现：三种 backbone 的下游生成质量几乎完全一样**（BERT-F1 三者都在 0.3823-0.3827，
BARTScore/BLEURT 差异都在小数点第三位）——尽管它们的链接预测 AUC 差了 0.76 vs 0.96，
**这个巨大的 AUC 差距完全没有传导到最终的解释文本质量上**。这是一个诚实、意外的发现，
不应该被回避或淡化。可能的原因（推测，未验证）：Phase 1 已经发现这张合成图上路径检索
经常退化为直连边（多跳路径候选被合成图的关系类型重复稀释），如果三种 backbone 在"该选哪条
退化路径"上给出的排序足够接近，最终序列化出的文本自然趋同——即 backbone 的差异化打分能力
在"路径候选本来就很有限"的场景下发挥不出来。**这不代表 backbone 选择不重要**：AUC
差距本身是真实的、可复现的（§2），只是在本次实验的具体检索管线（合成图 + power-link
+ 只取 2 条路径）里，这个差距没有变成生成质量的差距。真实多关系数据下的检索管线可能会有
不同结果，值得在数据缺口解决后重新验证。

## 4. 结论与 [需确认]

**推荐**：把 LightGCN 作为新的默认 GNN backbone（供 Phase 3+ 使用），R-GCN/KGAT 保留作对照。
LightGCN 检索延迟也最低（0.526s vs R-GCN 0.675s、KGAT 1.162s），效率上也是三者中最优的——
"效率+准确率"都改善（AUC 更高、检索更快），只是本次实验条件下**下游生成质量三者打平**，
不构成额外的采纳理由，但也不是反对理由。

**[需确认]**：本次 A/B 用的是 Phase 1 那张"退化"合成图（三种 etype 都是同一批边的复制），
不是真实的 buys/likes/bought_by 区分数据；且现有 `saved_models/yelp_model_trn.pth` 是
论文/前人在真实数据上训出的 R-GCN checkpoint，本 Phase 没有对应的真实数据可训出一份可比的
LightGCN checkpoint。所以：

- 本次没有把 `train_linkpred.py`/`pagelink.py` 的 `--encoder` 默认值从 `rgcn` 改成
  `lightgcn`——改了默认值但没有真实 checkpoint 可用会直接破坏 Phase 0/1 已验证的下游链路。
- 默认建议（已按此继续，不阻塞后续 Phase）：**保留 `rgcn` 默认值 + 现有真实 checkpoint**
  用于 Phase 3-5 的路径/节点检索；同时把"LightGCN backbone 应在真实数据上复训并替换"记为
  后续待办。一旦真实 `.pkl` 数据可用（或直接找到论文原始 buys/likes 区分方式），应优先补跑
  这一步——预期收益（AUC 从 ~0.76 到 ~0.95+ 量级的差距）大概率也会稳定复现，值得投入。

## 5. 未采纳/延后

| 方向 | 状态 | 原因 |
|---|---|---|
| IntTower(#5) sum-max 相似度 | 延后 | 依赖 `code/dense_retriever.py`，本机 `.pkl` 数据缺失，跑不起来；且 sum-max 需要多向量（多条评论）表征，当前节点表征是单一池化向量，需要更大的架构改动才能套用 |
| GNRR(#13) 候选重排 | 延后 | 依赖 dense_retriever 产出的真实 top-k 候选做重排输入，同上受阻 |
| RippleNet(#8) 偏好传播候选路径 | 延后 | 需要真实多关系图（区分 buys/likes）做多跳偏好传播，当前合成图关系退化，传播结果无意义 |
| KGRec(#2/#4) rationale 打分/去噪 | 延后 | 同上，边权去噪需要真实边语义区分才有意义 |

### 附：`download_data.py` manifest 复核（重要更正）

重新核对 `download_data.py` 的下载清单后发现：**`{split}.pkl` 从未在清单里出现过** ——
不是"本机没下载"，而是**论文作者的公开数据发布本身就不含这个中间文件**（只发布了
`data_trn.pt`、`dense_retrieval_results_*.json`、`saved_models/`、`saved_explanations/`、
`raft_data/` 这些"已经算好的产物"）。也就是说 Step1-2 的原始检索管线**在任何网络环境下都
无法从公开数据完整复现**，这不是本机网络限制问题。

顺带确认：`data/yelp/dense_retrieval_results_trn.json`（74212 对 user-item 的真实
IntTower/GNRR 候选，论文作者用真实 `.pkl` 跑出来的产物）确实存在且可读，但抽查发现
`code/dense_retriever.py:69` 有个数据 bug——`topk_user_similarities` 字段在保存前已经把
`topk_users` 重新赋值成纯 id 列表（第 57 行剪枝后），导致存下来的"相似度"其实是 id 本身
（`topk_user_ids == topk_user_similarities`，抽查样本完全一致）。这不影响本 Phase 的结论
（IntTower 的 sum-max 本来就因为节点表征是单一池化向量、没有多向量表征而无法套用，跟这个
bug 无关），记录下来供以后修复原始 pipeline 时参考。

## 6. 验收结论

**部分通过**：backbone 消融（KGAT/LightGCN vs R-GCN）方向完整跑通并给出明确、一致的结果
（LightGCN/KGAT 均显著优于 R-GCN）；其余 4 个方向因数据缺口延后，不阻塞进入 Phase 3
（子图级检索同样会受同一数据缺口影响，需要在 Phase 3 开始时先确认可用的数据范围）。
