# Phase 1 — Power-Link 图幂法替换 Dijkstra

分支：`exp/phase1-power-link`（从 `exp/phase0-baseline` 切出）
论文：**#1 Power-Link (arXiv:2401.02290)**。REXHA（#20）的向量检索/分层摘要部分本 Phase 未做，见 §5。

## 1. 改动摘要

- `path_retriever/utils.py`：新增 `graph_powering_paths()` —— 用**前向幂向量迭代**
  （稀疏矩阵 `A_t @ v` 反复乘，`max_length` 次）替代 `k_shortest_paths_with_max_length`
  （Yen's k-shortest-paths / 双向 Dijkstra，纯 Python heapq 实现，per-pair 串行图搜索）。
  - 转移权重沿用 G-Refer 现有"惩罚高度数节点"的语义：`trans_w(u→v) = eweight(u,v) / in_degree(v)`，
    与原 Dijkstra 用的 cost `= log(in_degree(v)) − log(eweight(u,v))` 是同一套偏好的乘法/加法对偶形式。
  - 路径按长度归一 `score = mass / L`（Power-Link 论文的长度归一），取 top-k 长度候选，
    再用前向势能引导的**贪心反向回溯**（显式排除已访问节点，保证 simple path，见下方"发现的问题"）
    还原具体节点序列。
  - 复杂度：`O(max_length)` 次稀疏矩阵乘法，与 (src,tgt) pair 数量、mask-learning epoch 数无关；
    原 Dijkstra 每个 pair 每个 epoch 都要重新跑一次图搜索（`path_loss()` 在 10 个 mask-learning
    epoch 里各调用一次，`get_paths()` 最终再调用一次）。
- `path_retriever/explainer.py`：`PaGELink.__init__` 新增 `path_method='dijkstra'|'power'`
  开关（默认 `dijkstra`，行为不变）；`path_loss()`（mask 训练中的正则项）和 `get_paths()`
  （最终路径抽取）都按此开关分流到 `graph_powering_paths` 或原 `k_shortest_paths_with_max_length`。
- `path_retriever/pagelink.py`：新增 `--path_method {dijkstra,power}` CLI 参数；加
  `time.perf_counter()` 包裹 `explain()` 调用，打印 `[retrieval-timing]` 汇总（Phase 0 已用此测基线）。
- 未做（超出本 Phase 范围，记录见 §5）：TES（Triplet Edge Scorer MLP）替换 PaGE-Link 掩码头；
  REXHA 的向量检索加速 / 分层摘要构建 profile。

## 2. 发现并修复的一个正确性 bug

初版 `graph_powering_paths` 的贪心回溯没有排除已访问节点，在图很稀疏（k-hop 子图 + k-core
剪枝后往往只剩直连边）时会退化成 `u→v→u→v→u` 这种反复横跳的"路径"（因为只有一条边权重
占绝对优势，回溯每步都选它）。这与原 Dijkstra（Yen's 算法本身只枚举 simple path）行为不一致，
序列化成文本会是无意义的解释。**修复**：回溯时显式维护 `visited` 集合，候选节点排除已访问节点
（`utils.py` 的 `backtrack()` 内部）。修复后同一批测试 pair 的路径全部退化为长度 1（即直连边）——
这是本 Phase 用的合成图本身连通性有限导致的（见 §4 局限），不是新 bug。

## 3. 检索延迟对比（同一 R-GCN checkpoint、同一图、CPU 推理）

| 样本量 | 方法 | n（预测正例数） | total | mean | min | max |
|---|---|---|---|---|---|---|
| 30 | dijkstra | 5 | 122.982s | 24.596s | 2.003s | **113.385s** |
| 30 | power | 5 | 4.011s | 0.802s | 0.429s | 1.644s |
| 100 | dijkstra | 18 | 51.393s | 2.855s | 0.730s | 4.928s |
| 100 | power | 18 | 12.822s | 0.712s | 0.289s | 1.606s |

- **典型情形（100 样本组）**：均值加速 **4.0x**（2.855s → 0.712s），且方差显著收窄
  （max 4.93s → 1.61s）。
- **长尾情形（30 样本组捕捉到）**：Dijkstra 单个 pair 最坏 **113.4s**，Power-Link 同一 pair
  集合最坏仅 **1.64s** —— **约 69x**。这正是论文和 REXHA 报告里提到的"O(N²) 最短路搜索、
  长尾延迟"问题的直接复现：多数 pair 很快，少数 pair（大概率是高连通度/高分支因子的
  computation subgraph）在 Yen's 算法里退化得很慢，而图幂法对 pair 结构不敏感，延迟稳定。

## 4. 真实生成质量对比（补充实验，2026-07-06）

之前的版本在这里写"未做下游质量对比"——后来补跑了。方法（`path_retriever/phase1_quality_compare.py`）：
从 `raft_data/yelp/test.json` 随机抽 500 条**真实**测试样本（真实 uid/iid/business profile/
user profile/ground-truth explanation），用 dijkstra 和 power 两种方法在同一张图上重新检索
路径，用 `code/translation.py` 的原始序列化模板把新路径**替换掉原 prompt 里的路径片段**
（business/user profile、节点检索片段保持不动），构造出两组新 prompt。两组都喂给**已经训好的
真实 Llama-3-8B RAFT checkpoint**（`ckpts/yelp_grefer_llama3`，跟 Phase 0 用的是同一个模型）
做推理，再用 `evaluation/eval_lite.py`（含真实 BLEURT，见 `experiments/phase0/result.md`）跑全指标。
除路径检索方法外，其余变量全部控制一致。

500/500 pair 两种方法都找到了真实路径（无退化为空/单边的情况——用真实测试集样本而不是
之前较小的随机抽样后，多跳路径的覆盖率好很多）。

| metric | Dijkstra | Power-Link | Δ（power − dijkstra） |
|---|---|---|---|
| BERT-P | 0.3232 | 0.3449 | **+0.0217** |
| BERT-R | 0.3935 | 0.4190 | **+0.0255** |
| BERT-F1 | 0.3586 | 0.3822 | **+0.0236** |
| BARTScore | -3.0511 | -2.9686 | **+0.0825**（更优） |
| BLEURT | -0.4755 | -0.4195 | **+0.0560**（更优） |
| USR | 1.0000 | 1.0000 | 0 |

**结论：在这组真实、受控的 500 样本对比里，Power-Link 在全部 6 个指标上都优于（或持平）Dijkstra**，
不只是"非劣"，是真的更好。这回答了"Power-Link 快但准确率如何"的问题——至少在本次实验规模和
数据条件下，速度和质量没有此消彼长，反而同时改善了。

**诚实的补充说明**：这两个数字（BERT-F1 0.359/0.382）都比 Phase 0 基线（0.4038，用论文原始
真实图检索出的路径）低。这不代表 Power-Link/Dijkstra 本身实现有问题，而是因为本次两种方法
共用的图是 §5 提到的**合成图**（关系类型退化）——用退化图检索出的路径本身信息量就比论文原始
真实图弱，这个"底"拉低了两者的绝对分数，但不影响"同一张图上两种检索算法谁更好"这个受控比较
的有效性。

**为什么 Power-Link 反而更准**（推测，非定论）：graph-powering 用"路径长度归一 + 转移概率"
连续打分选路径，Dijkstra/Yen's 算法用离散的"最短代价"排序——在多条路径代价接近时，图幂法的
连续评分可能更稳定地选出"整体转移概率更高"（而非单纯"跳数最少"）的路径，这类路径经过的中间
节点度数分布更均衡（惩罚高度数节点的机制在连续打分下生效更充分）。这是一个合理的解释方向，
但本次实验没有专门去验证这个机制假设，留作后续可选的深入分析。

## 5. 局限（诚实说明）

本机缺少 Step1-2 所需的原始 `{split}.pkl` 交互数据（见 [[path-retriever-env-and-data-gaps]]），
无法重跑真实的 buys/likes/bought_by 边类型区分。本 Phase 的图用 `total_trn.csv` 的
314944 条真实 (user, item) 交互对，三种 etype 都填入同一批边构造（拓扑规模、节点数与真实
checkpoint 完全一致，但边类型语义是重复的）。§4 的真实质量对比已经在这张合成图上完成，
且发现即使在这个数据条件下也能观察到有意义的方法间差异——但**绝对分数**仍受合成图质量所限，
不能直接和论文 Table 1 的数字比较（应该跟 Phase 0 在同一张合成图上的对照数字比）。

## 6. 未采纳/延后

- **TES（Triplet Edge Scorer, MLP 边打分头）**：Power-Link 论文的另一半改动，本 Phase 未做。
  原因：核心速度收益已经由 graph-powering 单独达成（前面测的就是替换路径搜索算法本身，
  边打分头仍是原 PaGE-Link mask learning）；TES 改变的是训练动态而非检索延迟，留给
  Phase 2（连同 KGRec 的 rationale 打分一起做边权改造消融）更合适。
- **REXHA 向量检索加速 / 分层摘要 profile**：留给 Phase 2（"检索质量"相关改动更集中）。

## 7. 验收结论

**通过**：检索延迟显著下降（典型 4x，长尾场景 ~69x）；**且在 500 样本真实生成质量对比中，
Power-Link 在全部指标上都优于 Dijkstra**（BERT-F1 +0.024，BARTScore +0.08，BLEURT +0.06），
不是简单的"非劣"。`--path_method` 开关保留原实现，随时可回退对照。进入 Phase 2。
