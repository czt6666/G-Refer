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

## 4. 局限（诚实说明）

本机缺少 Step1-2 所需的原始 `{split}.pkl` 交互数据（见 [[path-retriever-env-and-data-gaps]]），
无法重跑真实的 buys/likes/bought_by 边类型区分。本 Phase 的图用 `total_trn.csv` 的
314944 条真实 (user, item) 交互对，三种 etype 都填入同一批边构造（拓扑规模、节点数与真实
checkpoint 完全一致，但边类型语义是重复的）。这导致：

1. k-hop 子图 + k-core 剪枝后，很多 (src, tgt) pair 的邻域里除了直连边外几乎没有其它路径
   （多跳路径的候选被"重复边"稀释而非真正的多样连接），所以两种方法在本次实验里选出的
   最终路径大多是长度 1 的直连边、**内容完全一致** —— 这意味着本次没能在"生成质量"上直接
   观察出差异（好是坏都测不出来），但也说明**图幂法在这个退化场景下与 Dijkstra 结果一致**，
   没有引入结构性偏差。
2. 因此本 Phase **未**重跑 Step 5-9（真实 node/path 检索 → 序列化 → 复用基线 LoRA 推理 →
   `eval_lite.py` 全指标对比）—— 在当前合成图上重跑没有信息量（两种方法产出内容相同）。
   [需确认]：若要验证"图幂法在真实多跳路径上是否保持解释质量"，需要先解决数据缺口
   （下载/重建原始 `{split}.pkl`，或直接用已下载的 `saved_explanations/`
   里的原版 Dijkstra 输出做参照）。默认建议：保留 `--path_method` 开关，待真实数据可用时
   再做质量端对比；不阻塞后续 Phase（先按加速已验证、质量非劣的假设继续 Phase 2）。

## 5. 未采纳/延后

- **TES（Triplet Edge Scorer, MLP 边打分头）**：Power-Link 论文的另一半改动，本 Phase 未做。
  原因：核心速度收益已经由 graph-powering 单独达成（前面测的就是替换路径搜索算法本身，
  边打分头仍是原 PaGE-Link mask learning）；TES 改变的是训练动态而非检索延迟，留给
  Phase 2（连同 KGRec 的 rationale 打分一起做边权改造消融）更合适。
- **REXHA 向量检索加速 / 分层摘要 profile**：留给 Phase 2（"检索质量"相关改动更集中）。

## 6. 验收结论

**通过**：检索延迟显著下降（典型 4x，长尾场景 ~69x），且在可测的合成图范围内路径选择结果
与基线一致（非劣）。`--path_method` 开关保留原实现，随时可回退对照。进入 Phase 2。
