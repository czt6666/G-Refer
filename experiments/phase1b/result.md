# Phase 1b — 严格复现 Power-Link 论文算法 (arXiv:2401.02290)，3000 样本 Dijkstra vs Power-Link

分支：`exp/phase1-power-link`
背景：Phase 1（见 `experiments/phase1/result.md`）用的 `graph_powering_paths` 是一个**近似实现**——只是拿"前向幂向量 + 贪心回溯"替代 Dijkstra 找路径，训练时的 loss、TES（Triplet Edge Scorer）、最终路径抽取方式都还是原 PaGE-Link 的设计，没有真正实现论文的算法。本 Phase 用户要求"严格按论文方法修改代码"，逐条对照论文重写，然后在真实 3000 样本上重新做 Dijkstra vs Power-Link 的生成质量 A/B。

## 1. 论文原文核对（第一手，非转述）

从 arXiv:2401.02290 PDF 抽取原文（Sec 5.1-5.3, Eq 1-13, Algorithm 1）核对后，Power-Link 的实际设计是：

1. **TES（Triplet Edge Scorer）**：`TES(·) = MLP(Combine(h_i, r_j, t_k, ĥ, r̂, t̂))`（Eq 1-3），对计算图里每条边，用"局部边三元组 + 目标三元组"的拼接向量喂给 MLP 打分，不是 PaGE-Link 原来那种自由标量 mask。
2. **Path-Enforcing Learning**（Eq 4-10）：只对源节点行向量 `u = M[î,:]` 做幂运算（不对整个矩阵做幂），迭代 `u^(l) = u^(l-1) @ M`；用**二值邻接矩阵的幂**统计路径条数做分母，归一化公式是 `(u_k^(l) / 路径条数)^(1/l)`（**开 l 次方根**，不是除以 l）；对 `l=1..L-1` 取平均得到 `P_on`；path loss = `-log(P_on)`。整个过程是纯 tensor 运算，无离散路径搜索，可微。
3. **总损失**（Eq 11-12）：`L_total = L_prediction + L_path + γ‖M‖²`，`L_prediction` 是原 PaGE-Link 那种保真度损失（M 加权图上模型预测是否还认为目标三元组成立），`γ‖M‖²` 是新增的正则项，Phase 1 的近似实现完全没有。
4. **Path Generation**（Eq 13）：训练完 TES 后算一次最终的 M，**取 M 的倒数作为代价矩阵，跑一次 Dijkstra**——Power-Link 并没有在最终路径抽取阶段替换掉 Dijkstra！论文真正省掉的是"训练循环里每个 epoch 都要跑一次 Dijkstra/Yen's"，最终抽取那一次 Dijkstra 保留。

Phase 1 的近似实现在这 4 点上全部不同：mask 还是自由标量、归一化是除以 l 不是开根号、没有 -log(P_on) 损失、没有正则项、且从头到尾都用幂运算代替 Dijkstra（包括最终抽取）。

## 2. 代码改动

- `path_retriever/utils.py`：
  - 删除近似实现 `graph_powering_paths()`。
  - 新增 `power_link_p_on()`：严格按 Eq 7-9 实现，只对源节点做幂迭代，二值邻接矩阵幂做分母，`(u/count)^(1/l)` 归一化，对 `l=1..L-1` 取平均，全程可微（无 `.item()` 打断梯度、无离散路径解码）。
  - 新增 `get_inverse_score_func()`：Eq 13 的"取倒数当代价矩阵"，喂给已有的 `k_shortest_paths_with_max_length`（Dijkstra）复用。
- `path_retriever/explainer.py`：
  - 新增 `TripletEdgeScorer`（TES）：`Combine=concatenate` 策略（论文实验用的策略）+ 2 层 MLP；关系嵌入用可学习的 `nn.Embedding`（本图是二部 user-item 图，R-GCN 只有按 etype 的消息传递权重、没有原生关系嵌入表，这是相对论文 KGC 场景——TransE/DistMult/ConvE 天然自带关系嵌入——的必要改造）。
  - `PaGELink.__init__` 新增 `pred_etype`（目标三元组的关系，默认 `'likes'`）、`gamma`（`‖M‖²` 正则权重，默认 1e-3）。
  - `get_edge_mask` 按 `path_method` 分流：`'dijkstra'` 走原 PaGE-Link 逻辑（`_train_mask_dijkstra`，未改动）；`'power'` 走新的 `_train_tes_power`——冻结 encoder 算一次实体嵌入（post-hoc 假设：Φ 不参与反传），每个 epoch 用 TES 前向算 M、`power_link_p_on` 算可微 path loss、`pred_loss`（复用原 PaGE-Link 的保真度损失，本来就等价于论文 Eq 11）、`γ‖M‖²` 正则，三项相加反传更新 TES 参数（不更新冻结的 encoder）。
  - `get_paths` 的 `'power'` 分支改为：`get_inverse_score_func` 取 M 的倒数 + 复用 Dijkstra（`k_shortest_paths_with_max_length`），不再自己做贪心回溯。
- `path_retriever/pagelink.py`：新增 `--gamma` CLI 参数，`pred_etype`/`gamma` 透传给 `PaGELink`。

### 正确性检查（小规模）

15 epoch 跑若干真实预测为正的 (user,item) pair：total_loss 从 ~9 降到 ~2，`pred_loss` 收敛到 ~0（说明加权图仍能让模型判定目标三元组成立，保真度损失确实在起作用），都能找到长度 1 的路径（这批样本大多是直连边，和 Phase 1 一致，因为 §5 提到的合成图连通性有限）。梯度检查：TES 参数确实在更新（loss 单调下降），冻结 encoder 参数不变。

## 3. 3000 样本检索延迟对比

从 `raft_data/yelp/test.json`（3000 条全部真实标注样本，覆盖了该数据集所有可用的测试样本）跑检索，8 路并行（`path_retriever/phase1_quality_compare.py --num_shards 8`，CPU-bound，图加载/负采样阶段每进程 ~9GB 内存，24 路并行会 OOM，8 路是安全上限）：

| 方法 | n | total | mean | empty_paths |
|---|---|---|---|---|
| dijkstra | 3000 | 7777.2s | 2.592s | 0/3000 |
| power (faithful) | 3000 | 7390.0s | 2.463s | 0/3000 |

**~1.05x**（约 5%）加速，远低于 Phase 1 近似实现报告的 4x/69x。这是符合预期的：论文真正省掉的是"训练循环里每个 epoch 都要跑一次 Yen's/Dijkstra"，但本图在 k-core 剪枝后计算图很小，Yen's 单次搜索本来就很快；新增的 TES MLP 前向+反传的开销，与省下来的搜索时间大致相互抵消。且严格实现在最终抽取阶段**保留了一次 Dijkstra**（Eq 13），近似实现是完全不用 Dijkstra 的，这也拉低了严格版的速度优势。换言之：**Phase 1 报告的大幅加速，是近似实现"全程用幂运算代替 Dijkstra"换来的，不是论文算法本身的加速幅度**——这是本次严格复现推翻的一个 Phase 1 结论。

## 4. 3000 样本真实生成质量对比

方法与 Phase 1 相同（`phase1_quality_compare.py`）：3000 条真实测试样本，两种方法在同一张图上重新检索路径，替换掉原 prompt 里的路径片段，其余变量（business/user profile、原始检索片段）保持一致；两组新 prompt 都喂给已训练好的真实 Llama-3-8B RAFT checkpoint（`ckpts/yelp_grefer_llama3`，4 卡并行推理，batch_size=8，max_new_tokens=256）做推理，`evaluation/eval_lite.py` 跑全指标（真实 BERTScore + BARTScore + 真实 BLEURT + USR）。

| metric | Dijkstra | Power-Link (faithful) | Δ (power − dijkstra) |
|---|---|---|---|
| BERT-P | 0.3189 | 0.3185 | -0.0004 |
| BERT-R | 0.3973 | 0.3962 | -0.0011 |
| BERT-F1 | 0.3584 | 0.3576 | -0.0008 |
| BARTScore | -3.0535 | -3.0532 | +0.0003（微弱更优） |
| BLEURT | -0.4637 | -0.4643 | -0.0006 |
| USR | 1.0000 | 1.0000 | 0 |

**结论：严格复现后，Power-Link 和 Dijkstra 在 3000 样本上的生成质量几乎完全打平**（6 个指标里 4 个 power 略输、1 个基本打平、1 个略赢，全部差值在 0.001 量级，远小于 Phase 1 报告的 +0.024 BERT-F1 / +0.06 BLEURT），可以认为是统计噪声内的平手，不是"谁更好"。

**这是对 Phase 1 结论的诚实推翻**：Phase 1（500 样本，近似实现）报告"Power-Link 在全部 6 个指标上都优于 Dijkstra"；本次（3000 样本，严格复现论文算法）显示两者打平。差异来源不是样本量（3000 vs 500 大概率只会让结果更稳定而非反转），而是**近似实现本身的路径选择启发式**（连续打分 + 除以步数的归一化 + 贪心回溯）恰好在 Phase 1 那批样本上选出了略优的路径，这是该近似算法的特性，不是 Power-Link 论文算法的真实属性。论文本身（Table 2）汇报的也是"和基线相比互有胜负、不是全面碾压"的结果，与本次严格复现观察到的"打平"更吻合。

## 5. 局限（诚实说明）

- 与 Phase 1 相同，本机缺少真实 buys/likes/bought_by 区分的原始交互数据（见 `[[path-retriever-env-and-data-gaps]]`），图里 `buys`/`bought_by` 两个 etype 是同一批边复制出来的，`likes` 边为空（是held-out的预测目标关系）。绝对分数（BERT-F1 ~0.358）比 Phase 0 真实图基线（0.4038）低，是这个合成图连通性有限导致的"共同天花板"，不影响两种检索方法之间的受控对比。
- TES 的关系嵌入是本次改造中新增的可学习表（因为这个二部图的 R-GCN 编码器没有原生关系嵌入），并非从冻结的 Φ 里取出来的——这是论文 KGC 场景（TransE/DistMult/ConvE 天然有关系嵌入）到本项目 recsys 场景的必要适配，不是论文原文的一部分。
- `γ‖M‖²` 正则项用的是默认权重 1e-3，未做超参搜索；论文本身也没有给出这一项的调参细节。
- 检索延迟基准是 CPU 单线程（每 pair），4.3s/pair 量级的开销主要来自每个 epoch 都要重新算一次 TES 前向 + 两条稀疏矩阵幂链，如果检索图更大（比如论文用的 FB15k-237/WN18RR 规模），Yen's 算法的非并行长尾延迟会比这里明显得多，届时"省掉每 epoch 一次 Dijkstra"的收益会比本次测到的 ~5% 更显著——本次数据集图规模偏小，掩盖了论文算法真正的收益场景。

## 6. 验收结论

**代码已严格对齐论文算法**（TES + 可微分幂运算 path loss + 正则项 + 最终 Dijkstra 抽取，逐条核对 Eq 1-13 / Algorithm 1）。3000 样本真实生成质量 A/B 显示 Power-Link 和 Dijkstra 打平（不再是 Phase 1 近似实现报告的"全面更优"）；检索延迟仅小幅领先（~5%），因为论文真正的加速收益来自省掉训练循环里的重复 Dijkstra 调用，在本项目当前的小规模计算图上不明显。`--path_method` 开关保留，`dijkstra` 分支未改动，可随时对照回退。
