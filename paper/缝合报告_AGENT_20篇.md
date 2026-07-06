# G-Refer 可缝合论文 · Agent 执行版（Top-20）

> 用途：供 AI coding agent 参照修改 G-Refer 代码。每条含【机制】【落点模块】【改动要点】【预期收益】。
> G-Refer 现状回顾（缝合基线）：
> - `path_retriever/`：R-GCN 打分 → m-core 剪枝 → PaGE-Link 边掩码学边权 → **Dijkstra 取 top-k 路径**（O(N²)，慢）。
> - 节点检索：SentenceBERT 编码 → **点积相似度取 top-k 用户/商品节点**（各候选独立、无交互）。
> - `gen_explanations/`：路径/节点序列化为文本 prompt 喂 LLM。
> - `ds_training/`：LoRA 微调 LLaMA-2/3 生成解释。
> - 检索粒度：只用 node+path，**未用 subgraph**（作者留的坑 / 用户创新点）。

---

## A. 路径检索器（替换 PaGE-Link + Dijkstra 瓶颈）

### 1. Power-Link — Path-based Explanation for KG Completion
- **链接**：https://arxiv.org/abs/2401.02290
- **机制**：Triplet Edge Scorer (TES) 用 MLP 融合局部边 embedding 与目标三元组 embedding 打边权；用 **graph-powering**（幂向量迭代乘加权邻接矩阵、按路径长归一）替代最短路搜索，可并行、避免 on-path 误差累积。
- **落点**：`path_retriever/`。
- **改动**：把 Dijkstra top-k 路径抽取替换为 graph-powering 矩阵运算；TES 结构可直接替换 PaGE-Link 的边打分头。
- **收益**：解决 G-Refer 已知的 O(N²) 最短路瓶颈（对标 REXHA 的"4min→<1s"痛点），可扩展到大图。**最高优先级。**

### 2. KGRec — KG Self-Supervised Rationalization
- **链接**：https://arxiv.org/abs/2307.02759
- **机制**：注意力打分建模三元组作为"协同交互理由"的概率，用 **head 邻居数做全局归一**（消度数偏置）；masked autoencoding 重建高分边 + rationale 对比学习剪低分噪声边。
- **落点**：`path_retriever/` 边权模块 + 训练损失。
- **改动**：用 rationale 分数替代/补充 PaGE-Link 掩码分数；全局度数归一直接强化 G-Refer 现有"惩罚高度数节点"的路径损失；加自监督重建损失。
- **收益**：更鲁棒的边重要性、显式去噪，缓解图噪声。与 G-Refer 的"高度数节点惩罚"天然对齐。

### 3. 路径感知最小充分子图检索（Graph Foundation Models for Path-aware GraphRAG）
- **链接**：https://arxiv.org/abs/2603.07179
- **机制**：以路径为基本单元，用**信息论准则**筛出"最小且充分"的推理路径集，去冗余保完整。
- **落点**：`path_retriever/` 的 top-k 选择准则。
- **改动**：把"取 k 条最短路"改为"取信息论意义下最小充分路径集"，k 自适应而非固定=2。
- **收益**：解释更精简非冗余；k 从超参变为数据自适应。

### 4. PaGE-Link（基线参照）
- **链接**：https://arxiv.org/abs/2302.12465
- **说明**：G-Refer 路径检索器的直接来源。改动前先核对本仓库实现与原文一致性（k-core 剪枝 + mask learning + shortest-path budget B）。作为 #1/#2 的对照基线保留。

---

## B. 节点检索器（升级点积相似度）

### 5. IntTower — 双塔细粒度交互
- **链接**：https://arxiv.org/abs/2210.09890
- **机制**：Light-SE 特征注意力加权 + FE-Block 早期跨塔特征交互 + CIR 对比正则 + **sum-max 相似度**（替代点积）。保持双塔可预计算的效率。
- **落点**：节点检索的相似度打分函数（SentenceBERT 之后）。
- **改动**：将 `dot(user_emb, item_emb)` 换成 sum-max 多层相似度；加 Light-SE 对节点属性维度加权。
- **收益**：更细粒度的 user-item 相关性，检索到的 top-k 节点更准，且不牺牲离线索引效率。

### 6. KGAT — 注意力邻居聚合
- **链接**：https://arxiv.org/abs/1905.07854
- **机制**：在 user-item-attribute 图上堆叠 attentive embedding propagation 层，注意力区分邻居重要性，显式建模高阶连通性。
- **落点**：节点/路径检索所依赖的 R-GCN 节点嵌入。
- **改动**：把 R-GCN 换成 KGAT 式注意力聚合，得到更强的节点表征供检索。
- **收益**：邻居融合带注意力权重，节点嵌入质量↑，检索与链路预测双双受益。

### 7. LightGCN — 精简邻居聚合基座
- **链接**：https://arxiv.org/abs/2002.02126
- **机制**：只保留邻居聚合，去掉特征变换与非线性；各层 embedding 加权和作最终表征。
- **落点**：GNN 编码器基座（替代/对比 R-GCN）。
- **改动**：可作轻量基线替换 R-GCN 打分器，或与之集成。
- **收益**：更简单、更快、CF 表征更强的 top-k 打分基座。

### 8. RippleNet — 偏好在 KG 上传播
- **链接**：https://arxiv.org/abs/1803.03467
- **机制**：从用户历史点击项出发，沿 KG 边多跳"涟漪"扩散偏好，自动发现从历史项到候选项的路径，无需手工设计。
- **落点**：路径候选生成（Dijkstra 之前的候选来源）。
- **改动**：用偏好传播自动生成候选路径集，再交给检索器精选；天然带可解释性。
- **收益**：路径候选生成从"图搜索"转为"偏好传播"，减少对最短路的依赖。

---

## C. 子图级检索（用户创新点：node/path → subgraph）

### 9. K-RagRec — 子图索引 + 重排 + GNN→LLM 投影
- **链接**：https://arxiv.org/abs/2501.02226
- **机制**：SentenceBERT 编码节点/边 → GNN 逐跳聚合得 **l-hop 子图向量** 存向量库（粗/细粒度双索引）；**popularity-selective 检索策略**（按流行度阈值决定哪些 item 需要增强，省开销、偏向冷启动）；用 prompt 作 query **重排**子图；GNN encoder + MLP projector 把子图对齐进 LLM 语义空间当 soft prompt。
- **落点**：新增 `subgraph_retriever/` 模块 + `gen_explanations/`（soft prompt 注入）。
- **改动**：实现 l-hop 子图索引与向量库；用 LLM prompt 重排；加 GNN+projector 把子图作 soft prompt 注入（而非纯文本序列化）。
- **收益**：直接落地"子图级检索"创新点；soft-prompt 注入保留结构、避免冗长序列化。**创新主线首选。**

### 10. CHEST — 交互相关异构子图 Transformer
- **链接**：https://arxiv.org/abs/2106.06722
- **机制**：为每个 user-item 交互抽"交互相关异构子图"（含路径），异构子图 Transformer 编码结构+路径语义（multi-slot 序列），课程预训练由易到难。
- **落点**：`subgraph_retriever/` 的子图抽取与编码。
- **改动**：借其 interaction-specific subgraph 抽取法控制搜索空间；multi-slot 路径序列化可复用到 prompt 构造。
- **收益**：给"如何抽子图不爆搜索空间"一个成熟范式，配合 #9 使用。

### 11. KGIN — 意图感知关系聚合
- **链接**：https://arxiv.org/abs/1905.07854（KGAT 同组后续，检索 "KGIN Learning Intents behind Interactions"）
- **机制**：在聚合层注入关系 embedding，建模用户对不同关系的 intent。
- **落点**：节点/子图聚合层。
- **改动**：聚合时区分关系类型与用户意图，路径解释可按 intent 分组。
- **收益**：解释路径带"意图"语义，更贴 G-Refer 的语义 CF。

---

## D. top-k 选择 / 候选重排机制

### 12. DFTopK — 可微快速 Top-K 选择
- **链接**：https://arxiv.org/abs/2510.11472
- **机制**：`f_k(x)=σ(x−(x[k]+x[k+1])/2)`，用 sigmoid 在第 k/k+1 名中点做软阈值；O(n) 复杂度，梯度只在决策边界两维。
- **落点**：节点检索的 top-k 选择（**注意**：作者实测不宜直接替 Dijkstra 图搜索，但适合替代节点相似度的硬 top-k）。
- **改动**：把节点检索的硬 `topk()` 换成 DFTopK，使检索可微、能被下游 loss 反传。
- **收益**：为"检索器-LLM 端到端联合训练"（见 E 组）提供可微 top-k 桥梁。

### 13. GNRR — 语料图上的图重排
- **链接**：https://arxiv.org/abs/2406.11720
- **机制**：候选组成子图（边=语义相似），GNN 让候选互相传信息后再打分，融合单点相关性。
- **落点**：节点检索之后、喂 LLM 之前，新增重排层。
- **改动**：把独立选出的 top-k 用户/商品节点组成小图跑一轮 GNN 互相校正、去冗余，再序列化。
- **收益**：修复"top-k 节点各自独立、可能冗余"的问题。低成本高收益。

### 14. HetComp — 异构教师 easy-to-hard 蒸馏
- **链接**：https://arxiv.org/abs/2303.01130
- **机制**：从多个异构强模型的训练轨迹构造由易到难的排序知识，自适应蒸馏给轻量 student。
- **落点**：路径/节点检索器训练。
- **改动**：用训练好的强 GNN（teacher）的 top-k 轨迹，课程式蒸馏给可解释检索器（student）。
- **收益**：让轻量可解释检索器逼近强黑盒模型的 top-k，直接呼应 G-Refer"用可解释检索器还原 GNN 结果"。

---

## E. 检索器 ↔ LLM 联合训练 / 一致性

### 15. LCRON — 级联作为单一网络端到端训练
- **链接**：https://arxiv.org/abs/2503.09492
- **机制**：定义"ground-truth 穿过所有阶段的联合存活概率"作端到端 surrogate loss；NeuralSort 可微排序连接各阶段；加 per-stage 辅助损失防监督消失。
- **落点**：跨 `path_retriever/` + `ds_training/` 的联合训练脚本。
- **改动**：把"检索器→LLM"视为两阶段级联，用可微排序 + 联合存活概率损失端到端训练；LLM 生成质量梯度反传指导检索。
- **收益**：消除"检索器辛苦选的 top-k，LLM 未必用"的阶段错配。配合 #12 的可微 top-k。

### 16. Cooperative Retriever and Ranker
- **链接**：https://arxiv.org/abs/2206.14649
- **机制**：联合训练 retriever 与 ranker，处理分布漂移、假负、排序错配（互蒸馏思路）。
- **落点**：联合训练框架设计参考。
- **改动**：以其协同范式设计"检索器 ⇄ LLM"互学；注意训练/推理候选分布一致。
- **收益**：检索器与生成器协同，缓解假负与分布漂移。

### 17. Ranking Consistency of Pre-ranking Stage
- **链接**：https://arxiv.org/abs/2205.01289
- **机制**：主张粗排该优化与精排的排序一致性；提出 RCS 指标；样本选择 + 蒸馏提升一致性。
- **落点**：`evaluation/` 新增一致性指标 + 检索器训练目标。
- **改动**：引入 RCS 度量"检索器选的 top-k 边/节点"与"GNN 真实决策"一致性；作训练/评估目标。
- **收益**：给"检索器逼近 GNN"一个可量化目标与指标。

---

## F. CF 信息注入 LLM / LoRA 微调

### 18. GraphLoRA — 结构感知 LoRA
- **链接**：https://arxiv.org/abs/2606.07526
- **机制**：在 LoRA 低秩通路里嵌入可训练的图消息传递网络，让协同拓扑直接参与参数更新，捕获高阶关系依赖。
- **落点**：`ds_training/`（G-Refer 现用 rank=8 vanilla LoRA）。
- **改动**：把 vanilla LoRA 换成 GraphLoRA，微调时注入 user-item 图结构而非仅文本 prompt。
- **收益**：LoRA 微调同时吸收结构 CF 信号，解释更贴图结构。直接升级现有微调。

### 19. FACE — CF embedding 量化为 LLM token
- **链接**：https://arxiv.org/abs/2510.15729
- **机制**：向量量化 + codebook 把连续 CF 节点 embedding 离散成 token，作为特殊 token 与文本一起喂 LLM。
- **落点**：`gen_explanations/` prompt 构造 + tokenizer 扩展。
- **改动**：把检索到的 top-k 节点 CF embedding 量化成 token 注入 LLM（替代/补充纯文本 profile）。
- **收益**：LLM 直接消费 CF 结构信号，codebook 提供可解释瓶颈。与 #9 的 soft-prompt 是两条注入路线，可二选一或并用。

### 20. REXHA — 分层聚合的检索增强解释生成
- **链接**：https://arxiv.org/abs/2507.09188
- **机制**：分层聚合式 profiling，综合 user/item 评论多级摘要（治 profile deviation）；向量检索替代 O(N²) 最短路。
- **落点**：`gen_explanations/`（profile 构造）+ `path_retriever/`（检索加速）。
- **改动**：用分层聚合摘要替代随机评论采样构 profile；向量检索加速路径检索。
- **收益**：直击 G-Refer 两大痛点（profile 偏差 + 检索慢），BERTScore-Precision 报告高 7.46%–26.8%。**与 #1 互补的加速路线。**

---

## 落地优先级建议（给 agent 的执行顺序）
1. **先修瓶颈**：#1 Power-Link / #20 REXHA（检索加速）→ 立刻可测收益。
2. **升级检索质量**：#13 GNRR（节点重排，低成本）+ #5 IntTower（相似度）+ #6 KGAT（嵌入）。
3. **上创新主线**：#9 K-RagRec + #10 CHEST（子图级检索）。
4. **强化微调**：#18 GraphLoRA / #19 FACE（CF 注入 LLM）。
5. **进阶研究**：#15 LCRON + #12 DFTopK + #17 RCS（端到端联合训练，工作量大、创新度高）。
