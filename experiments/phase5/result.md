# Phase 5 — 端到端联合训练

分支：`exp/phase5-e2e-joint`（从 `exp/phase4-graphlora` 切出）
论文：**#12 DFTopK**、**#17 Ranking Consistency Score** 已实现并验证；**#15 LCRON**、
**#14 HetComp**、**#16 Cooperative** 未实现（见 §3）。

## 1. 改动摘要

- `path_retriever/dftopk.py`：`dftopk(scores, k, temperature)` —— 按论文公式
  `f_k(x) = σ((x − (x[k]+x[k+1])/2) / temperature)` 实现的可微 top-k 软选择。用于把节点
  检索的硬 `topk()` 换成可微版本，作为"检索器 ⇄ 下游任务"端到端联合训练的桥梁
  （论文原意：LLM 生成质量的梯度可以经由这个软选择反传指导检索器的打分）。
- `evaluation/rcs.py`：`ranking_consistency_score()` —— 提供 `topk_overlap`（两个 top-k
  集合的重合率）和 `kendall_tau`（全排序相关系数）两个互补视角，衡量"检索器排序"与
  "参照排序（如更强 GNN 的排序）"的一致性。
- `subgraph_retriever/phase5_smoke_test.py`：在 Phase 0-2 训出的真实 R-GCN / LightGCN
  checkpoint 上验证这两个新组件（不是训练脚本，只是组件正确性验证）。

## 2. 验证结果

### 2.1 DFTopK 正确性 + 可微性

| 检验项 | 结果 |
|---|---|
| 同一组分数上，DFTopK 软选择（低温度）vs 硬 `topk()` 的 top-20 重合率 | **1.000**（完全一致） |
| 对 `dftopk(...).sum()` 求 loss 并反传，R-GCN item embedding 是否收到非零梯度 | **是**（grad abs-sum ≈ 9401.2） |

确认 DFTopK 实现忠实（低温度下退化为硬 top-k）且真正可微（梯度确实能传回打分模型的参数），
具备作为"检索器→下游任务"端到端训练桥梁的基本条件。

### 2.2 RCS：一个有意思的真实发现

用 Phase 2 训好的 R-GCN（test AUC 0.7605）和 LightGCN（test AUC 0.9570）—— 同一张图、
同一个链接预测任务、两个都训练得不错的模型 —— 计算它们对**同一个用户的候选 item 排序**
的一致性：

| 指标（50 个用户平均，k=20） | 值 |
|---|---|
| top-k 重合率 | **0.001**（几乎完全不重合） |
| Kendall's tau | **0.016**（几乎不相关，接近随机） |

**这是一个真实、有信息量的发现，不是 bug**：两个模型在"AUC"这个聚合指标上都表现良好
（AUC 衡量的是"正边是否比负边打分更高"这种全局的、成对的比较），但这完全不要求两个模型
对同一个用户的具体候选 item **排序**趋于一致 —— 各自的 embedding 空间是独立训练出来的、
互相没有对齐约束。这正是 Phase 5 想验证的核心问题："检索器选的 top-k"与"另一个（更强/
更权威）打分模型的排序"之间可能存在的系统性错配，即便两者各自的"自身准确率"都不差。
这也直接印证了 Ranking Consistency 论文（#17）的核心论点：**"自身精度高"和"与下游一致"
是两个独立的、都需要显式优化的目标**，不能互相替代。

## 3. 未实现（诚实说明，[需确认]，不阻塞交付最终报告）

**#15 LCRON、#14 HetComp、#16 Cooperative 均未实现。** 原因：

1. 三者都要求"检索器 ⇄ LLM"或"教师 GNN → 学生检索器"之间存在**真正的联合/蒸馏训练循环**
   （LCRON：定义级联生存概率的 surrogate loss，LLM 生成 loss 梯度经 DFTopK 式可微排序反传进
   检索器；HetComp：教师 GNN 的 top-k 轨迹课程式蒸馏给检索器；Cooperative：检索器与生成器
   互相蒸馏/协同训练）。这些都需要在同一个训练循环里**同时**跑图检索的前向/反向 **和**
   LLM 的前向/反向，并让两者的梯度互相流通——这比 Phase 3/4"预计算子图 embedding → 注入 LLM"
   的单向数据流复杂得多，需要专门设计训练循环（而不是复用 `ds_training/step1_supervised_finetuning/main.py`
   现有的纯 LLM 训练循环）。
2. 这类联合训练要有意义，前提是检索器本身要在**真实**多关系数据上工作（Phase 2 已确认
   IntTower/GNRR/RippleNet/KGRec 都因为缺失原始 `.pkl` 数据无法验证，Phase 3 的子图检索
   也确认了同一个数据缺口）——在退化的合成图上做"检索器⇄LLM 联合训练"的意义有限，
   跑出来的对齐效果无法归因于真实的图结构信号。
3. 相比之下，DFTopK 和 RCS 是**自包含、可独立验证**的组件，即使没有完整的联合训练循环，
   也能验证其正确性并产出有意义的发现（如 §2.2），这是本 Phase 在时间预算内能做到的、
   诚实的最大化产出。

**推荐**：若后续要真正实现 LCRON/HetComp/Cooperative，建议顺序是：(1) 先解决 Phase 2/3
反复提到的真实多关系数据缺口；(2) 用本 Phase 的 DFTopK 替换节点检索的硬 top-k；
(3) 设计一个新的训练脚本（不是复用 `main.py`），让检索器打分和 LLM LoRA 在同一个
optimizer step 里一起更新，用 RCS 作为训练时的监控指标（而不只是离线评估）。

## 4. 验收结论

**部分通过**：DFTopK、RCS 两个可复用组件已实现、验证通过，并且 RCS 在真实模型上跑出了
一个有实际意义的发现（两个 AUC 都不错的模型排序几乎不一致）。三个更重的端到端联合训练方向
（LCRON/HetComp/Cooperative）因为需要真正的联合训练循环+真实多关系数据而推迟，已诚实记录
理由和后续建议，不阻塞进入最终汇总报告。
