# 引用 G-Refer 的论文分析

**被引论文**: G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation (WWW'25 Oral, arXiv:2502.12586)

**数据来源**: Semantic Scholar 引用接口，共检索到 **52 篇**引用论文（Google Scholar 显示的 67 篇额外包含未被 S2 索引的预印本/学位论文/二次引用）。原始数据见 `raw_citations.json`，摘要见 `abstracts_dump.txt`。

**分析目标**: 从 52 篇中筛选出**真正基于 G-Refer 继续做实验（把它当基线对比 / 直接扩展其框架）**的论文，并用一句话总结其改进点。

---

## 一、真正把 G-Refer 当实验基线 / 直接扩展的论文（核心结果）

> 判定标准：在实验表格里把 G-Refer 作为 baseline 对比，或直接在 G-Refer 框架上做扩展。经全文核实，只有 **2 篇**满足。

| # | 论文 | 相对 G-Refer 的改进（一句话） |
|---|------|------------------------------|
| 1 | **MMP-Refer: Multimodal Path Retrieval-augmented LLMs for Explainable Recommendation** (arXiv:2604.03666) | 把 G-Refer 从纯文本/图扩展到**多模态**：用 RQ-VAE 联合残差量化融合图文模态，用**规则启发式路径检索**替代 G-Refer 可学习边权（路径采集快 3–6 倍、更稳定），再经 MoE Adapter 做软提示指令微调；在 Amazon Baby/Sports/Clothing 上 BERTScore-F1 比 G-Refer 提升 0.56%–4.01%，BARTScore 提升约 10%–13%。 |
| 2 | **Retrieval-Augmented Recommendation Explanation Generation with Hierarchical Aggregation** (arXiv:2507.09188) | 直接针对 G-Refer 两大痛点：**(1) Profile Deviation**——用**分层聚合(HA)多级摘要**编码全部评论，取代 G-Refer 的随机评论采样；**(2) 高检索开销**——用伪文档向量相似检索取代 O(N²) Dijkstra 图遍历，推理从 4+ 分钟降到 <1 秒；BERTScore-Precision 在 Amazon/Yelp/Google 上比 G-Refer 高 7.46%–26.8%。 |

---

## 二、同一任务（可解释推荐）但 G-Refer 仅作相关工作、未进实验对比

> 这些论文和 G-Refer 做同一件事（LLM 可解释推荐），会在 related work 引用它，但**没有把它放进实验对比表**。属"同领域竞品/替代范式"，不算"基于 G-Refer 继续做实验"。其自身核心思路附上以供参考。

| # | 论文 | 自身核心做法（与 G-Refer 的差异） |
|---|------|-----------------------------------|
| 11 | Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation (2604.03724) | 换范式：把解释生成改成**从评论抽取的原子陈述里做 top-k 排序**，从构造上消除幻觉，并提出 StaR benchmark。 |
| 21 | On the Factual Consistency of Text-based Explainable Recommendation Models (2512.24366) | 提出**事实一致性**评测框架，指出包括检索增强在内的 SOTA 解释虽 BERTScore 高但语句级 precision 仅 4%–33%，普遍在"编造"。 |
| 9 | Curr-RLCER: Curriculum RL for Coherence Explainable Recommendation (2604.05341) | 用**课程强化学习 + 一致性奖励**对齐"预测评分"与"解释"，纯文本、不引入协同信号。 |
| 6 | LLMEKERec: Explainable Recommendation via KG Path Reasoning with LLMs (ICASSP'26) | 让 LLM 基于**知识图谱多跳路径**生成解释；MovieLens/Amazon-Book/Yelp 上 NDCG 提升至多 6.3%。 |
| 0 | Contrastive Learning for Explanation Ranking (CLER) (ML Journal 2026) | 后验解释**排序**，用 NT-BXent 对比损失学 user/item/explanation 表征；在 EXTRA benchmark(Amazon/TripAdvisor/Yelp) 评测。 |
| 12 | LLM-guided Data Distillation for Explainable Recommender System (J. Supercomputing 2026) | 用**数据蒸馏**把 LLM 解释能力迁移到小模型(SLM)，降算力同时保解释质量。 |

---

## 三、相关但不同任务（RAG推荐 / GraphRAG / 图学习等，仅顺带引用）

> 这些只是把 G-Refer 当作"LLM+图/检索增强推荐"的一个引用例子，任务不同，未做相关实验对比。

- **RAG 推荐（准确率向，非解释）**: ARAG 多智能体RAG推荐(2506.21931)、RETURN 检索增强净化鲁棒推荐(2504.02458)、RevBrowse/PrefRAG 评论驱动RAG推荐(2509.00698)、MoE KG-RAG 多智能体推荐(2605.28175)、DCGL 双通道图学习推荐(2605.07314)、KICR 课程推荐(DSA'25)、TRAIL(2602.04225)。
- **通用 GraphRAG / KG 推理**: GraphRAG-R1(2507.23581)、Graph-constrained Reasoning(2410.13080)、Structure Guided RAG(2604.22843)、HybRAG(app16052244)、KG-SMILE(2509.03626)、TKG-Thinker(2602.05818)、Faico(3770854)、Reasoning over User Preferences 会话推荐(2411.14459)、Grounded by Experience 医疗(2511.13293)。
- **综述**: Trustworthy Recommendation(2606.00540)、A Survey on Generative Recommendation(2510.27157)、Explainability of LLMs(2510.17256)、GNN for Collaborative Filtering Survey、Medical Multimodal RAG Survey、Integrating LLM & KG for AGI。
- **其他领域（图/LLM 但与推荐无关）**: 天气预报 MIGN(2509.20911)、分子优化 InversionGNN(2503.01488)、LoRA 变体(SeLoRA 2506.16787 / Circular Conv 2407.19342 / 2506.09400)、对抗攻击 TAG(2603.21155)/CTEA、电路发现 IBCircuit(2602.22581)、GNN解释 GSPELL(2508.07117)、多模态图 GraphGPT-o(2502.11925)、图推理奖励(2503.00845)、Graph Foundation Models(3711896)、错误信息 MisBench(2505.21608)、重排序可靠性(2508.18444)、微信小游戏LTV(2506.11037)、MemWeaver(2510.07713)、序列推荐 MLLMRec-R1(2603.06243)/Archetype(2606.11023)、多模态推荐 MSCF-net、超图治疗推荐、FS-KEN(ijcai512) 等。

---

## 结论

在 52 篇引用论文中：
- **仅 2 篇真正"基于 G-Refer 继续做实验"**（MMP-Refer、Hierarchical Aggregation），二者都把 G-Refer 当作最强基线并明确报告对其的提升。
  - MMP-Refer 的方向是**加多模态 + 规则化路径检索**；
  - Hierarchical Aggregation 的方向是**解决 G-Refer 的评论采样偏差与检索效率**（4分钟→<1秒）。
- 另有约 6 篇做同一任务（可解释推荐）但只把 G-Refer 当相关工作，未进实验对比。
- 其余约 44 篇为不同任务的顺带引用或综述。
