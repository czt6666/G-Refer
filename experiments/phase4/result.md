# Phase 4 — GraphLoRA 结构感知微调

分支：`exp/phase4-graphlora`（从 `exp/phase3-subgraph-retrieval` 切出）
论文：**#18 GraphLoRA (arXiv:2606.07526)** 已实现（简化版）；**#19 FACE** 未实现（见 §4）。

## 1. 改动摘要

- `ds_training/utils/module/lora.py` 新增：
  - `GraphLoRALinear(LinearLayer_LoRA)`：LoRA 低秩更新 `ΔW = (dropout(x) @ B @ A) * scaling`
    额外乘一个**逐样本**门控 `gate`（来自 Phase 3 的同一份子图 embedding），而不是像原版
    LoRA 那样对全体样本共用同一个静态 `ΔW`。**简化说明**：论文 GraphLoRA 的机制是把图消息
    传递直接嵌入低秩通路本身（重新参数化 A/B 的构造方式）；这里用的是更轻量的"每样本门控
    现有 LoRA 通路"，只做了"让 CF 结构影响 LoRA 更新幅度"这一层，没有改变 A/B 的参数化方式本身。
  - `GraphLoRAGate`：`subgraph_dim → 1` 的线性层 + `1 + tanh(·)`，**初始化为 0 权重/0 偏置**，
    使得 `tanh(0)=0`，门控初值严格等于 1.0 —— 训练刚开始时 GraphLoRA 和普通 LoRA 完全等价，
    随训练才学出逐样本的差异化调制（安全的默认值，不会一开始就扰乱训练）。
  - `set_graph_gate`/`clear_graph_gate`：模块级共享上下文，把当前 batch 的门控广播给模型内部
    所有被替换成 `GraphLoRALinear` 的层（HF transformer 内部调用链很深，没法把每层
    forward 签名都改一遍传参数，所以用这个轻量的"设置→调用→清除"模式）。
- `ds_training/step1_supervised_finetuning/main.py`：新增 `--use_graph_lora`（依赖
  `--subgraph_embed_path`，复用 Phase 3 的子图 embedding，不需要额外抽取）。`GraphLoRAGate`
  同样在 `only_optimize_lora_parameters()` **之后**挂载，保持可训练。`build_model_inputs()`
  扩展为同时处理"计算 soft-prompt 并 prepend"和"计算 GraphLoRA 门控并 set_graph_gate"两件事，
  调用方在 `model(**model_inputs)` 后调 `clear_graph_gate()`。

**当前实现是 GraphLoRA 叠加在 Phase 3 soft-prompt 之上**（因为两者复用同一份
`--subgraph_embed_path`），不是独立消融；要单独测"只有 GraphLoRA、没有 soft-prompt"需要再加
一个开关把 soft-prompt 单独关掉，本 Phase 未做（时间/收益取舍，见 §3）。

## 2. 训练冒烟测试

TinyLlama-1.1B + LoRA(rank=8) + `--use_graph_lora`，单卡（GPU1，与 Phase 3 冒烟测试所用
GPU0 隔开跑，避免互相干扰），ZeRO-2，同样跑了 ~191 步后主动停止：

| step | loss (GraphLoRA) | loss (Phase 3 纯 soft-prompt，对照) |
|---|---|---|
| 1 | 2.445 | 2.478 |
| 21 | 1.334 | — |
| 51 | 1.177 | 1.183 |
| 101 | 0.955 | 1.272 |
| 191 | 1.124 | — (跑到 261 步，见 phase3 result.md) |

训练前 eval perplexity：`(loss=2.549, ppl=12.79)`（对照 Phase 3 的 `(2.587, 13.29)`，
量级相近，符合两者都从同一个预训练 checkpoint 出发的预期）。

**结论**：GraphLoRA 门控机制的前向 + 反向 + 参数更新链路完整跑通（gate 参数、LoRA 参数一起
被优化器更新，无崩溃、无 NaN）。**两条 loss 曲线在这个规模（<200 步，同一份数据）下没有
统计上有意义的差异**——这符合预期：GraphLoRA 门控初值为 1.0（等价于普通 LoRA），
只有训练得足够久、门控真正学出跟样本相关的调制后，才可能观察到与 vanilla LoRA 的行为差异。
本次冒烟测试的目的是验证机制正确，不是验证效果提升，因此不构成"GraphLoRA 是否比 LoRA 更好"
的证据，不应误读。

## 3. 未做 / 局限（诚实说明，[需确认]，按推荐默认继续）

- **未做长程训练 + 下游质量对比**：与 Phase 3 一样，受限于时间预算和 Phase 3 already 提到的
  子图抽取规模限制（只有 5000/74212 pair 有真实子图信号），跑一次有统计意义的
  vanilla-LoRA vs GraphLoRA 对比（同样的数据、同样的步数、跑到收敛、`eval_lite.py` 全指标）
  需要更大的时间/算力投入，本 Phase 未做。
- **未做 GraphLoRA 单独消融**（不叠加 soft-prompt）：当前实现里 `--use_graph_lora` 总是和
  Phase 3 的 soft-prompt 一起生效（复用同一份 embedding）。若要单独衡量 GraphLoRA 的增量贡献，
  需要加一个"只要 GraphLoRA、不要 soft-prompt"的开关，本 Phase 未做，记录为后续待办。
- **推荐**：GraphLoRA 的机制验证已经通过，值得在真正意义上的全量训练里跟 vanilla LoRA
  做一次严肃对比（在 Phase 3 的规模问题解决之后一起做，成本上可以复用同一批数据/embedding）。

## 4. FACE (#19) 未实现

FACE（向量量化 + codebook，把检索到的节点 CF embedding 离散成 token 注入 LLM）本 Phase
**未实现**。原因：
1. 需要训练一个 VQ-VAE 风格的 codebook（straight-through estimator、codebook 坍塌/更新策略等），
   比 Phase 3 的连续 soft-prompt 复杂得多，是一项独立的、有相当工作量的子任务。
2. Phase 3 的连续 soft-prompt 已经是"把检索到的 CF 信号注入 LLM"这个核心思路的一个可工作实现；
   FACE 相对它的增量价值（离散化带来的可解释性/正则化收益）在当前数据规模限制下难以验证。
3. 任务说明里 FACE 本身标注为"可选"，与 Phase 3 soft-prompt 是"二选一或并用做消融"的关系——
   鉴于 soft-prompt 已验证可行，且 GraphLoRA 已经是本 Phase 的主线产出，FACE 留作后续工作。

## 5. 验收结论

**部分通过**：GraphLoRA 机制端到端验证通过（真实训练、loss 正常下降、gate 参数正确参与优化），
这是本 Phase 的核心技术风险点，已排除。规模化对比与 FACE 均记录为后续待办，不阻塞进入 Phase 5。
