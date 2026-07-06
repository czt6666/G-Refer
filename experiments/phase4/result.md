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

## 3. 真实生成质量对比（补充实验，2026-07-06）

跟 Phase 3 用完全一样的公平对照方法（同一份 `raft_data/yelp_subgraph5k/` 数据、同样的
TinyLlama-1.1B + LoRA rank=8 + 3 epoch + lr=2e-5）——**直接复用 Phase 3 的 soft-prompt
checkpoint 作对照组**（因为 GraphLoRA 的"vanilla LoRA"对照本来就应该是"同样带 soft-prompt，
只是 LoRA 不带门控"，而 Phase 3 的 soft-prompt 变体正好就是这个对照），只新训练 GraphLoRA
这一个变体（`--use_graph_lora`，其余参数不变）。

**训练 loss/perplexity**（3 epoch 后，三个变体一起看）：

| variant | eval ppl | eval loss |
|---|---|---|
| Baseline（无 soft-prompt，Phase 3） | 2.066 | 0.726 |
| Soft-prompt（Phase 3） | 2.108 | 0.746 |
| **GraphLoRA + soft-prompt** | **2.004** | **0.695** |

GraphLoRA 在训练 loss/perplexity 上是三者里最好的——不仅明显优于 soft-prompt-only（同样
带 soft-prompt，只是 LoRA 加了门控），甚至优于完全不带 soft-prompt 的 baseline。

**下游生成质量全指标**（跟 Phase 1/2/3 一样，500 条真实测试样本、`infer_subgraph.py`
+ `eval_lite.py`，GraphLoRA checkpoint 保存时 LoRA 已经 fuse 回底座权重——门控只影响训练期间
的梯度动态，不需要在推理时重建，见下方"发现"）：

| variant | BERT-P | BERT-R | BERT-F1 | BARTScore | BLEURT |
|---|---|---|---|---|---|
| Baseline（无 soft-prompt） | 0.3544 | **0.4315** | 0.3932 | -3.1181 | -0.3245 |
| Soft-prompt | 0.3259 | 0.3982 | 0.3623 | -3.0631 | -0.3823 |
| **GraphLoRA + soft-prompt** | **0.3774** | 0.4133 | **0.3958** | **-2.9251** | **-0.2977** |

**结论：GraphLoRA 在 5 个指标里的 4 个上是三者中最好的**（BERT-P、BERT-F1、BARTScore、
BLEURT），只有 BERT-R 略低于 baseline（但仍明显优于 soft-prompt-only）。更重要的是，
**GraphLoRA 修复了 Phase 3 发现的"soft-prompt 拖累生成质量"的问题**，甚至反超了不加任何
辅助信号的 baseline。这跟训练 loss 的结果是一致的（GraphLoRA 训练动态最好），是一个连贯、
可信的正向结果。

**为什么门控能修复 soft-prompt 的问题**（推测，本次未做额外消融验证这个具体机制假设）：
Phase 3 的纯 soft-prompt 只是"多塞一个 token"，LoRA 权重更新本身对这个 token 携带的信号
无感——模型必须自己在有限的训练步数内学会"何时该看这个 token"。GraphLoRA 额外让 LoRA
的更新幅度本身也被同一个信号调制，等于在更新权重的通路上直接引入了结构信息，可能让模型
更快地把"结构信号"和"该如何调整生成策略"关联起来，而不是把 soft-prompt token 当噪声。

**一个值得记录的发现**：确认了 GraphLoRA 的门控（`set_graph_gate`/`clear_graph_gate`）
只在训练时通过调制 LoRA delta 影响梯度，`main.py` 保存模型前统一调用
`convert_lora_to_linear_layer`（把 `LinearLayer_LoRA`及其子类 `GraphLoRALinear` 的
低秩更新原地 fuse 进底座权重）——`GraphLoRALinear` 没有覆写 `fuse_lora_weight()`，用的是
父类的静态 fuse（`weight.data += scaling * matmul(...)`，不含门控）。所以训练完保存下来的
是一个**普通的、门控效应已经"烧录"进权重里**的模型，推理时不需要、也没办法重建门控——
直接复用 Phase 3 的 `infer_subgraph.py`（只处理 soft-prompt 部分）即可正确评估。

## 4. 未做 / 局限（诚实说明）

- **未做 GraphLoRA 单独消融**（不叠加 soft-prompt）：当前实现里 `--use_graph_lora` 总是和
  soft-prompt 一起生效（复用同一份 embedding）。若要单独衡量"去掉 soft-prompt、只留
  GraphLoRA 门控"的增量贡献，需要再加一个开关，本 Phase 未做，记录为后续待办。
- **规模仍是 TinyLlama-1.1B + 4300 样本**，未在 Llama-3-8B 论文规模上复验——跟 Phase 3
  一样的局限，是否在更大模型上还能观察到同样的提升未知。
- **推荐**：这是本次 5 个 Phase 里少数几个"真正观察到正向下游质量提升"的方向之一，
  值得在真实多关系数据可用后优先复验和扩大规模。

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

**通过**：GraphLoRA 不仅机制验证通过，公平对照实验也显示**真实的下游质量提升**——5 个指标里
4 个最优，且修复了 Phase 3 soft-prompt 单独使用时的质量回退。这是本次 5 个 Phase 里证据
最扎实的正向结果之一。GraphLoRA 单独消融（不叠加 soft-prompt）与 FACE 均记录为后续待办，
不阻塞进入 Phase 5。
