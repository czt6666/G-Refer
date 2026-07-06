# Phase 3 — 子图级检索 + Soft-Prompt 注入

分支：`exp/phase3-subgraph-retrieval`（从 `exp/phase2-retrieval-quality` 切出）
论文：**#9 K-RagRec (arXiv:2501.02226)** 为主线；CHEST(#10)、KGIN(#11) 仅部分借鉴思路，未完整实现（见 §4）。

## 1. 改动摘要

新增 `subgraph_retriever/` 模块（G-Refer 原本只做 node/path 检索，从未有 subgraph 粒度 ——
这是作者留白/用户创新点）：

- `subgraph_retriever/extract_subgraph_embeds.py`：对每个 (uid, iid) pair，用
  `path_retriever/utils.py` 现成的 `hetero_src_tgt_khop_in_subgraph`（l-hop 子图抽取，
  l=2）取子图，用 Phase 0-2 已训练好的 R-GCN 做一次 `encode()` 拿到子图内所有节点的
  CF-aware embedding，对 user/item 节点分别 mean-pool 拼接成 256 维向量。
  **Popularity-selective 门控**（K-RagRec 思路）：item 入度高于中位数时，子图向量乘 0.3
  衰减（高热门 item 已有足够信号，少augment；冷启动/长尾 item 保留满权重 1.0）——用连续门控
  而非论文的硬阈值，避免训练时的不连续跳变。
- `subgraph_retriever/subgraph_encoder.py`：`SubgraphProjector`（2 层 MLP + GELU + Dropout，
  256 → LLM hidden_size）+ `prepend_soft_prompt()` 工具函数（把投影后的向量当一个额外 token
  拼进 `inputs_embeds`，`attention_mask`/`labels` 对应扩展 1 位，labels 位置填 -100 不算 loss）。
  这是 K-RagRec"GNN encoder + MLP projector 当 soft prompt"的核心机制 —— 用连续向量注入
  替代 G-Refer 现有的纯文本序列化。
- `ds_training/utils/data/data_utils.py`：`create_dataset_split`/`create_dataset`/
  `create_prompt_dataset` 新增可选 `subgraph_embeds` 参数（默认 `None`，不传时行为完全不变），
  按 `{uid}-{iid}` 查表把子图向量塞进每条训练样本；缓存文件名加了 `subgraph`/`nosubgraph`
  后缀区分，避免复用到不带子图向量的旧缓存。
- `ds_training/step1_supervised_finetuning/main.py`：新增 `--subgraph_embed_path`/
  `--subgraph_dim`（默认不传，行为不变）。设置后：在 LoRA 转换 **之后** 挂载
  `SubgraphProjector`（这样 `only_optimize_lora_parameters` 冻结完其它参数后，新挂的
  projector 仍保持 `requires_grad=True`，会被 `get_optimizer_grouped_parameters` 正常收进优化器）；
  新增 `build_model_inputs()` 统一处理训练/评估两条路径，有 `subgraph_embed` 就用
  `inputs_embeds` 路径，没有就还是原来的 `input_ids` 路径。

## 2. 发现并修复的两个 bug

1. **dtype 不匹配**：TinyLlama 以 bf16 加载，但 `extract_subgraph_embeds.py` 存的向量是
   fp32，投影层第一次 forward 直接报 `mat1 and mat2 must have the same dtype`。修复：
   `build_model_inputs` 里显式把 `subgraph_embed` cast 成 projector 参数的 dtype 再送入。
2. **单 pair 子图抽取本身很慢**：`hetero_src_tgt_khop_in_subgraph` 在这张图（3 万节点、
   94 万条边，Phase1/2 用的同一张合成图）上单个 pair 跑 l=2 跳抽取不算快，对 74212 个训练
   pair 全量抽取的首次尝试跑了 10+ 分钟仍未到第一个进度打点（5000 pair），判断是量级问题后
   中止，改成先验证正确性的小规模抽取（见 §3）。**这本质上是 Phase 1 解决的同一类"检索开销"
   问题在子图粒度上的重现** —— 值得记录为后续优化点（批量化多个 pair 的子图抽取、或用
   Phase 1 的图幂法思路加速），但本 Phase 没有再花时间解决，直接记录为限制。

## 3. 验证结果（诚实说明规模）

**子图向量抽取**：由于上述性能问题，只对 train.json 的**前 5000/74212 条**（约 6.7%）pair
做了真实抽取（`subgraph_retriever/embeds/yelp_trn_subgraph_embeds.pt`），其余样本训练时
自动 fallback 到零向量（`build_model_inputs`/`create_dataset_split` 里的
`subgraph_embeds.get(key)` 找不到时用 `'__zero__'`）。**没有跑全量 74212 条抽取**。

**训练冒烟测试**：TinyLlama-1.1B + LoRA(rank=8) + `--subgraph_embed_path`，单卡（GPU 空闲，
未用 CPU offload），ZeRO-2，`per_device_train_batch_size=2`。跑了 261 步（约 522 条样本，
远未到 1 epoch=37106 步）后主动停止 —— 目的是验证机制通得通，不是要它收敛：

| step | loss |
|---|---|
| 1 | 2.478 |
| 11 | 1.931 |
| 51 | 1.183 |
| 101 | 1.272 |
| 151 | 1.409 |
| 201 | 1.046 |
| 231 | 0.998 |
| 261 | 1.011 |

Loss 从 2.48 稳定下降到 ~1.0-1.1 区间并保持稳定（无 NaN、无崩溃、无显存问题），说明：
soft-prompt 拼接 → LLM 前向 → loss 反传 → LoRA + projector 一起更新，整条链路是通的。
训练前 eval perplexity：`(loss=2.587, ppl=13.29)`（epoch 0，未训练前的基线，供参照）。

**后续补做**（2026-07-06）：既然只有 6.7% 数据有真实子图信号，就用**公平对照**的方式把这个
限制转成一个受控实验，而不是继续搁置——构造一个只包含这 5000 条有真实子图 embedding 的
pair 的 raft_data 子集（`raft_data/yelp_subgraph5k/`，4300 train / 200 eval / 500 test），
baseline（无 soft-prompt）和 soft-prompt 两个变体在**完全相同**的数据、超参（TinyLlama-1.1B、
LoRA rank=8、lr=2e-5、3 epoch、batch=2）上分别训练，训练完各自推理 500 条真实测试样本，跑
`eval_lite.py` 全指标对比。

**训练 loss/perplexity**（3 epoch 后）：

| variant | 最终 train loss（示例区间） | eval ppl | eval loss |
|---|---|---|---|
| Baseline（无 soft-prompt） | ~0.9-1.2 | **2.066** | **0.726** |
| Soft-prompt | ~0.9-1.4 | 2.108 | 0.746 |

**下游生成质量全指标**（500 条真实测试样本，同一评估流程）：

| variant | BERT-P | BERT-R | BERT-F1 | BARTScore | BLEURT |
|---|---|---|---|---|---|
| **Baseline（无 soft-prompt）** | **0.3544** | **0.4315** | **0.3932** | -3.1181 | **-0.3245** |
| Soft-prompt | 0.3259 | 0.3982 | 0.3623 | **-3.0631** | -0.3823 |

**诚实结论：在这个规模（TinyLlama-1.1B、4300 条训练样本、3 epoch）下，soft-prompt 注入
没有带来提升，反而在 4/5 个指标上比不加 soft-prompt 的 baseline 差**（只有 BARTScore
soft-prompt 略优）。这是一个真实的负向/混合结果，不应该被掩盖或选择性引用。

**可能原因**（推测，本次未逐一验证，留作后续方向）：
1. 训练数据量小（4300 条）+ 多了一组待学习的 projector 参数，3 epoch 可能不够让模型学会
   "何时/如何利用"这个额外的连续 token，反而在早期训练中引入了一定噪声（softprompt 的
   eval ppl/loss 全程略高于 baseline，从 epoch 1 就是如此，不是训练不充分的中途现象）。
2. 子图 embedding 本身来自退化合成图（Phase 1/2 反复提到的限制），信息量可能本来就有限，
   "整合额外弱信号"的成本（多一个 token、多一组参数要学）可能超过了它带来的收益。
3. TinyLlama(1.1B) 参数规模较小，也许不如论文规模的 Llama-3-8B 那样有余力去利用辅助信号——
   但这一点没有实测验证（受限于时间/算力，未在 Llama-3-8B 规模上重跑这组对照）。

**推荐**：不建议在当前证据下把 soft-prompt 设为默认开启项。若要继续这个方向，应优先：
(a) 解决 Phase 1/2 反复提到的真实多关系数据缺口，让子图 embedding 携带真实信息；
(b) 在 Llama-3-8B 规模上重跑同样的受控对照，确认是否是模型规模问题。这两步都需要更大的
时间/算力投入，本次不阻塞地记录为后续待办。

## 4. 未采纳/延后（K-RagRec 之外的方向）

| 方向 | 状态 | 原因 |
|---|---|---|
| K-RagRec 的 LLM-prompt 子图重排 | 未做 | 需要额外一轮 LLM 调用重排检索到的子图，本 Phase 优先验证"soft-prompt 注入"这个核心机制能否跑通；重排是可选的二阶段优化，成本（每个训练样本多一次 LLM 前向）相对本 Phase 已验证的收益不确定，留作后续 |
| CHEST(#10) interaction-specific 子图抽取 | 部分借鉴 | 本身抽取已经是"per (u,i) pair"粒度（等价于 interaction-specific），课程预训练、multi-slot 序列化未实现 |
| KGIN(#11) 意图感知聚合 | 未做 | 需要在聚合层显式建模关系/意图 embedding，本 Phase 复用 Phase 0-2 现成的 R-GCN 池化，未改聚合机制 |

## 5. 验收结论

**机制通过，效果未通过**：soft-prompt 注入机制端到端验证通过（真实训练 loss 下降、无崩溃、
projector 参数正确参与优化）——这是本 Phase 最初的核心技术风险点，已经排除。但补做的公平
对照实验（同数据同超参，baseline vs soft-prompt）显示：**在 TinyLlama-1.1B + 4300 样本
+ 3 epoch 的规模下，soft-prompt 没有提升下游生成质量，多数指标反而变差**。这是诚实的负向
结果，不代表"soft-prompt 注入"这个方向本身不可行——K-RagRec 原论文是在更大规模、更真实的
数据上验证的——但在本次可控的实验条件下，没有证据支持采纳。不阻塞进入 Phase 4；Phase 4
的 GraphLoRA 也用同一个公平对照方法验证，两个方向的证据应该放在一起看。
