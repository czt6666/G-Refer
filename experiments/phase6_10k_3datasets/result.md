# Phase 6 — 10k×3数据集：Power-Link+GraphLoRA 组合方案 vs baseline

对比对象：**baseline**（原始 raft_data 真实 prompt + vanilla LoRA）vs **Power-Link+GraphLoRA
组合方案**（prompt 里的路径段替换成 Power-Link 在合成图上重新检索的版本 + GraphLoRA 门控 +
子图 soft-prompt），在 yelp / amazon / google 三个数据集上各用 8500 条真实训练数据、500 条
eval、1000 条 test（train/eval/test 都是从 `raft_data/<dataset>/train.json` 里筛出恰好有
真实子图 embedding 的 pair，保证两个变体训练/评估用的是完全相同的 (uid,iid) 集合，唯一变量
是 prompt 内容和微调方式）。TinyLlama-1.1B，LoRA rank=8，3 epoch，batch_size=2。

完整运行命令、日志路径见同目录 `RUN_LOG.md`。

## 1. 踩坑：max_seq_len 截断 bug（发现 + 修复过程）

第一次跑（`max_seq_len=256`）：google 的 graphlora 变体训练完之后质量断崖式下跌
（BERT-F1 只有 0.14，抽查发现 63.5% 的生成结果在复读 prompt 里的路径描述文字，而不是
生成解释）。排查后发现：`tokenizer(prompt+chosen, max_length=256, truncation=True)`
默认从右边截断，而 prompt（含检索到的路径文本）经常本身就超过 256 token——抽样 300 条
发现 85%-96% 的训练样本，排在 prompt 后面的 "chosen"（真实解释，约 39 token）在
256 token 预算已经被 prompt 用满的情况下被整个截断掉，训练信号本质上变成了"预测路径
文本的下一个 token"而不是"生成解释"。

修复：`max_seq_len` 从 256 改成 2048（抽样 300 条时观测到的最大长度是 1815）。6 个
checkpoint（yelp/amazon/google × baseline/graphlora）全部删除重跑。

修复后又在准备 Amazon-books 全量数据实验时发现：**2048 对全量数据仍然不够**——对
全部 94841 条真实数据完整扫描（不是抽样），真实最大长度是 2265 token。这提醒了一件事：
**本 Phase（10k 子集）用的 2048 是否也遗漏了个别超长样本？** 补充扫描了 yelp/amazon/google
各自 8500 条子集的真实 token 长度：

| 数据集 | 变体 | 超过 2048 的条数 | 最大长度 |
|---|---|---|---|
| yelp | baseline | 0/8500 | 1376 |
| yelp | powerlink | 0/8500 | 1252 |
| amazon | baseline | **1/8500** | **2072** |
| amazon | powerlink | 0/8500 | 1336 |
| google | baseline | 0/8500 | 1138 |
| google | powerlink | 0/8500 | 945 |

只有 amazon baseline 的 8500 条里有 1 条（0.012%）超出 2048，且只截掉 explanation
最后 24 个 token 里的一小部分，不是整段丢失。**这个量级的、单条样本的轻微截断，不足以
推翻本 Phase 的结论，未重新跑**——诚实记录在此。

## 2. 完整结果（500-1000 条真实测试样本，全部指标真实跑出）

| 数据集 | 变体 | BERT-P | BERT-R | BERT-F1 | BARTScore | BLEURT | USR |
|---|---|---|---|---|---|---|---|
| yelp | **baseline** | **0.3787** | **0.4412** | **0.4102** | **-2.8932** | **-0.2892** | 1.0 |
| yelp | Power-Link+GraphLoRA | 0.2968 | 0.3130 | 0.3052 | -3.2099 | -0.3987 | 1.0 |
| amazon | **baseline** | **0.3974** | **0.4440** | **0.4213** | **-2.4572** | **-0.1828** | 1.0 |
| amazon | Power-Link+GraphLoRA | 0.3700 | 0.3696 | 0.3704 | -2.6077 | -0.2492 | 1.0 |
| google | baseline | 0.4057 | 0.4477 | 0.4271 | -2.7483 | -0.2924 | 1.0 |
| google | **Power-Link+GraphLoRA** | **0.4102** | **0.4817** | **0.4461** | **-2.5849** | **-0.2704** | 1.0 |

训练 eval loss（供参照，越低越好）：

| 数据集 | baseline | Power-Link+GraphLoRA |
|---|---|---|
| yelp | 0.225 | 0.198 |
| amazon | 0.209 | 0.181 |
| google | 0.172 | 0.154 |

**一个反直觉但真实的现象**：三个数据集上，Power-Link+GraphLoRA 组合方案的训练 eval loss
**全部比 baseline 更低**（模型更"自信"地拟合训练分布），但下游生成质量（BERT-F1 等）却是
**yelp、amazon 两个数据集上 baseline 更好，只有 google 上组合方案更好**。也就是说训练
loss 低不代表生成质量一定更好——这是本次实验里另一个值得记录的诚实发现。

## 3. 诚实分析：为什么组合方案在 2/3 数据集上不如 baseline？

**这不是对 GraphLoRA 机制本身的纯净测试**——本实验的 "Power-Link+GraphLoRA" 是把两个
变量**同时**换掉：

1. **检索方式**：baseline 用的是 G-Refer 论文原始真实检索流程产出的 prompt；
   组合方案用的是我们在**合成图**（`path_retriever/build_synthetic_graph.py`，buys/likes/
   bought_by 三种关系用同一批边填充，语义退化——见 memory
   `path-retriever-env-and-data-gaps`）上重新跑 Power-Link 检索出来的 prompt。
2. **微调方式**：baseline 是 vanilla LoRA；组合方案是 GraphLoRA 门控 + 子图 soft-prompt。

而 **Phase 4** 曾经做过一个更干净的对照实验（`experiments/phase4/result.md`）：固定
检索/prompt 不变（都用同一份 soft-prompt 数据），只改变"LoRA 是否带 GraphLoRA 门控"这一个
变量——那次 GraphLoRA 在 5 个指标里 4 个跑赢 vanilla LoRA。两次实验放在一起看，可以推出
一个合理的假设：

> **GraphLoRA 微调机制本身大概率是有效的（Phase 4 的干净对照支持这一点），但本次
> Power-Link 在合成图上检索出的 prompt 质量，可能比 G-Refer 论文原始真实检索的 prompt
> 差一截——检索质量的损失盖过了 GraphLoRA 微调带来的增益，在 yelp/amazon 上表现为组合
> 方案整体不如 baseline。**

**google 是例外**：组合方案在 google 上全指标反超 baseline。这一点没有做进一步消融，
不确定具体原因，列几个可能但未验证的猜测：google 数据集的原始 prompt 可能因为某种
原因（比如更长、包含更多噪声性描述）反而不如 Power-Link 精简过的检索结果；也可能是
三个数据集各自的子图 embedding 覆盖率/图结构特性不同（google 合成图规模最大，见
`experiments/FINAL_REPORT.md` 的节点数统计），导致 GraphLoRA 门控在这张图上学到的调制
更有效。**这只是假设，没有验证，如实标注为待研究方向。**

## 4. 结论（10k 子集实验）

- 不建议把"Power-Link 检索 + GraphLoRA 微调"作为一个不做区分、直接替换 baseline 的
  打包方案——至少在这三个数据集、这个训练规模下，2/3 数据集上不如原始 baseline。
- **GraphLoRA 微调机制本身**（不叠加检索方式变化时，见 Phase 4）仍然有真实的正向证据，
  值得继续采纳；但如果同时要换检索方式，应该先确认新检索方式本身的质量不比原始检索差
  （本次合成图检索大概率是短板），而不是把两者捆绑判断。
- google 数据集上组合方案反而更好，这个反例说明结论有数据集依赖性，不能一概而论——
  没有做进一步消融来确认具体原因，如实标注为未解决的问题。

## 5. Amazon-books 全量数据三方法对比（baseline / Power-Link / GraphLoRA，各自独立消融）

针对上面 §3 提出的"检索方式和微调方式两个变量同时变化，无法判断谁是主因"这个局限，
在 Amazon-books **全量真实数据**（94841 条训练、1000 条 eval、3000 条 test，不是 10k
子集）上，把两个变量拆开，做三个独立方法的对比（而不是"组合方案 vs baseline"两个点）：

- **baseline**：原始真实 prompt + vanilla LoRA。
- **Power-Link**：prompt 换成 Power-Link 重新检索的版本，**其余都跟 baseline 一样**（vanilla
  LoRA，不叠加 GraphLoRA）——单独消融检索方式的影响。
- **GraphLoRA**：**跟 baseline 用完全相同的原始 prompt**（不换检索），只是 LoRA 换成
  GraphLoRA 门控 + 子图 soft-prompt——单独消融微调方式的影响。

TinyLlama-1.1B，LoRA rank=8，3 epoch，batch_size=2，`max_seq_len=2304`（全量数据实测最大
token 长度 2265，见 `RUN_LOG.md` 的踩坑记录）。三个方法因为 host 内存限制无法同时跑
tokenize 阶段，全部严格排队单卡跑完（每个训练约 26-46 小时，详见 `RUN_LOG.md` 的逐小时
进度记录）。

### 5.1 完整结果（3000 条真实测试样本）

| 方法 | BERT-P | BERT-R | BERT-F1 | BARTScore | BLEURT | USR | eval ppl（训练） |
|---|---|---|---|---|---|---|---|
| **baseline** | 0.3986 | **0.4656** | 0.4326 | -2.4091 | **-0.1013** | 1.0 | 1.0778 |
| **Power-Link**（仅换检索） | **0.4003** | 0.4647 | **0.4330** | **-2.4466** | -0.1097 | 1.0 | 1.0720 |
| GraphLoRA（仅换微调，原始输出） | 0.2540 | 0.2873 | 0.2712 | -3.1538 | -0.4800 | 1.0 | 1.0780 |
| GraphLoRA（去除 `## (` 格式前缀后） | 0.2572 | 0.3077 | 0.2829 | -2.8575 | -0.3564 | — | — |

### 5.2 诚实结论：这次证据很干净，问题出在 GraphLoRA，不是 Power-Link

跟 10k 子集实验（§1-4，两个变量绑在一起换）不同，全量数据这次把两个变量拆开了，结论
清楚很多：

1. **Power-Link 单独换检索，下游生成质量跟 baseline 几乎没有差别**（BERT-F1 0.4330 vs
   0.4326，BARTScore/BLEURT 也在同一量级）——这跟 Phase 2（backbone 换血，AUC 差很多但
   生成质量打平）的模式一致：**检索端的改动，在这条 pipeline 上似乎很难传导到最终生成
   质量**，无论是换 backbone 还是换路径搜索算法都是如此。这也从侧面支持了 §3 的猜测
   ("Power-Link 在合成图上检索出的 prompt 质量比原始真实检索差一截"）**其实站不住脚**——
   如果 Power-Link 检索本身就有明显质量损失，单独消融应该能看到下降，但实际没有。
2. **GraphLoRA 单独换微调方式，全量数据下质量断崖式下跌**——即使排除了一个真实的输出
   格式漂移问题（73.8% 的生成结果输出 `## (` 而不是训练模板约定的 `### `，见 §2.3 的
   踩坑记录），去除格式干扰后 BERT-F1 也只有 0.2829，仍然远低于 baseline 的 0.4326。
   **这次终于可以确定：10k 子集实验里"组合方案在 yelp/amazon 上不如 baseline"，主要
   问题出在 GraphLoRA 微调机制本身，不是检索方式的锅**。
3. 这跟 **Phase 4**（更小规模、soft-prompt 数据只有 5000/74212 pair 覆盖率）里 GraphLoRA
   赢过 vanilla LoRA 的结论明显矛盾。放在一起看，指向一个更明确的假设：**GraphLoRA 的
   门控机制在数据量较小（Phase 4 量级）时可能是有效的正则化/引导信号，但在数据量足够大
   （本次 94841 条全量数据）、vanilla LoRA 本身已经能学得很好的情况下，额外的门控参数
   和 soft-prompt 反而带来训练不稳定性**——具体表现就是这次观察到的输出格式漂移
   （`## (` 替代 `### `）和内容质量下降。这个假设现在有了更干净的证据支持（检索方式已经
   排除嫌疑），但**具体是门控机制的哪个环节导致格式漂移，仍未做进一步消融，如实记录为
   开放问题**。

### 5.3 最终建议

- **不建议在真实生产数据规模下使用 GraphLoRA 门控机制**，至少在 TinyLlama-1.1B +
  94841 条真实数据这个配置下，它带来了真实、可复现、排除了格式解析干扰后依然显著的
  质量下降。
- **Power-Link 路径检索方式可以放心替换 baseline**（不会拖累生成质量，Phase 1 的
  500 样本实验里甚至还有优势），但换 backbone/换检索算法这类"检索端"改动，不要指望
  它能带来生成质量的提升——至少在当前这条依赖合成图的 pipeline 上，检索端的改动很难
  传导到最终生成质量。
- 如果要继续探索 GraphLoRA 方向，下一步应该先定位清楚"数据量增大后不稳定性从哪来"，
  而不是直接在生产规模上使用它。
