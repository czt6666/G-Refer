# 运行记录：10k×3数据集实验 + Amazon-books 全量三方法实验

本文件记录所有启动命令、日志路径、GPU 分配，供复现和排查。结果解读见同目录 `result.md`。

## 一、10k×3数据集实验（yelp/amazon/google，各 8500/500/1000 train/eval/test）

### 数据准备

```bash
# 为 amazon/google 构建合成图（yelp 之前已构建）
cd path_retriever
python build_synthetic_graph.py --dataset amazon
python build_synthetic_graph.py --dataset google

# 为每个数据集抽取 10000 对 (uid,iid) 的子图 embedding
cd ../subgraph_retriever
python extract_subgraph_embeds.py --dataset yelp --split trn --limit 10000
python extract_subgraph_embeds.py --dataset amazon --split trn --limit 10000
python extract_subgraph_embeds.py --dataset google --split trn --limit 10000

# 构建 10k 公平对照子集（raft_data/{dataset}_10k/），train/eval/test = 8500/500/1000
# （从 raft_data/{dataset}/train.json 里筛选出恰好有真实子图 embedding 的 pair，shuffle seed=1234）

# 为每个数据集生成 Power-Link 检索版 prompt（train+eval+test 全部重新生成）
cd ../path_retriever
python generate_powerlink_prompts.py --dataset yelp --split_dir ../raft_data/yelp_10k --output_dir ../raft_data/yelp_10k_powerlink
python generate_powerlink_prompts.py --dataset amazon --split_dir ../raft_data/amazon_10k --output_dir ../raft_data/amazon_10k_powerlink
python generate_powerlink_prompts.py --dataset google --split_dir ../raft_data/google_10k --output_dir ../raft_data/google_10k_powerlink
```

日志：`/tmp/.../scratchpad/extract_{yelp,amazon,google}_10k.log`，`genprompt_{yelp,amazon,google}.log`

**中途踩的坑**：第一次跑 `generate_powerlink_prompts.py` 时 3 个数据集并行跑，每个进程默认开了约
172 个 torch 线程，3 个一起跑导致host严重过载（load average 120+/80核），单条耗时从预期的
~0.7s 飙到 ~17s/条。修复：脚本里加了 `torch.set_num_threads(20)`，3 个并行时 load 降到
~65-70，恢复到 ~0.5-0.9s/条正常速度。

### 训练：max_seq_len 踩坑与修复

**第一次跑（`max_seq_len=256`，错误）**：google_graphlora 跑完后质量断崖式下跌
（BERT-F1 仅 0.14，63.5% 的生成结果在复读 prompt 里的 path 文本而不是生成解释）。
排查后发现：`tokenizer(prompt+chosen, max_length=256, truncation=True)` 默认从右边截断，
而 prompt 本身（含检索到的路径文本）经常就超过 256 token——实测 85%-96% 的训练样本，
target explanation（"chosen"，约 39 token）在 prompt 已经用满 256 token 预算的情况下被
整个截断掉，训练信号本质上是"预测路径文本的下一个 token"而不是"生成解释"。

修复：把 `max_seq_len` 从 256 改成 2048（实测 amazon baseline 最长的 prompt+chosen
到 1815 token，2048 留了安全余量）。第一次跑的 6 个 checkpoint 全部删除重跑。

```bash
# 修改文件：ds_training/step1_supervised_finetuning/training_scripts/single_node/run_10k_ab.sh
#   --max_seq_len 256  ->  --max_seq_len 2048
```

### 训练启动命令（第二次，正确版本）

```bash
cd ds_training/step1_supervised_finetuning

# baseline: 原始 raft_data prompt，vanilla LoRA
bash training_scripts/single_node/run_10k_ab.sh yelp   baseline  0 29520 ../../ckpts/yelp_10k_baseline
bash training_scripts/single_node/run_10k_ab.sh amazon baseline  2 29522 ../../ckpts/amazon_10k_baseline
bash training_scripts/single_node/run_10k_ab.sh google baseline  0 29524 ../../ckpts/google_10k_baseline

# graphlora: Power-Link 检索版 prompt + GraphLoRA 门控 + soft-prompt
bash training_scripts/single_node/run_10k_ab.sh yelp   graphlora 1 29521 ../../ckpts/yelp_10k_graphlora
bash training_scripts/single_node/run_10k_ab.sh amazon graphlora 3 29523 ../../ckpts/amazon_10k_graphlora
bash training_scripts/single_node/run_10k_ab.sh google graphlora 2 29525 ../../ckpts/google_10k_graphlora
```

训练日志：`ckpts/{dataset}_10k_{baseline,graphlora}/training.log`
launcher 日志：`/tmp/.../scratchpad/train_{dataset}_{baseline,graphlora}_v2.log`

### 推理 + 评估命令

```bash
cd ds_inference

# baseline 用标准 infer.py
python infer.py --model_path ../ckpts/{dataset}_10k_baseline --streategy Parallel \
    --batch_size 8 --max_tokens 256 --save_dir ../convert_files/phase6_10k/{dataset}_baseline
# (amazon 在 GPU 显存紧张时用 --batch_size 4 重跑过一次，同样跑完 1000/1000)

# graphlora 用 infer_subgraph.py（soft-prompt 机制需要专门推理脚本）
python infer_subgraph.py --model_path ../ckpts/{dataset}_10k_graphlora \
    --subgraph_embed_path ../subgraph_retriever/embeds/{dataset}_trn_subgraph_embeds.pt \
    --subgraph_dim 256 --batch_size 4 --max_tokens 256 \
    --save_dir ../convert_files/phase6_10k/{dataset}_graphlora

cd ..
python evaluation/eval_lite.py --dataset {dataset} --gen_path convert_files/phase6_10k/{dataset}_{variant}/gen_datas.jsonl
```

推理日志：`/tmp/.../scratchpad/infer_{dataset}_{baseline,graphlora}.log`（部分因为 `infer.py`
末尾 `fire.Fire(main(args=args))` 的已知无害 bug 报 exit code 2，但检查 `gen_datas.jsonl`
行数确认数据完整写出）
评估日志：`/tmp/.../scratchpad/eval_{dataset}_{baseline,graphlora}.log`

**注意**：GPU 共享跑推理时出现过一次真实 OOM（amazon baseline 用 batch_size=8 跟同 GPU 上的
google 训练抢显存），重跑时改用 `batch_size=4` 解决，不是 bug。

---

## 二、Amazon-books 全量数据三方法实验（baseline / power-link / graphlora）

与上面 10k 子集实验不同，这次用 `raft_data/amazon/train.json` 的**全部 94841 条真实数据**
（留出 1000 条做 eval，test 复用原有的 3000 条），三个方法互相独立对照：

- **baseline**：原始真实 prompt，vanilla LoRA。无依赖，立即可跑。
- **graphlora**：**同样的原始真实 prompt**（不用 Power-Link）+ subgraph soft-prompt +
  GraphLoRA 门控，用现有的 `amazon_trn_subgraph_embeds.pt`（只覆盖 10000/94841 条，
  其余 fallback 零向量）。这次特意**不叠加 Power-Link**，单独消融 GraphLoRA 微调机制本身
  （对齐 Phase 4 的设计）。
- **power-link**：prompt 换成 Power-Link 在合成图上重新检索的版本，vanilla LoRA（不用
  GraphLoRA 门控）。单独消融检索方式的影响。**需要先把全量 98841 条 pair 的 Power-Link
  prompt 生成完才能跑**，预计耗时 16-20+ 小时（CPU 检索，不占 GPU，可以先跑着）。

### 数据准备

```bash
mkdir -p raft_data/amazon_full
head -n 94841 raft_data/amazon/train.json > raft_data/amazon_full/train.json
tail -n 1000 raft_data/amazon/train.json > raft_data/amazon_full/eval.json
cp raft_data/amazon/test.json raft_data/amazon_full/test.json

# Power-Link 全量重新生成（CPU，后台跑，预计 16-20+ 小时）
cd path_retriever
python generate_powerlink_prompts.py --dataset amazon \
    --split_dir ../raft_data/amazon_full \
    --output_dir ../raft_data/amazon_full_powerlink \
    --log_every 1000
```

日志：`/tmp/.../scratchpad/genprompt_amazon_full.log`

### 训练启动命令

新脚本：`ds_training/step1_supervised_finetuning/training_scripts/single_node/run_amazon_full_ab.sh`
（`<variant:baseline|graphlora|powerlink> <gpu_id> <port> [output_dir]`）

```bash
cd ds_training/step1_supervised_finetuning

# 第一次尝试：GPU1、GPU3 同时启动 baseline + graphlora
bash training_scripts/single_node/run_amazon_full_ab.sh baseline  1 29530 ../../ckpts/amazon_full_baseline
bash training_scripts/single_node/run_amazon_full_ab.sh graphlora 3 29531 ../../ckpts/amazon_full_graphlora

# power-link：等 Power-Link 全量 prompt 生成完 + 有 GPU 空出来再跑（GPU0/GPU2 当时被
# google_10k_baseline/graphlora 占用，预计跑完 10k 实验后释放）
# bash training_scripts/single_node/run_amazon_full_ab.sh powerlink <gpu_id> <port> ../../ckpts/amazon_full_powerlink
```

**踩坑 1：host RAM（不是 GPU 显存）是真正的瓶颈**——`baseline` 跑到
"Creating dataset jsonfile for train_phase=1 size=94841" 这一步（把 94841 条样本
tokenize 成张量）时被系统 OOM killer 杀掉（`dmesg` 确认：
`Out of memory: Killed process ... anon-rss:46828332kB`，单进程就用了 46.8GB
host 内存）。原因是这台机器总共 124GB 内存，`baseline` 和 `graphlora` 同时做这一步
tokenize+cache 时叠加起来超过了可用内存（当时 `graphlora` 自己就已经涨到 59-62GB）。

**教训 1**：这一规模（9万+条真实样本）的 tokenize 阶段内存开销很大，
`baseline`/`graphlora`/`powerlink` 三个方法**不能同时跑数据准备阶段**，必须错开
（等前一个进入稳定的训练循环、内存回落之后再启动下一个），即使它们分别用不同 GPU 也一样——
瓶颈在 CPU 内存，不是显存。查了 `ds_training/utils/data/data_utils.py` 的缓存逻辑
（`create_prompt_dataset`）：只有 `local_rank<=0` 负责实际 tokenize+`torch.save`，
但**所有 rank 最后都会各自 `torch.load()` 一份完整数据集到自己的内存**——这意味着
**用多卡（data-parallel）跑同一个任务，内存是按卡数近似倍增的，不是分摊**，在这个
host-RAM 瓶颈的场景下，多卡训练只会更容易 OOM，不是加速的免费手段。所以这一批实验
全部保持单卡跑，`baseline`/`graphlora`/`powerlink` 排队错开。

**踩坑 2：max_seq_len=2048 仍然不够**——之前只抽样了 300 条估算百分位数（最大 1815），
后来对**全部 94841 条**训练数据做了完整扫描，发现真实最大值是 **2265 token**（第 71248 行）。
这意味着 2048 会截断这一条（和另外至少一条，扫描时 tokenizer 报过一次
`2072 > 2048` 的警告）真实样本的 explanation。**修复**：`run_amazon_full_ab.sh` 的
`max_seq_len` 从 2048 改成 **2304**（略高于确认过的最大值，留安全余量）。
`amazon_full_graphlora` 当时已经跑了约 5741/142263 步（~4%），发现问题后立刻整个重跑，
避免用错误配置训练更久造成更大浪费。

**补充核查：之前跑完的 10k 子集实验有没有同样问题？** 对 yelp/amazon/google 三个数据集
各自的 `_10k`/`_10k_powerlink` train.json（8500 条）逐条扫描真实 token 长度：

| 数据集 | 变体 | 超过 2048 的条数 | 最大长度 |
|---|---|---|---|
| yelp | baseline | 0/8500 | 1376 |
| yelp | powerlink | 0/8500 | 1252 |
| amazon | baseline | **1/8500** | **2072** |
| amazon | powerlink | 0/8500 | 1336 |
| google | baseline | 0/8500 | 1138 |
| google | powerlink | 0/8500 | 945 |

**只有 amazon baseline 的 8500 条里有 1 条（0.012%）超出 2048**，超出 24 个 token
（不到"chosen"explanation 39 个 token 均长的三分之二），且只截掉 explanation 的最后
一小截，不是整段丢失。**结论**：这个量级的、单条样本的轻微截断，不足以推翻已经完成的
10k 实验的结论（yelp/amazon/google baseline vs graphlora 对比），不重新跑；但记录在此，
保持诚实透明。

**踩坑 3：max_seq_len 改成 2304 后，重新同时启动 baseline+graphlora 又 OOM 了一次**
（这次杀掉的是 `graphlora`，`anon-rss:72225704kB`）——进一步证实"两个方法不能同时做
tokenize 阶段"这个结论在 2304 长度下依然成立（更长的序列意味着更大的内存峰值，2304 比
2048 更容易撞到内存上限）。**教训 3**：这一批全量规模的实验，`baseline`/`graphlora`/
`powerlink` 三者必须严格排队跑（一个完全进入稳定训练循环之后，才能启动下一个），
不能像 10k 子集实验那样 4 个一起上——已经在两次尝试里都验证过，不再重复冒险。

训练日志：`ckpts/amazon_full_{baseline,graphlora,powerlink}/training.log`
launcher 日志：`/tmp/.../scratchpad/train_amazon_full_{baseline,graphlora}.log`

**规模提醒**：94841 条训练样本、batch_size=2、3 epoch = 142263 步。参照 10k 实验的每步耗时
（约 0.7-0.9s/步，2048 token 序列长度），预计单次训练要 **28-35+ 小时**。三个方法错开顺序跑
（不能并行做数据准备），含 power-link 要先等全量检索生成，总耗时会显著更长，需要有耐心持续等待。

### 进度追踪（持续更新）

| 时间 | baseline | graphlora | powerlink 生成 | host 内存 |
|---|---|---|---|---|
| 第二次重启后 | 5741/142263 步 (~4%) | 未跑（严格排队，等 baseline 跑完再启动） | 32310/94841 行 (34%) | 稳定 41GB available |
| +1小时后 | 11101/142263 步 (~7.8%) | 未跑 | 40367/94841 行 (42.5%) | 稳定 41GB available，无新 OOM |
| +2小时后 | 15801/142263 步 (~11.1%) | 未跑 | 47659/94841 行 (50.2%，仅 train，eval/test 还没开始) | 稳定 41GB available，无新 OOM |
| +3小时后 | 20471/142263 步 (~14.4%) | 未跑 | 54990/94841 行 (58%，仅 train) | 稳定 40GB available，无新 OOM |
| +4小时后 | 25131/142263 步 (~17.7%) | 未跑（严格排队，即使内存有余量也不提前启动） | 62204/94841 行 (65.6%，仅 train) | 内存余量回升到 71GB available（之前被杀掉的 graphlora 残留内存彻底释放），无新 OOM |
| +5小时后 | 30061/142263 步 (~21.1%) | 未跑 | 69868/94841 行 (73.7%，仅 train) | 稳定 70GB available，无新 OOM |
| +6小时后 | 34971/142263 步 (~24.6%) | 未跑 | 77486/94841 行 (81.7%，仅 train，接近跑完 train 分片) | 稳定 69GB available，无新 OOM |
| +7小时后 | 39721/142263 步 (~27.9%) | 未跑 | 84840/94841 行 (89.5%，仅 train，train 分片即将跑完) | 稳定 69GB available，无新 OOM |
| +8小时后 | 44571/142263 步 (~31.3%) | 未跑 | 92002/94841 行 (97%，train 分片即将完成) | 稳定 68GB available，无新 OOM |
| epoch 1 完成 | epoch_1 checkpoint 保存完成，ppl=1.1117，eval loss=0.1058 | 未跑 | train 94841/94841 完成、eval 1000/1000 完成、test 606/3000 进行中 | 稳定 68GB available，无新 OOM |
| powerlink 全部完成 | 48982/142263 步 (~34.4%，epoch 1 进行中) | 未跑 | **train/eval/test 全部完成**（94841/1000/3000，全程 0 empty_paths、0 out_of_bounds，总耗时约 50605+558+1474 ≈ 52637s ≈ 14.6 小时） | 稳定 68GB available，无新 OOM |

**当前严格排队顺序**：baseline（进行中）→ graphlora → powerlink（prompt 生成完成后再训练）。
不再尝试并行跑两个全量规模的方法。

**用户确认**：内存瓶颈的根本原因是 `data_utils.py` 把整份 tokenize 好的数据集完整加载进
每个进程的内存（不是内存映射/共享），要真正支持多任务并行需要改这部分缓存机制，有引入新
bug 的风险，且改造期间可能要重跑当前已经跑了 8+ 小时的 baseline。用户选择维持顺序跑
（零风险默认选项），不做这个改造。

| 时间 | baseline | 备注 |
|---|---|---|
| powerlink 全部完成之后 +若干小时 | 88332/142263 步 (~62.1%) | 内存稳定 60GB available，无新 OOM |
| +1小时后 | 94792/142263 步 (~66.6%) | 内存稳定 58GB available，无新 OOM |
| epoch 2 完成 | epoch_2 checkpoint 保存完成，ppl=1.0851，eval loss=0.0816 | 内存稳定 57GB available，无新 OOM |
| +1小时后（epoch 3 进行中） | 99723/142263 步 (~70.1%) | 内存稳定 57GB available，无新 OOM |
| +若干小时后 | 114153/142263 步 (~80.2%) | 内存稳定 53GB available，无新 OOM |
| +1小时后 | 119063/142263 步 (~83.7%) | 内存稳定 51GB available，无新 OOM |
| **baseline 完全跑完** | 3 epoch 全部完成，最终 ppl=1.0778，eval loss=0.0749，checkpoint 保存完成、进程正常退出 | 内存完全释放（113GB available） |

**baseline 跑完后立刻做的事**：
1. 启动 baseline 推理（`ds_inference/infer.py`，GPU0，`convert_files/phase6_amazon_full/baseline/`）
2. 内存完全清零后，立刻启动 graphlora 训练（GPU3，`ckpts/amazon_full_graphlora/`，
   `run_amazon_full_ab.sh graphlora 3 29534`）——严格排队规则下，baseline 完全退出、
   内存归零后才启动下一个，不再冒险

**baseline 完整结果（3000 条真实测试样本）**：

| metric | 10k 子集 baseline（对照） | **全量 baseline** |
|---|---|---|
| BERT-P | 0.3974 | **0.3986** |
| BERT-R | 0.4440 | **0.4656** |
| BERT-F1 | 0.4213 | **0.4326** |
| BARTScore | -2.4572 | **-2.4091** |
| BLEURT | -0.1828 | **-0.1013** |
| USR | 1.0 | 1.0 |

全量数据（94841 条训练样本，是 10k 子集的约 11 倍）训练出来的 baseline，全部指标都比
10k 子集版本更好——符合预期（更多真实训练数据，模型学得更好）。

**graphlora 训练速度**：初期看起来每 30 分钟只推进了 130 步，一度怀疑严重变慢；用进程
真实启动时间核实后，实际是 1421 步 / 27.6 分钟 ≈ 3090 步/小时，比 baseline
（约 4700-4900 步/小时）慢 ~35%——这是合理的开销（GraphLoRA 每个 forward 都要多算
门控 + soft-prompt 融合），不是异常。内存稳定 69GB available，无 OOM。按此速度全部
142263 步预计要 ~46 小时（比 baseline 的 ~28-30 小时长）。

| 时间 | graphlora | host 内存 |
|---|---|---|
| +1小时后 | 5671/142263 步 (~4%)，实测约 4250 步/小时（比之前估的 3090 更快） | 稳定 69GB available，无新 OOM |
| +2小时后 | 9941/142263 步 (~7%)，稳定约 4270 步/小时 | 稳定 69GB available，无新 OOM |
| +3小时后 | 14131/142263 步 (~9.9%)，稳定约 4190 步/小时 | 稳定 69GB available，无新 OOM |
| +4小时后 | 18301/142263 步 (~12.9%)，稳定约 4170 步/小时 | 稳定 69GB available，无新 OOM |
| +5小时后 | 22461/142263 步 (~15.8%)，稳定约 4160 步/小时 | 稳定 69GB available，无新 OOM |
| +6小时后 | 26631/142263 步 (~18.7%)，稳定约 4170 步/小时 | 稳定 75GB available，无新 OOM |
| +7小时后 | 30781/142263 步 (~21.6%)，稳定约 4150 步/小时 | 稳定 74GB available，无新 OOM |
| +8小时后 | 34941/142263 步 (~24.6%)，稳定约 4160 步/小时 | 稳定 73GB available，无新 OOM |
| +9小时后 | 39101/142263 步 (~27.5%)，稳定约 4160 步/小时 | 稳定 73GB available，无新 OOM |
| +10小时后 | 43281/142263 步 (~30.4%)，稳定约 4180 步/小时 | 稳定 71GB available，无新 OOM |
| +11小时后 | 47421/142263 步 (~33.3%，正好跑完 epoch 1，等 eval 算完存 checkpoint) | 稳定 71GB available，无新 OOM |
| epoch 1 完成 | epoch_1 checkpoint 保存完成，ppl=1.1093，eval loss=0.1038 | 稳定 70GB available，无新 OOM |
| +1小时后（epoch 2 进行中） | 51622/142263 步 (~36.3%) | 稳定 70GB available，无新 OOM |
| +2小时后 | 55712/142263 步 (~39.2%) | 稳定 69GB available，无新 OOM |
| +3小时后 | 59732/142263 步 (~42.0%) | 稳定 68GB available，无新 OOM |
| +4小时后 | 63852/142263 步 (~44.9%) | 稳定 67GB available，无新 OOM |
| epoch 2 完成 | epoch_2 checkpoint 保存完成，ppl=1.0843，eval loss=0.0809；94873/142263 步 (~66.7%，epoch 3 进行中) | 稳定 60GB available，无新 OOM |
| +1小时后 | 99073/142263 步 (~69.7%) | 稳定 59GB available，无新 OOM |
| +2小时后 | 103303/142263 步 (~72.6%) | 稳定 58GB available，无新 OOM |
| +3小时后 | 107513/142263 步 (~75.6%) | 稳定 57GB available，无新 OOM |
| +4小时后 | 111723/142263 步 (~78.5%) | 稳定 56GB available，无新 OOM |
| +5小时后 | 115953/142263 步 (~81.5%) | 稳定 55GB available，无新 OOM |
| +6小时后 | 120183/142263 步 (~84.5%) | 稳定 54GB available，无新 OOM |
| +7小时后 | 124413/142263 步 (~87.5%) | 稳定 53GB available，无新 OOM |
| +8小时后 | 128633/142263 步 (~90.4%) | 稳定 52GB available，无新 OOM |
| +9小时后 | 132853/142263 步 (~93.4%) | 稳定 51GB available，无新 OOM |
| +10小时后 | 137083/142263 步 (~96.4%，即将跑完) | 稳定 50GB available，无新 OOM |
| +11小时后 | 141303/142263 步 (~99.3%，还剩约960步) | 稳定 50GB available，无新 OOM |
| **graphlora 完全跑完** | 3 epoch 全部完成，最终 ppl=1.0780，eval loss=0.0751（跟 baseline 的 1.0778/0.0749 几乎一模一样），checkpoint 保存完成、进程正常退出 | 内存完全释放（113GB available） |

**graphlora 跑完后立刻做的事**：
1. 启动 graphlora 推理（`ds_inference/infer_subgraph.py`，GPU0，
   `convert_files/phase6_amazon_full/graphlora/`）
2. 内存完全清零后，立刻启动 powerlink 训练（GPU3，`ckpts/amazon_full_powerlink/`，
   `run_amazon_full_ab.sh powerlink 3 29535`）——三个方法排队顺序至此全部启动完毕：
   baseline（已跑完+评估）→ graphlora（已跑完，推理中）→ powerlink（刚启动）

**踩坑 4：`evaluation/eval_lite.py` 的 `load()` 函数有个真实 bug**（3000 条里踩中 1 条，
0.033%）——检查条件用的是 `"###" in output_str`（不带空格），但实际切分用的是
`.split("### ")`（带空格）；graphlora 在第 1748 条测试样本上生成到 `max_tokens=256`
上限时，正好在 "###" 后面被截断、没有跟着的空格，导致 `"###" in output_str` 为真但
`split("### ")` 只切出 1 段，`[1]` 越界崩溃。**修复**：改成先 `split("### ")`，
根据结果长度判断要不要取 `[1]`，不再用容易不一致的双重条件判断。已修复并重跑评估。

**重要发现：全量规模 graphlora 的输出格式漂移**——修复崩溃 bug 重跑评估后，
`amazon_full_graphlora` 的 BERT-F1 只有 0.2712，远低于 baseline 的 0.4326。排查发现：
**2214/3000（73.8%）的生成结果以 `## (` 开头，而不是训练数据模板约定的 `### `**
（baseline 和之前所有 10k 规模的 graphlora 跑法都正确输出 `### `，只有这次全量规模的
checkpoint 出现这个漂移）。检查这些 `## (` 开头的文本，发现 1679/2214（75.8%）后面的
`(` 从未在 400 字符内闭合——更像是格式标记漂移导致的伪影，而不是刻意的括号说明；
`## (` 之后的实际解释内容本身读起来是合理、切题的（比如 "Readers who enjoy historical
fiction and are intrigued by..."）。

为了公平对比，写了一个归一化脚本剥离这层 `## (`/`### ` 前缀（`re.sub(r'^#+\s*', '', s)`
+ 去掉不闭合的开头括号），重新跑了一遍评估：

| 版本 | BERT-P | BERT-R | BERT-F1 | BARTScore | BLEURT |
|---|---|---|---|---|---|
| 原始（含 `## (` 前缀） | 0.2540 | 0.2873 | 0.2712 | -3.1538 | -0.4800 |
| 去除格式前缀后 | 0.2572 | 0.3077 | 0.2829 | -2.8575 | -0.3564 |
| baseline（对照） | 0.3986 | 0.4656 | **0.4326** | **-2.4091** | **-0.1013** |

**诚实结论：格式漂移只解释了一小部分差距**——去掉 `## (` 前缀后 BERT-F1 只从 0.271
提升到 0.283，跟 baseline 的 0.433 仍然有巨大差距（0.15 左右）。**这是一个真实的、
全量规模下 GraphLoRA 明显不如 baseline 的负向结果**，不能归咎于格式解析问题。这跟
Phase 4（4300 条小规模）GraphLoRA 明显赢过 vanilla LoRA 的结论、以及 Phase 6 10k
规模（8500 条子集）yelp/amazon 上 GraphLoRA 输、google 上赢的混合结果，共同指向一个
值得关注的推测（**未做进一步消融验证**）：**GraphLoRA 的门控机制在小规模数据上可能是
有效的正则化/引导信号，但在数据量足够大、vanilla LoRA 本身就能学得很好的情况下，
额外的门控参数和 soft-prompt 可能反而引入训练不稳定性（表现为这次的输出格式漂移），
增益随数据规模增大而递减甚至转为负面**。这个假设需要更多受控实验才能确认，如实记录
为开放问题，不夸大也不掩盖。

**踩坑 5：尝试用 `torch.load(mmap=True)` 让多卡训练安全共享数据集缓存，实测无效**——
用户希望 powerlink 能用 4 卡并行跑（`data_utils.py` 目前的设计是每个 rank 各自完整
`torch.load()` 一份 tokenized 数据集，4 卡等于 4 倍内存，之前已经验证过这会直接 OOM）。
尝试了 PyTorch 2.1+ 的 `mmap=True` 参数（理论上能让多进程共享同一份文件的 page cache，
不必各自复制），但实测：单进程 `mmap=True` 加载后 RSS 仍然高达 42GB（跟不用 mmap 的
45.5GB 几乎没差别），因为这份缓存是 9.4 万个独立的小张量对象（每条训练样本一个），
反序列化 Python 对象结构本身就要读入大部分数据，mmap 的"惰性分页"优势对这种数据形状
基本失效（mmap 更适合"少数几个大张量"，比如模型权重）。测试第二个并发加载时内存一度
逼近极限（只剩 18GB 可用），主动杀掉测试进程保住了正在跑的 powerlink 训练。
**结论**：安全的多卡并行需要把数据加载换成能跨进程真正共享的格式（比如 HuggingFace
`datasets` 库的 Arrow 格式），是一次独立的、有一定工作量和风险的改造，不是简单加个参数
能解决的。已跟用户说明，继续保持 powerlink 单卡跑。

| 时间 | powerlink | host 内存 |
|---|---|---|
| mmap 实验后 | 6701/142263 步 (~4.7%) | 恢复稳定 45GB available，无新 OOM |
| +1小时后 | 11681/142263 步 (~8.2%) | 稳定 45GB available，无新 OOM |
| +2小时后 | 16241/142263 步 (~11.4%) | 稳定 45GB available，无新 OOM |
| +3小时后 | 20821/142263 步 (~14.6%) | 稳定 45GB available，无新 OOM |
| +4小时后 | 25391/142263 步 (~17.9%) | 73GB available（进程 RSS 波动，page cache 回收正常现象），无新 OOM |
| +5小时后 | 30001/142263 步 (~21.1%) | 稳定 71GB available，无新 OOM |
| +6小时后 | 34591/142263 步 (~24.3%) | 稳定 70GB available，无新 OOM |
| +7小时后 | 39161/142263 步 (~27.5%) | 稳定 68GB available，无新 OOM |
| +8小时后 | 43711/142263 步 (~30.7%) | 稳定 67GB available，无新 OOM |
| epoch 1 完成 | epoch_1 checkpoint 保存完成，ppl=1.1014，eval loss=0.0966 | 稳定 65GB available，无新 OOM |
| +1小时后（epoch 2 进行中） | 52052/142263 步 (~36.6%) | 稳定 67GB available，无新 OOM |
| +2小时后 | 56602/142263 步 (~39.8%) | 稳定 66GB available，无新 OOM |
| +3小时后 | 61182/142263 步 (~43.0%) | 稳定 65GB available，无新 OOM |
| +4小时后 | 65852/142263 步 (~46.3%) | 稳定 64GB available，无新 OOM |
| +5小时后 | 70442/142263 步 (~49.5%) | 稳定 62GB available，无新 OOM |
| +6小时后 | 75042/142263 步 (~52.8%) | 稳定 62GB available，无新 OOM |
| +7小时后 | 79612/142263 步 (~56.0%) | 稳定 62GB available，无新 OOM |
| +8小时后 | 84212/142263 步 (~59.2%) | 稳定 60GB available，无新 OOM |
| +9小时后 | 88812/142263 步 (~62.4%) | 稳定 59GB available，无新 OOM |
| +10小时后 | 93402/142263 步 (~65.7%) | 稳定 57GB available，无新 OOM |
| epoch 2 完成 | epoch_2 checkpoint 保存完成，ppl=1.0785，eval loss=0.0756；94853/142263 步 (~66.7%，epoch 3 进行中) | 稳定 57GB available，无新 OOM |
| +1小时后 | 99443/142263 步 (~69.9%) | 稳定 56GB available，无新 OOM |
| +2小时后 | 104013/142263 步 (~73.1%) | 稳定 56GB available，无新 OOM |
| +3小时后 | 108603/142263 步 (~76.3%) | 稳定 55GB available，无新 OOM |
| +4小时后 | 113293/142263 步 (~79.6%) | 稳定 53GB available，无新 OOM |
| +5小时后 | 118153/142263 步 (~83.1%) | 稳定 52GB available，无新 OOM |
| +6小时后 | 123103/142263 步 (~86.5%) | 稳定 51GB available，无新 OOM |
| +7小时后 | 127693/142263 步 (~89.8%) | 稳定 51GB available，无新 OOM |
| +8小时后 | 132283/142263 步 (~93.0%，即将跑完) | 稳定 50GB available，无新 OOM |
| +8.5小时后 | 134623/142263 步 (~94.6%) | 稳定 49GB available，无新 OOM |
| **完全跑完** | 3 epoch 全部完成，最终 ppl=1.0721，eval loss=0.0696，checkpoint 保存完成、进程正常退出 | 内存完全释放（111GB available） |

**powerlink 推理+评估结果**：3000 条真实测试样本，推理完成（`gen_datas.jsonl` 3000 行，
`fire.Fire()` 末尾报了同样的无害 exit code 2，数据已确认完整）。**检查了输出格式，
3000/3000 全部正确输出 `### ` 前缀，没有 graphlora 那种 `## (` 格式漂移**——确认格式漂移
是 GraphLoRA 机制特有的问题，不是全量训练规模的通病。

**Amazon-books 全量数据三方法（baseline / Power-Link / GraphLoRA）全部跑完**，完整对比
结果、诚实分析、最终建议已写入 `result.md` 第 5 节"Amazon-books 全量数据三方法对比"。
简要结论：baseline 和 Power-Link 几乎打平（BERT-F1 0.4326 vs 0.4330），GraphLoRA 即使
去除格式干扰后也明显更差（0.2829）——**这次干净的三方法独立消融证实了问题出在 GraphLoRA
微调机制本身，不是检索方式**。任务 #15 标记完成。

### 推理 + 评估命令（跑完训练后执行，模式同上）

```bash
cd ds_inference
# baseline / power-link 用 infer.py，graphlora 用 infer_subgraph.py（同上）
python infer.py --model_path ../ckpts/amazon_full_baseline --streategy Parallel \
    --batch_size 4 --max_tokens 256 --save_dir ../convert_files/phase6_amazon_full/baseline
python infer.py --model_path ../ckpts/amazon_full_powerlink --streategy Parallel \
    --batch_size 4 --max_tokens 256 --save_dir ../convert_files/phase6_amazon_full/powerlink
python infer_subgraph.py --model_path ../ckpts/amazon_full_graphlora \
    --subgraph_embed_path ../subgraph_retriever/embeds/amazon_trn_subgraph_embeds.pt \
    --subgraph_dim 256 --batch_size 4 --max_tokens 256 \
    --save_dir ../convert_files/phase6_amazon_full/graphlora

cd ..
python evaluation/eval_lite.py --dataset amazon --gen_path convert_files/phase6_amazon_full/{variant}/gen_datas.jsonl
```
