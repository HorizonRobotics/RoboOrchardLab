# 05 — 消融矩阵与正交性

> # ⛔ 本文全部数值已失效，不可用于任何结论（标注于 2026-08-04）
>
> **数据来源**：`$ROL_JFS/port/memoryvla/tools/run_gears.py --sampler sequential`。
> 那是一条**宿主到不了的装配**——`--sampler sequential` 是 harness 自己写的连续索引列表，
> 仓库里根本不存在这个东西；它碰巧产出 episode 连续的批，于是记忆库看起来一直在工作。
> 真实入口（`train.py`）当时走的是 `DistributedBatchFlagSampler` 全局随机排列，
> bank 恒为 `[1]`、融合精确恒等、7.47M 参数 grad `64 None / 4 零 / 0 非零`。
> 详见 `06-review-report.md` 的 **P1-1** 与 `08-review-response.md`。
>
> **失效状态**：下表 7 行**未重跑**，本轮也不打算重跑（消融本身不是本轮范围）。
> 数字**故意保留不删**——它们记录了失效是怎么发生的，删掉就看不见了。
>
> **现行有效数值**在 `PORT-STATUS.md`「验证结果」与 `08-review-response.md`，
> 全部从 `train.py` 真实入口产出。唯一**不受影响**的是 C 档（`06-verification.md` C 档），
> 它不经过 sampler。

全部跑在 RoboDojo Memory 维度任务上（`swap_T`），8 step，batch 4，单卡，固定 seed，`lr=0`。
~~**每一行都只靠 config 切换，没有改一行代码** —— 这是挂载点选对了的判据。~~
**订正（2026-08-04）**：这句**已不成立**。`mode=group` 那一行现在需要同时设两个键
（`dataloader_type=group` **且** `episode_stream_sampler=True`），见下面第 1 小节。
所有开关都是 `config["memoryvla"]` 里的键，可由 `train.py --kwargs` 直接传。

## 矩阵

> **⛔ 下表 7 行全部产自假路径，已失效，未重跑（2026-08-04 标注）。**
> `峰值显存` 一列另有一层问题：真实入口关闭态实测 `8.98 GiB`、开启态 `9.30 GiB`，
> 与下表的 `7.47 / 7.78 G` 差一档——那是 harness 不装 accelerate/checkpoint 的结果。

| 配置 | 开关 | memoryvla 参数 | step0 loss | vs base 最大逐 step 差 | 峰值显存 | bank 峰值长度 |
|---|---|---:|---:|---:|---:|---:|
| base（全关） | `enable=False` | 0 | 2.409832 | 0（定义） | 7.47 G | 0 （cite: logs/gearA_off.json） |
| **+full** | `enable=True` | 7,467,264 | 2.406754 | 4.029e-02 | 7.78 G | **16** （cite: logs/gearB_on.json） |
| 只开感知 | `use_cognitive=False` | 3,733,632 | 2.401623 | 6.273e-02 | 7.74 G | 16 （cite: logs/abl_per_only.json） |
| 只开认知 | `use_perceptual=False` | 3,733,632 | 2.401904 | 6.323e-02 | 7.50 G | 16 （cite: logs/abl_cog_only.json） |
| fusion=add | `fusion_type=add` | 6,876,672 | 2.402399 | 3.627e-02 | 7.77 G | 16 （cite: logs/abl_fusion_add.json） |
| consolidate=fifo | `consolidate_type=fifo` | 7,467,264 | 2.406754 | 4.030e-02 | 7.78 G | 16 （cite: logs/abl_consolidate_fifo.json） |
| **mode=group** | `dataloader_type=group` | 7,467,264 | 2.406754 | 3.887e-02 | 7.58 G | **4** （cite: logs/abl_mode_group.json） |

~~**每一行 vs base 都 > 0**，说明每个开关都真的生效了，没有哪个是摆设。~~

> **⛔ 这条结论建立在假路径上，已失效，未重跑（2026-08-04 标注）。**
> 「每一行 vs base 都 > 0」在 `--sampler sequential` 下成立，因为那条路径下 batch 恰好
> episode 连续、记忆库真的在工作。在**真实入口**上，当时这些配置**每一行都是恒等**——
> 差值全部来自随机性而非记忆库。也就是说这句话恰好把「全都是摆设」读成了「没有哪个是摆设」。
> **要重新回答「每个开关是否真的生效」，必须在真实入口上重跑整张表；本轮未做。**

## 三处值得单独说的

### 1. `mode=group` 的 bank 峰值是 4，其余都是 16 —— 在真实模型上复现了那个反直觉行为

> **状态（2026-08-04）**：本节描述的**现象是对的**，但**这一行的配方当时不可执行**，
> 且在被审 commit `2b739226` 上它是 **P1-B** —— `dataloader_type=group` 没有任何可用配置：
> 开 `episode_stream_sampler` 会 raise，关掉则静默退化成恒等且三道护栏一道都不响。
>
> **已修（第三轮第二段）。现在的可用配方是两个键一起设**：
>
> ```
> dataloader_type="group"   且   episode_stream_sampler=True
> ```
>
> 真实入口实测（`fix3/runs/2026-08-04/head_D_group_on.json`，`rc=0`，bs=4、20 step）：
> **bank 每步最大长度恒为 4**（= batch，记忆跨不出一个 batch，与本节的现象一致）、
> grad `0 None / 0 零 / 68 非零`、参数移动 62/68、峰值显存 9.1012 GiB。
> 完整数值与修法见 `PORT-STATUS.md`「`dataloader_type="group"` 的现状」。
>
> **但本行表格里的数字仍然作废**：现象一致不等于数值可用，那一行是在
> `--sampler sequential` 这条宿主到不了的装配上量的。要这一行的数，得在真实入口重跑。

batch 就是 4。也就是说 **group 模式下记忆跨不出一个 batch**，
`mem_length=16` 从头到尾没起过作用。这与 `01b` 里在纯模块上的实测
（group `[3,3,3]` vs stream `[3,4,4]`）互相印证，只不过这次是在完整的 HoloBrain 上。

**A 的默认值恰恰是 `group`**（cite: `MemoryVLA@0eef5c3 vla/memory_vla.py:369`）。
论文描述的 episode 级记忆对应的是 `stream`，所以本次默认用 `stream`。

### 2. 参数量的分解正好对得上，可以当结构自检

- 只开感知 / 只开认知都是 **3,733,632 = 7,467,264 / 2** → 两个 bank 结构完全相同、
  参数独立不共享（印证 `PerMemBank` 只是 `CogMemBank` 的空壳子类）。
- `fusion=add` 是 **6,876,672 = 7,467,264 − 590,592**，差的正好是两套 `GateFusion`
  的 `proj`（`Linear(768→384)`：`384×768 + 384 = 295,296`，两套 590,592）。
  → `fusion_type=add` 时确实没有构建 `GateFusion`。

### 3. `fifo` 与 `tome` 在这个长度上几乎无差别（4.030e-02 vs 4.029e-02）

8 step × batch 4 = 32 帧，bank 到 16 之后才开始巩固，所以只有最后几步走了巩固路径，
两种策略的差异还没来得及累积。**不能据此说两者等价** —— 要比较它们需要跑到
episode 尺度（本数据集 episode 长 276–1374 帧）。列为遗留问题。

## 正交性

本仓库**首次移植**，没有「已移植的其他方法」可以同开 → 该档记 **`N/A（首次移植）`**。
`docs_analysis/MIGRATIONS.md` 已把本次占用的挂载点写清楚，
下一个方法撞上同一处时按那里的说明处理。

## 新超参建议

| 超参 | 建议初值 | 建议范围 | 来源 |
|---|---|---|---|
| `dataloader_type` | **`stream`** | `stream` / `group` | A 默认 `group`（cite: `vla/memory_vla.py:369`），但实测其记忆跨度只有一个 batch；论文语义取 `stream` |
| `mem_length` | 16 | 8–32 | A 默认 16（cite: `vla/memory_vla.py:372`）。本数据集 episode 长 276–1374 帧，16 条巩固后的记忆覆盖整条 episode，压缩比很高，值得往上试 |
| `retrieval_layers` | 2 | 1–4 | A 默认 2（cite: `vla/memory_vla.py:373`） |
| `fusion_type` | `gate` | `gate` / `add` | A 默认 `gate`（cite: `vla/memory_vla.py:375`） |
| `consolidate_type` | `tome` | `tome` / `fifo` | A 默认 `tome`（cite: `vla/memory_vla.py:376`） |
| `use_timestep_pe` | True | — | A 默认 True（cite: `vla/memory_vla.py:374`） |
| `update_fused` | False | — | A 默认 False（cite: `vla/memory_vla.py:377`） |
| `group_size` | 16 | 仅 `group` 模式有意义 | A 默认 16（cite: `vla/memory_vla.py:370`） |

论文与代码不一致处：**论文强调 episode 级记忆，代码默认 `group` 只有 batch 级**。
以代码实测为准，配置上取 `stream` 来实现论文语义。
论文里的具体超参数值**未确认**（未逐条核对论文附录）。

## loss 权重量纲分析

**本次不需要。** 记忆库不产生任何新的 loss 分量 —— 它只改特征，
监督信号仍然全部来自宿主原有的 `decoder.loss`（B 档的分量表与 base 完全同构：
`loss_xyz` / `loss_angle` / `loss_rot` / `loss_depth` 及其 `_fk` 版本）。
所以没有「新 loss 与主 loss 量级差多少」的问题，也不需要给新权重。
