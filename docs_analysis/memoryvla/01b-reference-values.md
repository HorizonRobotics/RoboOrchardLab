# 01b — 参考数值（C 档的唯一硬证据）

生成日期：2026-08-03 · 生成环境 `memvla_cu128`（torch **2.8.0+cu128** / numpy 1.26.4）
落点：`/jfs-public/users/kun01.wu/robo_orchard_lab/port/memoryvla/ref/`
生成脚本：`$ROL_JFS/port/memoryvla/tools/gen_reference.py`

```bash
ssh <HOST> 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate memvla_cu128 \
  && cd ~/git_repo/MemoryVLA \
  && CUDA_VISIBLE_DEVICES="" python $ROL_JFS/port/memoryvla/tools/gen_reference.py'
```

## 三条设计选择及理由

1. **只 exec `vla/memory_vla.py` 的第 30–358 行**（`TimestepEmbedder` 到 `PerMemBank` 结束，
   停在 `class MemoryVLA` 之前）。这段是纯 `torch.nn`，不触发 `prismatic`/`transformers` import。
   **被移植的是这一段，被钉住的就必须是这一段**，不能拿整个 `MemoryVLA` 类来对。
   脚本里有两条 assert 守着切片边界，切片漂了会立刻失败而不是悄悄对错东西。
2. **CPU + float32**。C 档要回答的是「移植有没有改变数学」，GPU kernel 的不确定性会把这个
   问题和「硬件抖动」混在一起。放到 CPU 上，两边应当**逐位**一致。
3. **`.npz` 里同时存 state_dict**（键前缀 `sd::`）。协议要求「权重要用同一套」——
   靠 seed 重建权重是不行的：只要任何一边动了 RNG 消费顺序，权重就悄悄漂了，
   而表现出来是「数值对不齐」，会被误判成移植错误。落盘用临时文件 + `os.replace`
   （JFS 支持 rename；这也是这批文件**不能**放 bucket 的原因）。

## 文件清单

| 文件 | 靶子 | 参数数 | 覆盖的路径 |
|---|---|---:|---|
| `timestep_embedder.npz` | `TimestepEmbedder(64, freq=16)`，`t=0..36` | 4 | 纯函数 （cite: ref/manifest.json） |
| `cross_transformer_block.npz` | `CrossTransformerBlock(64)`，`q[2,5,64]` `k/v[2,11,64]` | 14 | 检索算子 （cite: ref/manifest.json） |
| `gate_fusion.npz` | `GateFusion(64)`，`x1,x2[2,5,64]` | 2 | 融合系数 （cite: ref/manifest.json） |
| `bottleneck_se.npz` | `BottleneckSE(96,64,32)`，`[2,64,96]`（N=64 → 8×8 方形） | 6 | **改写前基准**：原版只接受方形，这份证明改写没改数学 （cite: ref/manifest.json） |
| `cogmembank_s1_empty_then_accumulate.npz` | 1 批 × 3 帧，同 episode | 34 | `len(hist)==0` 分支 + 累积 （cite: ref/manifest.json） |
| `cogmembank_s2_tome_consolidate.npz` | 3 批 × 3 帧，同 episode，`mem_length=4` | 34 | **ToMe 巩固真的触发** （cite: ref/manifest.json） |
| `cogmembank_s3_cross_episode.npz` | 1 批内 episode 从 7 变 8 | 34 | `clear_episode` （cite: ref/manifest.json） |
| `cogmembank_s4_fifo.npz` | 同 S2，`consolidate_type=fifo` | 34 | FIFO 分支 （cite: ref/manifest.json） |
| `cogmembank_s5_group_mode.npz` | 同 S2，`dataloader_type=group` | 34 | group 模式清空行为 （cite: ref/manifest.json） |
| `cogmembank_s6_add_fusion.npz` | 2 批，`fusion_type=add` | **32** | 无 `GateFusion` 参数 （cite: ref/manifest.json） |
| `manifest.json` | 每个文件的 shape / dtype / mean / std / config | — | — （cite: ref/manifest.json） |

> S6 是 32 而其余是 34 —— 差的正是 `GateFusion` 的 `proj.weight` 与 `proj.bias`。
> 这条差异本身就是「`fusion_type=add` 时不构建融合模块」的证据。

## 脚本里的 assert 顺带证实了两条行为（**不是读代码读出来的，是跑出来的**）

`gen_reference.py` 记录了每批之后 `bank.bank` 里每条 episode 的长度：

| 模式 | 三批之后的 bank 长度 | 含义 |
|---|---|---|
| `group`（S5） | `[{7:3}, {7:3}, {7:3}]` | **每次 `process_batch` 开头都清空**，记忆跨度 = 一个 batch，`mem_length=4` 从未起作用 （cite: 实测 tools/gen_reference.py 输出） |
| `stream`（S2） | `[{7:3}, {7:4}, {7:4}]` | 跨 batch 累积，涨到 `mem_length=4` 后被巩固逻辑压住 → **ToMe 确实触发了** （cite: 实测 tools/gen_reference.py 输出） |

对应 `vla/memory_vla.py:267-274`。这直接决定了移植该用哪个模式，详见 （cite: vla/memory_vla.py:267-274）
[`01-source-anatomy.md`](01-source-anatomy.md) §4.1。

## C 档怎么用这批文件

对每个 `.npz`：

1. 用 `sd::` 前缀的数组构造 state_dict，`load_state_dict(..., strict=True)` 加载进**移植后**的模块
   （`strict=True` 很重要：参数改名会被当场抓住，而不是变成一个「数值差一点」的谜）
2. 喂 `in_*` 数组，CPU / float32
3. 与 `out_*` 比，报绝对与相对误差量级

判据：`atol < 1e-5` 判对齐（同 torch 版本 + CPU，**预期应当逐位一致**，即误差恰为 0）；
`1e-5 ~ 1e-3` 必须给出解释；更大一律视为移植错误。

对 `bottleneck_se.npz`：改写后的实现要能在 `(h,w)=(8,8)` 时**逐位复现**原版输出。
非方形输入没有参考值可比（原版根本跑不了），只能靠 code review + 形状断言。
这一点会写进最终汇报的风险项。
