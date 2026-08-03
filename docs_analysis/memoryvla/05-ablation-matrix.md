# 05 — 消融矩阵与正交性

全部跑在 RoboDojo Memory 维度任务上（`swap_T`），8 step，batch 4，单卡，固定 seed，`lr=0`。
**每一行都只靠 config 切换，没有改一行代码** —— 这是挂载点选对了的判据。
所有开关都是 `config["memoryvla"]` 里的键，可由 `train.py --kwargs` 直接传。

## 矩阵

| 配置 | 开关 | memoryvla 参数 | step0 loss | vs base 最大逐 step 差 | 峰值显存 | bank 峰值长度 |
|---|---|---:|---:|---:|---:|---:|
| base（全关） | `enable=False` | 0 | 2.409832 | 0（定义） | 7.47 G | 0 （cite: logs/gearA_off.json） |
| **+full** | `enable=True` | 7,467,264 | 2.406754 | 4.029e-02 | 7.78 G | **16** （cite: logs/gearB_on.json） |
| 只开感知 | `use_cognitive=False` | 3,733,632 | 2.401623 | 6.273e-02 | 7.74 G | 16 （cite: logs/abl_per_only.json） |
| 只开认知 | `use_perceptual=False` | 3,733,632 | 2.401904 | 6.323e-02 | 7.50 G | 16 （cite: logs/abl_cog_only.json） |
| fusion=add | `fusion_type=add` | 6,876,672 | 2.402399 | 3.627e-02 | 7.77 G | 16 （cite: logs/abl_fusion_add.json） |
| consolidate=fifo | `consolidate_type=fifo` | 7,467,264 | 2.406754 | 4.030e-02 | 7.78 G | 16 （cite: logs/abl_consolidate_fifo.json） |
| **mode=group** | `dataloader_type=group` | 7,467,264 | 2.406754 | 3.887e-02 | 7.58 G | **4** （cite: logs/abl_mode_group.json） |

**每一行 vs base 都 > 0**，说明每个开关都真的生效了，没有哪个是摆设。

## 三处值得单独说的

### 1. `mode=group` 的 bank 峰值是 4，其余都是 16 —— 在真实模型上复现了那个反直觉行为

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
