# 06 — 验证记录（五档 + ckpt 兼容性）

日期 2026-08-03 · 单卡（本机任意两卡 gather 必崩，协议已注明）· `ulimit -n 65536`
harness `$ROL_JFS/port/memoryvla/tools/run_gears.py` · 结果 JSON 在 `$ROL_JFS/port/memoryvla/logs/`

harness 刻意**不走 `train.py`**：accelerate / checkpoint / logging 会引入这套比对承受不了的
不确定性。它调的是真实的 `config_holobrain_common.build_model` 与真实的 RoboDojo 变换链，
所以数值描述的就是真正会训练的那个东西。优化器 `lr=0`，权重不动，
于是每步的数值是「数据 + seed」的纯函数，逐 step 比对才有意义。

---

## 第 0 步 — 确定性前置检查（决定后面用什么判据）

同一份 baseline（`3ce31c0c`，移植前）固定 seed 跑**两遍**，自己和自己比：

```
baseline 3ce31c0c × 2，20 step
最大逐 step 差值 = 0.000e+00
```

→ **逐位可复现**。因此 A 档用严格判据 `atol < 1e-6`，**不需要**退化成噪声地板。

---

## A 档 — 关闭态等价（红线）✅ PASS

| | baseline | 移植后 |
|---|---|---|
| commit | `3ce31c0c`（移植前，git worktree） | `111981a5`，`enable=0` （cite: logs/baseline_run1.json + logs/gearA_off.json） |
| `memoryvla` 模块是否构建 | False | False （cite: logs/gearA_off.json） |
| 参数量 | 1,136,284,265 | **1,136,284,265**（差 0） （cite: logs/gearA_off.json） |

```
最大逐 step 差值 = 0.000e+00   判据 1.000e-06
→ PASS（20 个 step 全部完全相同）
```

baseline 跑在 `git worktree add … 3ce31c0c` 出来的独立工作树里（`ckpt`/`data`/`urdf`/
`workspace` 四个符号链接手工补齐），**是真正的移植前代码**，不是「同一棵树关掉开关」。

这一档能过的**根本原因**是「关闭 = 不构建」：`cfg.memoryvla` 为 `None` → `build()` 返回
`None` → 模块不存在 → 不消耗任何全局 RNG。若做成「构建但早退」，
`GateFusion` 的 `normal(0,1e-3)` 初始化会吃掉 RNG，后续随机数全部错位。

## B 档 — 开启态前向/反向 ✅ PASS

```
memoryvla 参数: 7,467,264 (7.4673M)
step0 total loss = 2.406754
  loss_angle      0.590657      loss_rot        0.091413
  loss_angle_fk   0.590657      loss_rot_fk     0.089622
  loss_depth      0.000000      loss_xyz        0.770558
                                loss_xyz_fk     0.273846
记忆参数梯度: 68/68 个张量拿到梯度，总范数 8.392649e-02
梯度为 0 的 step: 无
开启 vs 关闭 最大逐 step 差值 = 6.203e-02   ← 必须 > 0，否则记忆库是摆设
```

**顺带验掉的 DDP 隐患**：`retrieval_blocks` 在 `len(hist)==0` 分支下完全不参与计算，
若一批内所有样本都无历史，这些参数拿不到梯度、DDP 会报 unused parameter。
实测 **68/68 全程有梯度**，该情形在 stream 模式下不出现。
⚠️ 但**本机单卡跑不了 DDP**，多卡下的行为**未验证**，列入风险。

## C 档 — 与 A 的原实现数值对齐 ✅ PASS（**10/10 逐位一致**）

同 torch 版本（2.8.0+cu128）、CPU、float32、**加载 A 的同一套权重**（`strict=True`）。

| 靶子 | max\|diff\| |
|---|---|
| `TimestepEmbedder` | **0.000e+00** （cite: ref/manifest.json） |
| `CrossTransformerBlock` | **0.000e+00** （cite: ref/manifest.json） |
| `GateFusion` | **0.000e+00** （cite: ref/manifest.json） |
| `BottleneckSE`（原版 sqrt 路径 + 改写后 `hw=(8,8)`） | **0.000e+00** （cite: ref/bottleneck_se.npz） |
| `CogMemBank` S1 空历史 → 累积 | **0.000e+00** （cite: ref/cogmembank_s1_empty_then_accumulate.npz） |
| `CogMemBank` S2 触发 ToMe 巩固 | **0.000e+00** （cite: ref/cogmembank_s2_tome_consolidate.npz） |
| `CogMemBank` S3 batch 内跨 episode | **0.000e+00** （cite: ref/cogmembank_s3_cross_episode.npz） |
| `CogMemBank` S4 FIFO | **0.000e+00** （cite: ref/cogmembank_s4_fifo.npz） |
| `CogMemBank` S5 group 模式 | **0.000e+00** （cite: ref/cogmembank_s5_group_mode.npz） |
| `CogMemBank` S6 add 融合 | **0.000e+00** （cite: ref/cogmembank_s6_add_fusion.npz） |

各场景的 bank 长度序列也与 A 完全一致。`BottleneckSE` 额外验了非方形输入
`hw=(8,11)` 不再触发 assert，输出 `(2, 88, 32)`。

`strict=True` 是有意的：参数改名会在这里当场炸掉，而不是变成后面一个「数值差一点」的谜。

## D 档 — 资源

| | 参数量 | 峰值显存 | 20 step 墙钟 |
|---|---:|---:|---:|
| baseline（移植前） | 1,136,284,265 | 7.47 GiB | 38.5 s （cite: logs/baseline_run1.json） |
| 移植后 关闭 | 1,136,284,265 | 7.47 GiB | 35.1 s （cite: logs/gearA_off.json） |
| 移植后 开启 | 1,143,751,529 | 7.78 GiB | 38.6 s （cite: logs/gearB_on.json） |

开启的增量：**参数 +7,467,264（+0.66%）· 显存 +0.31 GiB · 时间 +10%**。
无 NaN；记忆分量与主 loss 同量级（记忆库不产生独立 loss 分量，只改特征）。

## E 档 — RoboDojo Memory 任务冒烟 ✅ PASS

```
45 step, batch 8, sampler=episode, 任务 swap_T（Memory 六任务中 episode 最短）
见到的不同 episode 数 = 2                    ✅ 跨过了边界
episode 顺序: episode_0000083 → episode_0000041
bank 峰值长度 = 16 (mem_length=16)           ✅ 巩固逻辑触发
bank 回落的 step = [34]                      ✅ 换 episode 时 clear_episode 执行了
loss 范围 [2.2688, 47.5016]  NaN: 无
峰值显存 11.60 GiB · 141.52 s
```

**为什么非要跨过边界**：`clear_episode` 清的是**上一条** episode，
只跑一条 episode 时这条路径永远不执行。step 34 上 bank 长度回落，就是它执行过的证据。
这正是协议「冒烟必须跑到第 2 个单位」在本次的具体形态。

> loss 在个别 step 冲到 47 不是发散：优化器 `lr=0`，权重全程不动，
> 数值波动完全来自不同帧的动作幅度差异。

## 已有 ckpt 兼容性 ✅ PASS

v10 的 warm-start 权重在一个 URL 后面，本机无外网，所以不去拉它，而是**从这棵树自己造**
移植前的 state_dict：关闭态构建模型（A 档已证明它与移植前逐位相同）取 state_dict，
再往开启态的模型里加载。

```
移植前 state_dict: 1000 tensors
移植后 state_dict: 1068 tensors
  移植前有、移植后没了的 key : 0
  形状变了的 key             : 0
  新增的 key                 : 68（其中非 memoryvla.* 的: 0）
load_state_dict(strict=False):
  unexpected keys : 0
  missing 且非 memoryvla.* : 0
  missing 且是 memoryvla.* : 68（预期，就是新模块）
加载后跑一步: loss = 2.900933, backward 完成, NaN: False
```

→ **移植前的 checkpoint 原样可加载**，新增的 68 个张量全部落在 `memoryvla.*` 下。

---

## 降级说明（协议要求写明用了哪一档）

| 项 | 用的档位 | 原因 |
|---|---|---|
| 卡数 | **单卡** | 本机任意两卡 gather 必崩 `ILLEGAL_ADDRESS`，与显存无关 （cite: 本机已知约束） |
| batch | **4**（A/B/D 档）/ **8**（E 档），而非 config 默认 16 | 8 张卡都有同事的进程占着 12–18 GiB，留给本次的只有 ~19 GiB。降 batch 不影响本次结论：A 档比的是同 batch 的两棵树，C 档根本不过模型 （cite: 实测 nvidia-smi） |
| 优化器 | `lr=0` | 让逐 step 数值成为「数据 + seed」的纯函数；本次不追收敛，这样比更干净 （cite: tools/run_gears.py） |
| 训练时长 | 20–45 step | 验收线不含收敛，见 `PORT-STATUS.md` （cite: 验收线） |

⚠️ **batch 降到 1 时必须同时把 `dataloader_type` 切成 `stream`**（本次已是 stream）。
`group` 模式下一批一个样本 = 组内无历史，记忆库会变成恒等而**看起来仍在正常运行**。
