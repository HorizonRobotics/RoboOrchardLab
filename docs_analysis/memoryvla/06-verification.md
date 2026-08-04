# 06 — 验证记录（五档 + ckpt 兼容性）

> # ⛔ 本文 A / B / D / E 四档数值已失效（标注于 2026-08-04）
>
> | 档（按本文小节标题定位，不用行号——行号会漂） | 状态 | 去哪看现行数值 |
> |---|---|---|
> | `## 第 0 步 — 确定性前置检查` | **⛔ 结论被推翻** —— 「地板恰为 0，故用 `atol<1e-6` 严格判据」是 **harness 的性质**（`lr=0` 权重不动），不是宿主的性质 | `PORT-STATUS.md`「确定性」节 |
> | `## A 档 — 关闭态等价` | **⛔ 失效，已重测** | `PORT-STATUS.md`「验证结果」+ `08-review-response.md` |
> | `## B 档 — 开启态前向/反向` | **⛔ 失效，已重测** —— 「68/68 张量有梯度」是假路径产物，真实路径当时是 `64 None / 4 零 / 0 非零` | 同上 |
> | `## C 档 — 与 A 的原实现数值对齐` | ✅ **仍有效**，且复审已在 `2b739226` 上独立重跑 **10/10 逐位一致** | 本文即为现行 |
> | `## D 档 — 资源` | **⛔ 失效，已重测**；「开启 +10% 时间」结论已撤回（墙钟不可比） | `PORT-STATUS.md`「D 档：墙钟不用来下结论」 |
> | `## E 档 — RoboDojo Memory 任务冒烟` | **⛔ 失效，已重测** | `08-review-response.md` |
> | `## 已有 ckpt 兼容性` | ⚠️ 未重验（复审标「继承自上一轮，本轮未重验」） | 本文 |
>
> **失效的根因**：本文全部数值产自 `tools/run_gears.py`，A/B/D 档与全部 5 个消融跑的是
> `--sampler sequential`——一个**仓库里根本不存在**的手写连续索引列表，E 档则由 harness
> **自己实例化**了 `MemoryVLAEpisodeStreamBatchSampler`。**宿主没有任何路径能到达那两种装配。**
> 见 `06-review-report.md` 的 **P1-1**。
> **C 档是唯一不受影响的**：它拿 `ref/*.npz` 的固定输入直接喂模块，根本不经过 sampler。
>
> 数字**故意保留不删**——它们记录了「验证装置替换了被验证对象」这件事是怎么发生的。

日期 2026-08-03 · 单卡（本机任意两卡 gather 必崩，协议已注明）· `ulimit -n 65536`
harness `$ROL_JFS/port/memoryvla/tools/run_gears.py`（**⛔ 假路径，见上**） · 结果 JSON 在 `$ROL_JFS/port/memoryvla/logs/`

harness 刻意**不走 `train.py`**：accelerate / checkpoint / logging 会引入这套比对承受不了的
不确定性。它调的是真实的 `config_holobrain_common.build_model` 与真实的 RoboDojo 变换链，
~~所以数值描述的就是真正会训练的那个东西。~~ 优化器 `lr=0`，权重不动，
于是每步的数值是「数据 + seed」的纯函数，逐 step 比对才有意义。

> **⛔ 划掉的那句话是错的（2026-08-04）。** 它正是 **P1-1** 推翻的那一句：harness 调真实的
> `build_model` 与真实的变换链**不足以**让数值描述真正会训练的东西——**批的组织方式**
> 才是这次移植的关键变量，而 harness 恰好在这一点上与宿主不同。
> 「不走 `train.py` 的理由本身成立」也是真的，这正是它最阴的地方：
> 这个偏离在 code review 里读起来像谨慎，不像风险。

---

## 第 0 步 — 确定性前置检查（决定后面用什么判据）  ⛔ **结论已被推翻**

同一份 baseline（`3ce31c0c`，移植前）固定 seed 跑**两遍**，自己和自己比：

```
baseline 3ce31c0c × 2，20 step
最大逐 step 差值 = 0.000e+00
```

~~→ **逐位可复现**。因此 A 档用严格判据 `atol < 1e-6`，**不需要**退化成噪声地板。~~

> **⛔ 已被真实入口实测推翻（2026-08-04）。** 这个 `0` 是 harness 的性质：`lr=0`，权重不动，
> 逐 step 值是「数据 + seed」的纯函数，误差没有累积的机会。走真实入口（真 optimizer、真 lr、
> `num_workers=4`）实测地板是 step 0 精确 `0`、20 步内峰值 `2.899e-04`，
> 开确定性算子也只压到 `1.564e-04`，**压不到 0**。
> 更要紧的是复审的独立测量：**浮点判据在真实入口上没有分辨力**——
> 5 个关闭态 run、10 组两两比较，跨代码组区间 `[5.102e-05, 9.108e-05]`
> **完整落在**同代码组区间 `[4.101e-05, 1.159e-04]` 之内，
> 而 10 组里最大的那个差恰恰出现在**共用同一份 `train.py`** 的两个 run 之间。
> **现行判据见 `PORT-STATUS.md`「关闭态等价性：改用精确判据」节**——
> 用逐样本 id 序列 / batch key 集合 / 参数量 / 峰值显存 / sampler 链五项精确量，
> 浮点 loss 差只作参考量。

---

## A 档 — 关闭态等价（红线）~~✅ PASS~~  ⛔ **假路径产出，已重测**

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

## B 档 — 开启态前向/反向 ~~✅ PASS~~  ⛔ **假路径产出，已重测**

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

## C 档 — 与 A 的原实现数值对齐 ✅ PASS（**10/10 逐位一致**）  ✅ **仍有效**

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

## D 档 — 资源  ⛔ **假路径产出，已重测**

| | 参数量 | 峰值显存 | 20 step 墙钟 |
|---|---:|---:|---:|
| baseline（移植前） | 1,136,284,265 | 7.47 GiB | 38.5 s （cite: logs/baseline_run1.json） |
| 移植后 关闭 | 1,136,284,265 | 7.47 GiB | 35.1 s （cite: logs/gearA_off.json） |
| 移植后 开启 | 1,143,751,529 | 7.78 GiB | 38.6 s （cite: logs/gearB_on.json） |

开启的增量：**参数 +7,467,264（+0.66%）· 显存 +0.31 GiB · 时间 +10%**。
无 NaN；记忆分量与主 loss 同量级（记忆库不产生独立 loss 分量，只改特征）。

## E 档 — RoboDojo Memory 任务冒烟 ~~✅ PASS~~  ⛔ **假路径产出，已重测**

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

## 已有 ckpt 兼容性 ✅ PASS  ⚠️ **本轮与上轮均未重验**

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
| 优化器 | `lr=0` | 让逐 step 数值成为「数据 + seed」的纯函数；本次不追收敛，这样比更干净 （cite: tools/run_gears.py，**⛔ 假路径**） |
| 训练时长 | 20–45 step | 验收线不含收敛，见 `PORT-STATUS.md` （cite: 验收线） |

⚠️ **batch 降到 1 时必须同时把 `dataloader_type` 切成 `stream`**（本次已是 stream）。
`group` 模式下一批一个样本 = 组内无历史，记忆库会变成恒等而**看起来仍在正常运行**。

> **状态（2026-08-04）**：这条警告本身是对的，但它**只写在文档里，没有写进任何会被执行的东西**
> ——与 P0-1、与 `ulimit -n` 是同一个形状。本轮第二段把它换成了会执行的护栏：
> `MemoryVLAMemory` 里的 bank 存活性看门狗跑满 K 步后若从没见过长度 > 1 的 bank 就直接 raise，
> 而 `group` + batch=1 恰好满足「bank 恒为 1」。**判据不看配置项名字，看后果。**
