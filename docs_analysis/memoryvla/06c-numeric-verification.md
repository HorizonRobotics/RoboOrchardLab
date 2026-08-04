# 06c — 数值正确性：四档独立复跑

审查者独立执行 · 日期 2026-08-04 · 单卡 `cuda:6` · `ulimit -n 65536` · 任务 `swap_T` · batch 4

**全部自己跑，跑完再和 `PORT-STATUS.md` 对。** 复跑用的是审查者自写的
`$ROL_JFS/port/memoryvla/review/rev_agear.py`，**刻意不复用 `run_gears.py`**——
那个 harness 本身是被审对象（见 `06f` §1 与 `06-review-report.md` P1-1）。

基线树：`git worktree add --detach $R/baseline_3ce31c0c 3ce31c0c`，
并手工补齐 `ckpt`/`data`/`urdf`/`workspace` 四个符号链接（协议已警告 worktree 不带链接）。
已确认该树 `grep -c memoryvla config_holobrain_common.py` → **0**。

---

## 第 0 步 — 确定性前置检查

同一基线树、同 seed、跑两遍自比：

```
rev_baseline_run1  vs  rev_baseline_run2   (20 steps)
  MAX per-step |diff| on total loss = 0.000000e+00
  每个 loss 分量 max|diff| 全部 = 0.000000e+00
```

→ **逐位可复现**。因此 A 档用**严格判据** `atol < 1e-6`，不退化成噪声地板。

> 这一步必须先做：若基线自己就抖，后面拿 A 档的「0」当成功就毫无意义。
> 移植方也做了这一步且结论相同——这一条**独立复现一致**。

---

## A 档 — 关闭态等价（红线）✅ **PASS**

命令（两棵树，同 seed、同 batch、同一份顺序索引）：

```bash
MV_REPO=$R/baseline_3ce31c0c python3 rev_agear.py --enable 0 --steps 20 --batch-size 4 ...
                             python3 rev_agear.py --enable 0 --steps 20 --batch-size 4 ...
```

| | 基点 `3ce31c0c` | `port/memoryvla` `enable=0` | 差 |
|---|---:|---:|---:|
| 总参数量 | 1,136,284,265 | **1,136,284,265** | **+0** |
| 可训练参数 | 752,430,313 | 752,430,313 | +0 |
| 峰值显存 | 7.4683 GiB | 7.4683 GiB | 0 |
| batch key 集合 | 14 个 | **14 个，逐个相同** | 无 |

```
MAX per-step |diff| on total loss = 0.000000e+00   判据 1.000e-06
  loss_angle     0.000000e+00      loss_rot        0.000000e+00
  loss_angle_fk  0.000000e+00      loss_rot_fk     0.000000e+00
  loss_depth     0.000000e+00      loss_xyz        0.000000e+00
                                   loss_xyz_fk     0.000000e+00
→ PASS（20 步、7 个分量全部完全相同）
```

**与自述一致**（自述也是 0.000e+00 与参数量相同）。

这一档能过有**结构性原因**，不只是经验数字：`cfg.memoryvla=None` → `build()` 返回 `None`
→ 模块不存在 → 不消耗全局 RNG；且白名单追加受 `if enable` 保护，关闭态 batch 一字不变
（上表 14 vs 14 已实测坐实）。

### A 档补充：`num_workers=4` 下同样成立（补空白③）

移植方的 harness 用 `num_workers=0`，「新字段扰动 worker 随机流」这一失败模式
**在 0 worker 下根本不可能发生**。补跑：

```
A-gear with num_workers=4  (12 steps)
  batch keys identical : True
  params diff          : +0
  MAX per-step |diff|  : 0.000000e+00      → PASS
```

**并且这个测试有牙**——阳性对照证明这条流水线确实是 worker 敏感的：

```
同一棵树, workers=0 vs workers=4:  MAX per-step |diff| = 3.811359e-02
→ 变换里确实有 worker 播种的随机性，所以「移植后仍为 0」是有信息量的结论
```

---

## A' 档 — 已有移植回归 · **N/A（首次移植）**

独立确认 `docs_analysis/MIGRATIONS.md` 只有 2 个 `## ` 小节
（「已经定下来的约定」+「1. MemoryVLA」），仓库内无第二个已移植方法。
**记 N/A，不是跳过。**

---

## B 档 — 开启态前向/反向

走通 `dataloader → forward → loss → backward`，拿到全部要求的数值。
**但结论必须按 sampler 分开写**，因为两者天差地别（这正是 P0-1）：

| | 宿主真实 sampler（`train.py` 用的） | episode sampler（仅 harness 能选） |
|---|---|---|
| 总 loss | 有值，无 NaN | 有值，无 NaN |
| memoryvla 张量数 | 68（7,467,264 参数） | 68 |
| **拿到非零梯度的张量** | **0 / 68** | **68 / 68** |
| 梯度为 `None` 的张量 | 64（`retrieval_blocks` + `timestep_encoder`，从未被调用） | 0 |
| 梯度存在但**恰好为 0** | 4（两个 `GateFusion` 的 weight/bias） | 0 |

- **episode sampler 下 68/68 有梯度**——**独立复现了移植方的 B 档数字**。
  他们的测量没错，错的是这条路径在宿主里不存在。
- **真实 sampler 下 0/68**——详见 `06f` §2。

---

## C 档 — 与 A 的原实现数值对齐 ✅ **PASS（10/10 逐位一致）**

独立复跑 `check_reference.py`（CPU / float32 / torch 2.8.0+cu128，与生成参考值时同版本）：

```
  BIT-EXACT timestep_embedder                   max|diff| = 0.000e+00
  BIT-EXACT cross_transformer_block             max|diff| = 0.000e+00
  BIT-EXACT gate_fusion                         max|diff| = 0.000e+00
  BIT-EXACT bottleneck_se                       max|diff| = 0.000e+00  | hw=(8,8) 0.0e+00 | 8x11 ok -> (2,88,32)
  BIT-EXACT cogmembank_s1_empty_then_accumulate max|diff| = 0.000e+00  | bank sizes match
  BIT-EXACT cogmembank_s2_tome_consolidate      max|diff| = 0.000e+00  | bank sizes match
  BIT-EXACT cogmembank_s3_cross_episode         max|diff| = 0.000e+00  | bank sizes match
  BIT-EXACT cogmembank_s4_fifo                  max|diff| = 0.000e+00  | bank sizes match
  BIT-EXACT cogmembank_s5_group_mode            max|diff| = 0.000e+00  | bank sizes match
  BIT-EXACT cogmembank_s6_add_fusion            max|diff| = 0.000e+00  | bank sizes match

10 targets checked · 10 bit-exact · 0 failed
```

**与自述完全一致。**

### C 档的方法学核查（不能只看数字）

我在跑之前先确认了这个脚本**不是自己跟自己比**：

- `check_reference.py:19-21` —— `sys.path.insert(0, "~/git_repo/robo_orchard_lab")`
  然后 `from robo_orchard_lab.models.memoryvla import ...`。**比的是宿主里的移植实现。**
- `ref/*.npz` 由 `gen_reference.py` 从 `~/git_repo/MemoryVLA/vla/memory_vla.py` 切片 exec 产生，
  且 A 的工作树中**该文件未被修改**（R0 基线已确认）。

### C 档的覆盖边界（必须写明，不能被 10/10 掩盖）

C 档的 10 个靶子**全部是 A 里本来就有的类**。
**`wrapper.py`（274 行）与 `sampler.py`（179 行）在 A 里没有对应物，因此数值上零覆盖。**
它们恰恰是决定「存什么、取什么、插在哪一步」的胶水层。

本次审查对这两个文件的替代证据：
- `wrapper.py` —— `06d` 的 32 项接口语义逐项核对（mask 极性两路交叉验证、permute 往返验算、
  `uuid`/`step_index` 语义独立确认），**以及本次新增的恒等探针**（见 `06f` §2）。
- `sampler.py` —— `06d` §7 逐行确认时序单调性与 DDP 分片正确；
  但**它在宿主里从未被调用**，所以它「正确」这件事目前没有产品价值。

> 补充：我另外做了一件 C 档做不到的事——**逐行比对**移植后的 `CogMemBank`
> 与 A 的 `memory_vla.py:158-332`。两者除格式化外**逐字一致**，包含
> `@torch.no_grad()` 位置、`.detach().clone()`、`0.5*(f_i+f_j)`、
> `repeat_interleave(N,dim=1)`、以及 `block(query, episode_mem + pe, episode_mem)`
> 这个 **PE 只加在 key 不加在 value** 的非对称性。这解释了为什么 C 档能逐位一致。

---

## D 档 — 资源与 sanity

| | 参数量 | 峰值显存 | batch keys |
|---|---:|---:|---:|
| 基点 `3ce31c0c` | 1,136,284,265 | 7.4683 GiB | 14 |
| 移植后 关闭 | 1,136,284,265 | 7.4683 GiB | 14 |
| 移植后 开启 | 1,143,751,529 | 7.7797 GiB | 15（多 `step_index`） |

- **关闭态新增参数 = 0** ✅（协议硬要求）
- 开启增量：**+7,467,264 参数（+0.657 %）· +0.3114 GiB 显存**
- 无 NaN；记忆库不产生独立 loss 分量，无量级失配问题
- 开启 vs 关闭 逐 step 最大差 = **6.203079e-02**（顺序索引下，见下注）

**与自述对照**：自述「+7,467,264（+0.66 %）· +0.31 GiB · 开/关差 6.203e-02」——
**参数、显存、差值三项全部独立复现，差值甚至吻合到 4 位有效数字。**

> **注（重要）**：我的 A/D 档用的是**固定顺序索引**（0,1,2,3 / 4,5,6,7 …）。
> 由于 RoboDojo 的 episode 在全局索引上是连续排布的，顺序索引恰好给出
> **episode 连续**的 batch，因此这里的「开/关差 6.203e-02」反映的是
> **记忆库正常工作时**的效应。这是 D 档该测的东西。
> 但它**不是**真实训练路径的行为——真实路径见 `06f` §2。

**唯一与自述不符的一项**：自述「时间 +10 %」。
移植方自己的数据里 baseline 38.5 s vs **关闭态 35.1 s**（关闭态反而快 3.4 s ≈ 9 %），
说明墙钟噪声本身 ≥9 %，「+10 %」落在噪声内，**该结论不成立**（P3-1）。
本次未重复计时（同一张卡上有同事进程，墙钟不可比），**记为无法验证**。

---

## 与 PORT-STATUS 自述的逐项对照

| 档 | 自述 | **我实测** | 一致？ |
|---|---|---|---|
| 第 0 步 确定性 | 0.000e+00 逐位可复现 | **0.000000e+00** | ✅ 一致 |
| A 关闭态等价 | 0.000e+00，参数量一致 | **0.000000e+00**，参数 +0，显存相同，batch key 相同 | ✅ 一致 |
| A 档（多 worker） | 未测 | **0.000000e+00**（阳性对照 3.81e-02 证明有效） | ➕ 本次新增 |
| A' 已有移植回归 | 未提 | **N/A（首次移植）** | ✅ 一致 |
| B 68/68 有梯度 | 68/68，范数 8.39e-02 | **episode sampler 下 68/68 ✅；宿主真实 sampler 下 0/68 ❌** | ⚠️ **测量对，但不可迁移** |
| C 10/10 逐位一致 | 10/10，0.000e+00 | **10/10，0.000e+00** | ✅ 一致 |
| D 参数 +7,467,264 | +0.66 % | **+7,467,264（+0.657 %）** | ✅ 一致 |
| D 显存 +0.31 GiB | +0.31 GiB | **+0.3114 GiB** | ✅ 一致 |
| D 时间 +10 % | +10 % | **落在噪声内，结论不成立** | ❌ 不一致（P3-1） |
| D 开/关差 6.203e-02 | 6.203e-02 | **6.203079e-02** | ✅ 一致 |
| E bank 峰值 16 | 峰值 16，step 34 回落 | **episode sampler 下 4→8→12→16 封顶 ✅；宿主真实 sampler 下恒为 1** | ⚠️ **测量对，但不可迁移** |
| ckpt 兼容 1000→1068 | 0 unexpected，68 全在 `memoryvla.*` | **1000→1068，新增 68 全在 `memoryvla.*`，0 removed / 0 reshaped / 0 unexpected** | ✅ 一致 |

**结论**：移植方报的数字**基本都是真的**——12 项里 9 项独立复现一致，1 项不成立（时间），
2 项测量正确但**建立在宿主无法选择的代码路径上**。
问题从来不是他们编数字，而是**测的那条路不是将来会跑的那条路**。
