# 08 — 对 `06-review-report.md` 的逐条应答

**日期**：2026-08-04 · **被修分支**：`port/memoryvla`，基点 `18106b05`
**修复 commit**：`701679a9`（接线）· `166b8756`（护栏 + 文档订正）· 本文所在 commit（验证记录）
**复审**：`review/memoryvla` @ `97837e04`，判定 🔴 REJECT
**本次范围**：范围锁死的修复，只动「接线 / 护栏 / 文档订正」三处。不是继续移植。

**本文不自评 PASS。** 产出是修复 + 数值证据，裁决交给下一轮独立复审。

| ID | 级别 | 处置 |
|---|---|---|
| P0-1 | P0 | **已修复** |
| P1-1 | P1 | **已修复** |
| P2-1 | P2 | **延后**（记入遗留，理由见下） |
| P2-2 | P2 | **延后**（记入遗留） |
| P2-3 | P2 | **不修**（显式记录取舍） |
| P3-1 | P3 | **已重测**，原结论**不成立**，已撤 |
| P3-2 | P3 | **延后**，但影响已被本轮验证方式绕开 |

---

## P0-1 · `episode_stream_sampler` 是死开关 —— **已修复**

**改了什么**：`projects/holobrain_internal/common/train.py`，DataLoader 构造处
由硬编码 `DistributedBatchFlagSampler` 改成一个开关判断（commit `701679a9`）：

```python
if memoryvla_cfg.get("enable", False) and memoryvla_cfg.get(
    "episode_stream_sampler", False
):
    batch_sampler = MemoryVLAEpisodeStreamBatchSampler(...)
else:
    batch_sampler = DistributedBatchFlagSampler(...)   # 构造参数一字未动
```

判据是 **`enable ∧ episode_stream_sampler`**，不是只读后者。**这里是本次最容易引入新 P0 的地方**：
该键 ship 值为 `True`，却**挂在 `enable=False` 之下**（`config_holobrain_common.py:45,59`）。
只读它会让全关配置也换掉 sampler —— A 档当场破，而且破得像「数值有点飘」，不像「接错了」。

也不能改类名了事：`train.py:128` 传 `dataset_sample_weights=`，而
`MemoryVLAEpisodeStreamBatchSampler.__init__`（`sampler.py:108-117`）不接受该参数，
直接替换标识符会 `TypeError`。所以是 if/else 包住整个构造调用。

**证据**（全部走 `train.py` 真实入口，`$ROL_JFS/port/memoryvla/fix/runs/2026-08-04/`）：

| 观测 | 修复前（复审实测，真实路径） | 修复后（本轮） |
|---|---|---|
| 实际迭代的 sampler | `DistributedBatchFlagSampler` | **`MemoryVLAEpisodeStreamBatchSampler`**（post-`prepare()` 拆包后实测） |
| 每 batch 不同 episode 数 | **4 / 4** | **1**（B 档 20 步、E 档 60 步全程） |
| bank 长度 | **恒为 `[1]`** | bs=4：**4 → 8 → 12 → 16** 封顶；bs=8：**8 → 16** 封顶 |
| 感知流 `max\|out−in\|` | `1.192093e-07`（≈2⁻²³，1 ULP） | **`1.296956e+00`** |
| 认知流 `max\|out−in\|` | `5.960464e-08`（≈2⁻²⁴，1 ULP） | **`1.123835e+00`** |
| grad None / 精确零 / 非零 | **64 / 4 / 0** | **0 / 0 / 68** |
| 参数移动（68 张量） | **0 / 68** | **62 → 65 / 68** |
| optimizer 分组 | `{'1': 68}`，0 个游离 | `{'1': 68}`，0 个游离（**未被扰动**） |

未移动的 3~6 个张量与复审对控制组的解释一致：warmup 起始 lr 为 `1e-7`
（`ChainedScheduler` 的 `LinearLR(start_factor=0.001)`），部分梯度小到一次更新落在 fp32 分辨率以下。
**关键是 68 个全都拿到了非零梯度**，不再有 `None` 或精确零。

**静态判据**：复审 `07` §5.3 把「三条红一起变绿」定为修复的验收判据。
用**同一版本工具、同一组豁免**跑修复前后：

```
18106b05 : ORPHAN episode_stream_sampler / UNUSED MemoryVLAEpisodeStreamBatchSampler
           / DRIFT plan='False' shipped=True   ==== 3 finding(s) ====  preflight FAILED
166b8756 : （只剩 BottleneckSE 与两段 copy-fidelity 的既有豁免）
                                               ==== 0 finding(s) ====  preflight PASSED
```

两次的豁免完全相同，说明判据是活的、配置一致 —— 这就是这条对照的阳性控制。

> ⚠️ 工具位置在本次会话期间变了：`$ROL_JFS/port/_shared/` → `~/storage_policy/tools/port/`
> （2026-08-04 08:41），且新版**刻意不等价**（`--config`/`--subdir` 必须显式传，
> 扫不到东西时退出 2 而不是 0）。上表两次都用**新版**跑，以免比较被工具版本污染。

**顺带订正的两处文档**（复审列为加重情节）：`04-port-plan.md:78` 默认值 `False` → `True`；
`:58` 的「+ 可选 batch sampler 选择」承诺，标注它已兑现但**落在 `train.py` 而非 config 模块** ——
config 模块选不了 `train.py` 硬编码的 sampler，这正是该承诺当初静默死掉的结构性原因。

---

## P1-1 · 全部证据产自补齐了缺失集成的自建 harness —— **已修复**

复审说得对，而且**比原文更广**。不只是 E 档：**A / B / D 档与全部 5 个消融实际跑的是
`--sampler sequential`**（`run_gears.py:145-150`）——一个仓库里根本不存在的手写连续索引列表。
它碰巧产出 episode 连续批，所以 B 档「68/68 有梯度」是这条假路径的产物；
而真实路径上同一测量是 `64 None / 4 精确零 / 0 非零`。

**改了什么**：本轮所有档位一律从 `train.py` 真实入口进。观测装置是
`$ROL_JFS/port/memoryvla/fix/run_real.py`，规矩是**只 wrap，不 new**：

- `runpy.run_path("train.py", run_name="__main__")` 原样跑真实入口，train.py 一行未改；
- 包住 `SimpleTrainer.__init__`，从**宿主自己建好的** `self.dataloader`（post-`prepare()`）、
  `self.model`、`self.optimizer`、`self.batch_processor` 上读数；
- **不构造** sampler / dataloader / optimizer / model builder 中的任何一个；
- `train.py` 全程没有任何 seed 调用，`set_seed(0)` 由装置注入 ——
  baseline 与修复后施加**同一注入**，比较才公平。

于是复审 `06f` §1 列的三处空白同时被填上：真实 optimizer 分组（不再是 `SGD(flat, lr=0)`）、
**非零 lr**（参数是否真的动）、`num_workers=4` 的 worker 随机流（不再是 0）。

**顺带发现一条，已写进 `MIGRATIONS.md`**：护栏必须查 `accelerator.prepare()` **之后**那个
dataloader —— prepare 会把 batch_sampler 重新包一层（`BatchSamplerShard`），
**「构造出来的」不等于「被迭代的」**。本次单卡实测 accelerate 没有加包装层，
但护栏仍逐层拆包，单元测试里「藏在 shard 后面」与「压根没接上」两种情形都覆盖。

---

## P2-1 · deploy 分支白名单缺 `uuid` —— **延后**

**不修的理由**（不是「不重要」）：

1. 落点在 `config_robodojo_dataset.py` 与 `robodojo_eval.py`，**在本次三处允许改动之外**。
   范围锁死是为了保住已花算力买到的结论（C 档 10/10 逐位一致、接口 32 项一致、
   侵入度无虚报、68 张量全进 optimizer group 1），范围外改动会让它们全部作废重跑。
2. 它是**响亮的崩溃**（`wrapper.py:161` 抛 `KeyError`），不是静默失效。
   按复审自己的协议，「会崩的优先级天然低」。
3. 它与自陈遗留 1（`reset()` 未接 `robodojo_eval.py` 的 episode 循环）**必须一起修**：
   只加白名单会把 deploy 从「崩」变成「不崩但记忆库跨 episode 串味」——
   **把响亮失败换成静默失败，是净亏损。**

**记入遗留**：推理 / deploy 路径至今**从未被实际执行过**。两件事一起修，
且冒烟必须跑到**第 2 条 episode**（`reset()` 清的是上一条，N=1 永远不走那条路径）。

---

## P2-2 · cite 行号漂移 32% —— **延后**

复审的判断成立：内容零幻觉、`未确认` 用得诚实，**但行号不能照着翻**。
建议的修法（cite 改成「文件 + 符号名」或脚本生成校验）是对 `docs_analysis/memoryvla/`
全部 9 份文档的批量改写，**远超本次三处**。

本轮**新写**的 cite 一律带符号锚点或「行号 + 符号名」（如 `sampler.py:108-117` 同时写出
`__init__`），存量 cite 未动。

**记入遗留**：批量改写 + 一个校验脚本。注意复审 `07` §3 的教训 ——
「cite 指向的行存在」这条判据检测率为 **0**（漂移后的行号照样存在），
要检测必须比对**符号**，不是比对行是否存在。

---

## P2-3 · 8 个新类里 6 个没有方法名前缀 —— **不修，显式记录取舍**

违反的是移植方自己在 `MIGRATIONS.md:12` 立的约定。复审也写明可辩护。

**取舍**：`TimestepEmbedder` / `CrossTransformerBlock` / `BottleneckSE` / `GateFusion` /
`CogMemBank` / `PerMemBank` **保留 A 的原名**，因为这 6 个类是逐行对照移植的
（C 档 10/10 逐位一致，copy-fidelity ratio ≥ 0.998），原名让「这段对应源仓哪一段」一眼可查；
改名会切断这条溯源链。**代价**：`GateFusion` / `TimestepEmbedder` 在本仓极易撞名。
**缓解**：它们只从 `robo_orchard_lab.models.memoryvla` 导出，不进任何全局 registry，
真撞名时是 import 冲突这种响亮错误。

**约定被违反却没人记录，这才是真问题** —— 下一个人会以为是疏忽。现在记下来了。

---

## P3-1 · D 档「开启 +10% 时间」不成立 —— **已重测，结论撤回**

复审是对的，而本轮实测把它坐实得更死：**两次完全同配置（都是关闭态）的 baseline，
墙钟 `260.9 s` vs `203.6 s`，差 22%**。卡是共享的。

**重测后的 D 档**（旧数字全部作废——旧的是在 `--sampler sequential` 下测的，bank 恒为 `[1]`）：

| 运行 | 参数量 | 其中 memoryvla | 峰值显存 | 墙钟 |
|---|---|---|---|---|
| baseline（关闭态） | 1,136,284,265 | 0 | 8.9767 GiB | 260.9 s |
| A 档（关闭态，修复后） | 1,136,284,265 | 0 | 8.9767 GiB | 203.6 s |
| B 档（开启，bs=4） | 1,143,751,529 | 7,467,264 | 9.3024 GiB | 309.5 s |
| E 档（开启，bs=8） | 1,143,751,529 | 7,467,264 | 13.1525 GiB | 292.8 s |

- 参数 **+7,467,264（+0.657%）**，与复审复现值一致。
- 峰值显存 **+0.3257 GiB**（bs=4）。bank 从恒 `[1]` 变成真的累积到 16，显存上涨符合预期。
- **墙钟只记录不解释**：噪声 ≥22%，任何 ±10% 量级的结论都没有意义。
  要真测需独占卡，或改用 CUDA event + 多次取中位数。
- sampler 构造耗时（新增观测）：`MemoryVLAEpisodeStreamBatchSampler` 扫全部 328,975 帧，
  bs=4 用 **9.46 s**（82,006 batch）、bs=8 用 **16.15 s**（40,856 batch）；
  宿主 `DistributedBatchFlagSampler` 是 **0.005~0.012 s**（82,243 batch）。
  一次性开销，可接受，**未做优化**（本次禁止性能优化）。
  batch 数少 237 个，是每条 episode 末尾不足一批被 `drop_last` 丢掉，符合预期。

---

## P3-2 · `build` 顺序位移初始化 RNG —— **延后，但影响已绕开**

`self.memoryvla = build(self.cfg.memoryvla)` 插在 `data_preprocessor` **之前**，
开启态会位移其后所有模块的初始化 RNG，所以上一轮 B 档「开/关最大差 `6.203e-02`」
混有主干初始权重不同的成分，不能解读为记忆库净效应。复审判断成立。

**为什么不修**：落点在 `structure.py` / `structure_qwen3_5.py`，在三处之外。

**为什么影响已经绕开**：本轮 B 档**不再依赖开/关差值**这个量。
判据换成了 grad 三态、参数移动率、恒等间隙 —— 三个都是**开启态内部**的观测，
不需要与关闭态做差，因此不受初始化位移影响。
关闭态等价性（A 档）也不受影响：`_build_memoryvla_cfg` 关闭时返回 `None`，
`build(None)` 不消耗任何 RNG —— 这正是 A 档 step 0 能精确为 0 的机制。

**记入遗留**：真要比「记忆库净效应」，得先把 `build` 挪到所有既有模块之后，
或在构造前后存取 RNG 状态。

---

## 护栏本身的验证

复审说「护栏比修复本身更值钱」，所以护栏也要有阳性对照 —— 故意配错，确认**真的 raise**：

| 用例 | 配置 | 结果 |
|---|---|---|
| `stream` 但开关关 | `dataloader_type="stream"` + `episode_stream_sampler=False` | **raise**：`... disagree. stream memory is only meaningful with episode-ordered batches ...` ✅ |
| 开关开但模式是 `group` | `dataloader_type="group"` + `episode_stream_sampler=True` | **raise**：同上，方向相反 ✅ |
| 采样权重会被吞掉 | `dataset_sample_weights` + episode sampler | **raise** ✅（`dataset_sample_weights` 不是已声明的顶层 config 键，`train.py:84` 的 unknown_keys 会先拦住，所以这条走单元测试直接验护栏函数） |
| 迭代的不是那个 sampler | dataloader 挂着宿主 sampler | **raise** ✅（单元测试） |
| 藏在 accelerate 包装层后面 | `BatchSamplerShard(episode_sampler)` | **放行** ✅（拆包正确，不误报） |
| 关闭态 / `--eval_only` | `enable=False` / `dataloader=None` | **放行** ✅ |

单元测试 **10/10 通过**（`$ROL_JFS/port/memoryvla/fix/guard_unit_test.py`）。

运行期护栏在 B / E 档的真实日志里各触发一次，**只 log 一次，不刷屏**：

```
MemoryVLAMemory[stream]: first training batch holds 1 distinct episode(s) across 4 samples.
MemoryVLAMemory identity probe on the first forward that reads history:
    {'perceptual': '1.296956e+00', 'cognitive': '1.123835e+00'} (tolerance 1e-05)
```

**一个设计细节，写下来免得以后被当成 bug**：恒等探针触发在「**第一个真正读到历史的
forward**」，不是字面第一个 forward。episode 首帧 bank 本来就是空的、恒等旁路是**正确行为**，
按字面第一次触发会在修好的路径上必然误报。判据是「已有 episode 攒了条目」或
「同 batch 内有两个样本同属一个 episode」（`process_batch` 按顺序遍历并边写边读）。
另外探针要求**每一条启用的流都非退化**（取 `min` 而非 `max`），
否则一条死流可以藏在一条活流后面。

---

## E 档：Memory 冒烟必须跨过 episode 边界

60 步 / bs=8 / RoboDojo Memory 六任务。**关键证据是 bank 回落**：

```
step  0 : eps=1 maxlen=8      ← 新 episode 起步
step  1 : eps=1 maxlen=16     ← 撞到 mem_length=16 封顶
...
step 53 : eps=1 maxlen=16
step 54 : eps=1 maxlen=8      ← 跨过 episode 边界，clear_episode 真的跑了
step 55 : eps=1 maxlen=16     ← 新 episode 重新累积
...
step 59 : eps=1 maxlen=16
```

`eps` 始终为 1，说明上一条 episode 被**清掉**而不是堆积——这是 `stream` 模式该有的行为。
**单条 episode 的冒烟永远走不到 step 54 那一行**，绿色会是假绿色。

全程 `distinct_episodes_in_batch = 1`；grad 首尾都是 `0 None / 0 零 / 68 非零`；
参数移动 63/68（峰值 65/68）；恒等间隙 per `1.3496 → 1.4113`、cog `1.1261 → 1.1034`。

---

## 复审记为「无法验证」的两项 —— 修复后仍然无法验证

- **DDP / 多卡**：本机任意两卡 gather 必崩 `ILLEGAL_ADDRESS`，硬限制。
  **修复后风险反而更具体**：接上 sampler 后各 rank 拿到 `spans[rank::num_replicas]`，
  而 episode 长度差异极大（中位 276 → 1203 帧），**各 rank 的 `__len__` 不相等、收尾不齐**。
  **这一条连复审都没记**，已写进 PORT-STATUS 遗留。
- **外部真实 ckpt 加载**：bucket 只有 v9，config 是 v10，`vlm.*` 全线 size mismatch，
  且 v10 warm-start 在 http URL 后面而本机无外网。
  本轮所有档位一律 `checkpoint=null`（随机初始化 + 本地 `vlm_pretrain`），
  与移植当时的 harness 同口径。

## 本轮新增的遗留风险（修好了也不能略过）

1. **训练动力学变了**：每 batch 从 4 个 episode 变成 1 个 —— 梯度方差、epoch 内样本相关性、
   归一化层统计全都与关闭态不同。**A 档证明的是「关闭态没变」，不是「开启态已被验证」。**
2. **DDP 的 `__len__` 不齐**（上节）。
3. **`ulimit -n` 默认 1024 不够**：6 任务 × 3 个 LMDB env × (4 worker + 父进程)；
   接上 episode sampler 后 `_episode_spans` 要走遍 328,975 帧，**父进程也初始化 LMDB**，
   更紧。症状伪装成 `Pin memory thread exited unexpectedly`（真因是 worker 里的
   `OSError: [Errno 24]`），**看起来像 dataloader 抖动，不像资源限制**。
   本轮 A 档头两次、B 档三次尝试全折在这上面。→ 跑训练前 `ulimit -n 65536`。
