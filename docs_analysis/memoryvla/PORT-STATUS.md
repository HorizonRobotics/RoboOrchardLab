# PORT-STATUS — MemoryVLA → HoloBrain

**日期**：2026-08-03（`date +%Y-%m-%d`）
**日期（第二轮修复）**：2026-08-04 · **日期（第三轮修复）**：2026-08-04
**日期（第四轮修复）**：2026-08-05 · **日期（第五轮修复）**：2026-08-05
**总判定**：**不自评**。
第一轮自评 PASS → 独立复审 🔴 **REJECT**（P0-1：`episode_stream_sampler` 是死开关）。
第二轮修复 → 独立增量复审 🟡 **ACCEPT-WITH-FIXES**（P0×0 · P1×2 · P2×1 · P3×4，
见 `09-incremental-review.md`）：P0-1 与 P1-1 确认真闭环，剩下的是「修的覆盖面比声称的窄」。
第三轮修 P1-A / P1-B / P2-A / P3-A → 独立增量复审 🟡 **ACCEPT-WITH-FIXES**
（P0×0 · **P1×1** · P2×3 · P3×4，见 `11-incremental-review_v3.md`）：P1-A / P1-B 主干确认真闭环，
但 P1-B 的修法留了一个**配置可达**的角落（`group` + `batch_size=1`）。
第四轮修 P1-C / P2-A′ / P2-B / P2-C / P3-E / P3-F / P3-H → 独立增量复审 🟡 **ACCEPT-WITH-FIXES**
（P0×0 · **P1×1** · P2×3 · P3×2，见 `13-incremental-review_v4.md`）：P1-C 确认真闭环且超出契约要求，
七条 P2/P3 全部闭环；剩下的问题**全在风险册的记法上，没有一条在代码正确性上**。
第五轮（本轮）**纯文档**，清 P1-D / P2-D / P2-E / P2-F / P3-I 五条，**零代码改动**，
**裁决交给下一轮独立复审**。
逐条应答：第二轮见 `08-review-response.md`，第三轮见 `10-review-response.md`，
第四轮见 `12-review-response.md`；第五轮无独立应答文档 —— 五条修复全部就地写在被订正的位置。

| 项 | 值 |
|---|---|
| A 的 repo | `github.com/shihao1895/MemoryVLA` @ `0eef5c3`，MIT （判断/方案，非实测） |
| A 的环境状态 | **可运行**（`memvla_cu128`，实测 import 全栈成功，8 卡可见）；全程只读，未装/升任何包 （判断/方案，非实测） |
| 宿主基点 | `3ce31c0c`（`feature/memory_dev1` 的 tip，tag `memory_dev1-stage1-20260803`） （判断/方案，非实测） |
| 分支 | `port/memoryvla` （判断/方案，非实测） |
| 依赖档位 | **E0** —— 宿主主环境零改动，**差异包清单为空**，未建 `.venv_memoryvla` （判断/方案，非实测） |

## 移植了什么 / 放弃了什么

| 组件 | 处置 | 一句话理由 | 耦合类型 |
|---|---|---|---|
| `CogMemBank` / `PerMemBank` | 移植 | 方法本体：按 episode 巩固+检索+融合 | **T3 + T4** （判断/方案，非实测） |
| `CrossTransformerBlock` | 移植 | 检索算子 | T1 （判断/方案，非实测） |
| `GateFusion` | 移植 | 自适应融合 | T1 （判断/方案，非实测） |
| `TimestepEmbedder` | 移植 | 历史帧的时序编码 | T1 （判断/方案，非实测） |
| `BottleneckSE` | 移植但**未接入** | 是方法的一部分且已数值验证，但它把通道压到 256 会破坏与 decoder 的形状契约（宿主特征本来就是 `embed_dims=384`） | T2 （判断/方案，非实测） |
| `MemoryVLA` 壳类 | 放弃 | 绑死 `PrismaticVLM`，宿主的对应职责由 `HoloBrain_Qwen2_5_VL._forward` 承担 | — （判断/方案，非实测） |
| `ActionModel` / DiT | 放弃 | A 的动作头；宿主有自己的 `HoloBrainActionDecoder`，换掉等于换模型 | — （判断/方案，非实测） |
| FSDP 策略 / overwatch / CLI / trainer | 放弃 | 协议红线：A 的基础设施一律接宿主的 | — （判断/方案，非实测） |

## 侵入度：**L1**，触及宿主已有文件 **5 个**

> **订正 3（2026-08-05，第四轮，复审 P2-A′）：这一行的数字**第三次**带着过期值发布。**
> 标题原写 `train.py +38/−6 · sampler.py +94/−1`，而 `+94/−1` 与订正块里的
> `wrapper.py +118/−0` 都是 `18106b05..f6dfd1e8` 的量；它们随第二个 commit `49b2178c`
> 一起发布，而那个 commit 恰好让它们过期。
>
> **根因不是粗心，是标题那一行没有基点限定** —— 一个不带基点的数字无法自证过期。
> 现行口径因此改成「基点 + 截至 commit + 逐文件表」，任何一次代码提交后重跑下面的命令即可。
>
> **基点 `18106b05`：**
>
> | 文件 | 截至 `49b2178c`（第三轮 tip） | 截至 `fc33a5db`（第四轮代码提交） |
> |---|---:|---:|
> | `projects/holobrain_internal/common/train.py` | **+38 / −6** | **+38 / −6**（本轮一行未动） |
> | `robo_orchard_lab/models/memoryvla/sampler.py` | **+106 / −5** | **+202 / −5** |
> | `robo_orchard_lab/models/memoryvla/wrapper.py` | **+204 / −0** | **+330 / −0** |
> | `robo_orchard_lab/models/memoryvla/__init__.py` | **+2 / −0** | **+2 / −0** |
>
> 「触及宿主已有文件 **5 个**」在两个 commit 上都成立
> （`config_holobrain_common.py` · `config_robodojo_dataset.py` · `train.py` ·
> `structure.py` · `structure_qwen3_5.py`）。
> 第四轮另新增 3 个**测试文件**（`tests/test_robo_orchard_lab/models/memoryvla/`），
> 属新增而非触及。
>
> 重跑命令（**它不是闸门**，只是把「这几个数怎么算出来的」钉死，让下一次重算是机械动作
> 而不是回忆；脚本形式在 `$ROL_JFS/port/memoryvla/fix4/intrusion_line.sh`）：
>
> ```bash
> git diff --numstat 18106b05..HEAD -- robo_orchard_lab/models/memoryvla \
>     projects/holobrain_internal/common/train.py     # 逐文件 +/−
> git diff --numstat --diff-filter=M 3ce31c0c..HEAD -- "*.py"   # 「触及宿主已有文件 N 个」
> ```

> **订正（2026-08-04）**：原记 4 个文件。移植当时漏掉了 `common/train.py`，
> 而那正是 P0-1 —— sampler 开关没有读取者。修复后 `train.py` 是第 5 个。

> **订正 2（2026-08-04，复审 P2-A）**：本标题原写 **「0 删除」**，**不成立**。
> 实测 `git diff --stat 18106b05..f6dfd1e8`：`train.py` **+38/−6**、~~`sampler.py` **+94/−1**~~、
> ~~`wrapper.py` +118/−0~~（划掉的两个是 `f6dfd1e8` 的量，已被上面的订正 3 取代；
> 这个基点标注本身是诚实的，错的是标题那一行没有基点）。
>
> 那 6 行删除是**代码位移，不是逻辑改动**：`DistributedBatchFlagSampler(...)` 原本是
> `DataLoader(...)` 的一个实参，接线时被提到前面成了局部变量 `batch_sampler`，
> **构造参数一字未动**（含 `dataset_sample_weights=config.get(...)`）。
> 但「没动过」这件事**不能靠读 diff 判**——位移会不会改变关闭态只有实测能答。
> 复审用精确判据实测过：关闭态 5 个 run、10 组两两比较，**逐样本 id 序列 8/8 全部一致**，
> batch key `14 vs 14`、参数量严格相等、峰值显存逐位相同、sampler 链相同
> （见 `09-incremental-review.md` §4.1）。**位移成立，但判据是实测不是阅读。**
>
> 教训是「0 删除」这种**听起来最无害的自述最容易没人核**：它被重写过一次
> （上一轮把 4 改成 5）而同一行里的另一个数字照样错着。

| 文件 | 档 | 改动 |
|---|---|---|
| `models/holobrain/structure.py` | L1 | 一个 config 字段 + 一行 `build` + 一个 `if` + 一次调用 （判断/方案，非实测） |
| `models/holobrain/structure_qwen3_5.py` | L1 | 一行 `build`（它跳过父类 `__init__`，必须单独加） （判断/方案，非实测） |
| `configs/data_configs/config_robodojo_dataset.py` | L1 | 开关打开时给 ItemSelection 白名单加 `step_index` （判断/方案，非实测） |
| `configs/config_holobrain_common.py` | L1 | `cfg.memoryvla.*` 命名空间 + `_build_memoryvla_cfg()` （判断/方案，非实测） |
| `common/train.py` | L1 | 一个开关判断选 batch sampler + 一条装配期护栏调用（2026-08-04 修复 P0-1）|

新增文件（L0）：`models/memoryvla/{__init__,memory_bank,wrapper,sampler}.py`、
`configs/dataset_specs_memoryvla_robodojo_memory.py`、`docs_analysis/`。
**未触发 Gate B**（无 L3 改动）。**未触发 Gate E**（E0）。

## 验证结果（2026-08-04 全部重测；命令与证据见 `08-review-response.md`）

> **旧的五档数字全部作废。** 上一轮全部产自 `run_gears.py`：A/B/D 档与 5 个消融跑的是
> `--sampler sequential`（一个仓库里不存在的手写连续索引列表），E 档跑的是自建的
> `MemoryVLAEpisodeStreamBatchSampler`。**宿主没有任何路径能到达那两种装配。**
> 本轮所有档位一律从 `train.py` 真实入口进，观测装置只注入不构造。

| 档 | 判据 | 结果 |
|---|---|---|
| 第 0 步 确定性 | 同配置跑两遍自比 | **step 0 精确 `0.000000e+00`；20 步内峰值 `2.899e-04`**（step 11）。真实入口**不逐位可复现**，见下节 |
| **A 关闭态等价** | ① step 0 严格 0 ② 全程 ≤ 实测地板 | commit `701679a9`：step0 **`0.000000e+00`**，峰值 `1.249e-04`；commit `166b8756`：step0 **`0.000000e+00`**，峰值 `1.554e-04`。**两者都低于两次同配置 baseline 之间的 `2.899e-04`**。参数量 `1,136,284,265` 与移植前一致；关闭态 sampler 仍是 `DistributedBatchFlagSampler`；`memoryvla.*` 张量 **0 个**（模块根本不构建）→ **PASS** |
| **B 开启态** | 走真实入口，grad 与参数移动 | sampler 链实测 `['MemoryVLAEpisodeStreamBatchSampler']`；每 batch **1** 个 episode（原 4/4）；grad **0 None / 0 零 / 68 非零**（原 64/4/0）；参数移动 **62→65 / 68**（原 0/68）；恒等间隙 per `1.297` / cog `1.124`（原 `1.19e-07` / `5.96e-08`）；68 张量全在 optimizer group 1，0 个游离 → **PASS** |
| **C 数值对齐** | `atol < 1e-5` | **10/10 逐位一致（`0.000e+00`）**，修复前后各跑一次，结果相同 → 改动未溢出范围 → **PASS** |
| **D 资源** | 量级正常 | 参数 `1,136,284,265 → 1,143,751,529`（**+7,467,264 / +0.657%**）；峰值显存 `8.9767 → 9.3024 GiB`（**+0.3257 GiB**）；**墙钟不下结论**，见下 |
| **E Memory 冒烟** | 跨过 episode 边界 | 见 `08-review-response.md` |
| 护栏自验 | 故意配错必须 raise | 见 `08-review-response.md` |
| 静态判据 preflight | 三条红一起变绿 | `18106b05`：`ORPHAN` + `UNUSED` + `DRIFT` → **FAILED**；`166b8756`：0 finding → **PASSED**。同一版本工具、同一组豁免 → **PASS** |

### 确定性：真实入口不是逐位可复现的，判据因此改成两档

上一轮记「地板恰为 0，故 A 档用严格判据」。**那是 harness 的性质，不是宿主的性质**——
`run_gears.py` 用 `lr=0`，权重不动，逐 step 值是「数据 + seed」的纯函数，误差没有累积的机会。

走真实入口（真 optimizer、真 lr、`num_workers=4`）实测：

```
step 0   0.000000e+00     ← 前向逐位一致
step 1   0.000000e+00     （单分量 1.788e-07）
step 11  2.899170e-04     ← 20 步内峰值
```

误差从**反向/optimizer 的 float32 非确定性归约**进来，前向本身精确。
开 `cudnn.deterministic` + `use_deterministic_algorithms(warn_only=True)` 只把峰值压到
`1.564e-04`，**压不到 0**（有算子没有确定性实现，warn_only 下继续走非确定性路径）。

**所以 A 档判据是两档**：

1. **严格档**：step 0 的 7 个分量与 total 必须**精确** `0.000000e+00`。
   这才是 A 档真正要回答的问题——接线有没有改动关闭态的前向。
2. **地板档**：其余步 ≤ 实测地板 `2.899e-04`。

**阳性对照**（没有阳性对照的通过 = 未验证）：开启态与关闭态的恒等间隙相差 **7 个数量级**
（`1.19e-07` → `1.297`），远在地板之上；也就是说这套判据能分辨的变化，比「把开关打开」
小得多。

### 关闭态等价性：改用精确判据（2026-08-04 第三轮订正 —— 方法论级，不只是这次的事）

上面那个「两档判据」还是**浮点判据**。复审用 5 个关闭态 run 做了 10 组两两比较，
结论是**浮点判据在真实入口上没有分辨力**，不能拿它当通过：

```
同代码组（共用同一份 train.py）: [4.101e-05, 1.159e-04]
跨代码组（base vs head）      : [5.102e-05, 9.108e-05]     ← 完整落在同代码组区间之内
10 组里最大的那个差 1.159e-04 出现在【共用同一份 train.py】的两个 run 之间
```

**这个量级与「是否同代码」不相关**，所以「差异很小 ⇒ 没改动」这个推理在真实入口上是无效的。

**上一轮修复契约里写的「A 档仍应为 `0.000000e+00`，非 0 即回退重做」这条前提，据此作废。**
它继承自第一轮，而第一轮那个 `0` 是在 harness 路径上测的——harness 消除了真实入口的
不确定性来源（`lr=0`，权重不动）。**照字面套会得到一个不适用的严格判据。**

**现行判据：~~五项~~ 四项精确量 + 一条结构判据，逐项与基线严格比对。**
（**峰值显存这一项已在下一节被实测降级为参考量**，第四轮复审用独立装置再次确认了降级正当。）

| 精确量 | 判据 | 为什么它精确 |
|---|---|---|
| 逐样本 id 序列（每 batch 的原始 `uuid`） | 与基线完全一致 | 接线改的就是选 batch 的那段代码，而它唯一能破坏的就是这个。无噪声 |
| batch key 集合 | 完全一致 | 关闭态 14 个 / 开启态 15 个（多 `step_index`）；多一个 key 就说明数据管线被动了 |
| 参数量 | 严格相等 | 关闭态 `1,136,284,265` / 开启态 `1,143,751,529` |
| ~~峰值显存~~ | ~~严格相等~~ | ~~关闭态实测逐位相同 `8.975615978240967 GiB`~~ —— **已降级，见下一节** |
| sampler 链（类型与嵌套） | 完全一致 | `['DistributedBatchFlagSampler']`；查 `accelerator.prepare()` **之后**那个 |

**结构判据（比上面四项都强）**：关闭态 `sys.modules` 里 `robo_orchard_lab.models.memoryvla*`
**一个都不出现**（`train.py` 与 `_build_memoryvla_cfg` 两处 import 都在分支内）。
它证明的是「被改的文件根本没参与执行」，而不是「执行了但结果一样」——
所以关闭态的**运行时**那一档，主要作用是标定噪声地板与判据分辨力，而不是等价性的主要证据。

**每一条精确判据都必须配阳性对照。** 没有阳性对照的「一致」结论不算证据——
判据可能只是失灵了。已实测的对照：`num_workers` 4→0 使浮点差达 `1.028e-01`，
**比噪声地板高 3 个数量级**，而逐样本 id 序列仍 8/8 一致（sampler 决定索引，worker 只负责取）。
所以这套测量确实有牙。

**浮点 loss 差仍然记录，但只作参考量，不作判据。**

### 峰值显存也不是精确量 —— 本轮实测降级（方法论级）

上一节把峰值显存列进「五项精确量」，依据是上一轮 5 个 run 逐位相同。**本轮 5 个同配置
关闭态 run 并不逐位相同**：

```
base_A_stream_off      8.976670265197754
head_A_stream_off      8.971695899963379     ← 低 4.97 MiB（0.055%）
head_A_stream_off_r2   8.976670265197754     ← 同代码同卡重跑，回到基线值
base_A_group_off       8.976670265197754
head_A_group_off       8.976670265197754
```

**那个离群值不可能来自本轮改动**：关闭态每个 run 的 `port_imported` 都是 `[]`，
被改的两个文件根本没被 import。同代码、同卡、同配置重跑一次就回到了基线值——
这是分配器的 run-to-run 差异该有的样子，不是代码差异该有的样子。

**所以峰值显存降级成参考量，与浮点 loss 同级。** 这与本轮对 A 档判据做的是同一件事，
理由也是同一条：**一个量要成为判据，先得证明它在「除了被测对象之外的一切」下都稳定。**
上一轮 5 个 run 恰好一致，就被当成了「精确」——样本量小的一致性不是精确性。

**剩下四项精确判据全部成立**（逐样本 id 序列 / batch key 集合 / 参数量 / sampler 链），
外加结构判据 `port_imported == []`。

> **第四轮：降级已被独立复现（2026-08-05）。** 复审用**自己的**装置跑两次同配置、同卡、
> 同代码的关闭态 run，得到 `8.976670265197754` vs `8.971459388732910`（差 5.21 MiB），
> 且低值与本轮记的 `8.971695899963379` **也不相同** —— 与「分配器 run-to-run 差异」一致，
> 与「代码差异」不一致。**降级正当，不是把不利判据洗掉。**
>
> **第四轮又一次复现**：关闭态 `stream`，`f770afe0` 上 `8.976670265197754`、
> `fc33a5db` 上 `8.971459388732910`（差 **5.34 MiB**），而同一对 commit 的关闭态 `group`
> 两边**逐位相同**。同一份代码差异在两条路径上给出不同的显存差 ⇒ 差的不是代码。
>
> **观测器污染，本轮终于测出了数（补上 `09` §7.3 ⑥ 与 `MIGRATIONS.md` 教训 9 的欠账）。**
> 同一开启态配置（`stream` bs=4，20 步）跑两次，一次带 identity 探针一次 `--no-identity`：
>
> ```
> with identity probe : 9.302354336 GiB   (gpu 3, 20 forwards recorded)
> without             : 9.301393032 GiB   (gpu 0,  0 forwards recorded)
> difference          : +0.98 MiB  (+0.010%)
> ```
>
> **这个数要连着噪声地板一起读**：run-to-run 抖动本身就有 ~5 MiB 量级，
> 所以正确结论是「**观测器开销在这套测量下不可分辨，上界约 5 MiB**」，
> **不是**「开销是 0.98 MiB」。
> 顺带**订正 `MIGRATIONS.md` 教训 9 里那句「直接把 D 档显存读数抬高了 6 MiB 量级」**——
> 那是推断不是实测，实测差值比它小一个量级，且落在噪声里。
> 这恰好是同一条教训的第三次出现：**一个量在被当成任何东西之前，先得知道它的地板在哪。**

### 关闭态的批次顺序与训练 seed 无关（找阳性对照时挖出来的）

给「逐样本 id 序列」找阳性对照时，前两次尝试都失败了，两次都不是判据的问题：

| 尝试 | 结果 | 原因 |
|---|---|---|
| `--seed 1` vs `--seed 0` | **20/20 batch 完全相同** | `DistributedBatchFlagSampler._indices_generator`（`dataset_wrapper.py:133`）自建 `np.random.default_rng(self.seed + self._epoch)`，而 `self.seed` 来自一个 **`train.py` 从不传** 的构造参数。accelerate 的 `set_seed` 够不着它 |
| `set_epoch(7)` | **20/20 batch 完全相同** | `set_epoch` 确实跑在了 trainer 随后迭代的那个 `DistributedBatchFlagSampler` 上（已记进 JSON），顺序仍不变 ⇒ 排列在 trainer 存在之前就定死了 |
| **注入构造参数 `seed=99`** | **0/20 相同** ✅ | 这是唯一在排列决定之前能动的输入 |

第三种同时证明了它**只动了顺序**：参数量、batch key 集合、sampler 链三项全部不变。

**顺带一条值得写下来的宿主性质：关闭态的批次顺序与训练 seed 无关。**
指望换 seed 就换一批数据的人会得到「一模一样」，并且很容易把这个「一样」
当成别的东西的证据。

### D 档：墙钟不用来下结论

两次**完全同配置**（都是关闭态）的 baseline，墙钟 `260.9 s` vs `203.6 s`，**差 22%**。
卡是共享的（本次同卡上有同事进程，另有本人的 `collect_data` 作业占着别的卡）。
所以 D 档只报参数量与显存这两个可信量，**墙钟只记录不解释** ——
这也证实了复审 P3-1：上一轮「开启 +10% 时间」落在噪声内，结论不成立。

要真测时间需独占卡，或改用 CUDA event + 多次取中位数。

### `ulimit -n` —— 不是新发现，是「写了但没人执行」的活标本

默认软限 **1024**，这套数据会击穿它：6 个 RoboDojo 任务 × 3 个 LMDB env（meta/image/depth）
× (4 worker + 父进程)。**接上 episode sampler 后更紧**——`_episode_spans` 要走遍全部
328,975 帧，于是**父进程也初始化 LMDB**，而宿主 sampler 从不这么做。

症状极具欺骗性：worker 里是 `OSError: [Errno 24] Too many open files`，
浮到上层变成 `RuntimeError: Pin memory thread exited unexpectedly`，
**看起来像 dataloader 偶发抖动，不像资源限制**。本次 A 档头两次尝试、B 档三次尝试全折在这上面。

→ **跑训练前 `ulimit -n 65536`**（硬限 1048576，普通用户可自行提升）。

**订正**：这条**不是本轮新发现**。`06-verification.md` 的抬头里就写着 `ulimit -n 65536`，
移植方当时已经知道并设了。本轮照样折了 5 次（A 档 2 次、B 档 3 次），原因是它**只写在文档里、
没有写进任何会被执行的东西**——新的 runner 自然不会设。
这与 P0-1 是同一个形状：`04-port-plan.md` 三处预言了 sampler 风险，预言本身不会接线。
→ **凡是「跑之前必须先做 X」，就把 X 放进 runner，不要放进段落。**现已写进 `fix/gear.sh`。

## `dataloader_type="group"` 的现状（2026-08-04 第三轮，P1-B）

**结论先说：`group` 现在是一个可用配置，配方是 `dataloader_type="group"` **且**
`episode_stream_sampler=True`。** 被审 commit `2b739226` 上它没有任何可用配置。

### 为什么原来两条路都堵死

护栏 `assert_episode_stream_wired` 当时的判据是
`(dataloader_type == "stream") != bool(episode_stream_sampler)`，
背后是一条**假前提**：「episode sampler 只对 `stream` 有意义」。

`dataloader_type` **不是 dataloader 选择器**——`train.py` 全文零处读它
（`git grep dataloader_type` 可证）。它只被 `memory_bank.py` 消费，选的是**记忆跨度**：

| 值 | 行为 | 跨度 |
|---|---|---|
| `stream` | bank 跨调用存活，换 episode 才 `clear_episode`（`memory_bank.py:364-368`） | ≤ `mem_length`，跨 batch |
| `group` | 每次训练调用顶部 `self.bank.clear()`（`memory_bank.py:361`），batch 内按 `group_size` 轮转（`:374`） | 一个 batch 之内 |

**两者都需要 episode 连续的批**，只是跨度不同。于是那条 XOR 把
`group + sampler=True`（唯一能用的组合）判成冲突并 raise，
而 `group + sampler=False` 被放行——落进 P0-1 的失效签名，且三道护栏一道不响。

### 改了什么（只动 `sampler.py` 与 `wrapper.py`，`train.py` 一行未动）

1. **装配期判据从「两个键是否相符」换成「实际 sampler 链里有没有 episode sampler」。**
   判的是拿到的对象，不是配置项的名字，所以覆盖所有键的组合，包括还没被发明的。
2. **`wrapper.py` 的 batch 组成检查去掉了 `dataloader_type != "stream"` 的提前 return。**
   那个闸门恰好在最需要它的那种配置下把它关掉了。
3. **新增 bank 存活性看门狗**：跑满 K=8 次训练 forward 后，若从没有任何 episode 的
   bank 长度超过 1，直接 raise。它**不需要历史先存在**——而「历史永远不存在」正是失效形态本身。
   它不读任何配置键，判据是后果，所以任何路径、任何模式、将来任何新的
   `dataloader_type` 都覆盖得到。
4. **`_history_will_be_read` 在 `group` 下不再把「上一批残留的 bank」算作历史**：
   `group` 会在 `process_batch` 顶部清空，算进去会对**行为完全正确**的配置误报。
5. **三处会把使用者引向那个无护栏格子的报错文案全部改写**，
   并加了断言禁止它们回来（见下）。

### 护栏有牙 —— 人为构造退化场景，从真实入口确认它会触发

**未触发即等于没有护栏。** 所以每道护栏都成对交付：一个正常配置不响的用例，
一个已知坏配置必响的用例。故障注入用的是 runner 的 `--break-episode-order`：
它按 sampler **自己的** span 表重排它自己的输出，**sampler 对象本身不动**，
所以装配期护栏看到的链是对的、会放行——只有消费端能发现批次已经坏了。

| 场景 | 谁应该抓到 | 实测 |
|---|---|---|
| `group` + sampler 关（**P1-B 的洞**） | 装配期链判据 | ✅ `rc=1`，`train.py:236` → `sampler.py:264` raise（**行号截至 `49b2178c`**；在 `fc33a5db` 及之后是 `sampler.py:298`） |
| `stream` + sampler 关（既有阴性用例回归） | 装配期链判据 | ✅ `rc=1`，同一处 |
| 故障注入，bs=4 | batch 组成检查 | ✅ `rc=1`，`wrapper.py:302`，「4 samples from 4 different episodes」 |
| **故障注入，bs=1** | **只有看门狗能抓** | ✅ `rc=1`，`wrapper.py:354`，第 8 次 forward 报「no episode's memory ever grew past a single entry」 |

**最后一行是本轮最关键的一条证据**：batch=1 时 batch 组成检查判不了（它需要 >1 个样本），
恒等探针也永不 arm（没有历史），**三道护栏里只剩看门狗能响，而它响了**。
这正是 `06-verification.md` 末尾那条「`group` + batch=1 会恒等但看起来正常」的警告——
它从此不再只是一段文字。

> **第四轮重跑（commit `fc33a5db`）+ 新增第三档。**
> 前两档仍然触发，行号随本轮改动位移；第三档是**本轮新增的、专门去找残留洞的**一档。
>
> | 场景 | 实测 | 判定 |
> |---|---|---|
> | 故障注入，`stream` bs=4，4 步 | raise `wrapper.py:372 in _check_episode_stream`，**第 0 次 forward** | ✅ 仍触发 |
> | 故障注入，`stream` bs=1，12 步 | raise `wrapper.py:462 in _check_bank_liveness`，**第 8 次 forward**，`grad 68/0/0` | ✅ 仍触发 |
> | **故障注入，`stream` bs=1，4 步（新增）** | **`rc=0`**、bank `[1,1,1,1]`、`grad 64/4/0`、移动 `0/68`、护栏日志只有 1 行正常 INFO | ❌ **无人看守 —— 这就是残留洞** |
>
> 第三档**如实交付，不掩盖**：它需要主动注入（或 `_episode_spans` 在别的数据集上不成立，遗留 12），
> **不是配置可达**的。它证明的是 K 这个时间闸门在短跑下的窗口仍然存在，只是已经不再有配置能走进去。
>
> **每一档都核对了 raise 的栈帧函数名，不只看退出码** —— 上一轮复审踩过一次
> GPU OOM 被 `rc=3` 促成「按预期 raise」。`gear4.sh` 的期望写成 `raise:<函数名>`，对不上直接判 BAD。

单元测试（CPU，本机无 pytest，写成退出码脚本，均在 `fix3/`）：
`guard3_unit_test.py` **22/22**（15 个装配期用例 + 7 条文案卫生断言）·
`guard3_probe_test.py` **24/24**（看门狗 8 · batch 组成 6 · 恒等探针 5 · `_history_will_be_read` 5）。

> **订正（2026-08-05，第四轮，复审 P2-C）：这两个脚本当时不在 git 里。**
> `git ls-files | grep -c guard3` = `0`，仓内 memoryvla 相关测试为 **0 个**，
> 没有任何 runner 会再跑它们。也就是说上面那两个 `22/22` / `24/24` 是**一次性的手工结果**，
> 不是回归保护 —— 而下一段说「加了断言禁止它们回来」时，默认读者会读成后者。
>
> **已修（commit `fc33a5db`）**：改写进
> `tests/test_robo_orchard_lab/models/memoryvla/{test_sampler_guard,test_wrapper_guards}.py`，
> 即 `tests/Makefile` 的 `test_ut` 目标树（`pytest -c tests/pytest.ini tests/test_robo_orchard_lab`）。
> 合计 **84 项，全过**（46 项承接自 `fix3/`，其余为本轮新增，含 P1-C 的跨度用例与扩展的文案卫生断言）。
>
> ⚠️ **这句话的准确边界**：执行证据来自 `.git/run_tests_nopytest.py`
> （本机 `holobrain_internal` 没有 pytest，装它会破 E0「宿主主环境零改动」）。
> **「CI 里的 pytest 会不会真的收集到这两个文件」本轮无法验证**，已记入无法验证清单。
> 能给的结构性论证只到「文件落在 `test_ut` 的目标树内」。

**文案卫生断言是本轮新增的**，因为 P1-B 的成因不是缺护栏，是护栏的**文案**
把人指引进了那个洞。断言：任何 raise 文本都不得出现
`episode sampler is only meaningful` / `turn the episode sampler off` /
`switch the bank to dataloader_type='group'` / `episode_stream_sampler=False`，
且每条「sampler 不对」的报错都必须点名 `episode_stream_sampler=True` 这个正解。

### 结果矩阵（全部从 `train.py` 真实入口实测）

| 配置 | 结果 |
|---|---|
| `stream` + sampler `True` | 通过（不变） |
| `stream` + sampler `False` | **raise**（不变，文案改写） |
| **`group` + sampler `True`** | **通过 —— 新的可用配置**（前提见下面的订正：要 `batch_size ≥ 2` 且 `group_size ≥ 2`） |
| **`group` + sampler `False`** | **raise**（新；原来是静默恒等） |

> ### ⛔ 订正（2026-08-05，第四轮，复审 P1-C）：这张表**只有两个维度，而失效需要三个**
>
> 这里原本写着一句 **「不存在「memory 被构建 + 静默退化 + 无告警」的组合」**。
> **那句话已被实测证伪，本轮撤回。** 反例是**纯配置可达**的 —— 没有任何故障注入，
> sampler 也正确接在链上：
>
> ```
> dataloader_type="group"  +  batch_size=1  +  episode_stream_sampler=True  +  max_step=4
>
> error                     : None            rc = 0
> bank max_len / step       : [1, 1, 1, 1]
> grad none / zero / nonzero: 64 / 4 / 0      ← P0-1 的失效签名，逐项相同
> params moved              : 0 / 68          ← P0-1 的失效签名，逐项相同
> identity gap              : per=5.960464e-08   cog=0.000000e+00   ← 精确恒等
> guard log lines           : 2（两条都是正常 INFO，无任何告警）
> _bank_liveness_checked    : False           ← 看门狗从未裁决（4 < K=8）
> ```
>
> 同一配置跑满 12 步确实会 raise，但**触发时说错了原因**：文案断言
> 「The batches reaching this module are **not episode-contiguous**」——它们**是**连续的
> （`distinct_episodes_in_batch = 1`）；并建议 `memoryvla.episode_stream_sampler=True`
> ——它**已经**是 `True`。使用者照做会发现无事可做。
>
> **两条真正的病因**（比落点更值得记）：
> 1. **时间闸门** —— 看门狗要跑满 `K=8` 次 forward 才裁决，所以 **< 8 步的运行完全无保护**，
>    而 4–8 step 恰是本项目自己的冒烟长度（`05-ablation-matrix.md` 整张表 8 step，
>    `09` 的 `C_group_host.json` 4 step）。
> 2. **归因歧义** —— `bank 恒为 1` 既可能是「批次坏了」，也可能是「这个配置下记忆本来就不可能」，
>    而文案断言了前者。
>
> 而 `group` ∧ `batch_size == 1` **在构造期就是静态可判的**，根本不需要等任何 forward。
> **已在第四轮修复**（commit `fc33a5db`），实测矩阵见下。

### 支持矩阵（2026-08-05 第四轮，commit `fc33a5db`，全部从 `train.py` 真实入口实测）

**记忆跨度是这张表的组织原则**：`group` 在 `process_batch` 顶部 `bank.clear()`
（`memory_bank.py:361`），批内每 `group_size` 个样本再 `clear_episode` 上一组
（`:374-377`，配 episode sampler 时那是**同一条** episode）
⇒ **`group` 的记忆跨度 = `min(group_size, batch_size)`**，等于 1 就一定退化。
`stream` 的 bank 跨调用存活，所以 `batch_size=1` 在那边完全合法。

| `dataloader_type` | sampler | `batch_size` | `group_size` | `max_step` | 结果 | **谁在看守** |
|---|---|---:|---:|---:|---|---|
| `stream` / `group` | **False** | 任意 | 任意 | 任意 | **raise** `sampler.py:298` | 装配期**链**判据，`train_forwards=0` |
| **`group`** | True | **1** | 16 | **4** | **raise** `sampler.py:344` | 装配期**跨度**判据（**新**），`fwd=0`、护栏日志 0 行、训练从未开始 |
| **`group`** | True | **1** | 16 | 12 | **raise** `sampler.py:344` | 同上 —— **与步数无关** |
| **`group`** | True | 4 | **1** | 4 | **raise** `sampler.py:344`，`min(1, 4) = 1` | 同上 —— **`group_size=1` 是同一个失效的另一个键** |
| `group` | True | 4 | 16 | 4 | 通过，bank **恒为 4**，`grad 0/0/68`，移动 63/68 | 第一批检查 fwd0（批次坏了就在这里死） |
| `group` | True | 4 | 16 | 12 | 通过，同上 | 第一批检查 fwd0 **+** 看门狗 fwd8（`maxbank=4`） |
| `stream` | True | 4 | 16 | 4 | 通过，bank `4→8→12→16` | 第一批检查 fwd0 |
| `stream` | True | 4 | 16 | 20 | 通过，同上，恒等间隙 `1.296956` / `1.123835` | 第一批检查 **+** 看门狗 fwd8 |
| `stream` | True | **1** | 16 | 4 | 通过，bank `1→2→3→4` | 第一批检查 fwd0（判不了 bs=1，但会记录观测值） |
| `stream` | True | **1** | 16 | 12 | 通过，bank 涨到 8，`grad 0/0/68`，移动 62/68 | 第一批检查 **+** 看门狗 fwd8（`maxbank=8`） |
| `stream` | True | 4 | 16 (`mem_length=1`) | 12 | 通过，`grad 0/12/56`，移动 51/68 | 看门狗 fwd8 **主动站下**并 `WARNING`（见下） |

**`group` 现在没有「被构建 + 静默退化 + 无告警」的组合**：三种致命配置全部在
**第一次 forward 之前**就 raise，与 `max_step` 无关；能跑的两格都有 fwd0 的看守。

**一处主动去掉的误报**：`mem_length=1` 时巩固每写一条就把 bank 压回 1 条，
**bank 长度上界就是 1**，但那一条是真历史、检索照常发生 —— 实测 `grad 0/12/56`、
参数移动 51/68，模块**在工作**。旧判据会在第 8 次 forward 无故打死这个 run。
现在它 `WARNING` 说明「这条判据在此配置下失明，站下不裁决」，不 raise。
**误报是对「触发时必须指向真实原因」最彻底的违反**，所以一并修了。

> ### ⚠️ 订正（2026-08-05，第五轮，复审 P2-F）：上面这段只写了收益，代价必须一起写
>
> 站下**是永久的** —— `_bank_liveness_checked` 被置位后这道判据再也不运行。
> 所以在 `mem_length ≤ 1` 下，**批次真的坏掉也不会有任何护栏拦它**，
> 而且**不受 K 约束**（已记录的残留洞至少还有「跑够 K 步就会被抓」这个上界，这里没有）。
>
> 复审第四轮实测的一对（`stream`，`batch_size=1`，`mem_length=1`，**12 步 > K=8**）：
>
> | 档 | 注入 | 结果 |
> |---|---|---|
> | 健康对照 | 否 | `rc=0`、bank `[1]×12`、**`grad 0/12/56`、移动 51/68** ⇒ 模块在工作 |
> | 故障注入 | 是 | `rc=0`、bank `[1]×12`、**`grad 64/4/0`、移动 `0/68`** ⇒ **模块完全没算** |
>
> **两个 run 的护栏输出逐字相同**（都只有 fwd7 那一行 `WARNING ... standing down ...`）
> ⇒ **那行告警在「在工作」与「完全没算」之间不含任何分辨信息**，
> 一个见过健康 run 的使用者会学会忽略它。
>
> **这个交换本身是对的**（误报会打死正常 run，比漏报更立刻有害），
> 错的是只写了收益。⚠️ **注意 `mem_length ≤ 1` 只在 `batch_size=1` 时才真的没人看守**：
> `batch_size ≥ 2` 下第一批检查照样抓（复审实测 bs=4 + `mem_length=1` + 注入 →
> `wrapper.py:372`，第 0 次 forward）。→ 常设记录见**遗留 14**。

**残留洞，如实写明**：`stream` + `batch_size=1` + 批次**实际不连续** + 跑不满 K 步
⇒ 仍然无人看守。实测（第三档故障注入，`--break-episode-order`，4 步）：
`rc=0`、bank `[1,1,1,1]`、`grad 64/4/0`、参数移动 `0/68`、只有 1 行正常 INFO。
**它不是配置可达的** —— 要么 `_episode_spans` 在别的数据集上不成立（遗留 12），
要么像这里一样主动注入。同一注入跑满 12 步则被看门狗抓住（`wrapper.py:462`，第 8 次 forward）。

### D 档新增：`group` 路径的资源与行为（不覆盖 stream 的数值）

| 档 | bank 每步最大长度 | grad None/零/非零 | 参数移动 | 恒等间隙 per / cog | 参数量 | 峰值显存 | 20 step 墙钟 |
|---|---|---|---|---|---:|---:|---:|
| **B `stream`** bs=4 | `4→8→12→16` 封顶 | `0/0/68` | 62/68 | `1.296956` / `1.123835` | 1,143,751,529 | 9.3024 GiB | 299.76 s |
| **D `group`** bs=4, `group_size=16` | **恒为 4**（= batch） | `0/0/68` | 62/68 | `1.296956` / `1.123835` | 1,143,751,529 | 9.1012 GiB | 277.67 s |
| **D `group`** bs=4, `group_size=2` | **恒为 2** | `0/12/56` | 51/68 | `1.265461` / `1.132860` | 1,143,751,529 | 9.0717 GiB | 297.21 s |

三行都 `rc=0`、sampler 链都是 `['MemoryVLAEpisodeStreamBatchSampler']`、
每 batch 1 个 episode、68 张量全在 optimizer group 1、游离 0。**墙钟只记录不解释。**

**`group` 那行的 bank 恒为 4 正是它该有的样子**——记忆跨不出一个 batch，`mem_length=16`
从头到尾不起作用。这与 `05-ablation-matrix.md` 第 7 行（假路径上量到的 bank 峰值 4）
**在现象上一致**，但那一行的数字仍然作废：它是在宿主到不了的装配上产生的。

**`group_size=2` 那一行专门跑来走到组轮转分支**（`memory_bank.py:374`）：
`group_size=16 > batch=4` 时那条分支**永远不执行**，只跑第一行等于又一次 N=1 冒烟。
它顺带暴露了一个该记的事实：**`group_size=2` 时有 12 个张量拿到精确零梯度**
（bank 最长只有 2，巩固路径与更深的检索层从不激活），参数移动降到 51/68。
不是 bug，但**选小 `group_size` 等于关掉一部分模块**，值得写在这里。

## 新增 config 字段

`cfg.memoryvla.*`：`enable`(False) · `use_perceptual`(True) · `use_cognitive`(True) ·
`dataloader_type`("stream") · `group_size`(16) · `mem_length`(16) · `retrieval_layers`(2) ·
`use_timestep_pe`(True) · `fusion_type`("gate") · `consolidate_type`("tome") ·
`update_fused`(False) · `episode_stream_sampler`(True)。
**默认 `enable=False`，此时模块根本不构建。**

> **订正（2026-08-04）**：`episode_stream_sampler` 的读取者是
> `common/train.py`（DataLoader 构造处）。判据是 **`enable ∧ episode_stream_sampler`** ——
> 该键 ship 值为 `True` 但**挂在 `enable=False` 之下**，只读它会让全关配置也换掉 sampler。
> 装配期护栏 `assert_episode_stream_wired()` 会在开启态校验实际迭代的 sampler 类型。

## 降级说明

| 项 | 用的档位 | 影响 |
|---|---|---|
| 卡数 | 单卡 | 本机任意两卡 gather 必崩；**DDP 行为未验证** （cite: 本机已知约束） |
| batch | 4（A/B/D）/ 8（E），非默认 16 | 8 张卡都有同事进程占 12–18 GiB。不影响结论：A 档比的是同 batch 的两棵树，C 档不过模型 （cite: 实测 nvidia-smi） |
| 训练时长 | 20–45 step，`lr=0` | 验收线不含收敛（用户已确认） （cite: 验收线） |

## 已知问题

1. **感知记忆的语义与 A 不同**：A 记的是 LLM **之前**的视觉主干 patch，宿主记的是 VLM
   **之后**、已被语言条件化的特征。角色等价，内容不等价 —— **不能声称端到端与 A 数值可比**。
   模块级的 C 档对齐不覆盖这一点。
2. **认知记忆影响被稀释**：A 里那个 token 是 DiT 的全部条件输入；宿主 decoder 同时吃 264 个
   图像 token 和 L 个文本 token，改 1 个 token 的影响小得多。
3. **DDP 未验证**：`retrieval_blocks` 在无历史分支下不参与计算，一批内全部样本都无历史时
   会触发 DDP unused parameter。单卡实测 68/68 全有梯度，多卡未验证。
4. **`BottleneckSE` 是未接入的死代码**：已验证、有出处，但当前不在任何执行路径上。
5. **`process_batch` 是逐样本 Python 循环**，B 从 4 涨到 16 时开销线性增长。
   协议要求移植期不做性能优化，**未优化**。

## 遗留问题（3-strike 格式）

本次**没有出现任何需要 3-strike 的报错**——所有开关都是一次跑通的。下面是主动留下的口子：

1. **推理路径的 `reset()` 未接进评测循环**。模块已提供 `reset()` 并在推理态按 episode
   变化自动清理，但 `common/robodojo_eval.py` 的 50-episode 循环没有接。
   按用户确认的验收深度（不跑仿真评测）本次不做。**真要跑 benchmark 前必须接**，
   否则跨 episode 串记忆。
2. **`fifo` vs `tome` 未真正比较**：8 step 太短，差异 4.030e-02 vs 4.029e-02 不可区分。
   要比较需要跑到 episode 尺度。
   **订正（2026-08-04）**：引的那两个数产自 `run_gears.py --sampler sequential` 假路径
   （见 `05-ablation-matrix.md` 顶部标注），**不能用来支撑「不可区分」这个判断**。
   结论本身仍成立，但理由要换成「巩固逻辑要 bank 满 `mem_length` 才触发，8 step × batch 4
   只有最后几步走到那条路径」——这一条不依赖那两个数。
3. **多卡 / DDP 行为未验证**（见已知问题 3）。
4. **`cog_source="all_text"` 未实现**：需要先给 `CrossTransformerBlock` 加 attn_mask，
   那已经属于「改写」而非「搬运」。
5. **训练动力学变了，且原 A 档论证覆盖不到**（2026-08-04 新增）。接上 sampler 后每 batch
   从 4 个 episode 变成 **1 个**：梯度方差、epoch 内样本相关性、归一化层统计全都与关闭态不同。
   A 档证明的是「关闭态没变」，**不是**「开启态的训练行为已被验证」。
   这是新的遗留风险，不是「接完线就回到已验证状态」。
6. **DDP 多了一层新风险**（2026-08-04 新增，**复审也没记这条**）。
   `MemoryVLAEpisodeStreamBatchSampler` 按 episode 分片（`spans[rank::num_replicas]`），
   而 episode 长度差异极大（中位 276 → 1203 帧），所以**各 rank 的 `__len__` 不相等**、
   收尾不齐。本机任意两卡 gather 必崩 `ILLEGAL_ADDRESS`，无法本地验证。
7. **外部真实 ckpt 加载仍未验证**：bucket 上只有 v9，config 是 v10，`vlm.*` 全线 size
   mismatch，且 v10 warm-start 在 http URL 后面而本机无外网。本轮所有档位一律
   `checkpoint=null`（随机初始化 + 本地 `vlm_pretrain`），与移植当时同口径。
8. **`ulimit -n` 必须提到 65536**（2026-08-04 新增）。默认 1024 会被这套数据击穿，
   症状伪装成 dataloader 偶发抖动。详见「验证结果」末节。
9. **A 的采样频率 / 降采样未确认**（2026-08-04 补回，复审 P3-A）。
   这一条在 `06-review-report.md` §9 的「无法验证」六条里，上一轮承接时**漏掉了**。
   A repo 内只有消费端形参（`memory_vla.py:488`），**定义端在 A 的 RLDS 管线之外**；
   且 A 与宿主数据不同源，无法对跑。它的影响是：宿主一条 episode 的帧间隔与 A 的不一定同量级，
   于是 `mem_length=16` 在两边覆盖的**真实时间跨度**未必可比。
   **该怎样才能验**：读 A 的 RLDS builder 的 step 定义，或论文附录的数据处理节。
   自评影响中等偏低，但**掉一条和主动不承接是两回事**——补回来。

10. **`group` 现在能跑，但「能跑」不等于「该用」**（2026-08-04 第三轮新增）。
    本轮只证明了它不再静默退化，**没有**证明它训得好。它的记忆跨度只有一个 batch，
    `mem_length` 完全不起作用，训练动力学与 `stream` 不同且同样未验证。
    另外 `group` 每次调用清空 bank，所以它**走不到** `clear_episode` 与 tome 巩固
    两条路径——**`group` 的 D 档不能替代 `stream` 的 E 档冒烟**。
    还有一条选参陷阱：实测 `group_size=2` 时 12 个张量拿到精确零梯度，
    **把 `group_size` 调小等于关掉一部分模块**。
11. **看门狗的 K=8 没有调优**（2026-08-04 第三轮新增）。它只需要 ≥2
    （batch=1 的 `stream` 要到第 2 次 forward 才涨到 2），取 8 是与 B 档同量级的余量。
    没有实验支持 8 比 4 或 16 更好。若将来出现「合法配置要跑很多步才积累历史」的用法，
    这个数要重新定，**而且要用实测定，不是拍**。

    **第四轮补充 —— K 的取值依据与它现在的角色。**

    当时漏记了 K 最重要的性质：**它是时间闸门**，所以「跑不满 K 步」那一档
    **完全没有保护**，而 4–8 step 恰是本项目自己的冒烟长度。这就是 P1-C 的一半。

    | 问 | 答 |
    |---|---|
    | 下界 | **2**。`stream` + `batch_size=1` 的健康 run 要到**第 2 次** forward bank 才到 2（实测 `f4_B_stream_bs1_s12` 的 bank 序列 `1,2,3,4,…`）。取 1 会打死一个正常 run |
    | 为什么留到 8 | 余量。episode 可能只有 1 帧，连着几条极短 episode 会把「bank 涨过 1」推后；8 仍远小于任何真实训练 |
    | 上界怎么定 | **仍未做**。要枚举所有合法配置的「首次积累历史所需 forward 数」再取上确界，本轮没做 |
    | **它现在是不是唯一防线** | **不是了。** 所有**配置可达**的退化都被提前到装配期（`sampler.py:298` 链判据 / `sampler.py:344` 跨度判据）或第一次 forward（`wrapper.py:372` 第一批检查）。留给 K 的是**只有跑起来才知道**的一类：批次本该 episode 连续而实际不是 |
    | 剩下的窗口 | `stream` + bs=1 + 批次实际不连续 + 步数 < K。见支持矩阵末尾的「残留洞」。<br>⚠️ **`mem_length ≤ 1` 时这个窗口没有步数上界** —— 看门狗在该配置下站下，K 不再兜底。见下面的订正与遗留 14 |

    **一条方法论**：带计数 / 时间闸门的判据，**必须交付「闸门未到达时的行为」这一档证据**，
    否则它只在长跑里成立，而冒烟恰恰是短跑。已写进 `MIGRATIONS.md` 教训 13。

12. **`_episode_spans` 在其他数据集上的正确性未验证**（2026-08-05 第四轮**补回**，复审 P3-H）。
    这一条是 `09-incremental-review.md` §8 新增五条之一，**上一轮承接时掉了**。
    `sampler.py:_episode_spans` 假设「一条 episode 的帧在全局索引里连续」，
    只在 RoboDojo Memory 六任务上验过。换数据集若这个假设整体不成立，看门狗**会**报警
    （bank 涨不过 1）；但**若只是部分错位**（大部分连续、少数跨界），
    现有判据一条都抓不到 —— bank 照样涨得起来，恒等探针照样 arm。
    **该怎样才能验**：换一个数据集，比对 `_episode_spans` 的输出与该数据集自己的 episode 边界定义。

13. **长时训练稳定性未观测**（2026-08-05 第四轮**补回**，复审 P3-H）。
    同样是 `09` §8 新增五条之一，**上一轮承接时掉了**。第三轮最长 20 step、本轮最长 20 step，
    都远短于第二轮的 60 step。`tome` 巩固要 bank 满 `mem_length=16` 才触发、
    `clear_episode` 要跨 episode 边界才走到，**两者在 epoch 尺度上的行为仍未观测**。
    这条与遗留 2（`fifo` vs `tome`）相邻但不同：那条问「两种巩固谁更好」，
    这条问「跑久了会不会坏」（bank 泄漏、显存爬升、episode 键累积）。

14. **`mem_length ≤ 1` 时 bank 存活性看门狗永久站下**（2026-08-05 第五轮新增，复审 P2-F）。
    第四轮为消除误报加了 `mem_length > 1` 条件，代价是：该配置下看门狗发一行 `WARNING`
    后 `_bank_liveness_checked` 置位、**再也不运行**。于是
    **`stream` + `batch_size=1` + `mem_length ≤ 1` + 批次实际不连续**这一格
    **无人看守且没有步数上界**（12 步 > K 实测 `rc=0`、`grad 64/4/0`、移动 `0/68`），
    而健康对照发出**逐字相同**的那行 `WARNING` ⇒ 告警不含分辨信息。
    `batch_size ≥ 2` 不受影响（第一批检查照抓）。
    **该怎样才能修**：需要一条在 `mem_length=1` 下仍有分辨力的判据 ——
    bank 长度在这里恒为 1，所以只能改看别的量（例如检索是否真的命中过历史）。本轮未做。

15. **仓库 lint 门与 CI 的一致性未验证**（2026-08-05 第五轮**补回**，复审 P1-D）。
    这一条是 `11-incremental-review_v3.md` §8 的条目，第四轮**只写进了 `12-review-response.md`**，
    没进本文件。`holobrain_internal` 环境没装 ruff（装它破 E0「宿主主环境零改动」），
    历轮借用 `envs/RoboDojo/bin/ruff` 0.15.22，与 CI 实际版本是否相同**未验**。
    结论「零新增 lint 债」是**同一版本下 HEAD 与基线的相对比较**，不依赖版本正确性。

16. **`enable=False` + 非空 `dataset_sample_weights` 的真实入口行为未验证**
    （2026-08-05 第五轮**补回**，复审 P1-D）。同样是 `11` §8 的条目，第四轮只进了单轮应答。
    历轮只做到**函数级**取证（直接调 `assert_episode_stream_wired`，确认 `enable=False` 时返回不 raise）。
    要从真实入口验，需要一份带 per-spec `sample_weight` 的 `dataset_specs`，而那要改宿主 config。

17. **CI 的 pytest 是否真的收集到 `tests/.../memoryvla/` 下的三个文件未验证**
    （2026-08-05 第五轮**补回**，复审 P1-D）。本机无 pytest，执行证据来自
    `.git/run_tests_nopytest.py`（pytest stub，84 PASS / 0 FAIL / 0 SKIP）。
    结构性论证：三个文件在 `tests/Makefile: test_ut` 的目标树内
    （`pytest ... tests/test_robo_orchard_lab`），且 `memoryvla/` → `models/` →
    `test_robo_orchard_lab/` 三层 `__init__.py` 齐全，同目标树下已有 66 个 `test_*.py`。
    **剩下的不确定性是 CI 自己的 pytest 配置，不是本仓布局。**

> **为什么这两条会掉两次**（P3-A → P3-H 是同一形状的复发）：缺的不是这一次的细心，
> 是**承接动作没有清单化** —— 上一轮是「重写一遍风险清单」，重写就会漏。
> 第四轮起改成**逐条打勾**：每轮在 review-response 里放一张
> 「上一轮报告的无法验证清单 × 每一条承接到本文件哪一节」的对照表，逐条勾。
> 第四轮的那张表在 `12-review-response.md`。
>
> ### ⚠️ 订正（2026-08-05，第五轮，复审 P1-D）：清单化做对了，验收判据没做对
>
> 第四轮那张表 16/16 全勾，但其中**三条的「承接到」指向的是
> `12-review-response.md` 本身**（lint 门与 CI 一致性 · `dataset_sample_weights` 真实入口 ·
> CI 的 pytest 收集），而单轮应答文档下一轮就会被 `14-...` 取代 ——
> **等于没承接**。这是 P3-A → P3-H 的**第三次同形复发，而且发生在专门为了防它而造的那张表里**。
>
> **所以缺的既不是细心也不是清单，是清单的验收判据。现行判据改成：**
>
> > **承接目标只能是本文件（`PORT-STATUS.md`）的「遗留问题」或「已知问题」小节。**
> > 指向任何 `NN-review-response.md`、任何 review 报告、或任何一次性证据目录，**一律判未承接**。
> > 打勾时必须写出本文件的**小节号**（如「遗留 15」），写不出小节号就是没承接。
>
> 三条已按这条判据补进上面的**遗留 15 / 16 / 17**。

## 下一步建议

1. 真要用它训练：先把 `reset()` 接进评测循环（遗留 1），再跑一次 Memory 六任务的完整训练，
   与 `07_results.md` 里 20k/100k 的 Memory 维度数字对比 —— 那才是这次移植值不值的答案。
2. 训练时确认 `episode_stream_sampler=True` 且 `dataloader_type="stream"`。
   **订正（2026-08-04）**：这条建议在写下时是**无法执行**的 —— 该键当时没有读取者，
   设成什么都一样。现在它有读取者了，而且两者不匹配**会直接 raise**，不再是静默 no-op。
3. 若上多卡，先单独验 DDP 的 unused-parameter 行为。

## 合规

- A 为 **MIT**，宿主为 **Apache-2.0**，兼容，可移植、可分发。
- ⚠️ A 的 `pyproject.toml:15` 写 `license={file="LICENSE"}`，但**仓库里没有 LICENSE 文件**；
  MIT 的判据来自 `pyproject.toml:21` 的 classifier。已在 `00-phase0-record.md` 记录。
- 搬运处逐段留出处：`# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L<a>-L<b>`，
  文件头保留 MemoryVLA 的出处与许可证声明。
- 第三方权重：本次**未引入任何新权重**。
