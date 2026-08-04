# PORT-STATUS — MemoryVLA → HoloBrain

**日期**：2026-08-03（`date +%Y-%m-%d`）
**日期（第二轮修复）**：2026-08-04 · **日期（第三轮修复）**：2026-08-04
**总判定**：**不自评**。
第一轮自评 PASS → 独立复审 🔴 **REJECT**（P0-1：`episode_stream_sampler` 是死开关）。
第二轮修复 → 独立增量复审 🟡 **ACCEPT-WITH-FIXES**（P0×0 · P1×2 · P2×1 · P3×4，
见 `09-incremental-review.md`）：P0-1 与 P1-1 确认真闭环，剩下的是「修的覆盖面比声称的窄」。
第三轮（本轮）修 P1-A / P1-B / P2-A / P3-A，同样是范围锁死的修复 + 数值证据，
**裁决交给下一轮独立复审**。
逐条应答：第二轮见 `08-review-response.md`，第三轮见 `10-review-response.md`。

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

## 侵入度：**L1**，触及宿主已有文件 **5 个**，`train.py` +38/−6 · `sampler.py` +94/−1

> **订正（2026-08-04）**：原记 4 个文件。移植当时漏掉了 `common/train.py`，
> 而那正是 P0-1 —— sampler 开关没有读取者。修复后 `train.py` 是第 5 个。

> **订正 2（2026-08-04，复审 P2-A）**：本标题原写 **「0 删除」**，**不成立**。
> 实测 `git diff --stat 18106b05..f6dfd1e8`：`train.py` **+38/−6**、`sampler.py` **+94/−1**、
> `wrapper.py` +118/−0（这个确实是纯增量）。
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

**现行判据：五项精确量，逐项与基线严格比对。**

| 精确量 | 判据 | 为什么它精确 |
|---|---|---|
| 逐样本 id 序列（每 batch 的原始 `uuid`） | 与基线完全一致 | 接线改的就是选 batch 的那段代码，而它唯一能破坏的就是这个。无噪声 |
| batch key 集合 | 完全一致 | 关闭态 14 个；多一个 key 就说明数据管线被动了 |
| 参数量 | 严格相等 | 关闭态 `1,136,284,265` |
| 峰值显存 | 严格相等 | 关闭态实测逐位相同 `8.975615978240967 GiB` |
| sampler 链（类型与嵌套） | 完全一致 | `['DistributedBatchFlagSampler']`；查 `accelerator.prepare()` **之后**那个 |

外加一条结构性判据：关闭态 `sys.modules` 里**不应出现** `robo_orchard_lab.models.memoryvla*`
（`train.py` 与 `_build_memoryvla_cfg` 两处 import 都在分支内）。

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
| `group` + sampler 关（**P1-B 的洞**） | 装配期链判据 | ✅ `rc=1`，`train.py:236` → `sampler.py:264` raise |
| `stream` + sampler 关（既有阴性用例回归） | 装配期链判据 | ✅ `rc=1`，同一处 |
| 故障注入，bs=4 | batch 组成检查 | ✅ `rc=1`，`wrapper.py:302`，「4 samples from 4 different episodes」 |
| **故障注入，bs=1** | **只有看门狗能抓** | ✅ `rc=1`，`wrapper.py:354`，第 8 次 forward 报「no episode's memory ever grew past a single entry」 |

**最后一行是本轮最关键的一条证据**：batch=1 时 batch 组成检查判不了（它需要 >1 个样本），
恒等探针也永不 arm（没有历史），**三道护栏里只剩看门狗能响，而它响了**。
这正是 `06-verification.md` 末尾那条「`group` + batch=1 会恒等但看起来正常」的警告——
它从此不再只是一段文字。

单元测试（CPU，本机无 pytest，写成退出码脚本，均在 `fix3/`）：
`guard3_unit_test.py` **22/22**（15 个装配期用例 + 7 条文案卫生断言）·
`guard3_probe_test.py` **24/24**（看门狗 8 · batch 组成 6 · 恒等探针 5 · `_history_will_be_read` 5）。

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
| **`group` + sampler `True`** | **通过 —— 新的可用配置** |
| **`group` + sampler `False`** | **raise**（新；原来是静默恒等） |

**不存在「memory 被构建 + 静默退化 + 无告警」的组合。**

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
