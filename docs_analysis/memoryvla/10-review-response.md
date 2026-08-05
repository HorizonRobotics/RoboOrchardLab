# 10 — 对 `09-incremental-review.md` 的逐条应答

**日期**：2026-08-04 · **被修分支**：`port/memoryvla`，被审 commit `2b739226`
**修复 commit**：本文所在 commit（文档订正）· 第二个 commit（P1-B 护栏改挂消费端）
**复审**：`review2/memoryvla` @ `4268bca5`，判定 🟡 **ACCEPT-WITH-FIXES**（P0×0 · P1×2 · P2×1 · P3×4）
**本次范围**：范围锁死。第一段纯文档（零代码），第二段只改
`robo_orchard_lab/models/memoryvla/{sampler,wrapper}.py` 两个文件，**`train.py` 一行不动**。

**本文不自评 PASS。** 产出是修复 + 数值证据，裁决交给下一轮独立复审。

| ID | 级别 | 处置 |
|---|---|---|
| P1-A | P1 | **已修**（文档订正，本 commit） |
| P1-B | P1 | **已修**（护栏改挂消费端，第二个 commit）。`group` 由「无可用配置」变为**可用配置** |
| P2-A | P2 | **已修**（本 commit） |
| P3-A | P3 | **已修**（本 commit） |
| P3-B | P3 | **已修**（顺手做掉；本轮 runner 写 commit 绑定） |
| P3-C | P3 | **已修**（顺手做掉；改写那条被自身实测推翻的注释） |
| P3-D | P3 | **已改正做法**（本轮严格两个 commit） |

> 复审 §9 写「修完 1、2、3（即 P1-A / P2-A / P3-A）即可翻 ACCEPT」，
> 那三条全在**第一个 commit** 里，且都是纯 `.md`。P1-B 单独一个 commit，可分开判。

---

## P1-A · 失效的旧数值仍原样挂在 `05-ablation-matrix.md` / `06-verification.md` 上，零标注 —— **已修**

**改了什么**：给两份文件加失效标注，**一个数字都没删**。

复审给的处置选项是「重跑或明写未重跑，两者都行，留着不标不行」。
本轮选**明写未重跑**——消融矩阵本身不在本轮范围内，重跑它会把范围撑开，
而范围锁死正是上一轮守住的东西。

| 文件 | 改了什么 |
|---|---|
| `05-ablation-matrix.md` | 顶部加醒目失效横幅（数据来源 = `run_gears.py --sampler sequential`，宿主到不了的装配，不可用于任何结论，现行数值去哪看）；矩阵表上方单独再标一次，并点出 `峰值显存` 一列与真实入口差一档（`7.47/7.78 G` vs `8.98/9.30 GiB`）；**划掉** L4「每一行都只靠 config 切换」；**划掉**并反驳结论句「没有哪个是摆设」；`mode=group` 小节标为 P1-B |
| `06-verification.md` | 顶部加**逐档状态表**（按小节标题定位，不用行号——行号会漂）：第 0 步 / A / B / D / E 五处标失效并指向现行数值，**C 档单独标为仍有效**（不经 sampler，复审已在 HEAD 独立重跑 10/10）；**划掉**「所以数值描述的就是真正会训练的那个东西」；**划掉**「因此 A 档用严格判据 `atol < 1e-6`」并附推翻它的实测；`run_gears.py` 两处 cite 标为假路径；`group`+batch=1 那条警告标为「第二段已改成会执行的护栏」 |

**两处特意写进去的东西**（不只是贴标签）：

1. **结论句「每一行 vs base 都 > 0，说明没有哪个是摆设」的失效方式，比数字失效更值得记。**
   在假路径下这句成立；在真实入口上，当时那些配置**每一行都是恒等**，
   差值全来自随机性。**这句话恰好把「全都是摆设」读成了「没有哪个是摆设」**——
   同一个观测量，路径换了，结论完全反转。
2. **「harness 不走 `train.py` 的理由本身成立」这一点写在了划线旁边。**
   它是这次失效最阴的地方：偏离在 code review 里读起来像谨慎，不像风险。
   只划掉错句、不写这一层，下一个人还会做出同样的取舍。

**验证**：本 commit `git diff --name-only` 只含 `.md`（见文末 git 卫生）。

---

## P2-A · 本轮重写过的那一行仍写「0 删除」 —— **已修**

**改了什么**：`PORT-STATUS.md` 侵入度标题与 `MIGRATIONS.md`「改了宿主哪几处」标题，
「0 删除」换成实测值，并各加一条订正块。

```
git diff --stat 18106b05..f6dfd1e8
  projects/holobrain_internal/common/train.py    +38 / −6
  robo_orchard_lab/models/memoryvla/sampler.py   +94 / −1
  robo_orchard_lab/models/memoryvla/wrapper.py  +118 / −0   ← 只有这个是纯增量
```

**那 6 行是代码位移，不是逻辑改动**：`DistributedBatchFlagSampler(...)` 原本是
`DataLoader(...)` 的一个实参，接线时提到前面成了局部变量，构造参数一字未动。

**但订正块里写死了一句：「没动过」这件事不能靠读 diff 判。**
位移会不会改变关闭态，只有实测能答，判据引复审 §4.1 的五项精确量。

**顺带记下的教训**（写进了订正块）：「0 删除」这种**听起来最无害的自述最容易没人核**——
这一行被重写过一次（上一轮把「4 个文件」改成「5 个」），而同一行里的另一个数字照样错着。

---

## P3-A · 上一轮六条「无法验证」只承接了五条 —— **已修**

**改了什么**：`PORT-STATUS.md` 遗留问题新增第 9 条，把 `06-review-report.md` §9 第 3 条
「**A 的采样频率 / 降采样**」补回来，含「为什么验不了」与「该怎样才能验」两列的内容，
并补上一句上一轮没写的**影响**：宿主一条 episode 的帧间隔与 A 的不一定同量级，
于是 `mem_length=16` 在两边覆盖的**真实时间跨度**未必可比。

**为什么不因为「自评影响中等偏低」就略过**：掉一条和主动写「不承接，理由是 X」是两回事。
前者在报告上和「忘了」长得一样。

---

## P1-B · `dataloader_type="group"` 在 HEAD 上没有任何可用配置 —— **已修**

**勘察结论改变了修法。** 复审建议把恒等探针的判据换成「跑满 K 步后 bank 有没有超过 1」，
本轮采纳了这一条，但同时发现 `group` **本来就不该是死路**：

`dataloader_type` **不是 dataloader 选择器**——`train.py` 全文零处读它
（`git grep dataloader_type` 可证）。它只被 `memory_bank.py` 消费，选的是**记忆跨度**：
`stream` 让 bank 跨调用存活，`group` 在每次训练调用顶部 `bank.clear()`。
**两者都需要 episode 连续的批**，只是跨度不同。
所以 `sampler.py:230` 的 `(dl_type == "stream") != bool(stream_sampler)` 编码的是一条**假前提**
——「episode sampler 只对 stream 有意义」——而正是这条假前提同时造成了两件事：
把唯一能用的组合（`group` + sampler 开）判成冲突，并放行那个静默恒等的组合。

**改了什么**：只有 `robo_orchard_lab/models/memoryvla/{sampler,wrapper}.py` 两个文件。
**`train.py` 一行未动**，所以 P0-1 / P1-1 的修复成果与其全部已买到的证据保持有效。

| # | 改动 | 落点 |
|---|---|---|
| 1 | 装配期判据从「两个键是否相符」换成「**实际 sampler 链里有没有 episode sampler**」 | `sampler.py:assert_episode_stream_wired` |
| 2 | 去掉 batch 组成检查的 `dataloader_type != "stream"` 提前 return | `wrapper.py:_check_episode_stream` |
| 3 | **新增 bank 存活性看门狗**：K=8 次训练 forward 后 bank 从没超过 1 就 raise | `wrapper.py:_check_bank_liveness` |
| 4 | `group` 下不再把上一批残留的 bank 当历史（会对正确配置误报） | `wrapper.py:_history_will_be_read` |
| 5 | 三处误导文案改写 + 断言禁止其回归 | `sampler.py` ×3 |

**结果矩阵**（全部从 `train.py` 真实入口实测）：

| 配置 | 修复前 | 修复后 |
|---|---|---|
| `stream` + sampler `True` | 通过 | 通过（不变） |
| `stream` + sampler `False` | raise | raise（不变，文案改写） |
| **`group` + sampler `True`** | **raise（「disagree」）** | **通过 —— 可用配置** |
| **`group` + sampler `False`** | **静默恒等，护栏 0 行日志** | **raise，`rc=1`** |

**不存在「memory 被构建 + 静默退化 + 无告警」的组合。**

**护栏有牙（这是本条最重要的证据）**。故障注入用 runner 的 `--break-episode-order`：
按 sampler **自己的** span 表重排它自己的输出，**sampler 对象不动**，
所以装配期判据看到的链是对的、会放行 —— 只有消费端能发现批次已经坏了。

| 场景 | 谁应该抓到 | 实测 |
|---|---|---|
| `group` + sampler 关 | 装配期链判据 | ✅ `rc=1`，`train.py:236` → `sampler.py:264` |
| `stream` + sampler 关 | 装配期链判据 | ✅ `rc=1`，同一处 |
| 故障注入 bs=4 | batch 组成检查 | ✅ `rc=1`，`wrapper.py:302` |
| **故障注入 bs=1** | **只有看门狗能抓** | ✅ `rc=1`，`wrapper.py:354`，第 8 次 forward |

最后一行是决定性的：batch=1 时 batch 组成检查判不了（需要 >1 个样本），
恒等探针永不 arm（没有历史），**三道护栏里只剩看门狗，而它响了**。

> **一次失败的尝试，留在记录里**：故障注入的第一版是「把 sampler 输出在一个滑动窗口里转置」，
> **完全没有效果**——episode sampler 会连续吐出同一条 episode 的很多批
> （episode 长 276–1203 帧 = batch 4 下 69–300 批），所以 8 批的窗口里全是**同一条** episode，
> 转置是恒等。连续性在 span 表里，就得在 span 表上打断。
> 第一版若不核对就采信，会得到「护栏没触发」这个**完全相反**的结论。

**新增 D 档（`group` 路径，不覆盖 stream 的数值）**：bank 恒为 4（= batch）、
grad `0/0/68`、参数移动 62/68、峰值显存 9.1012 GiB。另跑了 `group_size=2` 一档专门走到
组轮转分支（`group_size=16 > batch=4` 时那条分支永不执行，只跑一档等于又一次 N=1 冒烟），
它顺带暴露了「`group_size=2` 时 12 个张量拿到精确零梯度」。全部数值见 `PORT-STATUS.md`。

**单元测试**：`fix3/guard3_unit_test.py` **22/22**、`fix3/guard3_probe_test.py` **24/24**。
后者把复审 §4.3 的 9 用例按新语义重写并扩到 24 个，其中 group 两条的期望**翻转**。

---

## P3-B · 证据 JSON 无 commit 绑定 —— **已修**

`fix3/run_real3.py` 在每个结果 JSON 里写 `provenance`：`git rev-parse HEAD`、`git_dirty`、
被执行的那个 `train.py` 的 sha256、`robo_orchard_lab` 的解析路径、torch 版本，
以及 **memoryvla 包四个文件各自的 sha256**（本轮改动期间工作树是 dirty 的，
只有 commit 号不足以定位代码）。

**怎么把本轮的 run 绑到本 commit**（下一轮复审需要）：

```
本轮全部 head 侧 run 的 provenance：git_head = 955fbe07（第一个 commit），git_dirty = true
被跑的 train.py            sha256 0087ec1b9b61fdeb2930c845b60a9377b1cc42f14e7e303a98a2e8d47604e2a3
                                 （与 2b739226 的 train.py 相同 —— 本轮未改它）
被跑的 sampler.py          sha256 fb14d11ce90c2649d0667cbce46dd377ef2d9dd004fd65a7c817e7b64c43107e
被跑的 wrapper.py          sha256 31ab3bcf16ea36377dcca89ce7bca438e21370908ec5b96d3820dfc598290b13
```

后两个哈希**就是本 commit 里这两个文件的内容**，`sha256sum` 可核。
基线侧 3 个 run 的 `git_dirty = false`，`git_head = 955fbe07`（纯文档提交，代码等同 `2b739226`）。
早于 `port_files` 字段加入的 3 个 wave-1 run 在 JSON 里没有这两个哈希，
但它们跑在同一份文件上：那两个文件在本轮最后一次修改之后再未变动，
**上面的哈希是在 wave-1 跑完当场取的，与现在逐位相同**。

---

## P3-C · `run_real.py` 关于确定性开关的说法被其自身实测推翻 —— **已修**

本轮 runner 是从 `fix/run_real.py` **复制**到 `fix3/run_real3.py` 后改的
（`fix/` 原文件一字未动，md5 `7a033e2218422d7f8aa68f98759fb451`，它是上一轮的证据）。
那条「These knobs pin that down so gear A can use a strict bar」已改写成实测结论：
它们压不到 0（`det_run1` vs `det_run2` = `1.564e-04`），A 档最终也没用它们。

**改完重新逐条确认了「只注入不构造」**，并顺手修掉了一处**观测装置污染被观测量**：

> 原 runner 在 `_install` 里**无条件** import 了 `MemoryVLAEpisodeStreamBatchSampler`
> （为了给构造计时）和 `MemoryVLAMemory`（为了包 forward）。两个 import 都发生在
> `train.py` 之前，于是**观测器自己**把 port 包放进了 `sys.modules`，
> 「关闭态从不 import port」这条结构判据**就没法测了**。
> 改法：装载期不 import port 包任何东西；哪个 sampler 真在跑由 `_sampler_chain`
> 读类型名回答（不需要 import），forward 探针改成在 `patched_init` 里、
> **确认宿主已经 import 了才装**。实测关闭态 `port_imported == []`，开启态 4 个模块——
> 判据活过来了。这正好是复审 §7.3 第 6 条提的那件事。

新增的记录字段（逐条对应本轮判据）：逐样本原始 `uuid`、`batch_keys`、
完整 bank 长度列表与 `max_len`、`port_imported`、`fault_injected`。
**新增的代码全是「读」与「记」，没有构造 sampler / DataLoader / optimizer / model builder 中的任何一个。**

---

## P3-D · 契约要 2 个 commit，实交 4 个 —— **已改正做法**

本轮严格两个 commit：`文档订正` 与 `P1-B 护栏改挂消费端`。

> 上一轮多出的两个是纯文档（`a81682e0` `f6dfd1e8`），动机是「边验边记」。
> 记录本身没错，错在**没有攒到一起提**。本轮的做法：第二段的验证数值全部攒在最后一次提交里。

---

## 对复审 §7.3 六条协议反馈的态度

**未修改任何协议文件**（那不是本轮范围），但其中三条已经落进本轮的做法，记在这里以便下轮判断：

| 反馈 | 本轮怎么处理的 |
|---|---|
| ① 输出文件名撞号，建议「紧接现有最大编号」 | 本文用 `10-`，紧接 `09-` |
| ③ S3 的「判据沿用上一轮的确定性结论」是错的前提 | 已按「每轮自测地板 + 额外给一条精确判据」执行，并写进 `PORT-STATUS.md` 与 `MIGRATIONS.md` 教训 9 |
| ⑤ 机械判据回放**必须钉住工具版本** | 本轮开工即记四个工具的 md5，与 `09` §7.1 逐个相同 |
| ⑥ 观测装置本身会不会污染被观测量要单列一条 | 已写进 `MIGRATIONS.md` 教训 9 的「顺带一条」 |
| ② 上一轮协议没要求产出「失效/仍有效/需重验」分类节 | 本文最后一节**主动提供**，不等下一轮自己推导 |
| ④ N/A 也要留痕并给判据 | 第二个 commit 里对「不做的档位」逐条给理由 |

---

## 本轮改动影响面自述（**改动方自述，待复审独立推导**）

**这一节是主动补的**：上一轮 `06-review-report.md` 没有这一节，复审只能自行推导并**明确降低了置信度**。
下面按 `09-incremental-review.md` 的小节逐条给。第二个 commit 会回填与代码改动相关的部分。

### 仍有效（本轮未触及，结论应原样继承）

| `09` 的哪一节 | 为什么不受影响 |
|---|---|
| §2 范围合规（全节） | 判的是 `18106b05..2b739226`，本轮不改历史 |
| §3 P0-1 闭环（全表） | `train.py` 一行不动；sampler 构造与装配期调用点都没碰 |
| §3 P1-1 闭环 | `fix/run_real.py` 原文件不动；本轮 runner 是**复制后改**，且改动只增记录字段 |
| §4.1 A 档精确判据成立 | 本轮沿用同一套精确量，并新增两条关闭态 run 复核 |
| §4.5 四条宿主语义 | `dataset_sample_weights` / per-spec `sample_weight` / `flags` / 第二入口，本轮均未触及 |
| §5.3 抽验 1（拷贝保真度 F） | 本轮不新增 `[port:]` 标记区间 |
| §5.4 全部「继承、未重验」项 | 方法要素、接口语义 32 项、cite 零幻觉、mask 极性、ckpt 1000→1068 |

### 需重验（本轮改动使其取证条件变化）—— **已全部重跑，结果如下**

| `09` 的哪一节 | 本轮实测 |
|---|---|
| §4.3 探针有效性 9 用例 | **已重写并扩到 24 用例，24/24**。其中 group 两条的期望**翻转**：用例 8「group 4 条不同 episode → 不检查」现在**必须 raise** |
| §7.2 判据 **I**（恒等探针） | 开启态 `1.296956e+00` / `1.123835e+00`（与上一轮**逐位相同**）；退化方向 5/5 会 raise |
| §7.2 判据 **G**（梯度三态） | `0 None / 0 零 / 68 非零`（stream 与 group 都是） |
| §7.2 判据 **P**（参数位移） | 62/68（stream 与 group 都是） |
| §7.2 判据 **B**（关闭态 batch key） | 14 vs 14 逐个相同，stream 与 group 两条路径都测了 |
| §5.2 **C 档** | **10/10 逐位一致，0 failed** —— 改动未溢出范围 |
| §5.3 抽验 2「纯增量」 | 本轮 `sampler.py` **+40/−32**、`wrapper.py` **+96/−10**（`git diff --numstat 2b739226`）。**不是纯增量**，删除全部是判据重写与闸门移除，逐条在上面列了。无格式化 / 无 import 重排 / 无重命名 / 无顺手重构 |
| §7.2 K/C/D 静态判据 | HEAD **0 finding**；**阳性对照**：同一份工具、同一组参数跑在 `18106b05` 的树上报 **2 findings（`ORPHAN episode_stream_sampler` + `UNUSED MemoryVLAEpisodeStreamBatchSampler`），EXIT=1** → HEAD 的绿是判据活着的绿。（复审在同一棵树上报 4 条，多出的两条是 `UNUSED BottleneckSE`（本轮显式 `--waive-class` 豁免）与 `DRIFT`（需 `--plan`，本轮未传）；**两侧参数完全相同**，所以对照成立） |
| preflight `--static` 全套 | **PASSED，EXIT=0**，带与上一轮相同的两组豁免；工具 md5 与 `09` §7.1 逐个相同 |

### 关闭态等价性（本轮重跑，用精确判据）

| 判据 | stream 路径 | group 路径 |
|---|---|---|
| 逐样本 id 序列 | **20/20 batch 完全一致** | **20/20 完全一致** |
| batch key 集合 | 14 vs 14 逐个相同 | 相同 |
| 参数量 | `1,136,284,265` 严格相等 | 严格相等 |
| sampler 链 | `['DistributedBatchFlagSampler']` 相同 | 相同 |
| `port_imported`（结构判据） | `[]` —— port 包从未被 import | `[]` |
| （参考量，非判据）loss max\|diff\| | `2.613e-04` | `1.507e-04` |

**阳性对照**：注入宿主 sampler 自己的构造参数 `seed=99` → **id 序列 0/20 相同**，
而参数量 / batch key / sampler 链 / `port_imported` **四项全不变**——只动了顺序，
正是这条判据该抓的东西。其余四项的对照用 `enable=True`：四项全部按预期改变。

### 本轮新增的两条「不算数」结论（都是方法论级，主动降级）

1. **峰值显存不是精确量。** 计划里它是五项精确判据之一，依据是上一轮 5 个 run 逐位相同。
   本轮 5 个同配置关闭态 run **不**逐位相同：4 个一致，1 个低 4.97 MiB（0.055%）。
   离群的那个不可能来自本轮改动（`port_imported == []`，被改的文件根本没被 import），
   且**同代码同卡重跑一次就回到基线值**。→ 降级为参考量，与浮点 loss 同级。
   **样本量小的一致性不是精确性**，这与本轮订正 A 档判据是同一个错误的两次出现。
2. **关闭态的批次顺序与训练 seed 无关。** 找阳性对照时发现 `--seed` 与 `set_epoch` 都动不了它
   （`dataset_wrapper.py:133` 自建 RNG，seed 来自 `train.py` 从不传的构造参数）。
   这条本身不是缺陷，但**指望换 seed 就换一批数据的人会得到「一模一样」**，
   并且很容易把这个「一样」当成别的东西的证据。

### 失效（本轮改动直接推翻，沿用旧值即报告失真）

> **订正（2026-08-05，第四轮，复审 P3-F）**：本节原有**两个内容近似重复的同名小节**
> （各 4 行，互有一行独占）。已合并为下面这一张，**取两者的并集，一行未丢**。

| `09` 的哪一节 | 怎么失效的 |
|---|---|
| **§4.4 全节（P1-B）** | 「`group` 的两条路都堵死了」不再成立。`group + episode_stream_sampler=True` 现在是**可用配置**；`group + False` 从静默恒等变成启动即 raise。该节的 `C_group_host.json` 证据描述的是**修复前**的行为 |
| **§8「`dataloader_type="group"` 是否还有意义」** | 该条写「需要一个能产出 episode 有序批又不与护栏冲突的配置，目前不存在」——现在存在了，本轮给了它的 D 档数值 |
| §4.2 引用的 raise 文案 | 三处误导文案已改写；引用旧文案的地方要换 |
| §4.1 / §5.2 里「峰值显存逐位相同」作为**判据** | 见上，降级为参考量。它作为**观测**仍然成立（上一轮 5 个 run 确实一致） |
| §7.2 判据 **C**（无人构造的类）的豁免列表 | 若第二段新增了类，豁免列表要重给（第二个 commit 说明） |

### 本轮**没有**做的事（明写，免得和「跳过」长得一样）

- **消融矩阵未重跑**（P1-A 选了「明写未重跑」这一支）。
- **DDP 仍未验证**：本机任意两卡 gather 必崩，与上一轮同一硬约束。
  接上 sampler 后各 rank `spans[rank::num_replicas]` 长度不齐这条风险**依旧**，
  且 `group` 变为可用**不改变**它。
- **开启态训练动力学仍未验证**：`group` 变为可用后，它的动力学同样未验证，
  与 `stream` 开启态是同一类遗留（见遗留 5）。**「能跑」不等于「该用」**——
  本轮只证明了 `group` 不再静默退化，没有证明它训得好。
- **外部真实 ckpt 仍未加载**：本轮全部 run 一律 `checkpoint=null`。
- **`group` 的长时行为未观测**：最长 20 step。`group` 每次调用清空 bank，
  所以它压根走不到 `clear_episode` 与 tome 巩固那两条路径——
  换句话说 `group` 的 D 档**不能**替代 stream 的 E 档冒烟。
- **K=8 这个数没有调优**：它只需要 ≥2（batch=1 的 stream 要到第 2 次 forward 才涨到 2），
  取 8 是与 B 档同量级的余量。没有实验支持 8 比 4 或 16 更好。

### 给下一轮复审的一条提醒

本轮有**两次**「判据看起来通过了，实际是装置坏了」被抓住，都不是靠读结论抓的：

1. **故障注入第一版完全无效**，日志里却是一片正常的训练——若不核对
   `distinct_episodes_in_batch` 与 bank 长度，会得出「护栏没触发」这个**相反**的结论。
2. **`guard3_run.sh` 最初复用固定 workspace**，第二次跑时 accelerate 在**解栈过程中**
   因残留 `checkpoint_0` 抛了 `ValueError`，把真正的 raise 埋在下面。
   `rc != 0` 与「期望字符串出现」两个判据**同时**满足，而过程是错的。

两次的共同点：**退出码与关键字都对，过程却不对**。所以本轮所有阴性用例的日志
最后一行都是护栏自己的 raise，可以直接核。
