# 13 — 增量复审 v4：MemoryVLA 移植的 P1-C 修复轮

| 项 | 值 |
|---|---|
| 基线 commit | `49b2178c` |
| 被审 commit | `28af78ab`（`port/memoryvla` HEAD） |
| 本轮四个提交 | `308730dc` 文档订正（纯 `.md`）· `f770afe0` P2-B 类文档字符串复位 · `fc33a5db` P1-C 护栏补齐 + 测试进仓 · `28af78ab` 实测回填（纯 `.md`） |
| 上一轮报告 | `11-incremental-review_v3.md` @ `review3/memoryvla` `d311b000`（🟡 ACCEPT-WITH-FIXES，P0×0 · **P1×1** · P2×3 · P3×4） |
| 本轮改动契约 | `~/storage_policy/protocols/robo_orchard_lab/port_memoryvla_4.md` |
| 改动方应答 | `docs_analysis/memoryvla/12-review-response.md`（514 行） |
| 方法论基准 | `review_memoryvla_1.md`（按需引用，未从 R0 全量执行） |
| 复审证据 | `$ROL_JFS/port/memoryvla/review4/`（自建观测器 + **24 个已促成的真实入口 run**，另 1 档因期望与栈帧不符判 INVALID 未促成 + 静态阳性对照，**不进 git**） |
| 算力 | 单卡 × 25 档，GPU 1/3/4/5（**6/7 按要求未使用**），`ulimit -n 65536` |
| 日期 | 2026-08-05（`date +%Y-%m-%d`） |

---

## 1. 裁决

# 🟡 ACCEPT-WITH-FIXES

| 级别 | 数 | 摘要 |
|---|---:|---|
| **P0** | **0** | 关闭态四项精确判据 + 结构判据全过（跨装置、跨轮）；C 档 10/10 逐位一致**在被审 commit 上**；无全局副作用；护栏不消耗 RNG |
| **P1** | **1** | **P1-D** `11` §8 的 16 条里有 **2 条没有进 `PORT-STATUS.md`**（lint 门与 CI 一致性、`enable=False`+`dataset_sample_weights` 真实入口），本轮新增的「CI pytest 是否收集到」同样只在 `12-review-response.md` 里 —— **逐条打勾表的「承接到」一栏允许指向单轮应答文档，而那正是 P3-A / P3-H 两次丢条目的地方** |
| **P2** | **3** | **P2-D** `PORT-STATUS.md` **三处**引用 `sampler.py:264`，被审 commit 上链判据在 **`sampler.py:298`**（实测 `G_group_nosampler`）。其中两处在**本轮新写**的支持矩阵与 K 表里 —— 与 P2-A′ 同形，出现在修 P2-A′ 的这一轮<br>**P2-E** `MIGRATIONS.md` 教训 9 把「6 MiB 量级」划掉、改成「+0.98 MiB，落在噪声里、不可分辨」，**这个订正不成立**：本复审**同卡**对照实测 **+8.01 MiB**，两个不带探针的 run（不同卡）**逐位相同**<br>**P2-F** `mem_length ≤ 1` 的站下分支使 **`stream` + bs=1 的残留洞不再受 K 约束**：实测跑满 12 步仍 `rc=0`、`grad 64/4/0`、移动 `0/68`，而健康对照发出**逐字相同**的那一行 WARNING ⇒ 该告警**不含分辨信息**。改动前这一格会在第 8 次 forward raise。文档只写了「主动去掉一个误报」，没写代价 |
| **P3** | **2** | **P3-I** `12` §8 的 git 卫生 `--stat` 块不是被审 commit 的输出（`10-review-response.md 13` vs 实测 `23`、`12-review-response.md 467` vs `514`、`1705/66` vs `1761/67`）<br>**P3-J** 「护栏日志 0 行」是装置边界产物：`run_real4.py` 的日志捕获挂在 trainer `__init__` 里，而 sampler 的构造期 INFO 发生在那之前，**结构上不可能被它看到** |

**一句话理由**：**P1-C 是真闭环，而且比契约要求的做得更好** —— 装配期跨度判据在三个致命格子上全部
`fwd=0` 就 raise、与步数无关；文案的四条验收条件逐条成立；它宣称的两条出路本复审各跑了一档，**都真的有效**；
而它宣称的「消费端兜底」本复审**旁路掉装配期护栏之后单独验到了**（`wrapper.py:338`，fwd 0）——
这是改动方无法交付的一档证据。P2-A′/P2-B/P2-C/P3-E/P3-F/P3-G/P3-H **七条全部真闭环**。
剩下的问题**一个都不在代码的正确性上**：唯一的 P1 是承接清单允许把条目放进会被下一轮替换的文档，
三个 P2 分别是过期行号、一条被订正错了的教训、和一处用误报换来的覆盖损失没写代价。

### 这个裁决该怎么读

**代码这一轮做得好，而且好在难的地方。** 上一轮点名的病因是两条 ——「时间闸门」与「归因歧义」——
本轮**两条都从根上处理了**，不是打补丁：静态可判的前提提到装配期（消掉闸门），
文案改成「先报观测值、再列两种成因与各自动作」（消掉歧义），还顺手消掉一个**没人点名的误报**
（`mem_length=1`）和一个**没人点名的等价失效**（`group_size=1`）。
`12` §3 的三分类如实、准确，本复审独立推导后**无分歧**。

**问题全部集中在同一个地方：什么算「记下来了」。** 本轮为 P3-H 造了一张逐条打勾表，
方向完全正确 —— 但表的验收标准只要求「能指到某处」，于是三条指进了 `12-review-response.md`。
`12` 是单轮应答，下一轮就会被 `14` 取代；**P3-A → P3-H → P1-D 是同一形状的第三次复发，
而这一次它发生在专门为了防止它而造的那张表里面**。这说明缺的既不是细心也不是清单，
是**清单的验收判据**：条目必须落在**常设**风险册里，指向别处一律不算承接。

---

## 2. S1 — 范围合规

### 2.1 `git diff --name-status 49b2178c..28af78ab`（全文）

```
M	docs_analysis/MIGRATIONS.md
M	docs_analysis/memoryvla/10-review-response.md
A	docs_analysis/memoryvla/12-review-response.md
M	docs_analysis/memoryvla/PORT-STATUS.md
M	robo_orchard_lab/models/memoryvla/sampler.py
M	robo_orchard_lab/models/memoryvla/wrapper.py
A	tests/test_robo_orchard_lab/models/memoryvla/__init__.py
A	tests/test_robo_orchard_lab/models/memoryvla/test_sampler_guard.py
A	tests/test_robo_orchard_lab/models/memoryvla/test_wrapper_guards.py
```

```
 9 files changed, 1761 insertions(+), 67 deletions(-)
```

### 2.2 `tests/` 算不算范围溢出 —— **不算**，理由要写清楚

契约正文没有一行写着 `tests/`，所以这个问题必须单独回答，不能默认放行。判定 **不溢出**，两条依据：

1. **契约 §1 的 P2-C 明写了这一支**：「二选一：**要么**把测试移进仓库并接进会被执行的入口（第三段一起做），
   **要么**先把这句话改成实话」。改动方选了前者，是契约给出的选项，不是自行扩大。
2. **新增文件与触及宿主已有文件是两回事**。这三个文件全部是 `A`（新增），
   落在 `tests/test_robo_orchard_lab/models/` 下的**新**子目录里，
   没有修改任何一个宿主已有测试文件，也不改 `tests/Makefile` / `pytest.ini`。
   宿主既有测试的行为面为零变化。

> 若换成「为了接进入口而改了 `tests/Makefile` 或 `pyproject.toml`」，判定会反过来 —— 那是触及宿主闸门。
> 本轮没有。

### 2.3 逐提交判定（契约要求「先文档、后零行为、再有行为」的分段）

| commit | 契约要求 | 实测 | 判定 |
|---|---|---|---|
| `308730dc` | 纯 `.md` | 3 文件：`MIGRATIONS.md` · `10-review-response.md` · `PORT-STATUS.md`，**全 `.md`** | ✅ |
| `f770afe0` | **零行为改动** | 只有 `wrapper.py`，`+12/−6`；逐 hunk 读：3 行 `#:` 注释 + 1 行赋值从类文档字符串**之前**移到**之后**，另加 6 行解释为什么要这样放。**无任何可执行语句变化** | ✅（精确说法：`__doc__` 由 `None` 变成字符串，这正是这一段要修的东西；**计算行为**零变化） |
| `fc33a5db` | 唯一有行为的改动 | `sampler.py` + `wrapper.py` + 3 个新增测试，**没有别的** | ✅ |
| `28af78ab` | 纯 `.md` | 4 文件全 `.md` | ✅ |

**契约要三个提交，实交四个。理由成立，代价说清楚了。** `12` §0 给的论证本复审复核后接受：
要闭 P3-G 就必须「先提交、再跑验证」，而文档要回填的数字只有跑完才有；
放进 `fc33a5db` ⇒ 证据长在脏树上（P3-G 原样复发），`--amend` ⇒ 所有 `git_head` 指向不存在的 commit。
第四个提交**确实是纯 `.md`**，与第一个同形。**这是一处主动的、有理由的偏离，不是漏做。**

⚠️ 第四个提交被 `amend` 过两次（`337193a8` → `ae598b85` → `28af78ab`，`git reflog` 可查），
**`fc33a5db` 未被 amend，全部代码证据绑在它上面**。

#### amend 的代价：`12` §8 的 git 卫生 `--stat` 块不是被审 commit 的输出 —— **P3-I**

契约 §6 要求「git 卫生逐条贴输出：`git diff --stat 49b2178c..HEAD`」。贴出来的那一块是：

| 项 | `12` §8 贴的 | 实测 `49b2178c..28af78ab` |
|---|---:|---:|
| `10-review-response.md` | `13 +-` | **`23 +-`** |
| `12-review-response.md` | `467` | **`514`** |
| 合计 | `1705 insertions(+), 66 deletions(-)` | **`1761 insertions(+), 67 deletions(-)`** |
| 其余四项（`PORT-STATUS 227`、`sampler 96`、`wrapper 210`、三个测试文件） | — | **逐项相同** |

成因可复原：`git diff --stat 49b2178c..337193a8`（第一次 amend 前）给 `1752/66`，
差的 47 行正好是 `514 − 467` ⇒ **这块输出取自第四个提交刚建好、而 `12-review-response.md` 本身
还只有 467 行的那一刻**；之后两次 amend 补了 `10-review-response.md:121` 与两处 `<C4>` 占位符，
它就过期了。

**实质结论本复审逐条独立复核，全部成立**：9 个文件全在范围内 · `git status --porcelain` 为空 ·
`git diff --stat -- .gitignore | wc -l` 为 0 · `review/` `review2/` `review3/` 三分支 tip 未动
（`97837e04` / `4268bca5` / `d311b000`）· A repo porcelain md5 `9815d522644f15ab4edd56e5b33d1d03`、20 个脏文件。
**所以这是 P3 不是更高** —— 但它和 P2-D 是同一句话：**贴出来的数字要么带基点，要么保证是终态取的**。
`12` §0 已经对**提交哈希**想清楚了这个问题（「它的哈希不写在这里」），只是没把同样的推理用在 `--stat` 上。

### 2.4 侵入度：自述 vs 实测 —— **本轮完全对上（P2-A′ 闭环）**

```
git diff --numstat 18106b05..28af78ab -- robo_orchard_lab/models/memoryvla \
    projects/holobrain_internal/common/train.py
```

| 文件 | 文档写的（截至 `49b2178c`） | 实测 | 文档写的（截至 `fc33a5db`） | 实测 `..28af78ab` |
|---|---|---|---|---|
| `train.py` | +38 / −6 | **+38 / −6** ✅ | +38 / −6 | **+38 / −6** ✅ |
| `sampler.py` | +106 / −5 | **+106 / −5** ✅ | +202 / −5 | **+202 / −5** ✅ |
| `wrapper.py` | +204 / −0 | **+204 / −0** ✅ | +330 / −0 | **+330 / −0** ✅ |
| `__init__.py` | +2 / −0 | **+2 / −0** ✅ | +2 / −0 | **+2 / −0** ✅ |

「触及宿主已有文件 **5 个**」实测同样成立：
`config_holobrain_common.py` · `config_robodojo_dataset.py` · `train.py` · `structure.py` · `structure_qwen3_5.py`。

> **`fc33a5db` 列 == `28af78ab` 列**，因为第四个提交零代码 —— 这一点下面 §3 P3-G 会再用一次。
> 侵入度实判 **L1**，与自述一致。

### 2.5 无关改动 —— 逐 hunk 读完的结论

`sampler.py` **+96/−0**（纯增量：`_effective_batch_size` 一个新函数 + 跨度判据一段 + 两处文档字符串扩写）。
`wrapper.py` **+168/−42**，删除集中在两处：`_check_episode_stream` / `_check_bank_liveness` 的**文档字符串重写**
与**旧 raise 文本整段替换**。**无格式化 / 无 import 重排 / 无重命名 / 无顺手重构**，lint 逐行对照可证（§7）。

---

## 3. S2 — 上一轮 P1/P2/P3 逐条闭环

> 全部为本轮实测；与 `11` 记录的**失效态数值**并排。

### P1-C `group`+`batch_size=1` 配置可达的静默退化 —— ✅ **已闭环，且超出要求**

| | `11` 记录的失效态（`49b2178c`） | 本轮实测（`28af78ab`） |
|---|---|---|
| `group` bs=1 gs=16 **4 步** | `rc=0`、bank `[1,1,1,1]`、`grad 64/4/0`、`params moved 0/68`、恒等 `per=5.96e-08 cog=0.00e+00`、护栏日志 2 行全 INFO、`_bank_liveness_checked=False` | **raise** `sampler.py:344 in assert_episode_stream_wired`；`train_forwards=0`、`steps=0`、`episode_check_done=False`、`peak_mem=None`（**训练从未开始**） |
| `group` bs=1 gs=16 **12 步** | 12 步会 raise，但文案两句都是错的 | **raise 同一处，与 4 步逐字相同 ⇒ 与步数无关** |
| `group` bs=4 **gs=1** 4 步 | `11` 未点名 | **raise 同一处**，`min(1,4)=1` |

**旧失效签名在被审 commit 上不可达。** 三格都在**第一次 forward 之前**终止。

**文案质量 —— 四条验收条件逐条实测（S3-7 升级项 2）**：

| 要求 | 判定 | 依据 |
|---|---|---|
| 不断言判据分辨不了的成因 | ✅ | 这是**静态**判据，成因唯一确定，可以断言，且断言的正是真实原因 |
| 不建议已经生效的开关 | ✅ | 文案明写 `The episode sampler is NOT the problem here ... Changing episode_stream_sampler will not affect this`；实测该场景 `episode_stream_sampler=True` |
| **建议的动作照做真的有效** | ✅ **两条各跑了一档** | `stream`+bs=1（`B_stream_bs1_s12`）：bank `1→2→…→12`、`grad 0/0/68`、移动 62/68、看门狗 fwd8 放行 ·<br>`group`+bs≥2∧gs≥2（`D_group_bs4_s4/s12`）：bank 恒 4、`grad 0/0/68`、移动 62–63/68 |
| 先报观测值 | ✅ | `Observed: dataloader_type='group', group_size=16, batch_size=1 (read from MemoryVLAEpisodeStreamBatchSampler.batch_size), episode_stream_sampler=True` |

**看门狗文案**（`T_break_bs1_s12` 的 `error` 原文）：`The batches reaching this module are not episode-contiguous`
这句断言**已不存在**；改成先报 `Observed: ... batch sizes seen=[1], distinct episodes per batch seen=[1], longest bank=1`，
再列 `(a)` 批次不连续 / `(b)` 配置本身不可能，各配动作。✅

**超出字面要求的两处，判定均为「不算溢出」**：

- `group_size == 1` 与 `batch_size == 1` 落在**同一条静态谓词** `min(group_size, batch_size) <= 1` 里，
  是同一个失效的另一个键，不是第二个特例。实测 `D_group_gs1_s4` 与 bs=1 两格 raise 在同一行。
- `mem_length > 1` 条件消除误报：**「原判据会误报」这个断言成立** —— `B_memlen1_s12` 实测
  `grad 0/12/56`、移动 **51/68**、bank 恒 1 ⇒ 模块**在工作**，而 bank 长度上界恒为 1，
  旧判据会在第 8 次 forward 打死一个正常 run。**但它的代价没被登记 —— 见 §4.6 与 P2-F。**

### P2-A′ 侵入度数字第三轮带着过期值发布 —— ✅ **已闭环**

见 §2.4，四个文件两列**逐格相同**。现行口径「基点 + 截至 commit + 逐文件表 + 重跑命令」是对的修法：
它让下一次重算变成机械动作。

#### 但同一份文档里另有三个过期**行号** —— **P2-D**

`sampler.py` 本轮 `+96/−0`，`_effective_batch_size` 插在 `assert_episode_stream_wired` **之前**，
把链判据的 raise 从 `sampler.py:264` 推到了 **`sampler.py:298`**。
`PORT-STATUS.md` 里 `sampler.py:264` 出现**三次**，全部指向链判据：

| 行 | 上下文 | 是不是本轮新写的 |
|---|---|---|
| 328 | 「护栏有牙」表，`group` + sampler 关那一格 | 否（第三轮留下的） |
| **430** | **本轮新写的支持矩阵**第一行 | **是** |
| **572** | **本轮新写的 K 表**「它现在是不是唯一防线」那一格 | **是** |

实测（`G_group_nosampler`，`r4_gear.sh` 按栈帧核对）：

```
raise sampler.py:298 in assert_episode_stream_wired  (sampler: episode sampler not in chain)
```

**这与 P2-A′ 是同一个形状 —— 一个不带基点的测量值随改动过期 —— 而且发生在专门修 P2-A′ 的这一轮里。**
`wrapper.py:338` / `:372` / `:462` 三处行号本复审逐个核对，**都是对的**；错的只有 `sampler.py:264` 这一个。
说明修法（给数字加基点）是对的，只是**没有推广到行号**。

### P2-B `MemoryVLAMemory.__doc__ is None` —— ✅ **已闭环**

`11` 实测 `is None == True`。本轮**从真实入口**读到（每个开启态 run 都记）：

```
MemoryVLAMemory.__doc__ is None : False
BANK_LIVENESS_FORWARDS          : 8
```

`__doc__` 含 `Args:` 段。19 个开启态 run 全部一致。

### P2-C 断言不在 git 里 —— ✅ **已闭环**，边界写得诚实

| 判据 | `11` | 本轮实测 |
|---|---|---|
| `git ls-files \| grep -c guard3` | 0 | 仍 0（那两个脚本没进仓，改写进仓） |
| 仓内 memoryvla 测试数 | **0 个** | **2 个测试文件 + `__init__.py`**，在 `tests/test_robo_orchard_lab/models/memoryvla/` |
| 执行 | 一次性手工结果 | `.git/run_tests_nopytest.py` 逐文件跑：**合计 84：PASS 84 / FAIL 0 / SKIP 0**（本复审独立复现） |

**`SKIP 0` 是这里最要紧的一格**：shim 把无法解析的用例记 SKIP 不记 pass，`SKIP 0` ⇒ 84 项全部真的执行了。
「收集到 0 个用例」这一失败模式也不成立（84 ≠ 0）。

**结构性论证本复审加强了一档**：`tests/Makefile: test_ut` 跑的是 `pytest ... tests/test_robo_orchard_lab`，
该目录下按默认 `python_files=test_*.py` 共 **66** 个文件，新增 2 个在其中；
且 `memoryvla/` → `models/` → `test_robo_orchard_lab/` 三层 `__init__.py` **全部存在**（本复审逐层核对）。
⇒ 剩下的不确定性只是「CI 自己的 pytest 配置」，不是本仓布局。
**改动方把边界写在了正确的位置**（明写「shim 结果不是 CI 里 pytest 的结果」），**这条边界诚实**。
⚠️ 但它被记进了 `12-review-response.md` 而不是常设风险册 —— 见 P1-D。

### P3-E 工作区未提交的纯改名 —— ✅ **已闭环**

`git status --porcelain` **为空**。`10-review-response_v2.md` 在 `fix4/attic/`，
md5 `83ecef02c5999853edd435bd13e9e86f`，与 **`49b2178c:10-review-response.md` 逐字节相同**（本复审直接比对
`git show 49b2178c:... | md5sum`）⇒ 改名没有丢信息，且用 `mv` 不用 `rm` 的理由（本机 hook deny 且返回空输出）本复审这一轮**又踩了一次**，成立。

### P3-F 两个近似重复的 `### 失效` 小节 —— ✅ **已闭环，且取的是并集**

合并后 5 行：第一份独有的「§4.1/§5.2 峰值显存作为判据」**在**，第二份独有的「§7.2 判据 C 豁免列表」**也在**。
**删掉任一份都会丢一行 ⇒ 这不是删重复，是合并。** 逐 hunk 读 diff 可证。

### P3-G 证据无代码绑定 —— ✅ **已闭环（机制 + 本轮全部产物）**

改动方称「17/17 evidence files bind to fc33a5db」。**本复审独立复核了这个数**
（`review4/r4_bind_check.py`，真值取自 `git show <commit>:<path>` 而非工作树）：

```
port files on each commit (robo_orchard_lab/models/memoryvla):
  fc33a5db  __init__=03677c247045a6e0  memory_bank=babe423323eab884  sampler=a69379d38b5f5b5b  wrapper=00e112fda56c0c9b
  28af78ab  __init__=03677c247045a6e0  memory_bank=babe423323eab884  sampler=a69379d38b5f5b5b  wrapper=00e112fda56c0c9b   ← 逐个相同
  f770afe0  ...                                                       sampler=fb14d11ce90c2649  wrapper=25c50d915b7728ef

f4_*.json  17 个已促成结果（+17 个 attempt）  全部 BOUND:fc33a5db, dirty=False
b2_*.json   4 个基线结果（+4 个 attempt）      全部 BOUND:f770afe0, dirty=False
未绑定：0
```

**17/17 属实。** 并且 —— **`fc33a5db` 与 `28af78ab` 的四个 port 文件哈希逐个相同**，
所以绑到 `fc33a5db` 的证据在代码层面就是被审 commit 的证据，这一点不需要推断。

### P3-H `09` §8 五条掉了两条 —— ✅ **两条已补回**，但承接动作只做对了一半

`_episode_spans` 换数据集 → **遗留 12**；长时训练稳定性 → **遗留 13**。两条都在 `PORT-STATUS.md` 里 ✅。
逐条打勾表在 `12` §4，16/16 打勾。
**但本复审逐条去核「承接到」那一栏时，发现 3 条指向的不是常设风险册 —— 这是 P1-D。**

---

## 4. S3 — 回归探测（本轮核心）

**24 档全部从 `projects/holobrain_internal/common/train.py` 真实入口进**，
装置 `review4/r4_probe.py`（自 `review3/r3_probe.py` 扩展，**review3/ 未改一个字节**，md5 已留档）。
每档的期望写成 `ok` / `raise:<函数名>@<行号>`，**由 `r4_gear.sh` 核对记录到的栈帧**，
退出码不参与促成 —— 这是上一轮踩过的第三次「退出码对 ≠ 过程对」。

### 4.1 关闭态：结构判据 + 四项精确量（S3-1/2/5）

```
=== per-run profile （4 档关闭态 + 1 档阳性对照）
A_off_stream_1   params=1136284265  n_keys=14  chain=['DistributedBatchFlagSampler']  port_imported=0
A_off_stream_2   params=1136284265  n_keys=14  chain=['DistributedBatchFlagSampler']  port_imported=0
A_off_group_1    params=1136284265  n_keys=14  chain=['DistributedBatchFlagSampler']  port_imported=0
A_off_group_2    params=1136284265  n_keys=14  chain=['DistributedBatchFlagSampler']  port_imported=0
ctrl_hostseed    params=1136284265  n_keys=14  chain=['DistributedBatchFlagSampler']  port_imported=0

=== pairwise（四档关闭态两两）
uuid 20/20   params=same  keys=same  chain=same  port=same  rng=same
```

**结构判据 `port_imported == []` 在 4 档关闭态全部成立** ⇒ 本轮改的两个文件在关闭态下**根本不参与执行**。
四项精确量逐项相同。RNG 三点指纹 `at_exit` 全部相同 ⇒ **护栏不消耗全局 RNG**（S3-10）。

### 4.2 阳性对照：判据有牙（S3-3）

```
ctrl_hostseed（注入宿主 sampler 构造参数 seed=99）vs 任一关闭态
    uuid  0/20      ← 必须是 0/20
    params=same  keys=same  chain=same  port=same  rng=same
```

**只动顺序，四项里另外三项 + 结构判据纹丝不动。** ⇒ §4.1 的「20/20 一致」是有分辨力支撑的一致。
`--seed` 与 `set_epoch` 对批次顺序无牙（前两轮实测），本轮**没有拿它们充数**。

### 4.3 精确判据本身还精不精确（升级项 1）—— 峰值显存的降级第三次被证明正确

同配置、同代码、同装置重跑：

| run | 卡 | 峰值显存 (GiB) |
|---|---|---|
| `A_off_stream_1` | 4 | `8.976670265197754` |
| `A_off_stream_2` | 1 | `8.976670265197754` |
| `A_off_group_1` | 5 | `8.976670265197754` |
| **`A_off_group_2`** | 3 | **`8.971695899963379`** ← 低 5.09 MiB |
| `ctrl_hostseed` | 4 | **`8.971163272857666`** ← 又一个新值 |

**连同前两轮，同一个关闭态配置至今观测到四个不同的峰值显存值**
（`8.976670265197754` / `8.971459388732910`（`11`）/ `8.971695899963379` / `8.971163272857666`）。
**降级正当，且这一轮产出了一个此前没出现过的新值。**
其余四项精确判据在同样这五个 run 上**逐项相同** ⇒ **它们在本轮重新证明了自己精确**（升级项 1 的要求）。

### 4.4 护栏：触发、指向、不误报、闸门未到达（S3-6/7/8/9）

| # | 档 | 配置 | 实测 | 判定 |
|---|---|---|---|---|
| 1 | `G_group_nosampler` | `group`，sampler 关 | **raise `sampler.py:298 in assert_episode_stream_wired`**，`train_forwards=0` | ✅ P1-B 主干未被破坏；**新增的跨度判据没有抢在链判据前面** |
| 2 | `D_group_bs1_s4/s12` | `group` bs=1 | raise `sampler.py:344`，`fwd=0` | ✅ |
| 3 | `D_group_gs1_s4` | `group` gs=1 | raise `sampler.py:344`，`fwd=0` | ✅ |
| 4 | `T_break_bs4_s12` | 故障注入，bs=4 | **raise `wrapper.py:372 in _check_episode_stream`，第 0 次 forward**，`grad 68/0/0`、移动 0/68 | ✅ 有牙 |
| 5 | `T_break_bs1_s12` | 故障注入，bs=1，12 步 | **raise `wrapper.py:462 in _check_bank_liveness`，第 8 次 forward**，`grad 68/0/0`、移动 0/68 | ✅ 有牙 |
| 6 | `T_break_bs1_s4` | 故障注入，bs=1，4 步 | **`rc=0`**、bank `[1,1,1,1]`、`grad 64/4/0`、移动 `0/68`、2 行 INFO、`_bank_liveness_checked=False` | ⚠️ **残留洞复现** —— 与改动方自述逐项相同 |

**残留洞「不是配置可达」这个断言 —— 本复审独立验证，成立。**
无注入的同一格 `B_stream_bs1_s4`（`stream` bs=1 4 步）实测 bank `1→2→3→4`、`grad 0/0/68`、移动 62/68。
⇒ 该格子在**没有注入**时是健康的，进入残留洞需要主动注入或 `_episode_spans` 在别的数据集上不成立（遗留 12）。

**合法配置不误报（升级项 3）—— 三档全过：**

| 档 | 实测 | 判定 |
|---|---|---|
| `B_stream_bs1_s12`（`stream` + bs=1 长跑） | bank `1→2→…→12`，看门狗 fwd8 裁决 `maxbank=8` **放行** | ✅ 不误报 |
| `D_group_bs4_s4` / `_s12`（`group` + bs≥2 + gs≥2） | bank 恒 4，`grad 0/0/68`，移动 62–63/68 | ✅ 不误报 |
| `B_memlen1_s12`（`mem_length=1`） | bank 恒 1 但 `grad 0/12/56`、移动 51/68 ⇒ 模块在工作；看门狗 fwd7 发 `WARNING` **不 raise** | ✅ 不误报 |

**闸门未到达档（升级项 3，本轮验收重点）—— `max_step=4` 全部开启态组合，逐格回答「谁在看守」：**

| 配置 | 结果 | 看门狗裁决了吗 | **谁在看守（实测，不是读代码推断）** |
|---|---|---|---|
| `group` bs=1 | raise，训练从未开始 | 不需要 | 装配期跨度判据 `sampler.py:344`（fwd 之前） |
| `group` bs=4 gs=1 | raise，训练从未开始 | 不需要 | 同上 |
| `group` bs=4 gs=16 | `rc=0`，bank 恒 4 | ❌ `bank_liveness_checked=False`（4 < 8） | `_check_episode_stream:ran@fwd0` —— **同配置注入实测在 `wrapper.py:372` 触发** |
| `stream` bs=4 | `rc=0`，bank `4→8→12→16` | ❌ | `_check_episode_stream:ran@fwd0` —— 同配置注入实测触发 |
| `stream` bs=1 | `rc=0`，bank `1→2→3→4` | ❌ | **第一批检查判不了 bs=1 ⇒ 此格无人裁决**（残留洞，已记入 `PORT-STATUS.md`） |

> 六格里五格答得上来，第六格答不上来**且已记入 `PORT-STATUS.md` 的残留洞**。
> 闸门未到达这一档因此**通过** —— 这是本轮的验收重点，改动方交付的结论本复审逐格独立复现。

### 4.5 消费端兜底是否真的存在 —— **本复审新增的一档，改动方交付不了**

`12` §2.1 的五个格子里，`group`+bs=1 两格**都停在装配期**。也就是说改动方新写进
`wrapper.py::_check_episode_stream` 的那条跨度兜底，**在真实入口下从未被执行过一次** ——
装配期护栏永远先赢。**一条从未被观测到执行的第二道防线是主张，不是防线。**

本复审用 `--bypass-assembly-guard`（把包属性 `assert_episode_stream_wired` 换成 no-op；
**只注入不构造**：不 new 任何对象，只替换一个宿主可调用对象）单独验它：

| 档 | 实测 | 判定 |
|---|---|---|
| `W_bypass_bs1_s4` | **raise `wrapper.py:338 in _check_episode_stream`，第 0 次 forward**；`assembly:BYPASSED-BY-OBSERVER` | ✅ **兜底真的存在** |
| `W_bypass_gs1_s4` | 同上 | ✅ |

⇒ 改动方文档里「第二训练入口没有装配期护栏时由消费端兜底」这个说法**成立**，本轮由本复审补齐了它的证据。

### 4.6 `mem_length ≤ 1` 的站下换来了什么 —— **本复审新增的一对，第一次还问错了**

「消除误报」只证明了**误报消失**。它没有回答**误报消失之后还剩什么**。
本复审为此补了一对 run —— 而且**第一次问错了地方，这件事本身值得记**：

**第一次（`T_break_memlen1_s12`，`stream` **bs=4** + `mem_length=1` + 注入 + 12 步）**：
期望「无人看守」，实测 **raise `wrapper.py:372 in _check_episode_stream`，第 0 次 forward** ——
bs=4 时批内 4 个样本来自 4 条不同 episode，**第一批检查照样抓到**。
`r4_gear.sh` 因为期望与栈帧对不上判了 INVALID，没有把它促成结果 —— **装置按设计工作**。
⇒ **`mem_length=1` 在 bs≥2 下覆盖未损失**，这是一条正面结论。

**第二次（bs=1，第一批检查天然判不了的那一格）**：

| 档 | 注入 | 实测 |
|---|---|---|
| `B_memlen1_bs1_s12` | 否（健康对照） | `rc=0`、bank `[1]×12`、**`grad 0/12/56`、移动 51/68** ⇒ 模块在工作 |
| **`T_break_memlen1_bs1_s12`** | **是** | `rc=0`、bank `[1]×12`、**`grad 64/4/0`、移动 `0/68`** ⇒ **模块完全没算** |

**两个 run 的护栏输出逐字相同**，都只有 fwd7 那一行：

```
WARNING  MemoryVLAMemory: no bank exceeded 1 entry in 8 training forwards, but mem_length=1
         caps bank length at 1, so this criterion cannot tell a working bank from a dead one
         here and is standing down rather than failing the run. ...
```

⇒ **这行 WARNING 在「模块在工作」与「模块完全没算」两种情况下一字不差，因此不含任何分辨信息。**
一个见过健康 run 的使用者会学会忽略它。

**与已记录的残留洞的区别，是本条的要害**：`PORT-STATUS.md` 记的残留洞是
「`stream` + bs=1 + 批次不连续 + **步数 < K**」——**有 K 这个上界**，跑够长就会被抓。
`mem_length ≤ 1` 把这个上界**去掉了**：实测 12 步（`> K=8`）仍然 `rc=0`。
**改动前**这一格会在第 8 次 forward raise（旧代码没有 `mem_length` 分支，`max_bank_len_seen <= 1` 直接 raise）。

**这个交换本身是对的**（误报会打死正常 run，比漏报更立刻有害），**错的是只写了收益**：
`PORT-STATUS.md` 支持矩阵那一行写「看门狗 fwd8 **主动站下**并 `WARNING`」，读起来是纯收益。
→ **P2-F**。

### 4.7 观测器污染量化（升级项 5）—— **改动方的结论不成立，见 P2-E**

同一开启态配置（`stream` bs=4，20 步），带 / 不带 identity 探针各两次：

| run | 卡 | 探针 | 峰值显存 (GiB) |
|---|---|---|---|
| `B_on_stream_s20` | **1** | 带 | `9.305203437805176` |
| `N_probe_2` | 4 | 带 | `9.302354335784912` |
| **`N_noprobe_1`** | **1** | 不带 | **`9.297196865081787`** |
| `N_noprobe_2` | 5 | 不带 | **`9.297196865081787`** |

- **同卡对照（gpu 1）**：`9.305203437805176 − 9.297196865081787` = **+8.01 MiB**
- **两个不带探针的 run 在不同卡上逐位相同** ⇒ 不带探针的值稳定、可复现，不是噪声
- 带探针两次相差 2.92 MiB ⇒ 探针侧抖动约 3 MiB，**小于 5.3–8.0 MiB 的差值**

⇒ **观测器开销可分辨，量级 5–8 MiB。**
改动方测到的 `+0.98 MiB` 是**跨卡**对照（他们的带探针 run 在 gpu 3、不带在 gpu 0），
且把关闭态的 ~5 MiB 噪声地板套到了另一个量上。**详见 P2-E。**

> 口径差异要明写：本复审的 `--no-identity` 只跳过两次 clone、**保留 forward 包装**（bank 仍被记录，
> 见 `N_noprobe_*` 的 bank `4→8→12→16`）；`run_real4.py` 的 `--no-identity` 跳过**整个探针**。
> 所以改动方测的量应当 **≥** 本复审测的量 —— 而实测反过来了，这本身就是那对 run 有问题的信号。

### 4.8 两套装置的一处对不上：「护栏日志 0 行」—— **P3-J**

`12` §2.1 与 `PORT-STATUS.md` 支持矩阵都写 `group`+bs=1 那几档「**护栏日志 0 行**」。
本复审同一档记到 **1 行 INFO**：

```
INFO  robo_orchard_lab.models.memoryvla.sampler:
      MemoryVLAEpisodeStreamBatchSampler: 600 episodes total, 600 on rank 0/1, 328975 batches of 1
```

**成因是装置边界，不是分歧**：`run_real4.py` 的 `_install_guard_log_capture()` 在
`SimpleTrainer.__init__` 里挂 handler，而这条 INFO 发生在 `train.py:131` 构造 sampler 的时候 ——
**在 trainer 存在之前**，结构上不可能被它看到。本复审的 handler 挂在 `runpy` 之前，所以看得到。

**实质结论不变**：这一行是 sampler 的构造期信息，**不是护栏**；两套装置都确认
「没有任何告警级日志」。但「护栏日志 0 行」这个说法**把装置的观测起点当成了运行的起点**，
下一轮如果拿它跟别的装置比会直接对不上。→ **P3-J**（记录，不影响任何结论）。

### 4.9 其余检查项

| # | 检查 | 结果 |
|---|---|---|
| 11 | 本轮新增开关 / config 键有真实读取者 | **本轮零新增 config 键**（`config_holobrain_common.py` 的 `mv.get(...)` 仍是 12 条，`git diff` 该文件为空）。`group_size` 早有读取者（`:158`），本轮只是多了一个读取者。preflight 判据 K **PASS** |
| 12 | 之前已移植的其他方法 | **N/A，留痕**：`ls -d docs_analysis/*/` 只有 `docs_analysis/memoryvla/` 一个方法目录 |
| 13 | 全局副作用 | 本轮 diff 内无 seed / 默认 dtype / device / hook 注册 / import 期动作；preflight 判据 **S PASS**；RNG 三点指纹关闭态与开启态 `at_exit` 均一致 |
| 14 | 新增探针的有效性 | 三档故障注入见 §4.4；跨度判据的阳性场景就是 `D_group_bs1_*` / `D_group_gs1_*` 三档 |
| — | 新增实例属性是否进 `state_dict` | **实测 `state_dict_has_guard_state == []`**（每个开启态 run 的 `at_exit` 都记）。这比「参数量不变」强：参数量相等只是必要条件 |
| — | `_effective_batch_size` 走的哪一支 | 实测全部走**首选支**：`source='MemoryVLAEpisodeStreamBatchSampler.batch_size'`，`chain_batch_size_attrs=[1]` / `[4]`，与 `config['batch_size']` 一致。回退支与 `WARNING` 支在本机配置下**不可达**（链判据保证 episode sampler 在链上，而它有 `batch_size` 属性） |

---

## 5. S4 — 继承基线三分类复核

> 协议 §0 前置条件满足：`12` §3 主动给了三分类并逐条标「（改动方自述，待复审独立推导）」。
> **本复审独立推导后与它无分歧。** 下面标注每一条是「本轮重验 / 抽验通过 / 继承未验」。

### 5.1 失效（须重新测量；沿用旧值即报告失真）

| 项 | 改动方处理 | 本复审判定 |
|---|---|---|
| `11` §1 / §4.7（P1-C 失效态数值） | 标为失效，并重新测量 | ✅ **正确失效，本轮重验**：旧签名在被审 commit 上不可达 |
| `11` §4.7 引用的看门狗文案 | 已重写 | ✅ **本轮重验**（§3 P1-C） |
| `11` §4.1（P2-B `__doc__ is None`） | 重新测量为 `False` | ✅ **本轮重验** |
| `11` §2.3（P2-A′ 侵入度） | 改成基点 + 两列表 | ✅ **本轮重验，逐格相同** |
| `11` §1 / §9-5（P2-C 仓内测试 0 个） | 不再成立 | ✅ **本轮重验**：3 文件 84 项 |
| `11` §8「观测器抬高量」 | 称已测出 `+0.98 MiB` 并据此改 `MIGRATIONS.md` | ❌ **本轮重验后不成立 → P2-E** |
| **沿用旧值的情况** | — | **未发现**。所有失效项都重新测量了 |

### 5.2 需重验（全部重跑）

| 项 | 本复审实测 | 判定 |
|---|---|---|
| A 档关闭态等价（`stream` / `group`） | 四项精确量 + 结构判据，4 档全同 | ✅ **本轮重验** |
| **C 档定输入数值对齐** | **10 targets / 10 bit-exact / 0 failed，在被审 commit `28af78ab` 上** | ✅ **本轮重验** |
| batch key 集合 | 关闭 14 / 开启 15 | ✅ **本轮重验** |
| 参数总量 | 关闭 `1,136,284,265` / 开启 `1,143,751,529` | ✅ **本轮重验**（与 `11` 逐位相同） |
| optimizer 分组 | `trainable_not_in_optimizer = 0`，68 张量 | ✅ **本轮重验** |
| 开启态 `stream` 回归 | bank `4→8→12→16` 封顶、`grad 0/0/68`、移动 63/68、恒等间隙 **`1.2969558238983154` / `1.1238346099853516`** | ✅ **本轮重验，与 `09`/`11` 记录的 `1.296956e+00` / `1.123835e+00` 逐位相同** |
| 开启态 `group` 回归 | bank 恒 4、`grad 0/0/68`、移动 62–63/68、恒等间隙与 `stream` 逐位相同 | ✅ **本轮重验** |
| 探针有效性 | 3 档故障注入 | ✅ **本轮重验**（2 触发 + 1 残留洞如实） |
| 判据 K/C/D/F/S | preflight `--static` EXIT=0 + 阳性对照 EXIT=1 | ✅ **本轮重验**（§7） |
| 仓库 lint | HEAD ≡ 基线 | ✅ **本轮重验**（§7，口径分歧已定因） |

> C 档在 `28af78ab` 上重跑，而不是沿用改动方在 `fc33a5db` 上的结果 —— 尽管两者代码哈希相同（§3 P3-G），
> **「哈希相同」是本复审自己算出来的，不是假设的**，先证再省。

### 5.3 仍有效 → 抽验 5 条（挑与本轮改动语义相邻的，非随机）

| # | 抽验项 | 为什么挑它 | 结果 |
|---|---|---|---|
| 1 | 拷贝保真 **F** | 改动方称「`[port:]` 标记全在 `memory_bank.py`」——**这是个可核的断言** | ✅ **抽验通过**：`grep -rn "\[port:"` 全部 7 处命中 `memory_bank.py`，`sampler.py`/`wrapper.py` **一处没有**；`memory_bank.py` 本轮 **0 行改动**；preflight F **PASS**（6 marker(s) checked, 0 finding) |
| 2 | `_build_memoryvla_cfg` 键转发（判据 **D**） | 护栏现在**多读了一个** `group_size` | ✅ **抽验通过**：`config_holobrain_common.py:158` `group_size=mv.get("group_size", 16)`，与 `sampler.py:277` 的 `mv.get("group_size", 16)` **默认值一致**；12 键全部有读取者 |
| 3 | 孤儿 config 键（判据 **K**） | 本轮是否引入死键 | ✅ **抽验通过**：零新增键，preflight K **PASS** |
| 4 | **ckpt 兼容性** | `__init__` 新增 `self.group_size` / `self.mem_length` + 两个 `set` | ✅ **抽验通过，且用了比参数量更强的判据**：每个开启态 run 实测 `state_dict` 中与这四个名字相关的键**为空集**；参数量 `1,143,751,529` 不变可作旁证 |
| 5 | mask 极性 | 本轮**未触及**，作「抽验无系统性偏向」的对照 | **（继承自上一轮，本轮未重验）**；`structure.py` 不在本轮 diff 内，`git diff --name-status` 可证 |

**5 条里 4 条本轮抽验通过、1 条按设计作为未重验对照。** 未触发「整类降级为需重验」。

### 5.4 明确标注为继承、本轮未重验的结论

以下全部标 **（继承自上一轮，本轮未重验）**，本复审既没复核也没推翻：
方法要素 12/15 与 A 逐行一致 · 接口语义 32 项一致 0 项不一致 · cite 零幻觉 ·
四个宿主文件的 L1 判定 · ckpt 兼容性 1000→1068 的历史结论 · `BottleneckSE` 不接入的理由 ·
mask 极性 · `09` §4.5 四条宿主语义 · `11` §5.4 列的那一批。

---

## 6. S5 — 新增风险是否如实记录

核对对象 **`PORT-STATUS.md`**（`12-review-response.md` 是单轮应答，不是常设风险册）。

### 6.1 协议要求的三类

| 类别 | 记了没有 | 位置 |
|---|---|---|
| **① 训练动力学变化** | ✅ 记了，且写得准确 | 遗留 5（stream）· 遗留 10（group：「能跑不等于该用」、记忆跨度只有一个 batch、`mem_length` 不起作用、`group_size=2` 时 12 个张量精确零梯度）· 支持矩阵给了 `batch_size` / `group_size` 两个新维度的边界条件 |
| **② 本轮之后才可能暴露** | ⚠️ **部分** | 遗留 11 补了「K 是时间闸门」这一性质、K 的下界依据、「剩下的窗口」；支持矩阵末尾如实写了残留洞。**但 `mem_length ≤ 1` 把残留洞的 K 上界去掉了这件事没写 → P2-F**；`_effective_batch_size` 全读不到时判据被 `WARNING` 跳过这条路径**只在 `12` 里** |
| **③ 本轮仍无法验证的项原样保留** | ⚠️ **16 条里 2 条没进常设风险册 → P1-D** | 见下表 |

### 6.2 `11` §8 的 16 条 —— 本复审逐条独立去核「在不在 `PORT-STATUS.md` 里」

| # | 条目 | 改动方打勾指向 | 本复审去核 |
|---|---|---|---|
| 1 | 外部真实 ckpt 加载 | 遗留 7 | ✅ 在 |
| 2 | DDP / 多卡 unused-parameter | 已知问题 3 + 遗留 3/6 | ✅ 在 |
| 3 | A 的采样频率 / 降采样 | 遗留 9 | ✅ 在 |
| 4 | A 与宿主端到端数值可比性 | 已知问题 1 | ✅ 在 |
| 5 | D 档墙钟时间 | 「D 档：墙钟不用来下结论」 | ✅ 在 |
| 6 | `fifo` vs `tome` | 遗留 2 | ✅ 在 |
| 7 | 开启态训练行为本身 | 遗留 5 + 10 | ✅ 在 |
| 8 | 关闭态浮点严格等价 | 「关闭态等价性」+「峰值显存也不是精确量」 | ✅ 在 |
| 9 | `_episode_spans` 换数据集 | 遗留 12 | ✅ 在（**本轮补回**） |
| 10 | 长时训练稳定性 | 遗留 13 | ✅ 在（**本轮补回**） |
| 11 | `group` 是否还有意义 | 已解决 | ✅ 合理移出 |
| 12 | K=8 是否合适 | 遗留 11 | ✅ 在 |
| 13 | `group_size` 中间取值 | 遗留 10 | ✅ 在 |
| 14 | 观测器抬高量 | 「本轮已测出数，不再是无法验证项」 | ⚠️ **自相矛盾**：`12` §5 同时把「观测器开销的真实值」列为**本轮新增**无法验证项。而本复审的结论是这个数**测错了**（P2-E），它应当**留在**清单里 |
| 15 | **仓库 lint 门与 CI 一致性** | 「无法验证清单（本轮沿用）」 | ❌ **`PORT-STATUS.md` 里没有**（`grep -ni "lint\|ruff\|CI"` 无命中）。只在 `12` §5 |
| 16 | **`enable=False` + 非空 `dataset_sample_weights` 真实入口** | 「无法验证清单（本轮沿用）」 | ❌ **`PORT-STATUS.md` 里没有**。只在 `12` §5 |

**14/16 落在常设风险册里，2 条没有；本轮新增的「CI 的 pytest 是否真收集到」同样只在 `12` 里
（`PORT-STATUS.md:369` 写「已记入无法验证清单」，但 `PORT-STATUS.md` 里并不存在这样一张清单，
常设的是遗留问题 1–13，而这条不在其中）。** → **P1-D**

### 6.3 `MIGRATIONS.md`

**教训 12–15：✅ 全部写成了方法无关的可复用判据**，本复审逐条读过：

| 教训 | 是否方法无关 | 依据 |
|---|---|---|
| 12 能静态判定的前提不要留给后果判据 | ✅ | 给了可操作问法「这个失效，需要看到数据才能确定吗？」 |
| 13 带闸门的判据必须交付「闸门未到达」那一档 | ✅ | 明确推广到「第一次才检查 / 每 N 步 / 采样检查」 |
| 14 判据分辨不了的成因不能断言 | ✅ | 给了机械判据「任何建议都要能回答『照做会发生什么变化』，答案是『什么都不变』就是错的」 |
| 15 证据从第一次写就绑代码哈希 | ✅ | 给了顺序规定（先提交再验证）与交付前的机械核对 |

#### 教训 9 的「6 MiB 量级」订正：❌ **不成立** —— **P2-E**

本轮把这句原文划掉了：

> ~~本次的探针每个 forward clone 两个张量，直接把 D 档显存读数抬高了 6 MiB 量级~~
> **订正（第四轮：终于实测了）**：差值是 **+0.98 MiB（+0.010%）**，……**落在 ~5 MiB 的 run-to-run 噪声里**，
> 所以正确说法是「观测器开销在这套测量下**不可分辨**，上界约 5 MiB」。
> **那个 6 MiB 是推断，被写成了实测。**

**本复审四个 run 的实测把它反过来了**（数据见 §4.7）：

- **同卡（gpu 1）对照**：带探针 `9.305203437805176` − 不带 `9.297196865081787` = **+8.01 MiB**
- **两个不带探针的 run 在不同卡（gpu 1 / gpu 5）上逐位相同** ⇒ 不带探针的值稳定，不是噪声
- 带探针两次相差 2.92 MiB ⇒ 探针侧抖动 ≈ 3 MiB，**小于要测的差值**

⇒ **观测器开销可分辨，量级 5–8 MiB。原来那句「6 MiB 量级」落在实测区间内，被划掉的是对的那句。**

**成因有两个，都值得记**：

1. **那对 run 跨卡** —— 改动方的带探针 run 在 gpu 3、不带在 gpu 0（`PORT-STATUS.md` 自己写着
   `(gpu 3, 20 forwards)` / `(gpu 0, 0 forwards)`）。本复审的同卡对照给出 8 倍大的差值。
2. **噪声地板取错了量** —— 用来判「落在噪声里」的 ~5 MiB 是从**关闭态** run 测出来的，
   而被判的是**开启态带/不带探针**的差。

> **这正是教训 9 自己那句话被违反的样子**：「一个数在被引用之前，先得知道它的噪声地板在哪」——
> 噪声地板必须是**同一个量、同一套条件**下的噪声地板。
> 讽刺的是，改动方在同一段里写对了方法（「不要靠估，跑一对 run 把它测出来」），
> 只是那一对 run 本身没有控制住卡这个变量。**修法不是回到 6 MiB，是把结论改成有区间的实测。**

---

## 7. S6 — 机械判据全量回放 + 协议反馈

### 7.1 工具版本（钉住）

四个工具的 md5 与 `09` §7.1 / `11` §7.1 **逐个相同** ⇒ 三轮之间工具没搬家、没变松，本轮的绿与前两轮可比：

```
8ad881d7f6ac955d79d4bd37f33f718f  preflight.sh
13f5ffdd6280fa9e6d0c467502dc61a9  copy_fidelity_check.py
7971d335083212f9bec576c387c52005  orphan_switch_check.py
de4cfc37d06c80d37314337ff8a4e350  port_probe.py
```

### 7.2 逐条结果

| 判据 | 命令 | 结果 | 是否误报 |
|---|---|---|---|
| K/C/D 孤儿开关 / 死类 / 默认漂移 | `preflight.sh --static`（同样两组豁免） | **PASS** | 否 |
| F 拷贝保真 | 同上 | **PASS**（6 marker 检查，0 finding） | 否 |
| S 关闭态全局副作用 | 同上 | **PASS** | 否 |
| **整体** | `preflight.sh ... --static` @ `28af78ab` | **`preflight PASSED`，EXIT=0** | — |
| **阳性对照** | 同一工具、同一组参数 @ `18106b05` | **EXIT=1，3 findings**：`ORPHAN episode_stream_sampler` · `UNUSED MemoryVLAEpisodeStreamBatchSampler` · `DRIFT episode_stream_sampler plan='False' shipped=True` | — |
| C 档定输入 | `tools/check_reference.py` @ `28af78ab` | **10 targets · 10 bit-exact · 0 failed**，`max\|diff\| = 0.000e+00` | 否 |
| 仓内测试 | `.git/run_tests_nopytest.py`（逐文件） | **84 PASS / 0 FAIL / 0 SKIP** | 否 |

**阳性对照的做法**：`git clone --shared --no-checkout` 到 `review4/base_18106b05` 再 `checkout 18106b05`
（`.git` 是**目录**，`preflight.sh:91` 才认）。**本复审没有复用 `fix4/clone_18106b05`。**

### 7.3 lint 口径分歧 —— **已定因**

`11` §2.4 记「HEAD 5 findings」，`12` §2.5 记「31」，改动方称「未能复原对方取值范围」。
本复审用同一 ruff（`envs/RoboDojo/bin/ruff` 0.15.22）复原了两者：

| 取值范围 | `49b2178c` | `28af78ab` |
|---|---:|---:|
| 全仓 `.`（`12` 的口径） | 31 | **31** |
| `robo_orchard_lab/` | 7 | **7** |
| `robo_orchard_lab/models/memoryvla/` | 7 | **7** |
| **`sampler.py` + `wrapper.py` 两个文件（`11` 的口径）** | **5** | **5** |
| 新增的 3 个测试文件 | — | **All checks passed!** |

⇒ **两个数都对，只是范围不同；两个范围下 HEAD 与基线都相等 ⇒ 本轮零新增 lint 债。**
`11` 当时审的就是这两个文件，取这个范围是自洽的。这条分歧到此关闭，不必再挂着。

### 7.4 协议反馈（`review-incremental` 第三次实战 —— **未修改协议文件**）

1. **上一轮提的六条升级项，本轮全部执行，且都产出了结论**：
   精确判据自证（§4.3，产出第四个显存值）· 文案指向真实原因（§3 P1-C，四条逐条实测）·
   合法配置不误报 + 闸门未到达（§4.4，两张表）· 证据哈希机械核对（§3 P3-G，独立复核 17/17）·
   观测器污染升格为可执行判据（§4.6，**直接推翻了一条已写进 `MIGRATIONS.md` 的订正**）·
   文档 commit 逐 hunk 读（§2.3，本轮四个提交里两个是纯 `.md`，P2-D 就是这么读出来的）。
   **六条建议全部有产出，其中两条直接产生了本轮的 P2。建议固化进协议正文。**
2. **建议新增一行 S3 检查：「新加的第二道防线，在真实入口下是否可达」。**
   本轮 `wrapper.py:338` 那条兜底**永远被装配期护栏抢先**，改动方交付的五个格子里一次都没执行过。
   要验它只能**旁路第一道**。协议应当明确：**当 A 与 B 检查同一件事且 A 总是先跑，
   B 的证据必须来自旁路 A 的一档**，否则 B 是未验证的。
3. **建议给「无法验证清单承接」加一条验收判据：目标必须是常设文档。**
   本轮 P1-D 的成因不是没做清单，是清单的「承接到」一栏允许指向单轮应答文档。
   判据可以很机械：**承接目标只能是 `PORT-STATUS.md`（或等价的常设风险册），
   指向任何 `NN-review-response.md` 一律判未承接。**
4. **建议 S2 增加一行：「上一轮报告里引用的行号，在被审 commit 上是否还指着同一个东西」。**
   P2-D 正是这一类：文档说 `sampler.py:264`，被审 commit 上那里是一段文档字符串。
   行号是最容易随改动过期的引用，而它看起来非常像已经核对过的事实。
   机械做法：文档里出现的 `<file>:<line>` 全部抽出来,与实测栈帧比对。
5. **两条给下一轮的提醒（本轮自己踩的）**：
   ① 本复审第一版矩阵**漏了 `mem_length=1` + 批次真的坏掉**这一格 —— 只测了「误报没了」，
   没测「误报没了之后还剩什么」。**任何「消除误报」的改动，验收必须成对：
   误报消失 + 真故障仍被抓。** 补跑后正是 P2-F 的证据。
   ② 补跑的**第一次还选错了配置**（bs=4，第一批检查照样抓得到），
   是 `r4_gear.sh` 用「期望的栈帧」判了 INVALID 才没被当成结论。
   **「期望写成栈帧、由装置核对」这条做法，本轮既抓了被审对象也抓了复审自己。** 建议写进协议。

---

## 8. 无法验证清单

**`11` §8 的 16 条原样承接**（本复审自己的清单，不因为主要问题修好了而消失）：

| 项 | 为什么仍验不了 | 本轮有无变化 |
|---|---|---|
| 外部真实 ckpt 加载 | bucket 只有 v9，config 是 v10，`vlm.*` 全线 size mismatch；本机无外网 | 无变化。本轮全部 24 档 `checkpoint=null` |
| DDP / 多卡 unused-parameter | 本机任意两卡 gather 必崩 | 无变化。**本轮新增一条相关不确定性**：`_effective_batch_size` 读的是 sampler 对象的 `batch_size`，单卡下它就是有效值；DDP 下它是 per-rank 还是全局未验 |
| A 的采样频率 / 降采样 | 定义端在 A 的 RLDS 管线之外 | 无变化 |
| A 与宿主端到端数值可比性 | 原理上不可比 | 无变化 |
| D 档墙钟时间 | 卡共享 | 无变化。本轮同配置 `B_on_stream_s20` 245.1 s vs `N_probe_2` 260.8 s（同代码、同配置、不同卡） |
| `fifo` vs `tome` 的实际差异 | 需要跑到 episode 尺度 | 无变化 |
| 开启态的训练行为本身 | 需要跑到收敛比指标 | 无变化 |
| 关闭态在浮点层面的严格等价 | 真实入口不逐位可复现 | 无变化；峰值显存**第三次**确认属于这一类 |
| `_episode_spans` 在其他数据集上的正确性 | 只在 RoboDojo Memory 六任务上验过 | 无变化；**已在 `PORT-STATUS.md` 遗留 12** |
| 长时训练稳定性 | 本轮最长 20 step | 无变化；**已在遗留 13** |
| K=8 是否合适 | 上界仍未实测确定 | **下界 2 本轮独立复现**（`B_stream_bs1_s12` 的 bank 序列 `1,2,3,…`）；上界仍未做 |
| `group` 在 `1 < group_size < batch_size` 区间的完整行为 | 本轮只跑了 `gs=1`（拒绝）与 `gs=16 > bs=4` | 无变化 |
| **观测器自身对峰值显存的抬高量** | — | ⚠️ **本轮测出可分辨的 5–8 MiB，但精确值仍未定**（需独占卡 + 多次取中位数）。**它应当留在清单里，而不是像 `12` §4 第 14 项那样打勾移出** |
| 仓库 lint 门与 CI 的一致性 | `holobrain_internal` 没装 ruff，借用 `envs/RoboDojo/bin/ruff` 0.15.22 | 无变化；**未进 `PORT-STATUS.md` → P1-D** |
| `enable=False` + 非空 `dataset_sample_weights` 的真实入口行为 | 需要一份带 per-spec `sample_weight` 的 dataset_specs，要改宿主 config | 无变化；本轮同样未做；**未进 `PORT-STATUS.md` → P1-D** |
| **CI 的 pytest 是否真的收集到新增测试** | 本机无 pytest，装它破 E0 | **本复审把结构性论证推进了一档**（三层 `__init__.py` 齐全、66 个同目标树文件），但 CI 侧配置仍未验；**未进 `PORT-STATUS.md` → P1-D** |

**本轮新增的无法验证项：**

| 项 | 为什么验不了 |
|---|---|
| **`_effective_batch_size` 的回退支与 `WARNING` 支** | 本机配置下**不可达** —— 链判据保证 episode sampler 在链上，而它总有 `batch_size`。这两支因此**从未被执行过**，它们的正确性只有静态论证 |
| **`mem_length ≤ 1` 下真实训练会不会自己退化** | 本轮实测「注入故障后 12 步仍无人裁决」（P2-F），但「不注入时 `mem_length=1` 的真实训练好不好」需要跑到收敛，属于「开启态训练行为本身」那一类 |
| **观测器污染的精确值** | 见上；本轮只给出「可分辨、5–8 MiB」这个区间 |
| **旁路装配期护栏这一档是否代表真实的第二训练入口** | 本复审用 no-op 替换模拟「入口不调用装配期护栏」。真实的第二训练入口有没有别的差异未验 |

---

## 9. 最短修复路径（按「修完能翻案」排序）

1. **P1-D（10 分钟）** —— 把三条搬进 `PORT-STATUS.md` 的遗留清单：仓库 lint 门与 CI 一致性 ·
   `enable=False`+`dataset_sample_weights` 真实入口 · CI 的 pytest 是否收集到新增测试。
   并把 `12` §4 那张打勾表的**验收判据**改成「承接目标必须是常设风险册」，
   顺手把 `PORT-STATUS.md:369` 那句「已记入无法验证清单」指向真实存在的一节。
2. **P2-F（10 分钟）** —— 支持矩阵里 `mem_length=1` 那一行补上代价：
   **`mem_length ≤ 1` 时看门狗一次性站下并永久失效，`stream` + bs=1 的残留洞因此不再受 K 约束**
   （实测 `T_break_memlen1_bs1_s12`：12 步 > K，仍 `rc=0`、`grad 64/4/0`、移动 `0/68`；
   健康对照 `B_memlen1_bs1_s12` 发出**逐字相同**的 WARNING）。
   并在遗留 11 的「剩下的窗口」那一格加一句：`mem_length ≤ 1` 时该窗口**没有步数上界**。
   写清楚这个交换本身是对的 —— 只是代价要写出来。
3. **P2-D（5 分钟）** —— `PORT-STATUS.md` 三处 `sampler.py:264` 改成 `sampler.py:298`
   （第 328 / 430 / 572 行），或者按 P2-A′ 的现行口径给行号加上「截至哪个 commit」。
   **建议后者** —— 行号一定会再过期，而带基点的行号至少能自证过期。
4. **P2-E（半小时，含一对 run）** —— 撤回 `MIGRATIONS.md` 教训 9 的「+0.98 MiB / 不可分辨」订正。
   正确说法是「**同卡对照实测 +8.0 MiB，两个不带探针的 run 跨卡逐位相同 ⇒ 开销可分辨，量级 5–8 MiB**」，
   原来那句「6 MiB 量级」落在实测区间内。
   **这一条的教训比数字本身重要**：那对 run **跨卡**（gpu 3 vs gpu 0），
   而用来判「落在噪声里」的 ~5 MiB 噪声地板是从**关闭态**测出来的、属于**另一个量**。
   —— 这正是教训 9 自己那句「一个数在被引用之前，先得知道它的噪声地板在哪」被违反的样子。
5. **P3-I（2 分钟）** —— `12` §8 的 `--stat` 块重取一次，或者按 §0 处理 commit 哈希的同样办法，
   明写「本节数字取自 `fc33a5db`，不含本提交自身」。
6. **P3-J（下次移植时）** —— 日志捕获要挂在 `runpy` 之前而不是 trainer `__init__` 里，
   否则「护栏日志 N 行」这个量里天然缺了构造期那一段，两套装置就对不上。

**修完 1 即可把唯一的 P1 清干净；2–5 是文档订正。全部不需要动代码。**

---

## 附：本轮证据清单

`$ROL_JFS/port/memoryvla/review4/`（不进 git；`fix4/` `fix3/` `review/` `review2/` `review3/` 全程只读未改）

| 文件 | 是什么 |
|---|---|
| `r4_probe.py` | 本复审的观测器，自 `review3/r3_probe.py` 扩展。新增：装配期护栏的进入/退出与 `_effective_batch_size` 返回值 · 两条 `_check_episode_stream` 分支可辨 · `mem_length` 站下分支 · `group_size`/`mem_length`/两个 set · `state_dict` 键差集 · `RAISE_SITES` 行号表 · **`--bypass-assembly-guard`** |
| `r4_gear.sh` | 单档 runner。**期望写成 `raise:<函数名>@<行号>`，由记录到的栈帧核对**，退出码不参与促成；对不上留 `.INVALID.json` |
| `r4_run_all.sh` | 矩阵驱动（`off`/`guard`/`backstop`/`regr`/`teeth`/`noise`），4 宽并行，**只用 GPU 1/3/4/5** |
| `r4_gen_cfg.py` · `cfg/` | 12 份配置，**生成而非手写** |
| `r4_summary.py` · `digest.json` | 「谁在看守」由护栏调用记录、装配期记录与日志级别**推导**，不靠眼看；答不上来记 `UNWATCHED` |
| `r4_offcmp.py` · `offstate_compare.txt` | 四项精确判据 + 结构判据的逐对比较；峰值显存打印但不参与比较 |
| `r4_bind_check.py` · `fix4_binding_recount.txt` | **独立复核改动方的 17/17**，真值取自 `git show <commit>:<path>` |
| `runs/2026-08-05/A_off_{stream,group}_{1,2}.json` | 关闭态 ×4（结构判据 + 噪声地板 + 精确判据自证） |
| `runs/2026-08-05/ctrl_hostseed.json` | **阳性对照**：注入宿主 sampler 构造参数 `seed=99` → uuid 0/20 |
| `runs/2026-08-05/D_group_bs1_s{4,12}.json` · `D_group_gs1_s4.json` | P1-C 三个致命格，全部 `sampler.py:344`，`fwd=0` |
| `runs/2026-08-05/D_group_bs4_s{4,12}.json` | 建议动作 ②「`group` + bs≥2 ∧ gs≥2」有效性验证 |
| **`runs/2026-08-05/W_bypass_{bs1,gs1}_s4.json`** | **旁路装配期护栏 → 消费端兜底 `wrapper.py:338` 确实触发**（改动方交付不了的一档） |
| `runs/2026-08-05/B_on_stream_s20.json` | `stream` 开启态回归，恒等间隙与 `09`/`11` 逐位相同 |
| `runs/2026-08-05/B_stream_bs1_s{4,12}.json` | 建议动作 ①「`stream` + bs=1」有效性 + 残留洞的无注入对照 |
| `runs/2026-08-05/B_memlen1_s12.json` | `mem_length=1` 不误报（bs=4） |
| `runs/2026-08-05/T_break_{bs4_s12,bs1_s12,bs1_s4}.json` | 三档故障注入：两档触发、一档残留洞 |
| **`runs/2026-08-05/B_memlen1_bs1_s12.json`** | **P2-F 的健康对照**：`grad 0/12/56`、移动 51/68 |
| **`runs/2026-08-05/T_break_memlen1_bs1_s12.json`** | **P2-F 的证据**：同配置 + 注入 → `grad 64/4/0`、移动 `0/68`、**WARNING 逐字相同** |
| `runs/2026-08-05/T_break_memlen1_s12.INVALID.json` | 本复审**问错的第一版**（bs=4，被第一批检查抓到）。装置判 INVALID 未促成，原样留痕 |
| `runs/2026-08-05/N_{probe_2,noprobe_1,noprobe_2}.json` | 观测器污染对照（升级项 5） |
| `runs/2026-08-05/G_group_nosampler.json` | P1-B 主干回归，`sampler.py:298`（**不是文档写的 `:264`**） |
| `preflight_head_28af78ab.txt` | 机械判据全量，EXIT=0 |
| `preflight_control_18106b05.txt` | **阳性对照，EXIT=1 / 3 findings** |
| `base_18106b05/` | `git clone --shared` 出来的基线仓（`.git` 是目录，preflight 才认）；**未复用 `fix4/` 那份** |
| `lint/tree_49b2178c/` | `git archive` 静态树，**只给 ruff 用**（ruff 不 import，meta path finder 的坑不适用） |
| `check_reference_28af78ab.txt` | C 档 10/10，**在被审 commit 上** |
| `tool_md5.txt` | 四个工具版本钉住 |

> **两条留给下一轮的提醒**（都是本复审自己踩的）：
>
> 1. 第一版矩阵只测了「`mem_length=1` 的误报没了」，**没测「误报没了之后还剩什么」** ——
>    补跑那一对才有了 P2-F。**凡是「消除误报」的改动，验收都必须成对：
>    误报消失 + 真故障仍被抓。** 单测前者会得到一个看起来完全正确、而且确实修好了一件事的结论。
> 2. 补跑的第一次**还选错了配置**（bs=4，第一批检查照样抓得到），
>    是 `r4_gear.sh` 拿「期望的栈帧函数名 + 行号」核对后判 INVALID 才没被当成结论 ——
>    如果沿用上一轮那种「rc=3 即视为按预期 raise」的 runner，它会被促成为
>    「`mem_length=1` 下无人看守」这个**错误结论**。
>    **把期望写成栈帧、由装置核对**，这一轮既抓了被审对象，也抓了复审自己。
