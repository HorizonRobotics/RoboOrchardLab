# 09 — 增量复审：MemoryVLA 移植的 P0-1 修复轮

复审者独立执行 · 日期 **2026-08-04** · 全程只读：未修改宿主代码、`port/memoryvla`、`review/memoryvla`、A repo 或任何环境

| 项 | 值 |
|---|---|
| 协议 | `review-incremental`（**首次实战**，反馈见 §7.3） |
| 基线 commit | `18106b05`（上一轮全量审查的对象） |
| 被审 commit | `2b739226`（`port/memoryvla` HEAD） |
| 修复方提交 | `701679a9` 接线 · `166b8756` 护栏+文档订正 · `a81682e0` `f6dfd1e8` 验证记录 |
| 本轮改动契约 | `~/storage_policy/protocols/robo_orchard_lab/port_memoryvla_2.md` |
| 上一轮裁决 | 🔴 **REJECT**（P0×1 P1×1 P2×3 P3×2） |
| 本轮证据 | `$ROL_JFS/port/memoryvla/review2/`（我自己的装置与 9 个真实入口 run，不进 git） |
| 上一轮 / 修复方证据 | `…/review/`、`…/fix/`　**只读，未覆盖、未重新生成** |
| 算力 | 单卡 × 9 个 run（GPU 1/2/3/4，均为空闲卡），另 2 个纯 CPU 单元测试 |

**标注约定**：凡引用上一轮结论而本轮未重测的，一律标 **（继承自上一轮，本轮未重验）**。

---

## 1. 裁决

# 🟡 ACCEPT-WITH-FIXES

**一句话理由**：P0-1 与 P1-1 都真的闭环了——`train.py:131` 现在构造 episode sampler、真实入口下
68 个张量全部拿到非零梯度、恒等间隙从 1 ULP 变成 ~1.30，且**关闭态 10 组两两比较全部逐样本 id 一致**；
剩下两条 P1 不是「修错了」，而是「修的覆盖面比声称的窄」，都不影响 ship 配置。

| 级别 | 数量 | 条目 |
|---|---:|---|
| **P0** | **0** | — |
| **P1** | **2** | **P1-A** 失效的旧数值仍原样挂在 `05-ablation-matrix.md` / `06-verification.md` 上，零标注<br>**P1-B** `dataloader_type="group"` 在 HEAD 上**没有任何可用配置**：开 sampler 会 raise，关 sampler 则静默恒等且三道护栏全不响——我实测复现出与 P0-1 **完全相同的失效签名** |
| **P2** | **1** | **P2-A** 本轮**重写过**的那一行仍写「0 删除」，实际 `train.py` −6、`sampler.py` −1 |
| **P3** | **4** | **P3-A** 上一轮六条「无法验证」只承接了五条<br>**P3-B** 证据 JSON 无 commit 绑定<br>**P3-C** `run_real.py` docstring 关于确定性开关的说法被其自身实测推翻<br>**P3-D** 契约要 2 个 commit，实交 4 个（多出两个纯文档） |

### 这个裁决该怎么读

上一轮 REJECT 的理由是「合入即意味着打开开关什么都不会发生，且没有任何信号会告诉使用者」。
**这两件事在 ship 配置上都不再成立**，且全部是我自己从真实入口测出来的，不是读它的自评得到的。

本轮修复的质量明显高于一次普通补丁：范围锁死守住了、护栏做成 fail-fast 而不是 warn、
A 档判据被**主动**改成两档，并诚实记录了「真实入口不是逐位可复现的」——
这一条推翻的是移植方自己上一轮的说法，而我独立复算它记的每一个数字，**逐位吻合，无一虚报**。

---

## 2. S1 — 范围合规

### 2.1 必须先做的一次拆分

`git diff --name-status 18106b05..2b739226` 有 16 个文件，但 `2b739226` 是一次**合并**，
其中 9 个是把 `review/memoryvla`（上一轮审查报告 `06*.md`、`07-postmortem.md`）并进来带的，
**不是修复方的改动**。按修复方那一侧的 4 个提交判才有意义：

```
$ git rev-list --parents -n1 2b739226
2b739226  f6dfd1e8(修复方一侧)  97837e04(review/memoryvla 一侧)

$ git diff --name-status 18106b05..f6dfd1e8        # ← 修复方实际改了什么
M	docs_analysis/MIGRATIONS.md
M	docs_analysis/memoryvla/04-port-plan.md
A	docs_analysis/memoryvla/08-review-response.md
M	docs_analysis/memoryvla/PORT-STATUS.md
M	projects/holobrain_internal/common/train.py
M	robo_orchard_lab/models/memoryvla/__init__.py
M	robo_orchard_lab/models/memoryvla/sampler.py
M	robo_orchard_lab/models/memoryvla/wrapper.py
```

**结论：契约内，不熔断。** 三处允许改动逐一对上——接线（`train.py`）、
护栏（`memoryvla/{__init__,sampler,wrapper}.py`，实现体全在方法子目录内）、
文档订正（其余四份）。**无契约外文件**，因此上一轮的继承基线整体有效。

### 2.2 逐 hunk 读完后单独拎出来的三点

**① `train.py` 的关闭态分支发生了代码位移，不是纯增量。**
`DistributedBatchFlagSampler(...)` 原本是 `DataLoader(...)` 的一个实参，现在被提到前面成了局部变量。
构造参数一字未动（含 `dataset_sample_weights=config.get(...)`），但「没动过」这件事**不能靠读 diff 判**——
位移会不会改变关闭态，只有实测能答。见 §4.1。

**② `train.py` 是第 5 个被触及的宿主文件，且是宿主主训练入口。**
`PORT-STATUS.md:30` 与 `MIGRATIONS.md:44` 都已订正成「5 个文件」——**这个订正是对的**；
形状仍是「一个开关判断 + 一次调用」，实现体没有外溢。判 **L1**，与自述一致。
但同两行里的「0 删除」是错的，见 §6 P2-A。

**③ `config_holobrain_common.py` 没有被改。**
上一轮的加重情节是「`04-port-plan.md:78` 写默认 `False`，实际 ship `True`」。修复方的处置是
**把文档改成 `True` 去迁就代码**，不是把代码改回 `False`。这可辩护（该键挂在 `enable=False` 之下，
单独设它无效），订正也写得很清楚。它的实际含义是**`enable=True` 一旦打开，数据顺序默认就变了**——
`PORT-STATUS.md`「新增 config 字段」订正段有写明，**记为已披露**。

### 2.3 git 卫生（逐条实测）

| 判据 | 命令 | 结果 |
|---|---|---|
| 工作树干净 | `git status --porcelain \| wc -l` | **0** ✅ |
| `.gitignore` 未触碰 | `git diff --stat 18106b05..HEAD -- .gitignore \| wc -l` | **0** ✅ |
| `review/memoryvla` 未被追加 | `git rev-parse --short review/memoryvla` | `97837e04`，与 `07` §7 记录一致 ✅ |
| A repo 与开审前快照一致 | `git -C ~/git_repo/MemoryVLA status --porcelain \| md5sum` | `9815d522644f15ab4edd56e5b33d1d03`，与 `07` §7 记录**逐字节相同**；HEAD 仍 `0eef5c3` ✅ |
| 契约「两个 commit，各自跑完 A 档」 | `fix/runs/2026-08-04/gearA_c1.json` `gearA_c2.json` | 两个 A 档都在 ✅；但非合并提交是 **4 个** → P3-D |

---

## 3. S2 — 上一轮 P0/P1 逐条闭环

### P0-1 `episode_stream_sampler` 是死开关 —— ✅ **已闭环**

**静态侧**，用同一版本工具（指纹见 §7.1）跑修复前后：

```
tree @ 18106b05（git archive，同参数）        tree @ 2b739226（HEAD）
  ORPHAN  episode_stream_sampler          →   ok   7 reader(s), e.g. train.py:119
  UNUSED  MemoryVLAEpisodeStreamBatchSampler →  ok   constructed at train.py:131
  DRIFT   plan='False' shipped=True       →   ok   episode_stream_sampler  True
  ==== 4 finding(s) ====  EXIT=1              ==== 1 finding(s) ====（只剩既有豁免 BottleneckSE）
```

**阳性对照做过**：同一份脚本、同一组参数跑在 `18106b05` 的树上确实报出那 4 条，
所以 HEAD 上的绿不是判据失灵。`07` §5.3 把「三条红一起变绿」定为验收判据——**达成**。

**运行时侧**，我自己的 8 步 B 档（`train.py` 真实入口，装置只 wrap 不 new）：

| 观测 | 上一轮**失效态**（真实路径） | **本轮我实测** | 修复方自述 |
|---|---|---|---|
| 实际被迭代的 sampler | `DistributedBatchFlagSampler` | **`['MemoryVLAEpisodeStreamBatchSampler']`** | 同 |
| 每 batch 不同 episode 数 | **4 / 4** | **1**（8 步全程） | 1 |
| bank 长度 | **恒为 `[1]`** | **4 → 8 → 12 → 16 封顶** | 同 |
| 感知 `max\|out−in\|` | `1.192093e-07`（1 ULP） | **`1.296956e+00`** | `1.296956e+00` |
| 认知 `max\|out−in\|` | `5.960464e-08`（1 ULP） | **`1.123835e+00`** | `1.123835e+00` |
| grad None / 精确零 / 非零 | **64 / 4 / 0** | **0 / 0 / 68** | 同 |
| 参数移动 | **0 / 68** | **62 → 63 / 68** | 62 → 65 / 68 |
| optimizer 分组 | `{'1': 68}` | **`{'1': 68}`，游离 0** | 同 |
| 参数总量 | 1,136,284,265 | **1,143,751,529（+7,467,264）** | 同 |

上面两个恒等间隙不是我的观测器算的，是**移植进去的护栏自己打的日志**：

```
wrapper.py:274 | MemoryVLAMemory[stream]: first training batch holds 1 distinct episode(s) across 4 samples.
wrapper.py:326 | MemoryVLAMemory identity probe on the first forward that reads history:
                 {'perceptual': '1.296956e+00', 'cognitive': '1.123835e+00'} (tolerance 1e-05)
```

**E 档我也独立重跑了**（60 步 / bs=8），关键证据是 bank 回落：

```
per_mem_bank maxlen per step: 8 16 16 ... 16 16 [step 54] 8 16 16 16 16 16
                                              ^^^^^^^^^^ 跨过 episode 边界，clear_episode 真的跑了
```

`step 54` 与修复方记录的**同一步**回落，`distinct_episodes_in_batch` 全程为 1，
grad 首尾 `0/0/68`，参数移动 63/68，峰值显存 13.153 GiB（其自述 13.1525）。
**单条 episode 的冒烟走不到 step 54 那一行**——这一档跑到了第 2 个单位，绿色不是假绿色。

### P1-1 全部证据产自补齐了缺失集成的自建 harness —— ✅ **已闭环**

协议明写这类问题极易复发，因为修复方往往直接复用那个有问题的脚本。所以我**逐行读了它实际执行的装置**
`$ROL_JFS/port/memoryvla/fix/run_real.py`（≈330 行），逐条核对「只注入，不构造」：

| 它做了什么 | 注入还是构造 | 判定 |
|---|---|---|
| `set_seed(seed)`（`train.py` 全程无 seed 调用） | 注入，且 baseline 与修复后**是同一注入** | 允许 ✅ |
| monkeypatch 两个 sampler 的 `__init__` 只为计时 | 包住已有构造，**没有自己 new** | ✅ |
| 包住 `MemoryVLAMemory.forward` 量进出张量 | 观测 | ✅ |
| 包住 `SimpleTrainer.__init__`，从 `self.dataloader` / `self.model` / `self.optimizer` / `self.batch_processor` 读数 | 读**宿主自己建好的**对象 | ✅ |
| `runpy.run_path("train.py", run_name="__main__")` | 原样跑真实入口，`train.py` 一行未改 | ✅ |
| 构造 sampler / DataLoader / optimizer / model builder | **一个都没有** | ✅ |

**判定：`run_gears.py` 的那条假路径没有被复用。** 交叉核对：本轮 10 个 run JSON 全部在
`fix/runs/2026-08-04/`，由 `fix/gear.sh` 经 `run_real.py` 产生；`tools/run_gears.py` 本轮无产物。

> 一点补充观测（不是缺陷）：`run_real.py` 的 `probing_forward` **每个 forward 都 clone 两个张量**，
> 所以 D 档「峰值显存」含观测装置自身开销。我用不 clone 的装置测同一配置得 9.296 GiB，
> 其记录为 9.302 GiB——量级很小，但下次量显存应把探针关掉再量一次。

---

## 4. S3 — 回归探测（本轮核心）

装置：`review2/r2_probe.py`（我自己写的，同样只注入不构造），额外记录**每个 batch 的原始样本 id**、
provenance、以及「port 包有没有被 import」——这三项修复方的装置都没记。两棵运行树只差 `train.py`：

```
common_base/train.py  sha256:3d94c209...  == git show 18106b05:.../train.py
common_head/train.py  sha256:0087ec1b...  == git show HEAD:.../train.py
diff -rq --no-dereference common_base common_head  →  仅 train.py 一处差异
```

> **为什么不用 git worktree 做基线**：`robo_orchard_lab` 是用 setuptools 的 **meta path finder**
> 装的（`__editable___..._finder.install()`），按模块名解析且**先于 `sys.path`**，`PYTHONPATH` 盖不住它——
> 基线 worktree 照样会 import HEAD 的库，测出来的「等价」是假的。而关闭态下 `18106b05` 与 HEAD 之间
> 唯一有效差异就是 `train.py`（port 包压根不被 import，本轮实测），所以只换 `train.py` 把变量隔离得更干净。

| # | 检查 | 取证 | 结果 |
|---|---|---|---|
| 1 | 关闭态实际走哪条分支 | 真实入口打印被迭代的 sampler | ✅ **`['DistributedBatchFlagSampler']`**。**A 档 config 里 `episode_stream_sampler` 故意保持 ship 值 `true`**——这正是接线最容易掉进去的坑，没掉进去 |
| 2 | **A 档等价性** | 5 个 run、10 组两两比较，见 §4.1 | ✅ **逐样本 id 10/10 组全部一致** |
| 3 | 关闭态资源不变量 | 参数量 / batch key / optimizer 分组 / 峰值显存 | ✅ 参数 `1,136,284,265` 严格相等；batch key **14 vs 14** 逐个相同；optimizer `[11, 626, 43]` 相同；峰值显存 `8.975615978240967 GiB` **逐位相同** |
| 4 | worker 随机流 + 阳性对照 | `num_workers` 4 vs 0 | ✅ 见 §4.1，**对照差 1.028e-01，比噪声地板高 3 个数量级** |
| 5 | 护栏真 fail-fast | 我自己构造违规 config 从真实入口跑 | ✅ **真的 raise**，见 §4.2 |
| 6 | 护栏无副作用 | 关闭态是否执行护栏 / 是否 import port | ✅ **`port_imported: {}`**——关闭态下 `robo_orchard_lab.models.memoryvla` 压根不在 `sys.modules` 里（`train.py` 分支内 import 与 `_build_memoryvla_cfg` 分支内 import 两处都成立）。探针只做 `.detach().clone()`，不碰 RNG |
| 7 | **人为构造退化场景，探针会不会触发** | 我自己写的 `r2_guard_probe_test.py`，9 用例 | ✅ **9/9**，见 §4.3。**探针不是摆设** |
| 8 | **护栏的结构性覆盖缺口** | 真实入口跑 `group` + 宿主 sampler | ❌ **P1-B**，见 §4.4 |
| 9 | 新增/新读 config 键有真实读取者 | 判据 K，12 键全扫 | ✅ 12/12 有读取者；`dataloader_type` 19 个、`episode_stream_sampler` 7 个 |
| 10 | episode sampler 会不会吞掉宿主语义 | `dataset_sample_weights` / per-spec `sample_weight` / `flags` / 第二入口 | ✅ **四条查完全是好的**，见 §4.5 |
| 11 | 全局副作用 | 判据 S，只扫 `*.py`、`--base 18106b05` 显式传 | ✅ **none** |
| 12 | 已移植的其他方法 | `MIGRATIONS.md` 仍只有 memoryvla 一节 | **N/A（首次移植）**，非跳过 |

### 4.1 A 档 —— 精确判据成立；浮点判据被证明分辨不了（两件事都要说）

我跑了 5 个关闭态 run（3 个基线 `train.py` + 2 个 HEAD `train.py`），做全部 10 组两两比较：

```
pair                     同一份 train.py?   loss max        逐样本 id 序列
A_base_1 vs A_base_2         SAME         4.625320e-05     8/8 identical
A_base_1 vs A_base_3         SAME         4.529953e-05     8/8 identical
A_base_2 vs A_base_3         SAME         4.100800e-05     8/8 identical
A_head_1 vs A_head_2         SAME         1.158714e-04     8/8 identical
A_base_1 vs A_head_1        differs       5.912781e-05     8/8 identical
A_base_1 vs A_head_2        differs       7.104874e-05     8/8 identical
A_base_2 vs A_head_1        differs       9.107590e-05     8/8 identical
A_base_2 vs A_head_2        differs       5.960464e-05     8/8 identical
A_base_3 vs A_head_1        differs       6.484985e-05     8/8 identical
A_base_3 vs A_head_2        differs       5.102158e-05     8/8 identical

同代码组: [4.101e-05, 1.159e-04]   跨代码组: [5.102e-05, 9.108e-05]
```

**两件事同时成立，缺一条结论就站不住：**

1. **精确判据全过。** 10 组比较**每一组**逐样本 id 序列 8/8 一致——关闭态拿到的是同样的样本、同样的顺序。
   接线改的就是选 batch 的那段代码，而它唯一能破坏的就是这个。这条是精确的、无噪声的。
   同时 batch key 14 vs 14、参数量严格相等、峰值显存逐位相同、`step 0` loss 恒为 `0.000000e+00`。
2. **浮点判据没有分辨力，不能拿它当通过。** **跨代码组的区间完整地落在同代码组区间之内**，
   而且全部 10 组里最大的那个差（`1.159e-04`）恰恰出现在**共用同一份 `train.py`** 的两个 run 之间。
   也就是说这个量级与「是否共用同一份代码」**不相关**。

**阳性对照**（没有阳性对照的通过 = 未验证）：同一套测量，把 `num_workers` 从 4 改成 0——

```
A_head_1(nw=4) vs A_head_w0(nw=0):   1.028390e-01     ← 比噪声地板高 3 个数量级
   而逐样本 id 序列仍 8/8 一致（sampler 决定索引，worker 只负责取）
```

**所以这套测量确实有牙**：真扰动了 worker 随机流会看到 `1e-01`，而 A 档跨代码组看到的是 `6e-05`。
换句话说，接线**没有**扰动 worker 流——这是有分辨力支撑的结论，不是「差异很小」。

> **修复方对这件事的处理是对的，且是本轮最值得肯定的一处。** 他们主动把 A 档判据改成
> 「step 0 严格 0 + 其余步 ≤ 实测地板」，并明写「上一轮那个 `0` 是 harness 的性质（`lr=0`，权重不动），
> 不是宿主的性质」。我独立复算 `base_run1/2`、`det_run1/2`、`gearA_c1/c2` 六个 JSON，
> `PORT-STATUS.md` 里每个数字都**逐位复现**：地板 `2.899e-04`@step11、c1 峰值 `1.249e-04`、
> c2 峰值 `1.554e-04`、step 0 恒为 `0.000000e+00`。**无一虚报。**

### 4.2 护栏确实是 fail-fast，不是 warn（从真实入口测的）

我自己配违规 config 从 `train.py` 跑：

```
config: enable=true, dataloader_type="stream", episode_stream_sampler=false
  File "train.py", line 236, in main
    assert_episode_stream_wired(config, trainer.dataloader)
  File ".../models/memoryvla/sampler.py", line 231, in assert_episode_stream_wired
    raise RuntimeError
RuntimeError: memoryvla.dataloader_type='stream' and memoryvla.episode_stream_sampler=False
disagree. ... Mismatched, the bank runs but never retrieves anything -- which does not
raise on its own and looks exactly like a healthy run.
```

`rc=1`，训练**一步没跑**就停了。真的 raise，不是 warning，报错文本直接说清了失效形态。
我也自己重跑了修复方的 `guard_unit_test.py`：**10/10 通过**，含「藏在 `BatchSamplerShard`
后面必须放行」这个**不误报**用例。

### 4.3 探针有效性 —— 我人为构造退化场景，它会触发

协议 S3 末行要求「人为构造退化场景确认它会触发，不触发 = 没有探针」。
修复方的证据里探针**只出现过通过方向**（真实 run 量到 1.297 / 1.124，远在容差之上），从没被看见响过。
我补了这一档（`review2/r2_guard_probe_test.py`）：

```
--- 恒等探针 ---
  [ok] 两条流精确恒等                       -> 必须 raise   ✅
  [ok] 两条流差 6e-08（实测退化态的量级）   -> 必须 raise   ✅
  [ok] 感知活(1.0)、认知死                  -> 必须 raise（取 min 不是 max）✅
  [ok] 两条流都活                           -> 必须不 raise ✅
  [ok] 差值恰为容差一半                     -> 必须 raise   ✅
--- batch episode 分布检查 ---
  [ok] stream 模式、4 样本来自 4 条 episode -> 必须 raise   ✅
  [ok] stream 模式、4 样本同 1 条 episode   -> 必须不 raise ✅
  [ok] group 模式、4 条不同 episode         -> 不检查（见 §4.4）✅
  [ok] eval 模式、4 条不同 episode          -> 不检查（推理态合法）✅
9/9
```

**探针是活的**，「取 min 而非 max」那条设计（一条死流不能藏在一条活流后面）也确实生效。

### 4.4 P1-B —— `dataloader_type="group"` 在 HEAD 上没有任何可用配置

`_history_will_be_read()` 决定探针跑不跑，它返回 True 的条件是**「历史已经存在」**。
这个设计有正当理由：episode 首帧 bank 本来就空，恒等旁路是**正确行为**，按字面第一次 forward 触发必然误报。

问题是 **P0-1 的失效形态正是「历史永远不存在」**。实测：

```
_history_will_be_read(4 条互不相同的 episode, 空 bank) = False
```

于是 `group` 模式的两条路都堵死了：

| 配置 | 结果 |
|---|---|
| `dataloader_type="group"` + `episode_stream_sampler=True`（ship 值） | **raise**（`(group=="stream") != True`）——护栏正确拦下 |
| `dataloader_type="group"` + `episode_stream_sampler=False` | **静默恒等**，见下 |

第二种我从真实入口实跑了（`review2/runs/C_group_host.json`，`enable=true`，4 步，rc=0）：

```
实际迭代的 sampler        : ['DistributedBatchFlagSampler']
每 batch 不同 episode 数  : 4
per_mem_bank 每步最大长度 : 1  1  1  1
grad None / 精确零 / 非零 : 64 / 4 / 0        ← 与 P0-1 失效签名逐项相同
参数移动                  : 0 / 68            ← 与 P0-1 失效签名逐项相同
护栏日志行数              : 0                 （wrapper.py:274 与 :326 一行都没有；B 档同位置有 2 行）
loss                      : 4.6106 / 6.6158 / 16.6960   一切正常
```

**三道护栏一道都不响**：`assert_episode_stream_wired` 因 `(group=="stream")==False==bool(False)` 放行；
`_check_episode_stream` 因 `dl_type != "stream"` 提前 return；恒等探针因无历史永不触发。
7,467,264 个参数照样冻结，loss 照样好看。**这是 P0-1 的失效签名在被审 commit 上原样复现。**

**为什么仍判 P1 而不是 P0**（按 `review.md` §严重度：P0 = 数值错误 / 污染宿主原逻辑 /
A 档不过 / 开关关闭时行为改变——本条一条都不占）：ship 配置是 `stream` + `True`，那条路已被焊死，
关闭态可证未变。要落进这个洞得**主动**把 `dataloader_type` 改成 `group` 且把 sampler 开关关掉。

**但有一条加重情节**：`05-ablation-matrix.md` 第 7 行的配方就是「`dataloader_type=group`」。
照它做 → 撞上 raise → **而那条报错信息写着「the episode sampler is only meaningful for `stream`」，
等于指示使用者把 sampler 关掉** → 正好落进无人看守的静默恒等态。
**护栏自己的报错文案把使用者引导进了唯一没有护栏的那个格子。**

**修法（不由我实施）**：把判据从「这一次 forward 是不是恒等」改成
「跑满 K 步之后，有没有任何一条 episode 的 bank 长度超过 1」。这个形式不需要历史先存在，
`group`+宿主 sampler 与 `stream`+宿主 sampler 两种退化都能抓，且不会在 episode 首帧误报。

### 4.5 四条「可能被吞掉的宿主语义」—— 全部查完，全部是好的

记下来是因为 `07` §5 的教训：harness 与生产的每处差异都要查，**事先分不清哪条藏着 bug**。

| 疑点 | 查法 | 结论 |
|---|---|---|
| episode sampler 不接 `dataset_sample_weights`，权重被静默丢掉 | 护栏查的 `config["dataset_sample_weights"]` 与 `train.py` 传给宿主 sampler 的**是同一个键**，且它由 `dataset_factory._finalize_dataset_sample_weights` 在 `build_dataset` 时填好、早于 sampler 分支 | ✅ **护栏位置正确**，会 raise |
| 现行 per-spec `sample_weight` 绕过那个键 | `dataset_specs_memoryvla_robodojo_memory.py` 无 `sample_weight` → 空 dict → 该键不被设置 | ✅ 当前 config 无此风险 |
| episode sampler 忽略 `flags`，batch 可能混数据集 | `dataset_wrapper.py:45` `flags.append(np.full(len(dataset), flag))` → flag **按子数据集恒定**；一条 episode ⊆ 一个子数据集；全仓无其他 `.flags` 消费者 | ✅ 语义保持 |
| 还有没有第二个训练入口也硬编码宿主 sampler | `grep -rn "DistributedBatchFlagSampler("` → `projects/holobrain/train.py:101` 确实有一个 | ✅ **不是第二个死开关**：`projects/holobrain/` 有自己的 config 树，全树 0 处 `memoryvla`，也不 import `config_holobrain_common` |

---

## 5. S4 — 继承基线三分类复核

> ⚠️ **`06-review-report.md` 没有协议 §0 要求的「哪些结论失效 / 仍有效 / 需重验」分类节**
> （它有 §10 最短修复路径和 §4 对照表，但那不是分类）。
> 因此**下面的分类由本轮自行推导，置信度低于原审查者给出的分类**。

### 5.1 失效（须重新测量；沿用旧值即报告失真）

| 项 | 处理 | 判定 |
|---|---|---|
| D 档峰值显存 / 墙钟 | 已重测，旧值明确标注作废；墙钟改为「只记录不解释」（两次同配置 baseline 差 22%） | ✅ **本轮重验** |
| B / E 档 grad、参数移动、bank 长度 | 已从真实入口重测 | ✅ **本轮重验** |
| A 档「0.000e+00」 | 已撤回，换成两档判据 | ✅ **本轮重验** |
| **`05-ablation-matrix.md` 全表（7 行）** | **未重测、未标注** | ❌ **P1-A** |
| **`06-verification.md` 五档全文** | **未标注** | ❌ **P1-A** |

**P1-A 详述**：`08-review-response.md` 自己承认「A/B/D 档**与全部 5 个消融**实际跑的是
`--sampler sequential`，一个仓库里根本不存在的手写连续索引列表」，`PORT-STATUS.md` 也写了
「旧的五档数字全部作废」。但是：

```
$ git diff --name-only 18106b05..f6dfd1e8 -- docs_analysis/memoryvla/05-ablation-matrix.md \
                                             docs_analysis/memoryvla/06-verification.md
(空 —— 两个文件都没被碰)
$ grep -n "作废\|失效\|订正\|2026-08-04" 05-ablation-matrix.md 06-verification.md
(空 —— 零标注)
```

而 `06-verification.md:5-8` 至今写着：

> harness 刻意**不走 `train.py`**……**所以数值描述的就是真正会训练的那个东西**。

**这正是 P1-1 推翻的那一句**，它还留在承载那些数字的文件里。`05-ablation-matrix.md` 更彻底——
它从头到尾没被任何地方点名作废，而其核心结论「每一行 vs base 都 > 0，说明每个开关都真的生效了，
没有哪个是摆设」完全建立在那条假路径上。附带一层：新护栏使 `mode=group` 那一行
**不再能只切一个键**，而该文件开头明写「每一行都只靠 config 切换」。

命中协议 S4 的「沿用旧值 → 报告失真，后续所有基于该数值的估算都建在假数上」。
**判 P1 而非 P0**：入口文档 `PORT-STATUS.md` 已把五档作废写在最显眼处，从入口进的读者不会被骗；
被骗的是直接翻到 `05` / `06-verification` 的人。

### 5.2 需重验（全部重跑）

| 项 | 结果 | 判定 |
|---|---|---|
| A 档关闭态等价 | §4.1，10 组两两比较逐样本 id 全一致 | ✅ **本轮重验** |
| C 档定输入数值对齐 | 我在 HEAD 上重跑 `check_reference.py`：**10 targets / 10 bit-exact / 0 failed** | ✅ **本轮重验** |
| batch key 集合 | 关闭态 14 vs 14 逐个相同；开启态 15（多 `step_index`，按设计） | ✅ **本轮重验** |
| 参数总量 | 关闭 `1,136,284,265` 严格相等；开启 `1,143,751,529`（+7,467,264 / +0.657%） | ✅ **本轮重验** |
| optimizer 分组 | 关闭 `[11, 626, 43]`；开启 68 张量全进 group 1，游离 0 | ✅ **本轮重验** |
| 关闭态峰值显存 | `8.975615978240967 GiB` 逐位相同 | ✅ **本轮重验** |

> C 档一点边界：它比的是 `CogMemBank` / `GateFusion` / `TimestepEmbedder` 这些**模块级**目标，
> **不覆盖** `MemoryVLAMemory.forward`——也就是本轮插探针的那一段。C 档通过是必要条件不是充分条件；
> 覆盖那段的是 B / E 档。

### 5.3 仍有效 → 抽验 5 条（挑与改动语义相邻的，非随机）

| # | 抽验项 | 为什么挑它 | 结果 |
|---|---|---|---|
| 1 | 拷贝保真度 F | `sampler.py` +94、`wrapper.py` +118，`[port:]` 标记区间可能错位 | ✅ **抽验通过**：6 标记，4 个 ratio ≥ 0.998，2 个 DRIFT 均为**已声明改写**（与上一轮同两条豁免） |
| 2 | 「纯增量、零无关改动」 | 两个文件本轮都长了 | ⚠️ **部分不成立**：`wrapper.py` +118/−0 纯增量 ✅；`sampler.py` +94/−1、`train.py` +38/−6。**无格式化 / 无 import 重排 / 无重命名 / 无顺手重构** ✅，但「0 删除」这个说法本身已不成立 → P2-A |
| 3 | `uuid` / `step_index` 接口语义 | 护栏现在每个训练 batch 都读 `episode_ids` | ✅ **抽验通过**：`_check_episode_stream` 收的是**已经提取好的** ids，没引入任何新 batch key 需求；关闭态 batch key 仍 14 个 |
| 4 | `_build_memoryvla_cfg` 键转发完整性（判据 D） | 新增了 `dataloader_type` 的第二个消费点 | ✅ **抽验通过**：12 行计划文档默认值与 ship 值**逐行一致**；该函数转发 11 键，唯一不转发的 `episode_stream_sampler` 由 `train.py` 消费——按设计 |
| 5 | mask 极性 | 上一轮点名的「最高频静默错误」，本轮**未触及**，作为「抽验没有系统性偏向」的对照 | **（继承自上一轮，本轮未重验）**；`structure.py` 不在本轮 diff 内，`git diff --name-status` 可证未被触碰 |

**5 条里 4 条抽验通过、1 条部分不成立且已单列为 P2-A。** 未触发「整类降级为需重验」。

### 5.4 明确标注为继承、本轮未重验的结论

以下全部标 **（继承自上一轮，本轮未重验）**，本轮既没复核也没推翻：
方法要素 12/15 与 A 逐行一致 · 接口语义 32 项一致 0 项不一致 · cite 零幻觉 ·
上一轮四个宿主文件的 L1 判定 · ckpt 兼容性 1000→1068 · `BottleneckSE` 不接入的理由成立 ·
mask 极性正确。

---

## 6. S5 — 新增风险是否如实记录

核对对象是 **`PORT-STATUS.md`**（不是 `08-review-response.md`）。

| 协议要求的类别 | 记了没有 | 位置 |
|---|---|---|
| **① 训练动力学变化** | ✅ 记了，写法准确 | 遗留 5：「每 batch 从 4 个 episode 变成 **1 个**……**A 档证明的是「关闭态没变」，不是「开启态的训练行为已被验证」**」 |
| **② 本轮之后才可能暴露** | ✅ 记了，**比上一轮更具体** | 遗留 6：各 rank `spans[rank::num_replicas]`、episode 长度中位 276→1203 帧 ⇒ `__len__` 不齐（**上一轮没记这条**）；遗留 7：v9/v10 size mismatch |
| **③ 本轮仍无法验证的项原样保留** | ⚠️ **六条承接了五条** | 见下 |

`06` §9 六条 → `PORT-STATUS.md`：外部 ckpt ✅(遗留7) · DDP ✅(已知3+遗留3+遗留6) ·
**A 的采样频率/降采样 ❌ 未承接** · 端到端可比性 ✅(已知1) · D 档墙钟 ✅(验证结果段) · fifo vs tome ✅(遗留2)。
掉的那条上一轮自评「影响中等偏低」，故判 **P3-A** 而非 P1。

另外新增了一条上一轮和契约都没要求的（遗留 8 `ulimit -n` 必须提到 65536），并且诚实地写了
「这条**不是本轮新发现**，`06-verification.md:3` 就写着，本轮照样折了 5 次，因为它**只写在文档里、
没写进任何会被执行的东西**」——已写进 `fix/gear.sh`。**这个自我诊断与 P0-1 是同一个形状，记得很到位。**

**`MIGRATIONS.md` 两条教训是否写成方法无关判据**：✅ 是。
第 7 条「config 键必须有真实读取者——写进文档、导出类、给了默认值，都不算接上」落到了可执行判据
（「每加一个 config 键，落地前 grep 一次它的读取者；只命中定义+表格+注释就是没接」）；
第 8 条「验证装置不许自建宿主装配」写出了可复用规则（「harness 只允许注入输入、抓取张量；
sampler / dataloader / optimizer / model builder 一个都不许自己 new」），
并附了「护栏要查 `accelerator.prepare()` **之后**那个 dataloader」这条具体判据。
**不是「某某 key 忘了接」式的记法。**

---

## 7. S6 — 机械判据全量回放 + 协议反馈

### 7.1 工具版本（不钉住 = 比较被污染）

`08` 记录工具在修复会话期间从 `$ROL_JFS/port/_shared/` 搬到 `~/storage_policy/tools/port/`，
且新版**刻意不等价**（`--config`/`--subdir` 必须显式传，扫空退出 2 而不是 0）。我全程只用新版：

| 文件 | md5 | mtime |
|---|---|---|
| `orphan_switch_check.py` | `7971d335083212f9bec576c387c52005` | 2026-08-04T08:36:51 |
| `copy_fidelity_check.py` | `13f5ffdd6280fa9e6d0c467502dc61a9` | 2026-08-04T08:34:54 |
| `port_probe.py` | `de4cfc37d06c80d37314337ff8a4e350` | 2026-08-04T08:34:54 |
| `preflight.sh` | `8ad881d7f6ac955d79d4bd37f33f718f` | 2026-08-04T08:44:26 |

### 7.2 逐条结果

```
bash ~/storage_policy/tools/port/preflight.sh --method memoryvla --base 18106b05 \
  --source-repo ~/git_repo/MemoryVLA \
  --config projects/holobrain_internal/common/configs/config_holobrain_common.py \
  --subdir robo_orchard_lab/models/memoryvla --static
```

| 判据 | 结果 | 是否误报 |
|---|---|---|
| **K** 孤儿配置键 | **12/12 键有读取者**（`episode_stream_sampler` 7 个，e.g. `train.py:119`） | 无 |
| **C** 无人构造的类 | 8 类，7 ok（`MemoryVLAEpisodeStreamBatchSampler` → `train.py:131`），1 UNUSED = `BottleneckSE` | 无（`BottleneckSE` 是**已声明**的不接入，需 `--waive-class`） |
| **D** 文档默认值漂移 | **12/12 行一致** | 无 |
| **F** 拷贝保真度 | 6 标记：4 个 ratio ≥ 0.998，2 个 DRIFT（`L105-L136` 0.902、`L335-L357` 0.105）均为**已声明改写** | 无（需 `--waive-copy`） |
| **S** 全局副作用 | **none** | 无 |
| **I** 恒等探针 | 开启态 `1.296956e+00` / `1.123835e+00`（护栏自打日志）；退化场景 5/5 会 raise | 无，**两个方向都验了** |
| **G** 梯度三态 | `0 None / 0 零 / 68 非零`（失效态为 `64/4/0`） | 无 |
| **P** 参数位移 | `62 → 63 / 68`（失效态为 `0/68`），非零 lr | 无 |
| **O** optimizer 覆盖 | `trainable_not_in_optimizer = 0`；68 张量全进 group 1 | 无 |
| **B** 关闭态 batch key 集合 | 14 vs 14 逐个相同 | 无 |
| **W** worker 随机流 + 阳性对照 | 跨代码 `6e-05`（噪声内）；`nw=4 vs 0` 对照 **`1.028e-01`**，高 3 个数量级 | 无，**判据有牙** |
| **X** harness/生产构造差异 | 未实现（`07` 标「推测」）。本轮以**人工逐行读 `run_real.py`** 代替（§3 P1-1） | — |

**带上与 `08` 相同的两组豁免**：

```
  PASS  K/C/D      PASS  F      PASS  S      preflight PASSED     EXIT=0
```

**阳性对照**：同一份脚本、同一组参数跑在 `git archive 18106b05` 的树上 →
`ORPHAN episode_stream_sampler` + `UNUSED MemoryVLAEpisodeStreamBatchSampler` +
`UNUSED BottleneckSE` + `DRIFT plan='False' shipped=True`，**4 findings，EXIT=1**。
所以 HEAD 上的绿是判据活着的绿。

### 7.3 协议反馈（`review-incremental` 首次实战 —— **未修改协议文件**）

1. **输出文件名撞号。** 协议规定输出 `08-incremental-review.md`，但修复方的逐条应答已经占了
   `08-review-response.md`。本轮改用 `09-`。建议协议写成「紧接现有最大编号」而不是写死 `08`。
2. **§0「继承基线的前置条件」在本轮直接落空。** 协议要求上一轮报告含「失效 / 仍有效 / 需重验」分类节，
   而 `06-review-report.md` **没有这一节**。协议给的兜底（自己推导 + 标低置信度）可执行，
   但这说明**上一轮的协议没有要求产出这一节**——两份协议之间有接缝。
   建议全量审查协议的输出清单里补上它，否则每次增量复审都要自己重推一遍。
3. **S3「关闭态等价性：判据沿用上一轮的确定性结论」这句在本轮是错的前提。**
   上一轮「地板恰为 0」是 harness 性质（`lr=0`，权重不动），换到真实入口就不成立。
   照字面套会得到一个不适用的严格判据。建议改成「**每轮自测地板，并要求报告额外给出至少一条
   不受浮点噪声影响的精确判据**」——本轮的「逐样本 id 序列」就是这样一条，比 loss 比对有力得多，成本还更低。
4. **S3「之前已移植的其他方法：逐个只开那一个」在首次移植时无对象**，协议没写 N/A 该怎么记。
   建议明确要求「N/A 也要留痕并给判据」——否则和「跳过了」在报告上长得一样。
5. **S6「机械判据可回溯套用」这条非常划算**（5 秒、零误报、可带阳性对照），建议保留，
   并加一句：**回放时必须钉住工具版本**。本轮工具在两轮之间搬过家且刻意不等价，
   不钉版本的话「变绿」可能是工具变松而不是代码变好。
6. **S2 要求「读改动方实际执行的脚本」，但没说要不要连它的观测器一起审。**
   本轮 `run_real.py` 的 `probing_forward` 每个 forward clone 两个张量，直接抬高了 D 档显存读数。
   建议补一句：**观测装置本身会不会污染被观测量，要单列一条**。

---

## 8. 无法验证清单

**上一轮 `06` §9 六条，原样承接**（不因为主要问题修好了而消失）：

| 项 | 为什么仍验不了 | 本轮有无变化 |
|---|---|---|
| **外部真实 ckpt 加载** | bucket 只有 v9，config 是 v10，`vlm.*` 全线 size mismatch；v10 warm-start 在 http URL 后而本机无外网 | 无变化。本轮全部 run 一律 `checkpoint=null` |
| **DDP / 多卡 unused-parameter** | 本机任意两卡 gather 必崩 `ILLEGAL_ADDRESS` | **风险变具体了**：接上 sampler 后各 rank `spans[rank::num_replicas]`、episode 长度中位 276→1203 帧 ⇒ `__len__` 不齐、收尾不齐。**修完 P0-1 才真正暴露，必须在多卡机器上单独验** |
| **A 的采样频率 / 降采样** | 定义端在 A 的 RLDS 管线外部 | 无变化，**且已从 `PORT-STATUS.md` 风险清单里掉了** → P3-A |
| **A 与宿主端到端数值可比性** | 原理上不可比（A 记 LLM 之前的 patch，宿主记 VLM 之后已被语言条件化的特征） | 无变化 |
| **D 档墙钟时间** | 卡共享；两次同配置 baseline 差 22% | 无变化；修复方已明确「墙钟只记录不解释」 |
| **`fifo` vs `tome` 的实际差异** | 需要跑到 episode 尺度 | 无变化 |

**本轮新增的无法验证项：**

| 项 | 为什么验不了 |
|---|---|
| **开启态的训练行为本身** | A 档只证明「关闭态没变」。开启态每 batch 从 4 条 episode 变成 1 条，梯度方差、epoch 内样本相关性、归一化层统计全都不同。**这需要跑到收敛去比指标，不是一次审查能答的**（修复方已记为遗留 5） |
| **关闭态在浮点层面的严格等价** | 真实入口不逐位可复现，噪声完全淹没这次改动的影响：10 组两两比较里最大差出现在**共用同一份 `train.py`** 的两个 run 之间。只能用精确判据（样本 id、key 集合、参数量、显存、sampler 类型）间接证明。实测 `use_deterministic_algorithms(warn_only=True)` 也压不到 0 |
| **`_episode_spans` 在其他数据集上的正确性** | 它依赖 `_get_indices`，且假设一条 episode 的帧在全局索引里连续。只在 RoboDojo Memory 六任务上验过，换数据集是否仍成立未验 |
| **长时训练稳定性** | 本轮最长 60 step。`tome` 巩固与 `clear_episode` 在 epoch 尺度上的内存行为未观测 |
| **`dataloader_type="group"` 是否还有意义** | 见 P1-B：两种设法一种 raise、一种静默恒等。**「group 模式本来应该是什么样」需要一个能产出 episode 有序批又不与护栏冲突的配置，目前不存在**，因此无法测量它本该有的行为 |

---

## 9. 最短修复路径（按「修完能翻案」排序）

1. **P1-A（10 分钟，纯文档）** — 在 `05-ablation-matrix.md` 与 `06-verification.md` 顶部各加醒目作废标注：
   「本文数值产自 `run_gears.py --sampler sequential`，那是宿主到不了的装配（见 P1-1），
   **不可用于任何结论**；现行数值见 `PORT-STATUS.md` 与 `08`」。
   `06-verification.md:5-8` 那句「所以数值描述的就是真正会训练的那个东西」应直接划掉。
   **消融矩阵要么重跑、要么明写「未重跑」——两者都行，留着不标不行。**
2. **P2-A（1 分钟）** — `PORT-STATUS.md:30` 与 `MIGRATIONS.md:44` 的「0 删除」改成实际值
   （`train.py` +38/−6、`sampler.py` +94/−1），并说明那 6 行是**代码位移**而非逻辑改动
   （可引用本报告 §4.1 的精确判据）。
3. **P3-A（1 分钟）** — 把 `06` §9 第 3 条「A 的采样频率 / 降采样」补回 `PORT-STATUS.md` 风险清单。
4. **P1-B（半天，代码）** — 把恒等探针判据从「这一次 forward 是不是恒等」换成
   「跑满 K 步后，有没有任何一条 episode 的 bank 长度超过 1」；同时把
   `assert_episode_stream_wired` 的报错文案改掉，**不要再指示使用者关掉 sampler**，
   而是明确「`group` 模式当前没有可用配置」。
5. **P3-B（下次移植时）** — 让 runner 把 `git rev-parse HEAD` 与被跑文件哈希写进结果 JSON。
   本轮 `fix/runs/` 里 10 个 JSON 只能靠 mtime 与 commit 时间反推是哪个 commit 跑的。
6. **P3-C（1 分钟）** — `run_real.py` docstring 里「These knobs pin that down so gear A can use a
   strict bar」与其自身实测矛盾（`det_run1` vs `det_run2` = `1.564e-04`，压不到 0；且 A 档最终没用这两个开关），
   删掉或改写。

**修完 1、2、3 即可翻 ACCEPT。** 第 4 条（P1-B）属于「这次没做到契约声称的覆盖面」，
可以记为遗留另起一轮，但**不要在没修的情况下把它从风险清单里划掉**。

---

## 附：本轮证据清单

`$ROL_JFS/port/memoryvla/review2/`（不进 git）

| 文件 | 是什么 |
|---|---|
| `r2_probe.py` | 我自己的「只注入不构造」观测器；额外记原始样本 id、provenance、port import 状态 |
| `r2_analyze.py` | 精确判据 + 浮点判据双轨比较器 |
| `r2_guard_probe_test.py` | 恒等探针与 episode 分布检查的**退化场景**测试，9 用例 |
| `r2_setup.sh` / `r2_gear.sh` | 建两棵只差 `train.py` 的运行树；单档 runner（含 `ulimit -n 65536`） |
| `common_base/` `common_head/` | 两棵运行树，`diff -rq` 证明只差 `train.py` |
| `tree_18106b05_static/` | `git archive 18106b05`，用于机械判据的阳性对照 |
| `runs/A_base_{1,2,3}.json` | 关闭态 × 基线 `train.py` × 3 |
| `runs/A_head_{1,2}.json` | 关闭态 × HEAD `train.py` × 2 |
| `runs/A_head_w0.json` | 关闭态 × HEAD × `num_workers=0`，W 判据阳性对照 |
| `runs/B_head.json` | 开启态 8 步 bs=4 |
| `runs/E_head.json` | 开启态 60 步 bs=8，跨 episode 边界 |
| `runs/G_mismatch.json` | 护栏阴性用例，真实入口 raise（rc=1） |
| `runs/C_group_host.json` | **P1-B 的证据**：`group` + 宿主 sampler，静默恒等，rc=0 |
