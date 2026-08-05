# 12 — 第四轮修复：对 `11-incremental-review_v3.md` 的逐条应答

| 项 | 值 |
|---|---|
| 被审基线 | `49b2178c`（`port/memoryvla`，第三轮 tip） |
| 上一轮报告 | `11-incremental-review_v3.md` @ `review3/memoryvla` `d311b000`（🟡 ACCEPT-WITH-FIXES，P0×0 · **P1×1** · P2×3 · P3×4） |
| 本轮提交 | `308730dc` 文档订正（纯 `.md`）· `f770afe0` P2-B 类文档字符串复位（零行为）· `fc33a5db` P1-C 护栏补齐 + 测试进仓 · **本文件所在的第四个提交** 实测回填（纯 `.md`；它的哈希不写在这里 —— 写进去就得再改一次文件，哈希又会变） |
| 本轮证据 | `$ROL_JFS/port/memoryvla/fix4/`（**不进 git**；`fix3/` `review/` `review2/` `review3/` 全程只读未改） |
| 日期 | 2026-08-05（`date +%Y-%m-%d`） |
| 算力 | 单卡 × 21 档（4 档基线 + 17 档验证），GPU 1/2/3/4/5，`ulimit -n 65536` |
| 自评 | **不自评。** 产出是修复 + 数值证据，裁决交给下一轮独立复审。 |

---

## 0. 关于提交数：契约要三个，实交四个 —— 理由与代价

上一轮的 **P3-G** 是「证据的 `git_head` 都是前一个 commit、`git_dirty=True`，只能靠文件哈希绑定」。
要真正闭掉它，只有一条路：**先提交代码，再跑验证**，让每份证据都是
`git_dirty=False` 且 `git_head == 被审 commit`。

但契约同时要求 `PORT-STATUS.md` 回填**本轮实测**的支持矩阵 —— 那些数字只有跑完才有。
两者不能同时塞进第三个 commit：

- 若把文档也放进 `fc33a5db`，就得先跑验证再提交 ⇒ 全部证据长在脏树上，**P3-G 原样复发**；
- 若先提交再 `--amend` 塞文档 ⇒ 提交哈希改变，**所有证据记录的 `git_head` 指向一个不再存在的 commit**，比不绑定更难审。

所以拆成第四个 commit，且它是**纯 `.md`**，与第一个 commit 同形。
**代价如实说明**：契约写的是三个提交，实交四个，这是一处**主动的、有理由的偏离**，
不是漏做。`git diff --stat fc33a5db..HEAD` 可证第四个提交不含任何代码。

---

## 1. 逐条应答

### P1-C —— `group` + `batch_size=1` 配置可达的静默退化 · ✅ **已修**

**修法**：把「能在构造期静态判定的前提」从后果判据里拿出来，放回构造期。

`group` 在 `process_batch` 顶部 `bank.clear()`（`memory_bank.py:361`），
批内每 `group_size` 个样本再 `clear_episode(episode_ids[i-group_size])`（`:374-377`，
配 episode sampler 时那就是**同一条** episode）⇒ **记忆跨度恒等于 `min(group_size, batch_size)`**，
等于 1 时没有任何样本有前驱可读。这是两个配置值加一个 batch size 的算术，**不需要任何 forward**。

| 位置 | 改了什么 |
|---|---|
| `sampler.py::assert_episode_stream_wired` | 链判据通过后追加**装配期跨度判据**：`group` ∧ `min(group_size, batch_size) <= 1` → raise。有效 batch size 从**链上的 sampler 对象**读（`prepare()` 会重新包一层，对象才是真的），依次退回链上任一 `batch_size` → `config['batch_size']`；**全部读不到就 `logger.warning` 明说这道判据被跳过**——一道跑不了又不出声的判据，与跑了并放行长得一模一样，那正是 P0-1 的形状 |
| `wrapper.py::_check_episode_stream` | 同一个问题在**第一次训练 forward** 再问一次，取真实到达的 batch。装配期护栏只在 `train.py:236` 一处被调用，而 `09` §4.5 记了存在第二训练入口。**后果判据是兜底，不是唯一**；反过来，静态前提也不该只靠运行期兜底 |
| `wrapper.py::_check_bank_liveness` | 文案不再断言判据分辨不了的成因：先报观测值，再列**两种成因与各自的动作** |

**顺带覆盖了一个 `11` 未点名、同样配置可达的格子**：`group_size == 1`。
它与 `batch_size == 1` 是**同一个失效**（bank 每个样本被清一次），与 batch 多大无关，
落在同一条谓词里，不是第二个特例。

**顺带修了一个误报**：`mem_length == 1` 时巩固每写一条就把 bank 压回 1 条，
**bank 长度上界就是 1**，而那一条是真历史、检索照常发生（`memory_bank.py:330-338` + `:386`）。
原判据会在第 8 次 forward **无故 raise 一个正常工作的 run**。现在改为一次性 `WARNING`
「这条判据在此配置下失明，站下不裁决」。**误报是对「触发时必须指向真实原因」最彻底的违反**，
所以一并修。

**K=8 不动，但角色变了**：所有**配置可达**的 group 退化被前两道提前到
「第一次 forward 之前或当时」，K 从唯一防线降级为**数据形状类失效**的兜底
（批次本该 episode 连续而实际不是）。取值依据与残留洞见 `PORT-STATUS.md`。

**证据**：见 §2。

---

### P2-A′ —— 侵入度数字第三轮带着过期值发布 · ✅ **已修**（commit `308730dc`）

标题原写 `sampler.py +94/−1`、订正块写 `wrapper.py +118/−0`，那是 `18106b05..f6dfd1e8` 的量。
在它们所在的 `49b2178c` 上实测：

```
$ git diff --numstat 18106b05..49b2178c -- robo_orchard_lab/models/memoryvla \
      projects/holobrain_internal/common/train.py
38   6   projects/holobrain_internal/common/train.py
2    0   robo_orchard_lab/models/memoryvla/__init__.py
106  5   robo_orchard_lab/models/memoryvla/sampler.py
204  0   robo_orchard_lab/models/memoryvla/wrapper.py

$ git diff --numstat --diff-filter=M 3ce31c0c..49b2178c -- "*.py"   →   5 个文件
```

`train.py +38/−6` 与「5 个宿主文件」两项**本来就是对的，未改动**。

**根因不是粗心，是标题那一行没有基点限定** —— 一个不带基点的数字无法自证过期。
现行口径改成「基点 + 截至 commit + 逐文件表」，并把重跑命令写进文档旁边；
脚本形式在 `fix4/intrusion_line.sh`（**如实说明它不是闸门**，只是把「怎么算出来的」钉死）。

---

### P2-B —— `MemoryVLAMemory.__doc__ is None` · ✅ **已修**（commit `f770afe0`，零行为改动）

三行 `#:` 注释 + `BANK_LIVENESS_FORWARDS = 8` 从类语句与文档字符串**之间**移到文档字符串**之后**。

```
python -c "from robo_orchard_lab.models.memoryvla.wrapper import MemoryVLAMemory as M; \
           assert M.__doc__ is not None and 'Args' in M.__doc__; print('doc OK')"
→ loaded from : .../robo_orchard_lab/models/memoryvla/wrapper.py
  __doc__ is None: False      has Args: True      K: 8      doc OK
```

**并从真实入口取证**（每个开启态 run 都记了 `module_doc`）：`is_none=False`。
`git show --stat f770afe0` 只有 `wrapper.py`，`+12/−6`，无夹带。

---

### P2-C —— 「有断言禁止那几句话回来」但断言不在 git 里 · ✅ **已修**（移进仓库）

选了「移进仓库」这一支。新增 `tests/test_robo_orchard_lab/models/memoryvla/`：
`__init__.py` · `test_sampler_guard.py` · `test_wrapper_guards.py`，
落在 `tests/Makefile: test_ut` 的目标树内（`pytest -c tests/pytest.ini tests/test_robo_orchard_lab`）。

- 自 `fix3/guard3_unit_test.py`（22 例）与 `guard3_probe_test.py`（24 例）改写；
  原脚本用 `importlib.util.spec_from_file_location` + 「从仓根跑」的相对路径，进仓后不成立，改成包内 import。
- **新增用例**：装配期跨度四死格 / 六活格 · 穿 accelerate shard 读 batch size ·
  退回 `config['batch_size']` · 读不到时必须 `WARNING` 不得静默 ·
  `mem_length=1` 不误报且要说自己站下了 · `stream` + bs=1 合法不误报 ·
  **文案卫生断言扩展**（见下）。
- **合计 84 项，全过。**

**文案卫生断言扩展**（P1-B 与 P1-C 的成因都是文案，而文案对 ruff / autoapi 完全隐形）：

| 断言 | 防的是什么 |
|---|---|
| 4 条 FORBIDDEN 短语在任何 raise 文本中都不得出现 | P1-B：把人指引进无护栏格子 |
| 「sampler 不对」类报错必须点名 `episode_stream_sampler=True` | 正解不能丢 |
| **跨度类报错不得出现 `the fix is memoryvla.episode_stream_sampler`，且必须写明「episode sampler is NOT the problem」** | P1-C：**建议一个已经生效的开关** |
| **跨度类报错必须给出两个在该场景下真的有效的动作** | 建议要能改变现状 |
| **看门狗文案不得出现 `The batches reaching this module are`（断言句），必须含 `Two different things produce this` / `cannot tell them apart` / `(a)` / `(b)`** | P1-C：**断言一个判据分辨不了的成因** |

**执行证据与它的边界**：本机 `holobrain_internal` **没有 pytest**，装它会破 E0
「宿主主环境零改动」，所以执行用 `.git/run_tests_nopytest.py`（会注入 pytest stub，
支持 `parametrize` / `raises` / `fixture`，**无法解析的一律报 SKIP 不报 pass**）：

```
===== 合计 84：PASS 84 / FAIL 0 / SKIP 0
```

> ⚠️ **必须说明的边界**：上面是 shim 的结果，**不是 CI 里 pytest 的结果**。
> 「CI 会不会真的收集到这两个文件」本轮**无法验证**，已记入无法验证清单。
> 结构性论证只到这一步：文件落在 `tests/test_robo_orchard_lab/` 下，
> 而 `tests/Makefile` 的 `test_ut` 目标就是 `pytest ... tests/test_robo_orchard_lab`。

`ruff check --config=pyproject.toml tests/test_robo_orchard_lab/models/memoryvla/` → `All checks passed!`

---

### P3-E —— 工作区未提交的纯改名 · ✅ **已修**

`10-review-response.md` 恢复（`11` 全篇引用这个名字），`10-review-response_v2.md`
**未用 `rm`**（本机 hook 会 deny 且返回空输出，看起来像远端没反应），
`mv` 到 `fix4/attic/` 留痕 —— 两者逐字节相同（`diff` 无输出，md5 `83ecef02c5999853edd435bd13e9e86f`），不丢信息。
终态 `git status --porcelain` 为空。

### P3-F —— 两个近似重复的 `### 失效` 小节 · ✅ **已修**

**合并为并集，一行未丢**：第一份独有「§4.1/§5.2 峰值显存作为判据」，
第二份独有「§7.2 判据 C 豁免列表」⇒ 合并后 5 行。**删掉任一份都会丢一行**，所以不是删重复。

### P3-G —— provenance 中途才生效，`head_B_stream_on` 无代码绑定 · ✅ **已修（机制）**，旧产物不回填

- **顺序改了**：先提交、再跑验证。本轮**全部 21 份证据**的
  `git_head` 都是被审 commit、`git_dirty=False`（见 §5 的核对表）。这是同一性，不是推断。
- `run_real4.py` 的 `provenance.port_files` **按磁盘读**（不按 `sys.modules`），
  所以关闭态 run 也有完整的 4 个 `.py` 哈希。
  （附带更正一处：`11` §3.7 推测「关闭态不 import port 所以 port_files 为空」——
  实际上 `run_real3.py` 的这段代码本来就是磁盘读，空是因为那几个 run 跑在这段代码存在**之前**。
  结论方向不变，成因不同。）
- **旧产物不回填**：`fix3/` 是上一轮的证据，只读不动。

### P3-H —— `09` §8 五条新增无法验证项掉了两条 · ✅ **已修，并把承接动作清单化**

补回 `PORT-STATUS.md` 遗留 12（`_episode_spans` 换数据集）与遗留 13（长时训练稳定性）。
**更重要的是改了做法**：P3-A → P3-H 是同一形状的复发，说明缺的不是细心，是
**承接动作没有清单化**（上一轮是「重写一遍风险清单」，重写就会漏）。
本轮起改成**逐条打勾**，表在 §4。

---

## 4. `11` §8 无法验证清单 —— 逐条打勾

**判据：每一条都要能指到 `PORT-STATUS.md` 的具体位置，指不到就是掉了。**

> ### ⚠️ 订正（2026-08-05，第五轮，复审 P1-D）：下表 16/16 全勾，但有三条勾错了
>
> 第 14 / 15 / 16 行的「承接到」写的是「无法验证清单」，而那张清单**在本文件里**
> （§5），不在 `PORT-STATUS.md` 里。**单轮应答文档下一轮就会被 `14-...` 取代 ⇒ 等于没承接。**
> 这是 P3-A → P3-H 的第三次同形复发，而且发生在**专门为了防它而造的这张表里**。
>
> **判据加严为：承接目标只能是 `PORT-STATUS.md` 的「遗留问题」或「已知问题」小节，
> 且打勾时必须写出小节号。** 指向任何 review-response、review 报告或证据目录一律判未承接。
>
> 三条已按新判据补进 `PORT-STATUS.md`：
> lint 门与 CI 一致性 → **遗留 15** · `dataset_sample_weights` 真实入口 → **遗留 16** ·
> CI 的 pytest 收集 → **遗留 17**。下表相应几行的「承接到」以这里为准。

| # | `11` §8 的条目 | 承接到 | ✓ |
|---|---|---|---|
| 1 | 外部真实 ckpt 加载 | 遗留 7 | ✅ |
| 2 | DDP / 多卡 unused-parameter | 已知问题 3 + 遗留 3 / 6 | ✅ |
| 3 | A 的采样频率 / 降采样 | 遗留 9 | ✅ |
| 4 | A 与宿主端到端数值可比性 | 已知问题 1 | ✅ |
| 5 | D 档墙钟时间 | 「D 档：墙钟不用来下结论」 | ✅ |
| 6 | `fifo` vs `tome` | 遗留 2 | ✅ |
| 7 | 开启态的训练行为本身 | 遗留 5 + 遗留 10 | ✅ |
| 8 | 关闭态在浮点层面的严格等价 | 「关闭态等价性」+「峰值显存也不是精确量」 | ✅ |
| 9 | **`_episode_spans` 在其他数据集上的正确性** | **遗留 12（本轮补回）** | ✅ |
| 10 | **长时训练稳定性** | **遗留 13（本轮补回）** | ✅ |
| 11 | `dataloader_type="group"` 是否还有意义 | 已解决，`11` 已移出清单；本轮的支持矩阵进一步给了**边界条件** | ✅ |
| 12 | K=8 是否合适 | 遗留 11（本轮补充了「它是时间闸门」这一性质） | ✅ |
| 13 | `group` 在 `group_size < batch_size` 中间取值的完整行为 | 遗留 10 | ✅ |
| 14 | 观测器自身对峰值显存的抬高量 | **本轮已测出数**，见 §2；不再是无法验证项 | ✅ |
| 15 | 仓库 lint 门与 CI 的一致性 | 无法验证清单（本轮沿用，见 §6） | ✅ |
| 16 | `enable=False` + 非空 `dataset_sample_weights` 的真实入口行为 | 无法验证清单（本轮沿用） | ✅ |

**16/16 承接完毕。**

---

## 2. 证据

**装置**：`fix4/run_real4.py`（自 `fix3/run_real3.py` 复制，**不是**复审的 `review3/r3_probe.py`）·
`gear4.sh`（每档写明期望，`raise:<函数名>`）· `cmp4.py`（精确判据比较器）· `summarize4.py`。
全部 21 档从 `projects/holobrain_internal/common/train.py` **真实入口**进。

**「只注入不构造」重新逐行确认**：本轮加的四样都是纯记录 ——
raise 的 traceback 帧、port 的日志记录、护栏计数器、`__doc__`，
加上一个 `--no-identity` 开关（**少**装一样东西）。
全文仍然没有 new 出任何 sampler / DataLoader / optimizer / model builder / trainer / 训练循环；
唯一被构造的对象是一个 `logging.Handler`，它挂在 logger 上，不参与被测系统。
`REC["port_imported"]` 仍在本文件 import port **之前**取。

### 2.1 修复目标 —— `group` 四格 + `group_size` 那一格

| 档 | 配置 | 结果 | 谁在看守 |
|---|---|---|---|
| `f4_D_group_bs1_s4` | `group` bs=1 gs=16 **4 步** | **raise** `sampler.py:344 in assert_episode_stream_wired`，`fwd=0`、`firstbatch_ran=False`、**护栏日志 0 行**、`peak_mem=None`（训练从未开始） | 装配期跨度判据 |
| `f4_D_group_bs1_s12` | 同上 **12 步** | 同上，**逐字相同** | 同上（与步数无关） |
| `f4_D_group_gs1_s4` | `group` bs=4 **gs=1** 4 步 | **raise** 同一处，`min(1, 4) = 1` | 同上 |
| `f4_D_group_bs4_s4` | `group` bs=4 gs=16 4 步 | `rc=0`，bank `[4,4,4,4]`，`grad 0/0/68`，移动 63/68，恒等间隙 `1.296956e+00` / `1.123835e+00` | 第一批检查 fwd0（`liveness_ruled=False`，4 < K） |
| `f4_D_group_bs4_s12` | 同上 12 步 | `rc=0`，bank 恒 4，`grad 0/0/68`，移动 63/68 | 第一批检查 fwd0 **+** 看门狗 fwd8（`maxbank=4`） |

**上一轮的反例（`group`+bs=1+4 步、`rc=0`、`grad 64/4/0`、`0/68`、零告警）在 `fc33a5db` 上已不可达。**

### 2.2 文案指向真实原因

装配期跨度判据的完整文本（`f4_D_group_bs1_s4` 的 `error` 字段原文）：

```
memoryvla.enable=True with dataloader_type='group', but this configuration cannot
hold any memory at all. `group` clears the bank at the top of every training call
and again every group_size samples within the batch, so its memory reaches
min(group_size, batch_size) = min(16, 1) = 1 sample(s). At 1, no sample ever has a
predecessor to read: every retrieval finds an empty history, every fusion reduces
to an exact identity, and 7.47M parameters receive no gradient while the loss looks
perfectly normal.
Observed: dataloader_type='group', group_size=16, batch_size=1 (read from
MemoryVLAEpisodeStreamBatchSampler.batch_size), episode_stream_sampler=True.
Two ways out, both effective here:
  * dataloader_type='stream' -- it carries the bank across calls, so batch_size=1
    is a perfectly good configuration there; this is also the episode-level memory
    the paper describes.
  * keep 'group' but use batch_size >= 2 AND group_size >= 2 -- memory then spans
    min(group_size, batch_size) frames inside each batch, and nothing beyond it.
The episode sampler is NOT the problem here: it is wired, it is in the chain, and
the batches it produces are episode-contiguous. Changing episode_stream_sampler
will not affect this.
```

逐条对照契约的验收条件：

| 要求 | 这段文案 |
|---|---|
| 不断言判据分辨不了的成因 | 这条判据是**静态**的，成因唯一且确定，所以它**可以**断言 —— 并且它断言的正是真实原因 |
| 不建议已经生效的开关 | 明写 `episode_stream_sampler` **不是**问题、改它没用；实测该场景下它确实已经是 `True` |
| 建议的动作照做真的有效 | 两条都实测过：`stream` + bs=1 → `f4_B_stream_bs1_s12` 通过且 bank 涨到 8；`group` + bs=4 + gs=16 → `f4_D_group_bs4_s12` 通过且 bank 恒 4 |
| 先报观测值 | `Observed:` 一行给了 `dataloader_type` / `group_size` / `batch_size` **及其读取来源** / `episode_stream_sampler` |

看门狗文案（`f4_T_break_bs1_s12` 的 `error` 原文，节选）：

```
MemoryVLAMemory ran 8 training forwards and no episode's memory ever grew past a
single entry, ...
Observed: dataloader_type='stream', group_size=16, mem_length=16,
batch sizes seen=[1], distinct episodes per batch seen=[1], longest bank=1.
Two different things produce this and they need different fixes. Bank length
cannot tell them apart, so both are listed rather than one being asserted:
  (a) the batches are not episode-contiguous. If memoryvla.episode_stream_sampler
      is off, turn it on -- every bank mode needs episode-ordered batches. If it
      is already on, then MemoryVLAEpisodeStreamBatchSampler's episode spans do
      not match this dataset: _episode_spans (sampler.py) assumes one episode's
      frames are contiguous in the global index.
  (b) the configuration cannot hold memory at all -- under dataloader_type='group'
      that is min(group_size, batch_size) == 1. assert_episode_stream_wired
      rejects that before training starts, so reaching here that way means it was
      never called on this path.
```

**「The batches reaching this module are not episode-contiguous」这句断言已经不存在**，
`episode_stream_sampler=True` 只作为 (a) 的**条件分支**出现，不再是无条件的「the fix」。
仓内测试对这两点都有断言（见 P2-C）。

### 2.3 闸门未到达档（`max_step=4`，本轮验收重点）

| 配置 | 结果 | 看门狗裁决了吗 | **谁在看守** |
|---|---|---|---|
| `group` bs=1 | **raise**，训练从未开始 | 不需要 | 装配期跨度判据（fwd 之前） |
| `group` bs=4 gs=1 | **raise**，训练从未开始 | 不需要 | 装配期跨度判据（fwd 之前） |
| `group` bs=4 gs=16 | `rc=0`，bank 恒 4 | ❌ `liveness_ruled=False`（4 < 8） | 第一批检查 fwd0：批次坏了会在这里死（同配置注入实测触发） |
| `stream` bs=4 | `rc=0`，bank `4→8→12→16` | ❌ `liveness_ruled=False` | 第一批检查 fwd0（同配置注入 → `wrapper.py:372` 触发） |
| `stream` bs=1 | `rc=0`，bank `1→2→3→4` | ❌ `liveness_ruled=False` | **第一批检查判不了 bs=1，此格无人裁决** —— 见残留洞 |

**最后一格是本轮唯一没有守住的格子，如实交付**：同配置注入故障后
（`f4_T_break_bs1_s4`）`rc=0`、bank `[1,1,1,1]`、`grad 64/4/0`、移动 `0/68`、只有 1 行正常 INFO。
它**不是配置可达**的 —— 无注入的同一格（`f4_B_stream_bs1_s4`）bank 正常涨到 4。

### 2.4 `stream` 回归与关闭态

| 项 | 结果 |
|---|---|
| 关闭态 `stream`（`f770afe0` vs `fc33a5db`） | **四项精确量全同 + 结构判据 `port_imported == []`**：id 序列 20/20、batch key 14、参数量 `1,136,284,265`、sampler 链 `['DistributedBatchFlagSampler']` |
| 关闭态 `group`（同上） | **同样全同** |
| 开启态 `stream` bs=4 20 步 | bank `4→8→12→16` 封顶、`grad 0/0/68`、移动 63/68、参数量 `1,143,751,529`、batch key 15、恒等间隙 **`1.296956e+00` / `1.123835e+00`（与 `09`/`11` 逐位相同）** |
| **`stream` + bs=1 长跑（12 步）** | `rc=0`，bank `1→2→…→8`，`grad 0/0/68`，移动 62/68，看门狗 fwd8 裁决 `maxbank=8` ⇒ **合法配置不误报** |
| `mem_length=1`（12 步） | `rc=0`，bank 恒 1 但 `grad 0/12/56`、移动 51/68 ⇒ **模块在工作**；看门狗 fwd8 **主动站下并 WARNING**，不 raise |

**每条精确判据都配了阳性对照：**

| 判据 | 阳性对照 | 结果 |
|---|---|---|
| 逐样本 id 序列 | 注入宿主 sampler 构造参数 `seed=99`（`f4_ctrl_seed99`） | **0/20 相同**，而其余四项**纹丝不动** ⇒ 它只看得见顺序，且看得见 |
| batch key 集合 | 关闭 vs 开启 | 14 vs 15（多 `step_index`）✅ 有牙 |
| 参数量 | 关闭 vs 开启 | `1,136,284,265` vs `1,143,751,529` ✅ |
| sampler 链 | 关闭 vs 开启 | `DistributedBatchFlagSampler` vs `MemoryVLAEpisodeStreamBatchSampler` ✅ |
| 结构判据 `port_imported` | 关闭 vs 开启 | `[]` vs 4 个模块 ✅ |

> `--seed` 与 `set_epoch` 对批次顺序**无牙**（两轮实测），本轮没有拿它们充数。
> 「关闭 vs 开启」这一对本身就是后四项的天然阳性对照 —— 一次 run 同时给两个方向。

### 2.5 机械判据

| 项 | 结果 |
|---|---|
| 工具 md5（四个） | `preflight.sh 8ad881d7…` · `copy_fidelity_check.py 13f5ffdd…` · `orphan_switch_check.py 7971d335…` · `port_probe.py de4cfc37…` —— **与 `09` §7.1 / `11` §7.1 逐个相同**，两轮之间工具没搬家 |
| preflight `--static` @ `fc33a5db` | **EXIT=0**，`preflight PASSED`（同样两组豁免：`--waive-class BottleneckSE --waive-copy L105-L136 --waive-copy L335-L357`） |
| **阳性对照 @ `18106b05`** | **EXIT=1，3 findings**：`ORPHAN episode_stream_sampler` · `UNUSED MemoryVLAEpisodeStreamBatchSampler` · `DRIFT episode_stream_sampler plan='False' shipped=True` |
| C 档 `tools/check_reference.py` @ `fc33a5db` | **10 targets · 10 bit-exact · 0 failed**，`max\|diff\| = 0.000e+00` |
| 仓库 lint（`envs/RoboDojo/bin/ruff` 0.15.22，`--config=pyproject.toml .`） | 基线 `49b2178c` **31 findings** vs HEAD **31 findings**，`diff` 只有两行**行号位移**（`wrapper.py:145/146 → 151/152`，是 P2-B 移动造成的，那两行是既有的 `PerMemBank`/`CogMemBank` 构造行）⇒ **本轮零新增 lint 债** |
| memoryvla 子目录 lint | 改动前后同为 7 findings；新增的 3 个测试文件 **`All checks passed!`** |

> ⚠️ **一处与 `11` 的口径差异，明写**：`11` §2.4 记「HEAD 5 findings」，本轮同一命令在全仓上得到 **31**。
> 未能复原对方的取值范围（可能限定了子目录或只算 tracked 文件）。
> 本轮结论**不依赖绝对值** —— 它是**同一命令、同一 ruff 版本下 HEAD 与基线的逐行 diff**，
> 而那个 diff 只有两行行号位移。
>
> ⚠️ **阳性对照这次真跑成了。** `11` §7.2 记录上一轮那次「磁盘上只留下一行报错」，
> 成因是 `preflight.sh:91` 认死 `[[ -d .git ]]`，而 `git worktree` 的 `.git` 是文件、
> `git archive` 的树没有 `.git`。本轮用
> `git clone --shared --no-checkout` + `checkout 18106b05`（`.git` 212 K，**真目录**）跑通，
> 输出留在 `fix4/runs/2026-08-05/preflight_control_18106b05.txt`。

### 2.6 证据与代码的绑定（P3-G）

**先提交、再跑验证**，所以是同一性不是推断：

```
17/17 evidence files bind to fc33a5db (dirty=False, all 4 file hashes equal)
```

逐个核对了 `provenance.git_head_short == fc33a5db` ∧ `git_dirty is False` ∧
`port_files == {__init__,memory_bank,sampler,wrapper}.py 在 fc33a5db 上的 sha256[:16]`。
另有 4 份**基线**证据（`b2_*`）绑定到 `f770afe0`、`dirty=False`。

---

## 3. 本轮改动影响面自述（三分类）

> **全部标「（改动方自述，待复审独立推导）」。** 按协议 §0，这一节是给复审省时间的，不是证据。

### 3.1 失效（`11` 的哪些结论被本轮改动直接推翻）

| `11` 的哪一节 | 怎么失效的 |
|---|---|
| **§1 / §4.7 全节（P1-C）** | `group`+bs=1 的静默退化**不再可达**：4 步与 12 步都在第一次 forward 之前 raise。该节引用的 `R_D_group_bs1_short4` / `R_D_on_group_bs1` 描述的是**修复前**行为 |
| **§4.7 引用的看门狗文案** | 两句错话已删除；引用旧文案的地方要换。新文案见 §2.2 |
| **§4.1（P2-B）** | `MemoryVLAMemory.__doc__ is None == True` 不再成立；开启态 run 实测 `is_none=False` |
| **§2.3（P2-A′）** | 侵入度数字已更新，且改成基点 + 截至 commit 的两列表；`fc33a5db` 上是 `sampler.py +202/−5`、`wrapper.py +330/−0` |
| **§1 / §9 第 5 条（P2-C）** | 「仓内 memoryvla 测试为 0 个」不再成立：3 个文件、84 项 |
| **§2.4（P3-E）** | 工作区已干净 |
| **§6（P3-H）** | 两条无法验证项已补回，且承接动作已清单化（§4） |
| **§3.7 关于 `port_files` 为空的成因推测** | 那段代码本来就是**磁盘读**，空是因为那几个 run 跑在这段代码存在之前。结论方向不变，成因不同 |
| **§8「观测器自身对峰值显存的抬高量」** | 已测出数：`+0.98 MiB`，且落在噪声里。`MIGRATIONS.md` 教训 9 里那句「6 MiB 量级」是推断被写成实测，已划掉订正 |

### 3.2 需重验（本轮改动可能触及，已全部重跑）

| 项 | 本轮实测 |
|---|---|
| A 档关闭态等价（`stream` / `group` 两条路径） | 四项精确量 + 结构判据全同（§2.4） |
| 开启态 `stream` 回归 | bank `4→8→12→16`、`grad 0/0/68`、恒等间隙**逐位相同** |
| 开启态 `group` 回归 | bank 恒 4、`grad 0/0/68`、移动 63/68 |
| 护栏 fail-fast 与有牙 | 3 档注入 + 3 档链判据/跨度判据（§2.1、`PORT-STATUS.md`） |
| C 档 | 10/10 逐位一致 **在 `fc33a5db` 上** |
| 判据 K/C/D/F/S | preflight `--static` EXIT=0 + 阳性对照 EXIT=1 |
| 仓库 lint | HEAD ≡ 基线（只有两行行号位移） |
| optimizer 分组 | 68 张量全进 group 1，`trainable_not_in_optimizer = 0` |

### 3.3 仍有效（本轮未触及）

`memory_bank.py` 一行未动 ⇒ **所有 `[port:]` 拷贝保真结论、C 档的模块级对齐、
`BottleneckSE` 的判定**不受影响（判据 F 的 6 个标记全部在 `memory_bank.py`，
`sampler.py` / `wrapper.py` 一个都没有）。
`train.py` / 四个宿主 config 文件一行未动 ⇒ **L1 判定、宿主 seam 分析、ckpt 兼容性、
mask 极性、方法要素 12/15、接口语义 32 项**不受影响。
`11` §5.4 列的「继承、未重验」那批同样不受影响。

**唯一需要提醒的边界**：`wrapper.py` 的 `__init__` 新增了两个普通属性
（`self.group_size` / `self.mem_length`）与两个 set。它们**不进 `state_dict`**
（与既有的 `_episode_check_done` 等同处、同性质），所以 ckpt 兼容性不变；
参数量实测仍为 `1,143,751,529`，与 `11` 逐位相同，可作旁证。

---

## 5. 本轮新增 / 沿用的无法验证项

**沿用**（`11` §8 的两条，本轮同样没条件验）：

| 项 | 为什么仍验不了 |
|---|---|
| 仓库 lint 门与 CI 的一致性 | `holobrain_internal` 没装 ruff，借用 `envs/RoboDojo/bin/ruff` 0.15.22。结论是**同版本下 HEAD 与基线的相对比较**，不依赖版本正确性 |
| `enable=False` + 非空 `dataset_sample_weights` 的真实入口行为 | 需要一份带 per-spec `sample_weight` 的 dataset_specs，那要改宿主 config。本轮同样只做到函数级 |

**本轮新增：**

| 项 | 为什么验不了 |
|---|---|
| **CI 的 pytest 是否真的收集到新增的三个测试文件** | 本机无 pytest，装它破 E0。执行证据来自 `.git/run_tests_nopytest.py`（pytest stub）。只能给结构性论证：文件在 `tests/Makefile: test_ut` 的目标树内 |
| **`stream` + bs=1 + 批次实际不连续 + 步数 < K 这一格** | 已实测**无人看守**（`f4_T_break_bs1_s4`，`rc=0`、`grad 64/4/0`）。要关掉它需要一条不带时间闸门、又能在 bs=1 下判「批次是否连续」的判据 —— 而 bs=1 的单批里没有任何可比对的东西，**只能跨批看**，那就必然带闸门。本轮没找到解法，如实留着 |
| **K 的上界仍未实测确定** | 要枚举所有合法配置的「首次积累历史所需 forward 数」再取上确界。本轮只确定了下界是 2（实测 `f4_B_stream_bs1_s12` 的 bank 序列 `1,2,3,…`） |
| **观测器开销的真实值** | 实测差 `+0.98 MiB`，但 run-to-run 噪声就有 ~5 MiB ⇒ **不可分辨**。要真测出来需要独占卡 + 多次取中位数 |
| **`group_size` 在 `1 < group_size < batch_size` 区间的完整行为** | 本轮只跑了 `gs=1`（拒绝）与 `gs=16 > bs=4`（组轮转分支不执行）。中间取值沿用 `11` 的记录：`gs=2` 时 12 个张量拿到精确零梯度 |

---

## 6. 本轮**没有**做的事（明写，免得和「跳过」长得一样）

- **消融矩阵未重跑**（沿用第三轮的选择：明写未重跑）。
- **DDP 仍未验证**：本机任意两卡 gather 必崩，与前三轮同一硬约束。全部 21 档单卡。
- **`checkpoint=null`**：外部真实 ckpt 仍加载不了（遗留 7），全部档位随机初始化 + 本地 `vlm_pretrain`。
- **`fix3/` `review/` `review2/` `review3/` 一个字节未改**，只读引用。
- **`memory_bank.py` 一行未动** —— 它是 `[port:]` 拷贝保真的全部载体，动它会让判据 F 的结论作废。
- **没有新增任何 config 键** —— 新键会立刻落进判据 K 的射程。`group_size` 已有读取者
  （`config_holobrain_common.py:158`），本轮只是多了一个读取者。
- **`train.py` 一行未动**。

---

## 7. 装置与证据清单

`$ROL_JFS/port/memoryvla/fix4/`（**不进 git**）

| 文件 | 是什么 |
|---|---|
| `run_real4.py` | 观测器，自 `fix3/run_real3.py` 复制。新增：raise 的 traceback 帧 · port 的日志记录（带 forward 序号）· 护栏计数器 · `__doc__` · `--no-identity` |
| `gear4.sh` | 单档 runner。**每档写明期望**（`ok` / `raise:<函数名>`），期望 raise 的档核对栈帧函数名 —— 退出码对 ≠ 过程对 |
| `run_all4.sh` | 矩阵驱动（`core` / `regr` / `teeth`），4 宽并行，GPU 由 `nvidia-smi` 当场挑 |
| `gen_cfg.py` · `cfg/` | 17 份配置，**生成而非手写**（三个轴同时动，手抄必然抄错一格） |
| `cmp4.py` | 精确判据比较器；峰值显存与 loss 打印但标注为参考量 |
| `summarize4.py` | 「谁在看守」由栈帧与护栏计数器**推导**，不靠眼看 |
| `intrusion_line.sh` | 侵入度数字的重跑命令（**不是闸门**） |
| `runs/2026-08-05/b2_*.json` | 4 档对照基线，`f770afe0`（= `49b2178c` 的行为） |
| `runs/2026-08-05/f4_*.json` | 17 档验证，全部 `fc33a5db` / `dirty=False` |
| `runs/2026-08-05/preflight_head_fc33a5db.txt` | 机械判据全量，EXIT=0 |
| `runs/2026-08-05/preflight_control_18106b05.txt` | **阳性对照，EXIT=1 / 3 findings** —— 上一轮没跑成的这一条 |
| `runs/2026-08-05/check_reference_fc33a5db.txt` | C 档 10/10 |
| `clone_18106b05/` | `git clone --shared` 出来的基线仓（`.git` 是**目录**，preflight 才认） |
| `tree_49b2178c/` | `git archive` 出来的静态树，**只给 ruff 用**（ruff 不 import，所以 meta path finder 的坑不适用） |
| `lint/{base_49b2178c,head_*}.txt` | lint 逐行对照 |
| `attic/10-review-response_v2.md` | P3-E 那份未提交改名的副本，逐字节相同，留痕不删 |
| `msg{1,2,3,4}.txt` | 四个提交的信息 |

---

## 8. git 卫生

```
$ git diff --stat 49b2178c..HEAD
 docs_analysis/MIGRATIONS.md                        |  62 ++-
 docs_analysis/memoryvla/10-review-response.md      |  13 +-
 docs_analysis/memoryvla/12-review-response.md      | 467 +++++++++++++++++++++
 docs_analysis/memoryvla/PORT-STATUS.md             | 227 +++++++++-
 robo_orchard_lab/models/memoryvla/sampler.py       |  96 +++++
 robo_orchard_lab/models/memoryvla/wrapper.py       | 210 +++++++--
 .../models/memoryvla/__init__.py                   |  15 +
 .../models/memoryvla/test_sampler_guard.py         | 307 ++++++++++++++
 .../models/memoryvla/test_wrapper_guards.py        | 374 +++++++++++++++++
 9 files changed, 1705 insertions(+), 66 deletions(-)

$ git diff --stat 49b2178c..HEAD -- .gitignore | wc -l
0

$ git status --porcelain
(空)
```

> ### ⚠️ 订正（2026-08-05，第五轮，复审 P3-I）：上面那块 `--stat` 不是被审 commit 的输出
>
> 它取自第四个提交刚建好、而本文件自身还只有 467 行的那一刻；之后本提交被 `amend` 两次
> （`337193a8` → `ae598b85` → `28af78ab`），补了 `10-review-response.md:121` 与两处 `<C4>` 占位符。
> **交付的 `28af78ab` 上实测是：**
>
> ```
>  docs_analysis/memoryvla/10-review-response.md      |  23 +-      (上面写 13)
>  docs_analysis/memoryvla/12-review-response.md      | 514 ++++    (上面写 467)
>  9 files changed, 1761 insertions(+), 67 deletions(-)             (上面写 1705/66)
> ```
>
> 其余四项（`PORT-STATUS 227` · `sampler 96` · `wrapper 210` · 三个测试文件）**逐项相同**，
> 下面那三条结论（9 文件全在范围内 · `.gitignore` 未动 · porcelain 为空）
> 复审已逐条独立复核，**全部成立**。
>
> **根因与 P2-A′ 是同一句话**：贴出来的数字要么带基点，要么保证是终态取的。
> 本文件 §0 已经对**提交哈希**想清楚了这个问题（「它的哈希不写在这里」），
> 只是没把同样的推理用在 `--stat` 上 —— 而 `--stat` 里含着本文件自己的行数，
> **它天然是自指的，写完就会过期**。
> → 今后这一节要么在**最后一次改动之后**重取，要么明写「取自 `<代码提交>`，不含本提交自身」。

**9 个文件全部落在契约范围内**：memoryvla 子目录（2）+ 文档（4）+ 新增测试（3）。
`train.py` 与四个宿主 config 文件**一行未动**；`memory_bank.py` 一行未动。

**范围锁**：

| 项 | 结果 |
|---|---|
| A repo `git status --porcelain \| md5sum` | `9815d522644f15ab4edd56e5b33d1d03` ✅ 与 R0 快照相同 |
| A repo 20 个脏文件 md5 | 逐个与 `review/A_repo_baseline.txt` 相同（`md5sum -c` 全 `OK`）✅ |
| `review/memoryvla` | `97837e04` ✅ 未移动 |
| `review2/memoryvla` | `4268bca5` ✅ 未移动 |
| `review3/memoryvla` | `d311b000` ✅ 未移动 |
| `fix/` `fix3/` `review/` `review2/` `review3/` 证据目录 | 只读，未写入 |

**四个提交**：

| # | commit | 内容 | 判据 |
|---|---|---|---|
| 1 | `308730dc` | 文档订正 | `git diff --name-only 49b2178c..308730dc` **全是 `.md`** |
| 2 | `f770afe0` | P2-B 类文档字符串复位 | `git show --stat` **只有 `wrapper.py`**，`+12/−6`，零行为 |
| 3 | `fc33a5db` | P1-C 护栏补齐 + 测试进仓 | 代码 + 测试，**全部证据绑定到它** |
| 4 | `HEAD` | 实测回填 | **纯 `.md`**，理由见 §0 |
