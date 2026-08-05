# 11 — 增量复审 v3：MemoryVLA 移植的 P1-A / P1-B 修复轮

| 项 | 值 |
|---|---|
| 基线 commit | `2b739226` |
| 被审 commit | `49b2178c`（`port/memoryvla` HEAD） |
| 本轮两个提交 | `955fbe07` 文档订正（5 文件，全 `.md`）· `49b2178c` P1-B 护栏改挂消费端 |
| 上一轮报告 | `09-incremental-review.md` @ `4268bca5`（🟡 ACCEPT-WITH-FIXES，P0×0 · P1×2 · P2×1 · P3×4） |
| 本轮改动契约 | `~/storage_policy/protocols/robo_orchard_lab/port_memoryvla_3.md` |
| 方法论基准 | `review_memoryvla_1.md`（按需引用，未从 R0 全量执行） |
| 复审证据 | `$ROL_JFS/port/memoryvla/review3/`（自建观测器 + 15 个真实入口 run + 静态阳性对照，**不进 git**） |
| 算力 | 单卡 × 15 档，GPU 1/2/4/5，`ulimit -n 65536` |

---

## 1. 裁决

# 🟡 ACCEPT-WITH-FIXES

| 级别 | 数 | 摘要 |
|---|---:|---|
| **P0** | **0** | 关闭态五项判据 + 结构判据全过（**跨装置**与基线对比）；C 档 10/10 逐位一致；无全局副作用 |
| **P1** | **1** | **P1-C** `group` + `batch_size=1` 是一个**配置可达**的格子：跑满 8 次 forward 会 raise 但**文案指向错误原因、并给出一个已经生效的修复**；跑不满 8 次则**静默退化、零告警、rc=0**，`grad 64 None/4 零/0 非零`、`参数移动 0/68` —— **P0-1 的失效签名逐项复现**。`PORT-STATUS.md` 写的「不存在「memory 被构建 + 静默退化 + 无告警」的组合」**不成立** |
| **P2** | **3** | **P2-B** `BANK_LIVENESS_FORWARDS = 8` 插在类语句与类文档字符串之间 ⇒ `MemoryVLAMemory.__doc__ is None`（实测），autoapi 覆盖该包，而仓库 ruff 忽略 D101 ⇒ **无任何闸门会发现**<br>**P2-A′** 侵入度那一行**第三次**带着过期数字发布：`sampler.py +94/−1` / `wrapper.py +118/−0` 是 `f6dfd1e8` 的量，在它们所在的 `49b2178c` 上实为 `+106/−5` / `+204/−0`<br>**P2-C** `PORT-STATUS.md` 称「现在有断言禁止那几句话回来」，但两个测试脚本（22/22、24/24）**都不在 git 里**（仓内 memoryvla 测试为 0 个），没有任何 runner 会再跑它们 |
| **P3** | **4** | **P3-E** 工作区不干净（未提交的纯改名）<br>**P3-F** `10-review-response.md` 有两个近似重复的 `### 失效` 小节<br>**P3-G** P3-B 的 provenance 中途才生效，`head_B_stream_on`（stream 开启态回归的唯一证据）**无任何代码绑定**，且 A 重跑了、B 没有<br>**P3-H** `09` §8 五条新「无法验证」项**掉了两条**（`_episode_spans` 换数据集、长时训练稳定性）—— 与上一轮 P3-A 同一形状的复发 |

**一句话理由**：P1-A、P1-B（主干）、P2-A（部分）、P3-A（六条老项）、P3-C、P3-D **全部真闭环，且由本轮独立测量确认**；
但 P1-B 的修法留了一个**配置可达**的角落（`group` + `batch=1`），而交付文档恰恰把这个角落当作看门狗的战果写了进去，
**没有核对它触发时说了什么、也没有核对跑不满 K 步时它是否根本不触发**。

### 这个裁决该怎么读

**这一轮的主体是好的，而且比上一轮更好。** 三处最容易糊弄的地方都没糊弄：
旧数字**保留并划掉**而不是删掉；峰值显存被**主动降级**（而不是留着当一条好看的通过项）；
「本轮没有做的事」单列成节。改动方主动补了协议 §0 要求的三分类，**本轮因此不必自行推导**。

P1-C 不是「又错了一次」，是**同一条修复没走到底**：把判据从「配置项名字」换成「可观测后果」这个方向完全正确，
但落地成的那条后果判据带了**时间闸门**（K=8）与**归因歧义**（`bank 恒为 1` 既可能是批坏了、也可能是这个配置下记忆本来就不可能）。
而 `group` + `batch=1` 这件事**在构造期就是静态可判的**，根本不需要等 8 次 forward。

---

## 2. S1 — 范围合规

### 2.1 `git diff --name-status 2b739226..49b2178c`（全文）

```
M	docs_analysis/MIGRATIONS.md
M	docs_analysis/memoryvla/05-ablation-matrix.md
M	docs_analysis/memoryvla/06-verification.md
A	docs_analysis/memoryvla/10-review-response.md
M	docs_analysis/memoryvla/PORT-STATUS.md
M	robo_orchard_lab/models/memoryvla/sampler.py
M	robo_orchard_lab/models/memoryvla/wrapper.py
```

**7 个文件全部落在契约范围内**（memoryvla 子目录 + 文档）。**无范围溢出 ⇒ 不熔断，继承基线有效。**

- 契约要「两个提交，先文档后代码」：实交 **2 个非合并提交**。`955fbe07` 只含 `.md`（契约 §1 的 gate 满足）。
  上一轮 **P3-D**（契约要 2 个、实交 4 个）**闭环**。
- 契约允许改 `train.py`，**本轮一行未动** —— 自述属实。P0-1 的修复成果未被触及。

### 2.2 逐 hunk 读完后单独拎出来的三点

| 观察 | 判定 |
|---|---|
| 本轮净改动 `sampler.py` **+40/−32**、`wrapper.py` **+96/−10**（`git diff --numstat 2b739226..49b2178c`）——与改动方自述**逐字相同** | ✅ 不是纯增量，但删除全部是判据重写与闸门移除，逐条对得上 |
| 无格式化 / 无 import 重排 / 无重命名 / 无顺手重构 | ✅ 见 2.3 的 lint 对照 |
| `BANK_LIVENESS_FORWARDS = 8` 被放在 `class MemoryVLAMemory(nn.Module):` 与类文档字符串**之间** | ❌ **P2-B**，见 §4.1 |

### 2.3 侵入度：自述 vs 实测

`PORT-STATUS.md` 标题：`侵入度：L1，触及宿主已有文件 5 个，train.py +38/−6 · sampler.py +94/−1`

| 断言 | 实测 | 判定 |
|---|---|---|
| 触及宿主已有文件 **5 个** | `git diff --numstat --diff-filter=M 3ce31c0c..49b2178c -- "*.py"` = `config_holobrain_common.py` / `config_robodojo_dataset.py` / `train.py` / `structure.py` / `structure_qwen3_5.py` | ✅ **正确** |
| `train.py` **+38/−6** | `38 6` | ✅ **正确** |
| `sampler.py` **+94/−1** | **`106 5`** | ❌ **P2-A′** |
| `wrapper.py` **+118/−0**（订正块） | **`204 0`** | ❌ **P2-A′** |

`94/1` 与 `118/0` 是 `18106b05..f6dfd1e8` 的量（订正块**明确标了**这个基点，属诚实标注），
但**标题那一行没有基点限定**，它随第二个 commit 一起发布，而那个 commit 恰好让它过期。
上一轮 P2-A 的教训原文是「「0 删除」这种**听起来最无害的自述最容易没人核**：它被重写过一次而同一行里的另一个数字照样错着」——
**这一行第三次带着错数字发布了。**

### 2.4 git 卫生（逐条实测）

| 项 | 结果 |
|---|---|
| `git status --porcelain` | ❌ **不干净** —— `D docs_analysis/memoryvla/10-review-response.md` + `?? docs_analysis/memoryvla/10-review-response_v2.md`，两者 `diff` **逐字节相同**，即一次未提交的纯改名 → **P3-E** |
| `git diff --stat -- .gitignore \| wc -l` | `0` ✅ |
| A repo 与 R0 快照 | `git status --porcelain \| md5sum` = `9815d522644f15ab4edd56e5b33d1d03` ✅，**且 20 个脏文件的 md5 与 `review/A_repo_baseline.txt` 逐个相同** ✅（`train.py` mtime 变过但内容一致 ⇒ touch 非编辑） |
| `review/memoryvla` / `review2/memoryvla` 分支 | `97837e04` / `4268bca5`，与 `06` / `09` 记录一致，**未移动** ✅ |
| 仓库 lint 门（ruff 0.15.22，`--config=pyproject.toml`） | HEAD **5 findings**（I001×2 · B905 · E501×2）；**基线 `2b739226` 同一命令同样 5 findings，逐条相同** ⇒ **本轮新增 lint 债为 0** ✅ |

> ⚠️ 仓库环境 `holobrain_internal` 里**没有装 ruff**，上面用的是 `envs/RoboDojo/bin/ruff`（0.15.22）。
> 与 CI 实际版本是否一致**未验**，列入 §8。

---

## 3. S2 — 上一轮 P1/P2/P3 逐条闭环

判据：闭环证据必须是**从宿主真实入口跑出来的数值**，与 `09` 记录的失效态数值对照。
下表全部由**本复审自建的观测器**独立测得（见 §3.7），不采信改动方的自评数字。

### P1-A 失效数值零标注 —— ✅ **已闭环**

逐条核对 `49b2178c` 上的两份文档：

| 契约要求 | 实测 |
|---|---|
| 逐处标注数据来源与失效状态 | ✅ 两份文件顶部各有醒目 `⛔` 横幅，注明数据来源 `run_gears.py --sampler sequential` 是宿主到不了的装配 |
| 明确区分 harness 产出（已失效）/ 真实入口产出（有效） | ✅ `06-verification.md` 顶部给了**逐档状态表**（按小节标题定位，不用行号），C 档单独标为**仍有效** |
| **不要删掉旧数字** | ✅ 一个数字都没删，全部 `~~划掉~~` 保留 |
| `06-verification.md:5` 那句「所以数值描述的就是真正会训练的那个东西」 | ✅ **已划掉**，并附了推翻它的说明 |
| `05-ablation-matrix.md` 的结论句「没有哪个是摆设」 | ✅ 已划掉并反驳（指出它恰好把「全都是摆设」读成了反面） |
| 消融矩阵重跑或明写未重跑 | ✅ 明写「未重跑，本轮也不打算重跑」 —— 契约允许的那一支 |

### P1-B `group` 没有任何可用配置 —— ✅ **已闭环（主干）**，但留了 P1-C

`09` §4.4 记录的失效态（`review2/runs/C_group_host.json`，`group` + sampler 关，4 步，**rc=0**）：

```
每 batch 不同 episode 数  : 4
per_mem_bank 每步最大长度 : 1  1  1  1
grad None / 精确零 / 非零 : 64 / 4 / 0
参数移动                  : 0 / 68
护栏日志行数              : 0
```

本轮从 `train.py` 真实入口实测（`review3/runs/2026-08-05/`）：

| 配置 | 实测 | 判定 |
|---|---|---|
| `group` + sampler **False** | **raise**，`sampler.py:264 in assert_episode_stream_wired`，`train_forwards=0` | ✅ 失效签名**不再可达** |
| `stream` + sampler **False** | **raise**，同一处（既有阴性用例回归） | ✅ |
| **`group` + sampler True**（新的可用配置） | rc=0，sampler 链 `['MemoryVLAEpisodeStreamBatchSampler']`，bank 每步最大长度**恒为 4**，`grad 0/0/68`，参数移动 **62/68**，恒等间隙 `per=1.296956e+00 / cog=1.123835e+00`，看门狗日志「longest episode history seen = 4」 | ✅ **确认可用** |
| `stream` + sampler True（回归） | bank `4→8→12→16` 封顶，`grad 0/0/68`，参数移动 **63/68**，恒等间隙**与 `09` 逐位相同** | ✅ 无回归 |

新的 raise 文案已不再指引使用者关掉 sampler，实测原文含
「Turning it off is never the fix; it is how this state is reached.」—— `09` 点名的**加重情节已消除**。

### P2-A 「0 删除」 —— ⚠️ **修法引入了新问题（转 §2.3）**

数字换上去了，但换上去的是**上一个 commit 的**数字。判 **P2-A′**。

### P3-A 六条「无法验证」只承接了五条 —— ✅ **已闭环**

`06` §9 六条在 `PORT-STATUS.md` 上逐条命中：外部 ckpt（遗留 7）· DDP（已知 3 + 遗留 3/6）·
**A 的采样频率/降采样（遗留 9，本轮补回，写明「掉一条和主动不承接是两回事」）**· 端到端可比性（已知 1）·
D 档墙钟（验证结果段）· fifo vs tome（遗留 2）。

> 但 `09` 自己新增的五条里**掉了两条** → **P3-H**，见 §6。

### P3-B 证据 JSON 无 commit 绑定 —— ⚠️ **机制已修，旧产物未回填**

`run_real3.py` 新增 `provenance`：`git_head` / `git_dirty` / `train_py_sha256` / `port_files`（每个 `.py` 的 sha256[:16]）。
**机制有效**：`49b2178c` 上 `sampler.py=fb14d11ce90c2649`、`wrapper.py=31ab3bcf16ea3637`，
与 `git show 49b2178c:…| sha256sum` 逐位相同 ✅。

但**它是会话中途才生效的**，逐个文件核对 `fix3/runs/2026-08-04/`：

| run | `git_head` | `port_files` | 说明 |
|---|---|---|---|
| `head_D_group_on` / `head_D_group_rot` / `head_T_teeth` / `head_T_teeth_bs1` / `head_G_group_nosampler` / `head_G_stream_nosampler` / `head_A_stream_off_r2` / `ctrl_A_epoch7` / `ctrl_A_hostseed` | `955fbe07` + dirty | **4 个，与 `49b2178c` 相符** | ✅ 可绑定 |
| **`head_B_stream_on`** | `955fbe07` + dirty | **空** | ❌ **stream 开启态回归的唯一证据，无任何代码绑定** |
| `head_A_stream_off` / `head_A_group_off` | `955fbe07` + dirty | 空 | ⚠️ 关闭态不 import port，`train_py_sha256` 已绑定，影响可控 |
| `base_A_stream_off` / `base_A_group_off` / `base_A_seed1` | `955fbe07` **clean** | 空 | ✅ `955fbe07` 未改代码 ⇒ 等价于基线代码，`git_dirty=False` 本身即绑定 |

**所有 run 记录的 `git_head` 都是 `955fbe07`（文档提交），而被审 commit 是 `49b2178c`** ——
靠 `port_files` 才能把它们绑到真正跑的代码上。改动方**重跑了 A**（`head_A_stream_off_r2`，已绑定）**却没重跑 B**。
判 **P3-G**（不升级：我独立重跑了 B，逐项复现，见 §4.4，所以实质结论无损）。

### P3-C `run_real.py` docstring 自相矛盾 —— ✅ **已闭环**

`run_real3.py:228-234` 已改写为「These knobs do NOT buy a strict bar，其自身实测 `det_run1` vs `det_run2` = `1.564e-04`，A 档最终没用它们」。矛盾消除，且**没有复制进新文件**。

### P3-D 契约要 2 个 commit —— ✅ **已闭环**（见 §2.1）

### 3.7 改动方实际执行的验证脚本 —— 逐行审计 `run_real3.py`（606 行）

协议 S2 要求读改动方实际跑的脚本，且**这类问题极易复发**（上一轮 P1-1 的成因就是 harness 自建宿主装配）。

**「只注入不构造」仍然成立** ✅ —— 全文没有 new 出任何 sampler / DataLoader / optimizer / model builder / trainer / 训练循环；
测量对象全部是宿主自己构造的对象，只做 wrap，最后 `runpy.run_path("train.py", run_name="__main__")` 原样交还控制权。

本轮新增的三个开关逐个查：

| 开关 | 做了什么 | 判定 |
|---|---|---|
| `--break-episode-order` | 替换 `MemoryVLAEpisodeStreamBatchSampler.__iter__`，按 sampler **自己的 span 表**跨 episode 走 | ✅ 是**故意的故障注入**，且 sampler 对象留在链上 ⇒ 装配期护栏照样放行，只有消费端能发现 —— 正是要测的盲区 |
| `--host-sampler-seed` | 给 `DistributedBatchFlagSampler.__init__` 注入 `seed` kwarg（`train.py` 从不传） | ✅ 是**阳性对照**，改的是构造参数不是构造行为，且只在显式传入时生效 |
| `--host-sampler-epoch` | 在 post-`prepare()` 链上找到第一个有 `set_epoch` 的对象并调用；找不到就 raise | ✅ 没有静默 no-op（协议 S3「未触发即等于没有护栏」的同一纪律） |

**`REC["port_imported"]` 在本文件 import port 之前取** ✅ —— 关闭态的 `[]` 是关于宿主的陈述，不是关于观测器的。

⚠️ **观测装置污染被观测量：`09` §7.3 ⑥ 提的问题原样存在。**
`probing_forward` 每次 forward `clone` 两个张量，`wrapped_step` 每步 `clone` 全部 68 个 memoryvla 参数。
关闭态不装探针（`port_imported` 为空 ⇒ 不安装）**所以关闭态数值干净**；
**开启态的峰值显存读数确实带着这份开销**。改动方在 `MIGRATIONS.md` 教训 9 里**写下了这条**（「本次的探针每个 forward clone 两个张量，直接把 D 档显存读数抬高了 6 MiB 量级」）却**没有改装置**。
不单列为 finding：它被如实记录了，且开启态显存本轮也不作判据。

---

## 4. S3 — 回归探测（本轮核心）

全部 15 档从 `projects/holobrain_internal/common/train.py` **真实入口**进，观测器为**本复审自建**的
`review3/r3_probe.py`（同样「只注入不构造」；与 `run_real3.py` 字段名刻意对齐，以便跨装置比对）。

| # | 检查 | 取证 | 结果 |
|---|---|---|---|
| 1 | 关闭态实际走的分支 / 实例化的类型 | 运行时打印 sampler 链与 `sys.modules` | ✅ §4.2 |
| 2 | 关闭态等价性（五项精确判据） | **跨装置**：改动方基线 run vs 本复审 HEAD run | ✅ §4.2 |
| 3 | 每条精确判据配阳性对照 | 注入 `seed=99` | ✅ §4.3 |
| 4 | 峰值显存是否真该降级 | 本复审两次同配置 run | ✅ §4.3 |
| 5 | 新增护栏真的 fail-fast（raise 非 warn） | 4 个阴性用例 | ✅ §4.4 |
| 6 | 护栏无副作用（不消耗全局 RNG） | RNG 指纹 | ✅ §4.5 |
| 7 | 新增探针的有效性（人为构造退化场景） | 3 档故障注入 | ✅ §4.4 |
| 8 | 本轮新增开关/config 键有真实读取者 | preflight K | ✅ 本轮未新增 config 键 |
| 9 | 之前已移植的其他方法 | **N/A**，判据见 §4.6 | — |
| 10 | 全局副作用 | preflight S | ✅ `none` |
| 11 | **类文档字符串被挤掉** | 运行时读 `__doc__` | ❌ **P2-B**，§4.1 |
| 12 | **`group` + `batch=1` 的护栏行为** | 4 档实跑 | ❌ **P1-C**，§4.7 |

### 4.1 P2-B —— `MemoryVLAMemory.__doc__` 已经是 `None`

`wrapper.py:64-70` 把类属性放在了类语句与文档字符串之间：

```python
class MemoryVLAMemory(nn.Module):
    #: Training forwards to watch before ruling on bank liveness. ...
    BANK_LIVENESS_FORWARDS = 8

    """Perceptual + cognitive memory over HoloBrain's VLM features.

    Args:
        ...
    """
```

赋值语句在前，后面那段字符串**不再是 `__doc__`**，退化成一条无副作用的表达式语句。
真实入口实测（每个开启态 run 都记了这一项）：

```
MemoryVLAMemory.__doc__          : None
MemoryVLAMemory.__doc__ is None  : True
BANK_LIVENESS_FORWARDS           : 8
source_file                      : .../robo_orchard_lab/models/memoryvla/wrapper.py
```

**为什么不是纯审美问题**：

- `docs/conf.py:118` `autoapi_dirs = ["../robo_orchard_lab"]` —— autoapi 扫**整个包**，
  这个类会以**空描述**出现在生成的 API 文档里，而它的 `Args:` 是该模块参数的唯一说明。
- **没有任何闸门会发现**：`pyproject.toml` 的 ruff `ignore` 列表里有 **`D101`**（缺类文档字符串），
  且 ruff 不把字符串表达式报成 B018。实跑确认：HEAD 与基线的 findings **完全相同**，这条不在其中。

**这与本轮要消灭的缺陷同形**：无报错、无告警、无闸门，只有读源码才看得出来。修法是把那三行移到文档字符串**之后**。

### 4.2 关闭态等价性 —— 五项精确判据 + 结构判据，跨装置成立

**跨装置对比**（改动方的基线 run 用他们的装置，HEAD run 用我的装置；共享缺陷无法同时骗过两套）：

```
=== change-side BASELINE (base_A_stream_off, 955fbe07 clean) vs review-side HEAD (49b2178c)
    [ok  ] uuid_sequence   20/20 batches identical
    [ok  ] batch_keys      same  (14 keys)
    [ok  ] param_total     same  1136284265
    [ok  ] sampler_chain   same  ['DistributedBatchFlagSampler']
    [ok  ] port_imported   same  []
    [ref ] peak_mem_gib    same  8.976670265197754
    (reference only, NOT a criterion) loss max|diff| = 9.536743e-05
```

`stream` 与 `group` 两条关闭态路径**互相之间**也逐项相同（`uuid_sequence 20/20`、`param_total`、`sampler_chain`、`port_imported == []`）。

> **一条比运行时更强的静态论证**：关闭态下 `train.py:118` 的分支不成立，
> `robo_orchard_lab.models.memoryvla*` **一个模块都不被 import**（`port_imported == []`，4 个关闭态 run 全部如此）。
> 本轮改的两个文件在关闭态下**根本不参与执行** —— 所以关闭态不可能被本轮改动影响。
> 运行时的 A 档在本轮因此主要充当**噪声地板与判据分辨力的标定**，而不是等价性的主要证据。这一点值得写进下一轮的判据里。

**关闭态 + 非空 `dataset_sample_weights`**（本轮删掉了 `if not stream_sampler: return`，触发条件从「sampler 开」变成「enable 开」）：
直接对护栏函数取证 —— `enable=False` + `dataset_sample_weights={'a':1.0}` **返回，不 raise**（`stream` 与 `group` 都测）；
`enable=True` 同样输入才 raise。`enable` gate 在函数最前（`sampler.py:243-245`）⇒ **关闭态行为未变** ✅。
（此项为函数级取证，非真实入口，已在 §8 标注。）

### 4.3 阳性对照 —— 判据有牙，且峰值显存确实该降级

**逐样本 id 序列的阳性对照**（`--seed` 与 `set_epoch` 已被上一轮证明**对批次顺序无牙**，不能充数）：

```
=== 注入宿主 sampler 构造参数 seed=99
    [FAIL] uuid_sequence   0/20 batches identical      ← 必须是 0/20，否则判据看不见重排
      A ['press_by_number_...41', 'match_and_pick_...34', ...]
      B ['imitate_sorting_...56', 'press_by_number_...90', ...]
    [ok  ] batch_keys / param_total / sampler_chain / port_imported   四项全部不变
```

**只动了顺序，四项精确量纹丝不动** —— 正是这条判据该抓的东西。**所以 §4.2 的「20/20 一致」是有分辨力支撑的一致。**

**峰值显存：本复审独立复现了抖动，改动方的降级成立。**

```
review3 两次同配置、同卡、同代码的关闭态 run：
    R_A_off_stream_1   8.976670265197754
    R_A_off_stream_2   8.971459388732910      ← 低 5.21 MiB
```

改动方降级它的依据是 5 个 run 里 1 个离群；本复审用**自己的装置**跑出了**第二个不同的低值**
（`8.97145938873291` ≠ 他们的 `8.971695899963379`），与「分配器 run-to-run 差异」一致，与「代码差异」不一致。
**判定：降级正当，不是把不利判据洗掉。** 这一条本轮从「继承」升级为「本轮独立重验」。

> 剩下四项精确判据（逐样本 id 序列 / batch key 集合 / 参数量 / sampler 链）+ 结构判据 `port_imported == []` 全部成立。

### 4.4 护栏 fail-fast 与有牙 —— 全部从真实入口取证

| 场景 | 谁应该抓到 | 实测 |
|---|---|---|
| `group` + sampler 关（**P1-B 的洞**） | 装配期链判据 | ✅ raise，`sampler.py:264`，`train_forwards=0`，护栏日志 0 行后直接终止 |
| `stream` + sampler 关（既有阴性用例回归） | 装配期链判据 | ✅ raise，同一处 |
| 故障注入 `stream` bs=4 | batch 组成检查 | ✅ raise，`wrapper.py:302`，「4 samples from 4 different episodes」，第 0 次 forward |
| 故障注入 `group` bs=4，`max_step=4` | batch 组成检查 | ✅ raise，同上（短跑不影响这一道） |
| **故障注入 bs=1，`max_step=12`** | **只有看门狗能抓** | ✅ raise，`wrapper.py:354`，**第 8 次 forward**，「no episode's memory ever grew past a single entry」 |

**bs=1 那一档是决定性的**：batch 组成检查需要 >1 个样本、恒等探针永不 arm，**三道护栏里只剩看门狗**，而它响了。
`09` §4.3 的探针有效性由 9 用例扩到 24 用例，本复审用**独立实现的**故障注入（同样走 span 表，因为滑窗重排在这套数据上是恒等）复现了同一结论。

**开启态回归（B 档）与 `09` 逐项对照**：

| 量 | `09` 记录 | 本复审实测 |
|---|---|---|
| bank 每步最大长度 | `4→8→12→16` 封顶 | **相同** |
| grad None/零/非零 | `0 / 0 / 68` | **相同** |
| 参数移动 | `62 → 63 / 68` | `63/68` ✅ 落在区间内 |
| 恒等间隙 per / cog | `1.296956e+00` / `1.123835e+00` | **逐位相同** |
| 参数量 | `1,143,751,529` | **相同** |
| optimizer 分组 | 68 张量全进 group 1，游离 0 | **相同** |
| batch key | 开启 15（多 `step_index`） | **相同**（关闭 14） |

### 4.5 护栏无副作用 —— RNG 指纹

| run | `after_set_seed` | `trainer_built` | `at_exit` |
|---|---|---|---|
| 4 个**关闭态** run（stream / group / 重跑 / 阳性对照） | `1ccf1725…` | `1656c212…` | `38c878cd…` |
| 2 个**开启态** run（stream / group） | `1ccf1725…` | `b6c3f4b3…` | `c0123107…` |

- 关闭态四个 run **三个检查点全部逐位相同** ⇒ 本轮改动没有扰动关闭态的随机流。
- 两个开启态 run 的护栏走了**不同分支**（日志模式不同、`max_bank_len_seen` 4 vs 16、`_history_will_be_read` 在 group 下直接返回 False），
  而 `at_exit` 指纹**相同** ⇒ **护栏不消耗全局 RNG**。静态上也成立：两个方法只做 `len(set(...))` 与字典长度遍历，不触任何 RNG API。
- 开启态与关闭态在 `trainer_built` 处不同 —— 这是**预期**的（memoryvla 模块构造消耗 RNG，`06-verification.md` 早有记载）。

### 4.6 「之前已移植的其他方法」= **N/A**，但留痕

按 `09` §7.3 反馈 ④（N/A 要留痕并给判据，否则与「跳过了」长得一样）：
判据 —— `docs_analysis/` 下只有 `memoryvla/` 一个方法目录；`git ls-files | grep -i` 未命中第二个方法包。
**本仓当前只移植了 memoryvla 一个方法，本项无对象。**

### 4.7 P1-C —— `group` + `batch_size=1`：一个配置可达的角落

这一格**不需要任何故障注入**：`dataloader_type="group"` + `batch_size=1` + `episode_stream_sampler=True`，
sampler 正确接上（链里就是 `MemoryVLAEpisodeStreamBatchSampler`，`328975 batches of 1`），批次也确实 episode 连续。
但 `group` 在每次 `process_batch` 顶部 `bank.clear()`，而一个 batch 只有 1 个样本 ⇒ **bank 长度永远是 1**，记忆恒等。

**跑满 K=8 次 forward（`max_step=12`）**：

```
raise at wrapper.py:354 in _check_bank_liveness
"... The batches reaching this module are not episode-contiguous, so every
 retrieval finds an empty history ... so the fix is
 memoryvla.episode_stream_sampler=True. Turning it off is how this state is
 reached, never how it is left."
```

**触发本身是对的**（模块在这个配置下确实什么都不算），**但文案里两句话都是错的**：

1. 「The batches reaching this module are **not episode-contiguous**」—— 它们**是**连续的（`distinct_episodes_in_batch = 1`）。
2. 「the fix is **memoryvla.episode_stream_sampler=True**」—— 它**已经**是 `True`。

使用者照着做会发现无事可做，只能去读源码。**真实原因是 `group` 的记忆跨度只有一个 batch，而 batch=1 里没有历史** ——
这句话 `06-verification.md` 末尾早就写着（「batch 降到 1 时必须同时把 `dataloader_type` 切成 `stream`」），护栏没说。

**跑不满 K=8（`max_step=4`，纯配置、无注入）**：

```
error            : None            rc = 0
bank max_len/step: [1, 1, 1, 1]
grad none/zero/nz: 64 / 4 / 0   ->  64 / 4 / 0          ← P0-1 的失效签名，逐项相同
params moved     : 0 / 68                                ← P0-1 的失效签名，逐项相同
identity gap     : per=5.960464e-08  cog=0.000000e+00    ← 精确恒等
guard log lines  : 2（两条都是正常 INFO，无告警）
_bank_liveness_checked : False                           ← 看门狗从未裁决（4 < K=8）
```

**三道护栏全部沉默**：装配期判据放行（sampler 确实在链上）· batch 组成检查判不了（bs=1 需要 >1 个样本）·
看门狗**没到 8 次 forward，不裁决**。`7,467,264` 个参数照样冻结，loss 照样好看，**rc=0**。

> `PORT-STATUS.md`「结果矩阵」下面那句 **「不存在「memory 被构建 + 静默退化 + 无告警」的组合」不成立。**
> 这个组合**只靠配置就能到达**，而且 4–8 step 的短跑在本项目自己的实践里是常态
> （`05-ablation-matrix.md` 整张表跑的是 8 step；`09` 的 `C_group_host.json` 跑的是 4 step）。

**为什么判 P1 而不是 P0**（沿用 `09` 对 P1-B 的同一口径）：ship 配置是 `stream` + `True` + config 默认 batch，
那条路已被焊死；关闭态可证未变（§4.2）。要落进这个洞得**主动**同时选 `group`、把 batch 设成 1、且跑得比 8 步短。

**为什么仍是 P1 而不是 P2**：本轮契约 §5 把「触发时给出的信息指向真实原因」写成了**验收条件**，
而 `PORT-STATUS.md` 恰恰把 `group`+batch=1 这一格当作看门狗的战果写了进去
（「而 `group` + batch=1 恰好满足「bank 恒为 1」……它从此不再只是一段文字」）——
**声称覆盖了，实际覆盖得比声称的窄**，这正是 `09` 给 P1 的那条口径。

**根因（比落点更值得记）**：把判据从「配置项名字」换成「可观测后果」方向正确，但这条后果判据带了两个副作用——
**时间闸门**（要 K 次 forward 才裁决 ⇒ 短跑无保护）与**归因歧义**（`bank 恒为 1` 分不清「批坏了」和「这个配置下记忆本来就不可能」）。
而 `group` ∧ `batch_size == 1` **在构造期就是静态可判的**，根本不必等 forward。

---

## 5. S4 — 继承基线三分类复核

> ✅ **协议 §0 的前置条件本轮满足**：`10-review-response.md` 主动给了「仍有效 / 需重验 / 失效」分类节
> （上一轮 `06` 缺这节，`09` 只能自行推导并降置信度）。
> 但按协议立场 2，**改动方的分类不是证据**。下面是本复审独立推导的结果与分歧。

### 5.1 失效（须重新测量；沿用旧值即报告失真）

| 项 | 改动方处理 | 本复审判定 |
|---|---|---|
| `09` §4.4 全节（P1-B）「group 两条路都堵死」 | 明确标为失效，并说明 `C_group_host.json` 描述的是修复前行为 | ✅ **正确失效，且已重新测量** |
| `09` §8「`group` 是否还有意义」 | 更新为「现在存在可用配置」并给出 D 档数值 | ✅ **如实更新，没有被悄悄划掉** |
| `09` §4.2 引用的 raise 文案 | 三处已改写 | ✅ 本轮重验（§4.4） |
| 峰值显存作为**判据** | 主动降级为参考量 | ✅ **本轮独立重验并确认**（§4.3） |
| **侵入度 `+94/−1`、`+118/−0`** | **未随第二个 commit 更新** | ❌ **P2-A′** —— 正是本节要防的「沿用旧值」 |

### 5.2 需重验（全部重跑）

| 项 | 本复审实测 | 判定 |
|---|---|---|
| A 档关闭态等价 | 五项精确判据 + 结构判据，跨装置成立 | ✅ **本轮重验** |
| **C 档定输入数值对齐** | **10 targets / 10 bit-exact / 0 failed，在被审 commit `49b2178c` 上** | ✅ **本轮重验** |
| batch key 集合 | 关闭 14 / 开启 15（多 `step_index`） | ✅ **本轮重验** |
| 参数总量 | 关闭 `1,136,284,265` / 开启 `1,143,751,529` | ✅ **本轮重验** |
| optimizer 分组 | 68 张量全进 group 1，`trainable_not_in_optimizer = 0` | ✅ **本轮重验** |
| 探针有效性 | 3 档独立故障注入全部触发 | ✅ **本轮重验** |
| 判据 I / G / P | 恒等间隙、`0/0/68`、62–63/68 | ✅ **本轮重验** |

> ⚠️ **补一个上一轮的边界**：`09` §5.2 记录 C 档是「在 HEAD 上重跑」，但那个 HEAD 是 `2b739226`，
> 也就是**本轮改动之前**。本轮动了 `MemoryVLAMemory.forward` 的尾部（新增 `_check_bank_liveness()` 调用），
> 所以 C 档**必须在 `49b2178c` 上重跑才算数** —— 已跑，10/10 逐位一致，**改动未溢出范围**。

### 5.3 仍有效 → 抽验 5 条（挑与本轮改动语义相邻的，非随机）

| # | 抽验项 | 为什么挑它 | 结果 |
|---|---|---|---|
| 1 | 拷贝保真度 **F** | `sampler.py` 本轮 −32 行，`[port:]` 标记区间可能错位 | ✅ **抽验通过**：6 标记，4 个 ratio ≥ 0.998，2 个 DRIFT 均为**与上一轮相同的已声明改写** |
| 2 | `_build_memoryvla_cfg` 键转发（判据 D） | 护栏现在读 `dataloader_type` 的方式变了 | ✅ **抽验通过**：12/12 行计划默认值与 ship 值一致 |
| 3 | 孤儿 config 键（判据 K） | 本轮是否引入了新的死键 | ✅ **抽验通过**：12 键全部有读取者；**本轮未新增 config 键**（`BANK_LIVENESS_FORWARDS` 是类属性，不是 config 键） |
| 4 | 「纯增量、零无关改动」 | 两个文件本轮都有删除 | ⚠️ **部分不成立**（已知）：`sampler.py +40/−32`、`wrapper.py +96/−10`；但**无格式化 / 无 import 重排 / 无重命名 / 无顺手重构**，lint 对照可证（§2.4）。改动方**主动自述了这一点** |
| 5 | mask 极性 | 上一轮点名的「最高频静默错误」，本轮**未触及**，作为「抽验无系统性偏向」的对照 | **（继承自上一轮，本轮未重验）**；`structure.py` 不在本轮 diff 内，`git diff --name-status` 可证 |

**5 条里 4 条通过、1 条部分不成立且改动方已自述。** 未触发「整类降级为需重验」。

### 5.4 明确标注为继承、本轮未重验的结论

以下全部标 **（继承自上一轮，本轮未重验）**，本复审既没复核也没推翻：
方法要素 12/15 与 A 逐行一致 · 接口语义 32 项一致 0 项不一致 · cite 零幻觉 ·
四个宿主文件的 L1 判定 · ckpt 兼容性 1000→1068 · `BottleneckSE` 不接入的理由成立 · mask 极性正确 ·
`09` §4.5 四条宿主语义（`dataset_sample_weights` / per-spec `sample_weight` / `flags` / 第二训练入口）。

---

## 6. S5 — 新增风险是否如实记录

核对对象 `PORT-STATUS.md`（`10-review-response.md` 是单轮应答，不是常设风险册）。

| 协议要求的类别 | 记了没有 | 位置 |
|---|---|---|
| **① 训练动力学变化** | ✅ 记了，且写得准确 | 遗留 5（stream：每 batch 从 4 个 episode 变 1 个）+ **遗留 10**（group：「能跑不等于该用」、记忆跨度只有一个 batch、`mem_length` 完全不起作用、`group_size=2` 时 12 个张量拿到精确零梯度 ⇒「把 `group_size` 调小等于关掉一部分模块」） |
| **② 本轮之后才可能暴露** | ✅ 记了 | 遗留 10 指出 `group` 变可用后它的动力学首次变得相关，且 `group` 走不到 `clear_episode` 与 tome 巩固 ⇒「`group` 的 D 档不能替代 stream 的 E 档冒烟」；**遗留 11** 主动记了「K=8 没有调优，没有实验支持 8 比 4 或 16 更好，将来要用实测定不是拍」 |
| **③ 本轮仍无法验证的项原样保留** | ⚠️ **`06` 六条全部承接（P3-A 闭环），但 `09` 新增五条掉了两条** | 见下 |

`09` §8 新增五条 → `PORT-STATUS.md`：
开启态训练行为 ✅（遗留 5/10）· 关闭态浮点严格等价 ✅（「关闭态等价性」节）·
**`_episode_spans` 在其他数据集上的正确性 ❌ 未承接**（全文无「其他数据集」「RoboDojo Memory 六任务」相关表述）·
**长时训练稳定性 ❌ 未承接**（`PORT-STATUS.md` 无「长时」「epoch 尺度」；`10-review-response.md` 只有一句更窄的「`group` 的长时行为未观测」，而那不是常设风险册）·
`group` 是否还有意义 ✅（如实更新为已解决）。

判 **P3-H**。**这是上一轮 P3-A 的同形复发**：老的六条被认真补回来了，新的五条又掉了两条 ——
说明缺的不是这一次的细心，是**承接动作没有清单化**。

> `_episode_spans` 那条尤其值得留着：它假设「一条 episode 的帧在全局索引里连续」，
> 只在 RoboDojo Memory 六任务上验过。换数据集若不成立，看门狗**会**报警（bank 涨不过 1），
> 但若只是部分错位，现有判据一条都抓不到。

**`MIGRATIONS.md` 三条方法无关教训**（契约 §6 逐条要求）：✅ **全部写成了可复用判据，不是「某某 key 忘了接」式的记法**。

| 契约要求 | 落地 |
|---|---|
| 护栏挂生产端会被绕过，应挂消费端，判据用后果不用配置项名字 | 教训 11，且拆出了**两条病因**：「判据写成了配置项的名字」与「判据依赖前提已成立才能启动 ⇒ 要抓「X 从未发生」，判据不能以 X 已发生为前提」。附带记了「报错文案本身会制造事故」 |
| 真实入口下浮点不可逐位复现，关闭态等价性要用精确量 | 教训 9，给了区间对比与「最便宜也最有力的一条是逐样本 id 序列」的可操作结论 |
| 任何「无差异」结论都要先用阳性对照证明判据有牙 | 教训 10，「每条判据都成对交付：一个阴性用例 + 一个阳性用例」 |

> 教训 11 里「报错文案要说真实原因和唯一正解，不要给出一个更省事的错误出口」这句**写得完全正确**，
> 而 §4.7 表明它**在自己新写的那条看门狗文案上没有被执行**。教训是对的，落实差一格。

---

## 7. S6 — 机械判据全量回放 + 协议反馈

### 7.1 工具版本（钉住）

四个工具的 md5 与 `09` §7.1 **逐个相同**，mtime 也相同 ⇒ **两轮之间工具没搬家、没变松**，
本轮的绿与上一轮的绿可比：

| 文件 | md5 | mtime |
|---|---|---|
| `copy_fidelity_check.py` | `13f5ffdd6280fa9e6d0c467502dc61a9` | 2026-08-04T08:34:54 |
| `orphan_switch_check.py` | `7971d335083212f9bec576c387c52005` | 2026-08-04T08:36:51 |
| `port_probe.py` | `de4cfc37d06c80d37314337ff8a4e350` | 2026-08-04T08:34:54 |
| `preflight.sh` | `8ad881d7f6ac955d79d4bd37f33f718f` | 2026-08-04T08:44:26 |

### 7.2 逐条结果

```
bash ~/storage_policy/tools/port/preflight.sh --method memoryvla --base 18106b05 \
  --source-repo ~/git_repo/MemoryVLA \
  --config projects/holobrain_internal/common/configs/config_holobrain_common.py \
  --subdir robo_orchard_lab/models/memoryvla --static \
  --waive-class BottleneckSE --waive-copy L105-L136 --waive-copy L335-L357
→ preflight PASSED   EXIT=0
```

| 判据 | 命令 | 结果 | 是否误报 |
|---|---|---|---|
| **K** 孤儿配置键 | preflight `--static` | **12/12 键有读取者**（`episode_stream_sampler` 8 个 e.g. `train.py:119`；`dataloader_type` 26 个） | 无 |
| **C** 无人构造的类 | 同上 | 8 类，7 ok（`MemoryVLAEpisodeStreamBatchSampler` → `train.py:131`），1 UNUSED = `BottleneckSE` | 无（已声明不接入，需 `--waive-class`） |
| **D** 文档默认值漂移 | 同上 | **12/12 行一致** | 无 |
| **F** 拷贝保真度 | 同上 | 6 标记：4 个 ratio ≥ 0.998，2 个 DRIFT（`L105-L136` 0.902、`L335-L357` 0.105）均为**已声明改写** | 无（需 `--waive-copy`） |
| **S** 全局副作用 | 同上 | **none** | 无 |
| **I** 恒等探针 | 真实入口 B/D 档 | 开启态 `1.296956e+00` / `1.123835e+00`；退化方向 3 档注入全部 raise | 无，**两个方向都验了** |
| **G** 梯度三态 | 真实入口 | `0 None / 0 零 / 68 非零`（stream 与 group 都是；失效态为 `64/4/0`） | 无 |
| **P** 参数位移 | 真实入口 | `62–63 / 68`（失效态为 `0/68`） | 无 |
| **O** optimizer 覆盖 | 真实入口 | `trainable_not_in_optimizer = 0`；68 张量全进 group 1 | 无 |
| **B** 关闭态 batch key 集合 | 真实入口 | 14 vs 14 逐个相同（stream 与 group 两条路径） | 无 |
| **W** worker 随机流 + 阳性对照 | 真实入口 | 关闭态 4 个 run 的 RNG 指纹三点全同；阳性对照 `seed=99` → id 序列 `0/20` | 无，**判据有牙** |
| **X** harness/生产构造差异 | 人工逐行读 `run_real3.py` | 「只注入不构造」成立（§3.7） | — |
| **仓库 lint 门** | `ruff check --config=pyproject.toml` | HEAD 5 findings ≡ 基线 5 findings ⇒ **本轮零新增** | 无（ruff 版本未钉，见 §8） |

**阳性对照（本轮独立跑成）**：同一份工具、同一组参数跑在 `18106b05` 上 →

```
  ORPHAN  episode_stream_sampler     default=True  NO READER ANYWHERE
  UNUSED  MemoryVLAEpisodeStreamBatchSampler   NEVER CONSTRUCTED OR REFERENCED
  DRIFT   episode_stream_sampler     plan='False'  shipped=True
  ==== 3 finding(s) ====        preflight FAILED     EXIT=1
```

**所以 HEAD 上的 `EXIT=0` 是判据活着的绿。**

> ⚠️ **改动方声称的那次阳性对照，磁盘上没有留下工具输出。**
> `10-review-response.md` 与 `49b2178c` 的提交信息都写了「同一份工具、同一组参数跑在 `18106b05`
> 的树上报 **2 findings**，EXIT=1」。但 `fix3/` 下**只有两个 preflight 输出文件**：
> `runs/2026-08-04/preflight_head.txt`（HEAD，正常）与 `runs/2026-08-04/preflight_base_control.txt`，
> 而后者全文只有一行：
> `not a git repo: .../fix3/wt_18106b05 -- pass --repo, or cd to the host repo first`。
> `fix3/` 下另一处出现 `ORPHAN` 的是 `msg2.txt` —— 那是提交信息草稿，
> **是同一条声称的复述，不是工具输出**。
> 成因很清楚：preflight 的判据是 `[[ -d .git ]]`，而 **`git worktree` 的 `.git` 是文件不是目录**，
> `git archive` 出来的树则完全没有 `.git` —— 两种「建基线树」的常用做法**都过不了这一关**。
> 本复审改用 `git clone --shared --no-checkout` + `checkout 18106b05`（`.git` 212 K，真目录）跑通。
> 不单列 finding：结论方向由本复审独立确认，且改动方对差异（2 vs 3 条）的解释自洽；
> 但**「声称跑过而产物是一条失败信息」这件事本身**并入 P3-H 的同一条纪律：**声称即须留痕**。

### 7.3 协议反馈（`review-incremental` 第二次实战 —— **未修改协议文件**）

1. **`09` §7.3 ③ 的建议本轮生效了，且应固化。** 「每轮自测地板 + 额外给出至少一条不受浮点噪声影响的精确判据」
   在本轮直接产出了结论：逐样本 id 序列有牙、峰值显存没有。建议再加一句：
   **精确判据本身也要每轮重新证明它精确** —— 峰值显存就是「上一轮 5 个 run 恰好一致」被误当成精确性的例子。
2. **协议 S3 表格缺一行：「护栏触发时给出的信息是否指向真实原因」。**
   现表只问「新增护栏真的 fail-fast（raise 而非 warn）」。本轮 P1-C 的一半正是**触发了但说错了原因**，
   按现表逐条打勾会全绿。建议改成「触发 + 文案指向真实原因 + 建议的动作在该场景下确实有效」。
3. **协议 S3「新增探针的有效性」只要求「构造退化场景确认它会触发」，没要求「确认它在合法配置下不误报」，
   也没要求「确认它在**短跑**下仍然覆盖」。** 本轮的看门狗带时间闸门（K=8），
   一个 4 步的冒烟就绕过去了 —— 而 4–8 步恰是这个项目自己的冒烟长度。
   建议补：**带计数/时间闸门的护栏，必须同时给出「闸门未到达时的行为」这一档证据。**
4. **协议没有要求「被审 commit 的证据必须绑定到被审 commit」。**
   本轮所有 `fix3` 证据的 `git_head` 都是**前一个** commit（代码尚未提交、`git_dirty=True`），
   靠 `port_files` 哈希才能绑定。建议 S2 补一条机械判据：
   **逐个证据文件核对其记录的代码哈希 == 被审 commit 的对应文件哈希；对不上的证据降级为「不可采信」。**
5. **`09` §7.3 ⑥（观测装置污染被观测量）本轮被写进了 `MIGRATIONS.md` 却没被执行。**
   建议协议把它从「建议单列一条」升格为**可执行判据**：
   同一配置跑「带观测器」与「不带观测器」各一次，差值进报告。（本复审的 `r3_probe.py` 留了 `--no-identity` 开关，
   但本轮算力用在了 P1-C 上，这一档**没跑**，见 §8。）
6. **S1「逐 hunk 读完」对纯文档 commit 成本很高但价值很高。** 本轮 P1-A 的闭环判定完全来自读 `.md` 的 diff
   （划掉了没有、旧数字删没删、那句话改没改）。建议明确写「文档 commit 同样逐 hunk 读，不得因为是 `.md` 就跳过」。

---

## 8. 无法验证清单

**`09` §8 的 11 条原样承接**（六条老 + 五条新，不因为主要问题修好了而消失）：

| 项 | 为什么仍验不了 | 本轮有无变化 |
|---|---|---|
| 外部真实 ckpt 加载 | bucket 只有 v9，config 是 v10，`vlm.*` 全线 size mismatch；v10 warm-start 在 http URL 后而本机无外网 | 无变化。本轮全部 run `checkpoint=null` |
| DDP / 多卡 unused-parameter | 本机任意两卡 gather 必崩 `ILLEGAL_ADDRESS` | 无变化。**`group` 变可用不改变这条**，各 rank `spans[rank::num_replicas]` 长度不齐的风险依旧 |
| A 的采样频率 / 降采样 | 定义端在 A 的 RLDS 管线之外 | 无变化；**本轮已补回风险清单**（遗留 9） |
| A 与宿主端到端数值可比性 | 原理上不可比 | 无变化 |
| D 档墙钟时间 | 卡共享 | 无变化，**本轮再次实证**：同一配置的 B 档，改动方 `299.76 s`、本复审 `193.72 s`，差 55% |
| `fifo` vs `tome` 的实际差异 | 需要跑到 episode 尺度 | 无变化 |
| 开启态的训练行为本身 | 需要跑到收敛比指标 | 无变化，**且 `group` 现在也进入了这一类** |
| 关闭态在浮点层面的严格等价 | 真实入口不逐位可复现 | 无变化；**峰值显存本轮从「精确量」掉进这一类** |
| `_episode_spans` 在其他数据集上的正确性 | 只在 RoboDojo Memory 六任务上验过 | 无变化，**且已从 `PORT-STATUS.md` 掉了** → P3-H |
| 长时训练稳定性 | 本轮最长 20 step | **比上一轮更短**（上一轮 60 step），`tome` 巩固与 `clear_episode` 在 epoch 尺度的行为仍未观测 → 承接情况见 P3-H |
| `dataloader_type="group"` 是否还有意义 | — | ✅ **本轮已解决**：可用配置存在，D 档数值已给。此条**移出**无法验证清单（但「group 训得好不好」进入上面第 7 条） |

**本轮新增的无法验证项：**

| 项 | 为什么验不了 |
|---|---|
| **K=8 这个闸门值是否合适** | 只知道它必须 ≥2，且 4 步的短跑会绕过它（§4.7 实测）。什么值既不误报又不留窗口，需要枚举合法配置的「首次积累历史所需 forward 数」，本轮没做。改动方已如实记为遗留 11 |
| **`group` 在 `batch_size ≥ 2` 且 `group_size < batch_size` 时的完整行为** | 只跑了 `group_size=16`（>batch，组轮转分支不执行）与改动方的 `group_size=2`。中间取值未测；已知 `group_size=2` 时 12 个张量拿到精确零梯度 |
| **观测器自身对峰值显存的抬高量** | `r3_probe.py` 留了 `--no-identity`，本轮算力用在 P1-C 上，**这一档没跑**。开启态峰值显存因此仍带着观测开销，不作判据 |
| **仓库 lint 门与 CI 的一致性** | `holobrain_internal` 环境没装 ruff，本轮借用 `envs/RoboDojo/bin/ruff`（0.15.22）。与 CI 实际版本是否相同未验 —— 结论「本轮零新增 lint 债」是**同一版本下 HEAD 与基线的相对比较**，不依赖版本正确性 |
| **`enable=False` + 非空 `dataset_sample_weights` 的真实入口行为** | 本轮以**函数级**取证（直接调 `assert_episode_stream_wired`）确认返回不 raise。要从真实入口验，需要一份带 per-spec `sample_weight` 的 dataset_specs，而那要改宿主 config —— 复审只读，没做 |

---

## 9. 最短修复路径（按「修完能翻案」排序）

1. **P2-B（1 分钟）** —— 把 `wrapper.py` 的三行 `#:` 注释与 `BANK_LIVENESS_FORWARDS = 8` 移到类文档字符串**之后**。
   验收判据：`python -c "from ...wrapper import MemoryVLAMemory; assert MemoryVLAMemory.__doc__ is not None"`。
2. **P2-A′（2 分钟）** —— `PORT-STATUS.md` 侵入度标题与订正块的数字换成被审 commit 上的实测值
   （`sampler.py +106/−5`、`wrapper.py +204/−0`，`train.py +38/−6` 与「5 个宿主文件」不变），
   或者给标题加上基点限定。**建议顺手把这一行的数字改成由脚本生成** —— 它已经错了三轮。
3. **P3-H（5 分钟）** —— 把 `_episode_spans` 换数据集、长时训练稳定性两条补回 `PORT-STATUS.md` 风险清单。
   并把「承接上一轮无法验证清单」做成一个逐条打勾的动作，而不是重写一遍。
4. **P1-C（半天，代码 + 文案）** —— 两处：
   - **构造期静态判据**：`MemoryVLAMemory.__init__`（或装配期护栏）里，
     `dataloader_type == "group"` 且实际 `batch_size == 1` 时**直接 fail-fast**，
     文案说真实原因（「`group` 的记忆跨度是一个 batch，batch=1 时没有历史可取；用 `dataloader_type="stream"`，
     或把 batch 提到 >1」）。这条不需要等 forward，消除时间闸门这一半。
   - **看门狗文案去掉误导**：`bank 恒为 1` 有两种成因，文案现在只写了一种，且把已生效的开关当成修复建议。
     改成同时列出两条可能成因与各自的动作，或在文案里先报 `dataloader_type` / `batch_size` / `distinct_episodes_in_batch`
     三个观测值再给建议。
5. **P2-C（半天）** —— 把 `guard3_unit_test.py` / `guard3_probe_test.py`（含文案卫生断言）
   移进仓库的测试树并接进会被执行的入口；在那之前，把 `PORT-STATUS.md` 里
   「现在有断言禁止那几句话回来」改成「断言写在 `$ROL_JFS/.../fix3/`，**尚未进仓，不会被自动执行**」。
   （本机无 pytest，可沿用 `.git/run_tests_nopytest.py` 那种退出码脚本。）
6. **P3-E（1 分钟）** —— 把工作区那次未提交的改名收干净（提交或还原），使 `git status --porcelain` 为空。
7. **P3-F（1 分钟）** —— `10-review-response.md` 两个重复的 `### 失效` 小节合并成一个。
8. **P3-G（下次移植时）** —— runner 在**每次**写结果时都带 provenance（本轮是中途才加的）；
   并在提交前对所有引用进报告的证据文件跑一次「记录的代码哈希 == 被审 commit」的核对。

**修完 1、2、3、6、7 即可把 P2/P3 清干净；P1-C 是唯一需要动代码的一条**，
它不影响 ship 配置，可以记为遗留另起一轮，但**不要在没修的情况下把
「不存在「memory 被构建 + 静默退化 + 无告警」的组合」这句话留在 `PORT-STATUS.md` 里** —— 那句话现在是错的。

---

## 附：本轮证据清单

`$ROL_JFS/port/memoryvla/review3/`（不进 git；`fix3/` `review2/` `review/` 全程只读未改）

| 文件 | 是什么 |
|---|---|
| `r3_probe.py` | 本复审自建的「只注入不构造」观测器。相对 `run_real3.py` 多记：护栏日志（带 forward 序号）、两道消费端护栏的**进入/退出与内部计数**、`MemoryVLAMemory.__doc__`、三点 RNG 指纹、raise 的栈帧函数名、`--no-identity` 开关 |
| `r3_gear.sh` | 单档 runner（`ulimit -n 65536`、每 attempt 独立 workspace、rc 3 视为「按预期 raise」） |
| `r3_cmp.py` | 精确判据比较器；字段名与 `run_real3.py` 对齐，**支持跨装置比对** |
| `cfg/` | 12 份配置（含本复审新增的 `D_on_group_bs1` / `D_group_bs1_short4` / `T_short4_bs1` / `T_short4_group`） |
| `runs/2026-08-05/R_A_off_stream_{1,2}.json` | 关闭态 stream × 2（噪声地板 + 峰值显存稳定性） |
| `runs/2026-08-05/R_A_off_group_1.json` | 关闭态 group |
| `runs/2026-08-05/R_ctrl_hostseed.json` | **阳性对照**：注入宿主 sampler 构造参数 `seed=99` |
| `runs/2026-08-05/R_B_on_stream.json` | 开启态 stream 回归（bank `4→8→12→16`） |
| `runs/2026-08-05/R_D_on_group.json` | 开启态 group —— **新可用配置**（bank 恒为 4） |
| `runs/2026-08-05/R_G_{group,stream}_nosampler.json` | 两条 fail-fast 阴性用例（rc=3，`sampler.py:264`） |
| `runs/2026-08-05/R_T_break_stream.json` | 故障注入 bs=4 → `wrapper.py:302` |
| `runs/2026-08-05/R_T_break_bs1.json` | 故障注入 bs=1 → **只有看门狗能抓**，`wrapper.py:354`，第 8 次 forward |
| `runs/2026-08-05/R_T_short4_group.json` | 故障注入 group bs=4 短跑 → 仍被 batch 组成检查抓到 |
| `runs/2026-08-05/R_T_short4_bs1_rerun.json` | 故障注入 bs=1 + `max_step=4` → **rc=0，零告警** |
| **`runs/2026-08-05/R_D_group_bs1_short4.attempt1.json`** | **P1-C 的证据**：纯配置、无注入、`group`+bs=1+4 步 → **rc=0，`64/4/0`，`0/68`，三道护栏全沉默** |
| `runs/2026-08-05/R_D_on_group_bs1.json` | **P1-C 的另一半**：同配置跑满 12 步 → raise，但文案指向错误原因 |
| `preflight_head_49b2178c.txt` / `preflight_head_waived.txt` | 机械判据全量，`EXIT=0` |
| `preflight_control_18106b05.txt` | **阳性对照，`EXIT=1`，3 findings** —— `fix3` 那次没跑成的这一条 |
| `clone_18106b05/` | `git clone --shared` 出来的基线仓（`.git` 是**目录**，preflight 才认） |
| `tree_2b739226_static/` `tree_18106b05_static/` | `git archive` 静态树（**不用 `git worktree`**：可编辑安装的 meta path finder 会让它静默跑到新代码上） |

> **一条留给下一轮的提醒**：本复审的 `r3_gear.sh` 把 `rc=3` 一律当成「按预期 raise」而促成结果，
> 结果一档 GPU OOM（同事进程占满卡）也被促成了，是靠比对 JSON 里的 `error` 文本才发现的。
> **退出码对 ≠ 过程对** —— 与 `10-review-response.md` 末尾改动方自己记的两次是同一形状。
> 该档已在空闲卡上重跑（`R_T_short4_bs1_rerun`），原始 OOM 记录保留在 `R_T_short4_bs1.json` 里未删。
