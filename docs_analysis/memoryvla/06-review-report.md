# 06 — 审查报告：MemoryVLA → HoloBrain

> **⚠️ 本文当前是「第一批：静态审查」的中期报告。**
> R3（四档数值复跑）与 R6（训练动力学 + 补三项空白）属于第二批，尚未执行。
> **裁决为暂定**，第二批完成后本文整体重写为终版。

审查者独立执行 · 日期 **2026-08-04** · 只读，未修改宿主 / A repo / `port/memoryvla` 任何文件

| 项 | 值 |
|---|---|
| 被审分支 | `port/memoryvla` @ `18106b05` |
| 受审**代码**实际止于 | `111981a5`（`18106b05` 已核实为纯文档，`git diff --name-only` 仅 5 个 `.md`） |
| 宿主基点 | `3ce31c0c` |
| A repo | `~/git_repo/MemoryVLA` @ `0eef5c39`（与自述 `0eef5c3` 一致 ✅） |
| 报告分支 | `review/memoryvla`（从 `18106b05` 拉出） |
| 临时证据 | `$ROL_JFS/port/memoryvla/review/`（不进 git） |

---

## 1. 裁决（暂定）

# 🔴 P0 × 1 —— 暂定 `REJECT`（待第二批复核后定终版）

**一句话理由**：方法本体搬得忠实完整、接口语义全部核对一致、侵入度干净无虚报，
**但 `episode_stream_sampler` 是个没有任何消费者的死开关，配套 sampler 从未接入真实训练路径**——
在 `train.py` 下打开 `memoryvla.enable` 会得到一个**精确恒等**的记忆库：
数值出得来、loss 正常降、日志干净、无任何告警，而方法根本没有运行。

| 级别 | 数量 | 条目 |
|---|---:|---|
| **P0** | **1** | P0-1 死开关，方法在真实训练路径上不生效 |
| **P1** | **1** | P1-1「记忆库在工作」的全部证据来自补齐了该缺失集成的自建 harness |
| **P2** | **3** | P2-1 deploy 分支缺 `uuid`；P2-2 cite 行号漂移 32 %；P2-3 命名违反自定约定 |
| **P3** | **2** | P3-1 D 档时间结论落在噪声内；P3-2 构造顺序使 B 档「开/关差值」不可解读为净效应 |

**为什么暂定 REJECT 而不是 ACCEPT-WITH-FIXES**：协议规定 REJECT 的判据之一是
「必需组件缺失导致方法失效」。当前状态**符合该判据的后果**（方法失效且静默），
但**不符合其通常成因**——组件不缺，`sampler.py` 已实现且实现正确，缺的只是几行接线。
这是一个**修复成本极小、后果极重**的缺口。终版裁决取决于第二批能否实测坐实恒等退化
（R6 已列入取证计划），若坐实则维持 REJECT；若第二批发现我的推演有误则改判。
**我倾向于：只要这条被修好并补一次真实路径的 E 档，其余部分足以支撑 ACCEPT-WITH-FIXES。**

---

## 2. P0 清单

### P0-1 · `episode_stream_sampler` 是死开关，方法在真实训练路径上静默失效

| | |
|---|---|
| **现象** | 配置键 `memoryvla.episode_stream_sampler=True` 没有任何代码读取；`MemoryVLAEpisodeStreamBatchSampler` 从未被宿主实例化；`train.py` 硬编码 `DistributedBatchFlagSampler`（全局随机排列）。开启 `enable=True` 训练时，记忆库退化为**精确恒等**，`retrieval_blocks` 拿不到梯度，**不报错** |
| **证据** | 完整命令与输出见 `$ROL_JFS/port/memoryvla/review/P0-1_dead_sampler_switch.txt`，摘要于 `06e-intrusion-audit.md` §3.1，六条独立证据：① 配置键只有定义无读取 ② sampler 类零实例化 ③ `train.py:117-131` 硬编码且**未被本次改动触及**（`git diff --name-only … -- train.py \| wc -l` → `0`）④ `dataset_wrapper.py:132-134` 确为 `generator.permutation(n)` ⑤ `run_gears.py:138-145` 由 harness 自行实例化 sampler ⑥ `_build_memoryvla_cfg()` 不转发该键、`MemoryVLAMemory.__init__` 无该形参 |
| **恒等性的代数证明** | `hist` 恒空 → `retrieved = working_mem` → `gate` 融合 `scale*w+(1-scale)*w = w`（与 `scale` 无关）；`add` 融合 `(w+w)*0.5 = w`。**两种融合方式都是精确恒等** |
| **影响** | 任何人按 `PORT-STATUS.md:99` 的指引「确认 `episode_stream_sampler=True`」后训练，会得到与 baseline 数值等价的模型 + 7.47 M 个冻结的死参数，并把「指标没提升」误判为「这方法在我们数据上没用」。**这是本次移植最坏的失败形态：不是错，是白跑** |
| **加重情节** | `04-port-plan.md:58` 承诺过 `config_holobrain_common.py` 含「+ 可选 batch sampler 选择」——未实现；`04-port-plan.md:78` 规定该键默认 **False**，实际 ship 的是 **True**。默认值与设计文档相反，使问题从「缺失」变成「误导」 |
| **建议修法** | 在 `train.py` 构造 `DataLoader` 处按 `enable ∧ episode_stream_sampler` 选择 batch sampler；并让 `MemoryVLAMemory` 在 `stream` 模式下检测到「一个 batch 内 episode 数 == batch_size」时主动告警，使该路径**结构上不可能再静默** |

---

## 3. P1 清单

### P1-1 · 「记忆库确实在工作」的全部证据都产自补齐了缺失集成的自建 harness

| | |
|---|---|
| **现象** | B 档「68/68 张量有梯度」、E 档「跨 2 条 episode、bank 峰值 16、step 34 回落」——这些证明方法生效的关键数字，全部来自 `run_gears.py`，而该脚本在 `:138-145` **自己 import 并实例化了 sampler**。宿主没有任何路径能做到这件事 |
| **证据** | `run_gears.py:61-62` 有 `--sampler {episode,sequential}` 选项，默认 `episode`；`:138-143` 直接构造 `MemoryVLAEpisodeStreamBatchSampler`。而宿主侧零实例化（见 P0-1 证据②③） |
| **影响** | 这些数字**不能迁移到真实训练路径**。它们描述的是「如果 sampler 接上了，方法会正常工作」——这是有价值的信息，但**不是**「移植完成」的证据。P0-1 之所以能躲过五档验证，根因就在这里：**验证装置替被审代码补上了它缺的那块集成** |
| **为什么单列而不并入 P0-1** | P0-1 是「缺陷是什么」，P1-1 是「为什么没被发现」。后者对下一次移植的价值更大：**harness 若比被审代码多做了任何一件事，那件事就是审查盲区** |
| **建议修法** | 验证 harness 只允许走宿主已有的构造路径；harness 需要而宿主没有的东西，一律先补进宿主再验 |

---

## 4. P2 / P3 清单

| 编号 | 级别 | 现象 | 出处 |
|---|---|---|---|
| **P2-1** | P2 | `config_robodojo_dataset.py` 的白名单补丁注释称「`uuid` is already whitelisted」，对 training(`:202`)/validation(`:240`) 成立，**对 deploy(`:259-273`) 不成立**。开启态 deploy 会在 `wrapper.py:161` 抛 `KeyError`。**响亮的崩溃**，但与自陈遗留 1（`reset()` 未接评测循环）叠加说明推理路径从未被执行过 | `06e` §2.2 |
| **P2-2** | P2 | cite 抽样 38 条：命中 26 / **行号漂移 12（32 %）/ 幻觉 0**。漂移集中于指向宿主 `structure.py` 的中段引用（+3~+8），且与基点、HEAD **两个版本都对不上** → 非机械提取。指向 A 的类定义行 13/13 精确 | `06a` |
| **P2-3** | P2 | `MIGRATIONS.md:12` 自定约定「类名带方法前缀」，但 8 个新类中 **6 个无前缀**（`TimestepEmbedder`/`CrossTransformerBlock`/`BottleneckSE`/`GateFusion`/`CogMemBank`/`PerMemBank`）且全部进 `__all__`。`GateFusion`/`TimestepEmbedder` 在本仓极易撞名。**违反的是移植方自己立的规矩**（可辩护：保留原名利于溯源，但应显式记录取舍） | `06e` §5 |
| **P3-1** | P3 | D 档结论「开启 +10 % 时间」不成立：baseline 38.5 s vs **关闭态 35.1 s**（关闭态反而快 3.4 s，≈9 %），说明墙钟噪声 ≥9 %，+10 % 落在噪声内 | `06-verification.md` D 档 |
| **P3-2** | P3 | `self.memoryvla = build(...)` 插在 `data_preprocessor` **之前**，开启态会位移其后所有模块的初始化 RNG。故 B 档「开/关最大差 6.203e-02」混有主干初始权重不同的成分，**不能解读为记忆库净效应** | `06e` §2.1 |

---

## 5. 验证结果对照表

> 第一批未跑数值档。下表先记录**自述值**与**本批的独立静态复核**，
> 「我实测」列由第二批填入。

| 档位 | 我实测（第一批） | PORT-STATUS 自述 | 是否一致 |
|---|---|---|---|
| 第 0 步 确定性 | ⏳ 待第二批 | 0.000e+00 逐位可复现 | 待定 |
| **A 关闭态等价** | ⏳ 待第二批。**静态已确认机制成立**：`build(None)→None`，模块不构建，不消耗 RNG | 0.000e+00，参数量一致 | 机制可信，数值待验 |
| A' 已有移植回归 | ✅ **N/A（首次移植）**，已独立确认 `MIGRATIONS.md` 仅 2 节、无第二个方法 | 未提 | 一致 |
| B 开启态前向 | ⏳ 待第二批 | 68/68 有梯度，范数 8.39e-02 | **已知不可迁移**（P1-1） |
| C 数值对齐 | ⏳ 待第二批复跑。**静态已确认**：`check_reference.py:19-21` 确实 import 宿主实现而非 A，**非自比**；`CogMemBank` 与 A `:158-332` 逐行一致 | 10/10 逐位一致 | 方法学可信，数值待验 |
| D 资源 | ⏳ 待第二批 | 参数 +0.66 %、显存 +0.31 GiB、时间 +10 % | **时间项已判不成立**（P3-1） |
| E Memory 冒烟 | ⏳ 待第二批 | 2 episode，bank 峰值 16，step 34 回落 | **已知不可迁移**（P1-1） |
| 已有 ckpt 兼容 | ⏳ 待第二批（将用 bucket 上的**真实** `checkpoint_20000`，而非自造 state_dict） | 1000→1068，0 unexpected | 待验 |

---

## 6. 完整性结论（详见 `06b`）

| | 数量 |
|---|---:|
| 论文/A 侧方法要素 | 15 |
| **已移植且与 A 逐行一致** | **12** |
| 声明不移植且理由成立 | 2（`BottleneckSE` 未接入、DiT 动作头） |
| **静默缺失** | **1**（episode 有序数据流 → P0-1） |
| 多搬 | **0** |

「最容易静默丢掉」六类**逐类通过**：公式系数（ToMe 的 `0.5`、add 的 `*0.5`、
`max_period=10000`、`freq_emb=token_size//4`）· detach/no_grad 位置（5 处逐一对应）·
warmup（A 无此逻辑，N/A）· norm 位置（post-norm，逐字一致）· loss 归约（不产生 loss，N/A）·
`__init__` 隐式副作用（已剥离且未连带丢功能）。

---

## 7. 接口语义结论（详见 `06d`）

| 类别 | 已核对一致 | 不一致 | 未确认 |
|---|---:|---:|---:|
| 动作语义 / 归一化 / 时序 / mask·padding / 张量 / 常量 / 额外三项 | **32** | **0** | **1** |

**零不一致。** 协议点名的「最高频静默错误」——mask 极性——我用**构造端**
（`structure.py:334-342`，`~img_feature_mask & not_pad_mask`）与**消费端**
（三处消费者一律 `key_padding_mask=~mask`）两条独立路径交叉验证，移植方判断正确。
`uuid` per-episode、`step_index` episode 内步序、permute 往返可逆、`scatter` 非原地、
零硬编码 device，均独立坐实。

**唯一 `未确认`**：A 的采样频率/降采样策略。**为什么验不了**——A 的帧率由其 RLDS 管线决定，
`vla/memory_vla.py` 内只出现消费端形参，A repo 中读不到定义端，且 A 与宿主数据不同源无法对跑。
**影响**：中等偏低，PE 编的是整数 step 序号故实现不会错，受影响的是 `mem_length=16` 的物理时长语义
→ 属调参而非 bug。**该怎样才能验**：读 A 的 RLDS builder 的 step 定义或论文附录数据处理节。

---

## 8. 侵入度实况（详见 `06e`）

| 文件 | 自述 | 实际 | 偏差 |
|---|---|---|---|
| `models/holobrain/structure.py`（3 hunk） | L1 | **L1** | 无 |
| `models/holobrain/structure_qwen3_5.py`（1 hunk） | L1 | **L1** | 无 |
| `configs/data_configs/config_robodojo_dataset.py`（1 hunk） | L1 | **L1** | 无 |
| `configs/config_holobrain_common.py`（3 hunk） | L1 | **L1** | 无 |

**无虚报。** 8 个 hunk 全为纯增量（+67/-0），形状均为「一个开关判断 + 一次调用」，
实现体全在子目录内，**零无关改动**（无格式化、无 import 重排、无顺手重构）。
全局副作用扫描（seed / dtype / device / backends / hook / 原地操作 / import 期动作）**全部为零**。

---

## 9. 事实性（详见 `06a`）

**命中 26 / 漂移 12 / 幻觉 0**（38 条抽样，三份文档各 ≥4 条）。
协议规定的「≥1 条内容不符 → 翻倍重抽」**未触发**。
Gate A 形式判据成立：`03-interface-diff.md` 中不存在既无 `cite:` 又无 `未确认` 的事实行；
且读不到的项确实写了 `未确认` 而非拿常识充数。

---

## 10. 无法验证的部分（第一批）

| 项 | 为什么验不了 | 该怎样才能验 |
|---|---|---|
| A 的采样频率 / 降采样 | A repo 内只有消费端形参，定义端在 RLDS 管线外部 | 读 A 的 RLDS builder step 定义或论文附录 |
| DDP / 多卡下的 unused-parameter 行为 | 本机硬约束：任意两卡 gather 必崩 `ILLEGAL_ADDRESS` | 换一台多卡可用的机器，或在 AIDI 上单独起一个 2 卡 job |
| A 端到端与宿主端到端的数值可比性 | 感知记忆语义不同（A 记 LLM 前的视觉 patch，宿主记 VLM 后已被语言条件化的特征），**原理上不可比**，非工程问题 | 不可验证；只能做「加/不加记忆库」的宿主内 A/B，不能做跨 repo 数值对齐 |
| 四个数值档（A/B/C/D/E）与 ckpt 兼容 | **本批按约定不占 GPU** | 第二批执行，已列取证计划 |

---

## 11. 第二批取证计划（因 P0-1 而调整）

P0-1 改变了第二批的重心。除原定的复跑五档 + 补三项空白外，**新增一项决定性实验**：

1. **【新增·最高优先】实测恒等退化**：`enable=True` + **宿主真实 `DistributedBatchFlagSampler`**，
   跑 20 step，取证三项：① `retrieval_blocks` 各参数 `grad.norm()` 是否为 0
   ② 输出与 `enable=False` 的逐 step 差是否为 0 ③ 每 batch 内 `len(set(episode_ids))` 是否 == batch_size。
   三项全中即坐实 P0-1，裁决维持 REJECT；任一项不中则我的推演有误，改判。
2. 复跑五档（A/B/C/D/E），与自述数字逐个对照。
3. 补三项空白：真实 `build_optimizer` 分组打印 · **非零 lr** 跑 2 step 验参数确实更新 ·
   `num_workers>0` 的 worker RNG 流。
4. 用 bucket 上**真实** `checkpoint_20000` 做 ckpt 兼容性（只读 bucket），替代自造 state_dict。

---

## 12. 给移植方的最短修复路径（按「修完能翻案」排序）

1. **接上 sampler**（P0-1）：`train.py` 按 `enable ∧ episode_stream_sampler` 选 batch sampler，
   并把 `04-port-plan.md:78` 与 `config_holobrain_common.py:59` 的默认值对齐。**这一条不修，其余都是空谈。**
2. **加一条静默不了的护栏**：`stream` 模式下检测到 batch 内 episode 全不相同即告警/报错。
   P0-1 的本质是「失效时没有任何外部症状」，护栏比修复本身更值钱。
3. **补 deploy 分支的 `uuid`**（P2-1），并把 `reset()` 接进 `robodojo_eval.py` 的 episode 循环。
4. **重跑 E 档，走真实训练入口**（不用 `run_gears.py` 的自建 sampler），证明修复生效。
5. cite 改为符号锚点（P2-2）；`episode_stream_sampler` 默认值与文档对齐。

---

## 13. 给下次审查的提醒

- **最值钱的一条检查**：把「配置里出现的每一个开关」拿去 grep 消费者。
  本次 P0-1 就是一条 `grep -rn "episode_stream_sampler"` 找出来的，成本 10 秒，
  而它躲过了五档验证 + 三份文档 + 一次自评 PASS。
  **「声明了但没人读」是配置驱动系统里最廉价也最致命的 bug 类型。**
- **第二值钱**：读验证 harness 的源码，逐条问「它做了哪些被审代码没做的事」。
  凡是 harness 自己补上的集成，都是审查盲区。本次 `run_gears.py` 自建 sampler 即是。
- **最难查的一类**：本次是「分析对、实现对、验证过、唯独集成没做」。
  三份文档都正确预警了这个风险（`03:111`、`04:99`、`sampler.py:19-27` 的 docstring
  甚至精确描述了失效形态），**文档质量高反而制造了安全感**。
  → 审查时不要因为「文档里已经写了这个风险」就认为它被处理了；**要找那段处理它的代码。**
- **一个反直觉的观察**：本次移植在所有传统高危项上都是干净的（mask 极性、dtype、layout、
  detach 位置、原地操作、全局副作用、侵入度、许可证）。**质量高的移植，剩下的缺陷会集中在
  「接线」而不是「算法」**——下次可以把更多预算直接投给「端到端到底跑没跑通」。
