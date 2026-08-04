# 06 — 审查报告：MemoryVLA → HoloBrain（终版）

审查者独立执行 · 日期 **2026-08-04** · 全程只读：未修改宿主代码、`port/memoryvla` 分支、A repo 或任何环境

| 项 | 值 |
|---|---|
| 被审分支 | `port/memoryvla` @ `18106b05` |
| 受审**代码**实际止于 | `111981a5`（`18106b05` 经核实为纯文档：`git diff --name-only` 仅 5 个 `.md`） |
| 宿主基点 | `3ce31c0c` |
| A repo | `~/git_repo/MemoryVLA` @ `0eef5c39`（与自述 `0eef5c3` 一致 ✅） |
| 报告分支 | `review/memoryvla` |
| 分项报告 | `06a` 事实性 · `06b` 完整性 · `06c` 数值 · `06d` 接口语义 · `06e` 侵入度 · `06f` 训练动力学 |
| 临时证据 | `$ROL_JFS/port/memoryvla/review/`（复跑 JSON、脚本、基线 worktree，不进 git） |

---

## 1. 裁决

# 🔴 REJECT

**一句话理由**：`episode_stream_sampler` 是一个没有任何代码读取的死开关，配套 sampler
从未接入宿主任何执行路径——**已实测坐实**：在真实训练入口下打开 `memoryvla.enable`，
两个记忆库都是**精确恒等函数**（差 1～2 ULP），7,467,264 个新参数**一个都不会被更新**，
而 loss 正常、无告警、无 NaN。方法没有运行。

| 级别 | 数量 | 条目 |
|---|---:|---|
| **P0** | **1** | P0-1 死开关 → 真实路径上方法精确失效（**已实测**） |
| **P1** | **1** | P1-1 全部「记忆库在工作」的证据产自补齐了该缺失集成的自建 harness |
| **P2** | **3** | P2-1 deploy 分支缺 `uuid`；P2-2 cite 行号漂移 32 %；P2-3 命名违反自定约定 |
| **P3** | **2** | P3-1 D 档时间结论落在噪声内；P3-2 构造顺序使「开/关差值」不可解读为净效应 |

### 这个裁决该怎么读

REJECT 是按协议判据机械得出的（「存在 P0」）。但它**不是**「这份移植质量差」的意思——
恰恰相反，本次移植在几乎所有传统高危项上都是干净的：

- 方法本体 12/15 个要素与 A **逐行一致**，系数、`detach` 位置、norm 位置、
  PE 只加 key 不加 value 的非对称性，**一处没丢**
- 接口语义 **32 项核对一致、0 项不一致**，含协议点名的「最高频静默错误」mask 极性
- **A 档红线独立复跑 = 0.000000e+00**，关闭态新增参数 0、显存相同、batch key 相同
- **C 档 10/10 逐位一致**独立复现
- 侵入度 4 文件 × L1、8 个 hunk 全为纯增量、**零无关改动、零全局副作用**
- cite **零幻觉**

**缺的是一根接线，不是一块算法。** 修复成本极小（`train.py` 里几行 + 一条护栏），
修完重跑一次真实路径的 E 档即可翻案。之所以仍判 REJECT 而非 ACCEPT-WITH-FIXES，
是因为当前状态下**合入即意味着「打开开关什么也不会发生」，且没有任何信号会告诉使用者这件事**。

---

## 2. P0 清单

### P0-1 · `episode_stream_sampler` 是死开关，真实训练路径上方法精确失效

| | |
|---|---|
| **现象** | 配置键 `memoryvla.episode_stream_sampler=True` 在全仓**没有任何读取者**；`MemoryVLAEpisodeStreamBatchSampler` **从未被宿主实例化**；`train.py:124` 硬编码 `DistributedBatchFlagSampler`（全局随机排列）且**未被本次移植触及**。开启 `enable=True` 后记忆库退化为**精确恒等**，`retrieval_blocks` 与 `timestep_encoder` 拿不到梯度，**不报错** |
| **证据（静态）** | `$ROL_JFS/port/memoryvla/review/P0-1_dead_sampler_switch.txt`：① `grep -rn "episode_stream_sampler"` 全仓仅命中定义(`config_holobrain_common.py:59`)+注释+3 处文档，**无读取** ② `grep -rn "MemoryVLAEpisodeStreamBatchSampler"` 仅命中自身定义/`__all__`/包再导出/2 处 docstring，**无调用** ③ `git diff --name-only 3ce31c0c..port/memoryvla -- .../train.py \| wc -l` → **0** ④ `dataset_wrapper.py:132-134` 确为 `generator.permutation(n)` ⑤ `_build_memoryvla_cfg()` 不转发该键、`MemoryVLAMemory.__init__` 无该形参 ⑥ `run_gears.py:138-145` 由 harness 自行实例化 |
| **证据（实测，决定性）** | `review/decisive_host_sampler.json` vs `decisive_episode_sampler.json`，同 seed/config/batch/数据，**唯一变量是 sampler**：<br>　　　　　　　　　　　　**host（真实路径）** ／ episode（对照）<br>每 batch 不同 episode 数　　**4/4** ／ 1/4<br>恒等差 感知 `max\|out−in\|`　**1.192093e-07** ／ 1.473746e+00<br>恒等差 认知 `max\|out−in\|`　**5.960464e-08** ／ 1.154852e+00<br>grad `None` / 恰好 0 / **非零**　**64 / 4 / 0** ／ 0 / 0 / **68**<br>bank 长度　　　　　　　　　**恒为 [1]** ／ 4→8→12→**16 封顶**<br>参数实际移动（lr=1e-4）　　**0 / 68** ／ 66 / 68 |
| **代数解释** | `hist` 恒空 ⇒ `retrieved = working_mem` ⇒ `gate`：`s·w+(1−s)·w = w`（**与 s 无关**）；`add`：`(w+w)·0.5 = w`。两种融合都是精确恒等。`5.96e-08 = 2⁻²⁴`、`1.19e-07 = 2⁻²³` 即 float32 的 1～2 ULP 舍入 |
| **影响** | 任何人按 `PORT-STATUS.md:99` 的指引「训练时务必确认 `episode_stream_sampler=True`」照做后训练，得到的是 baseline + 7.47 M 个冻结死参数，并会把「指标没提升」归因为「这方法在我们数据上没用」。**这是最坏的失败形态：不是错，是白跑，且没有任何外部信号** |
| **加重情节** | `04-port-plan.md:58` 承诺 `config_holobrain_common.py` 的改动含「+ **可选 batch sampler 选择**」——**未实现**；`04-port-plan.md:78` 规定该键默认 **False**（语义「用宿主原 sampler」），而实际 ship 的是 **True**。默认值与设计文档相反，使问题从「缺失」升级为「误导」 |
| **建议修法** | 在 `train.py` 构造 `DataLoader` 处按 `enable ∧ episode_stream_sampler` 选择 batch sampler；并让 `MemoryVLAMemory` 在 `stream` 模式下检测到「batch 内 episode 数 == batch_size」时主动告警/报错，使该路径**结构上不可能再静默** |

---

## 3. P1 清单

### P1-1 · 「记忆库确实在工作」的全部证据产自补齐了缺失集成的自建 harness

| | |
|---|---|
| **现象** | B 档「68/68 张量有梯度」、E 档「跨 2 条 episode、bank 峰值 16、step 34 回落」——这些证明方法生效的关键数字，全部来自 `run_gears.py`，而它在 `:138-145` **自己 import 并实例化了 sampler**。宿主没有任何路径能做到这件事 |
| **证据** | `run_gears.py:61-62` 有 `--sampler {episode,sequential}` 且默认 `episode`；`:138-143` 直接构造 `MemoryVLAEpisodeStreamBatchSampler`。宿主侧零实例化（P0-1 证据②③）。**我用对照组独立复现了这些数字**（68/68 有梯度、bank 涨到 16 封顶）——**移植方测得没错，错在那条路宿主选不到** |
| **影响** | 这些数字不能迁移到真实训练路径。P0-1 之所以能躲过五档验证 + 三份文档 + 一次自评 PASS，根因就在这里：**验证装置替被审代码补上了它缺的那块集成，于是缺口被验证过程本身掩盖** |
| **为什么单列** | P0-1 是「缺陷是什么」，P1-1 是「为什么没被发现」。后者对下次移植的价值更大 |
| **建议修法** | 验证 harness 只允许走宿主已有的构造路径；harness 需要而宿主没有的东西，一律**先补进宿主再验** |

---

## 4. 验证结果对照表

| 档位 | **我实测** | PORT-STATUS 自述 | 一致？ |
|---|---|---|---|
| 第 0 步 确定性 | **0.000000e+00**（逐位可复现 → 用严格判据） | 0.000e+00 | ✅ |
| **A 关闭态等价** | **0.000000e+00**（20 步 × 7 分量）；参数 **+0**；显存相同；batch key **14 vs 14 相同** | 0.000e+00，参数量一致 | ✅ |
| A 档 · `num_workers=4` | **0.000000e+00**（阳性对照 3.81e-02 证明测试有效） | 未测 | ➕ 本次补 |
| A' 已有移植回归 | **N/A（首次移植）**，已确认 `MIGRATIONS.md` 仅 2 节 | 未提 | ✅ |
| B 开启态前向/梯度 | episode sampler **68/68 非零** ✅ ／ **真实 sampler 0/68** ❌ | 68/68，范数 8.39e-02 | ⚠️ 测量对，**不可迁移** |
| C 数值对齐 | **10/10 逐位一致**（0.000e+00） | 10/10 | ✅ |
| D 参数增量 | **+7,467,264（+0.657 %）** | +7,467,264（+0.66 %） | ✅ |
| D 显存增量 | **+0.3114 GiB** | +0.31 GiB | ✅ |
| D 时间 | **落在噪声内，结论不成立** | +10 % | ❌ P3-1 |
| D 开/关逐 step 差 | **6.203079e-02** | 6.203e-02 | ✅（吻合 4 位有效数字） |
| E Memory 冒烟 | episode sampler bank 4→8→12→**16 封顶** ✅ ／ **真实 sampler 恒为 1** ❌ | 峰值 16，step 34 回落 | ⚠️ 测量对，**不可迁移** |
| ckpt 兼容 | **1,000 → 1,068；新增 68 全在 `memoryvla.*`；removed 0 / reshaped 0 / unexpected 0** | 同 | ✅ |
| **真实 optimizer 分组** | **68 张量全进 group 1（`other_params`，base_lr, wd=5e-4）；0 张量掉在 optimizer 外** | **未测**（harness 用 lr=0 扁平列表） | ➕ 本次补，**结果正确** |
| **非零 lr 下参数是否更新** | episode **66/68 移动** ／ **真实 sampler 0/68** | **未测**（lr=0 无法测） | ➕ 本次补 |

**12 项自述里 9 项独立复现一致，1 项不成立，2 项测量正确但建立在宿主选不到的路径上。**
移植方没有编造数字。

---

## 5. 完整性结论（详见 `06b`）

| | 数量 |
|---|---:|
| 论文/A 侧方法要素 | 15 |
| **已移植且与 A 逐行一致** | **12** |
| 声明不移植且理由成立 | 2（`BottleneckSE` 未接入、DiT 动作头） |
| **静默缺失** | **1**（episode 有序数据流 → P0-1） |
| 多搬 | **0** |

「最容易静默丢掉」六类**逐类通过**：公式系数（ToMe `0.5`、add `*0.5`、`max_period=10000`、
`freq_emb=token_size//4`）· `detach`/`no_grad` 位置（5 处逐一对应）· warmup（A 无此逻辑，N/A）·
norm 位置（post-norm 逐字一致）· loss 归约（不产生 loss，N/A）· `__init__` 隐式副作用（已剥离且未连带丢功能）。

---

## 6. 接口语义结论（详见 `06d`）

| 类别 | 已核对一致 | 不一致 | 未确认 |
|---|---:|---:|---:|
| 动作语义 / 归一化 / 时序 / mask·padding / 张量 / 常量 / 额外三项 | **32** | **0** | **1** |

**零不一致。** mask 极性用**构造端**（`structure.py:334-342`：`~img_feature_mask & not_pad_mask`）
与**消费端**（三处消费者一律 `key_padding_mask=~mask`）两条独立路径交叉验证，移植方判断正确。
`uuid` per-episode、`step_index` episode 内步序、permute 往返可逆、`scatter` 非原地、
零硬编码 device，均独立坐实。

**唯一 `未确认`**：A 的采样频率/降采样策略。影响中等偏低（PE 编的是整数 step 序号，
实现不会错；受影响的是 `mem_length=16` 的物理时长语义 → 属调参非 bug）。

---

## 7. 侵入度实况（详见 `06e`）

| 文件 | 自述 | **实际** | 偏差 |
|---|---|---|---|
| `models/holobrain/structure.py`（3 hunk） | L1 | **L1** | 无 |
| `models/holobrain/structure_qwen3_5.py`（1 hunk） | L1 | **L1** | 无 |
| `configs/data_configs/config_robodojo_dataset.py`（1 hunk） | L1 | **L1** | 无 |
| `configs/config_holobrain_common.py`（3 hunk） | L1 | **L1** | 无 |

**无虚报。** 8 个 hunk 全为纯增量（+67/−0），形状均为「一个开关判断 + 一次调用」，
实现体全在子目录内，**零无关改动**。全局副作用扫描（seed / dtype / device / backends /
hook / 原地操作 / import 期动作 / 多搬 A 基础设施）**全部为零**。

---

## 8. 事实性（详见 `06a`）

**抽样 38 条：命中 26 / 行号漂移 12（32 %）/ 幻觉 0。**
协议的「≥1 条内容不符 → 翻倍重抽」**未触发**。Gate A 形式判据成立
（`03-interface-diff.md` 无「既无 cite 又无未确认」的事实行），
且读不到的项确实写 `未确认` 而非拿常识充数。

漂移有结构：指向 A 的**类定义行 13/13 精确**；指向宿主 `structure.py` 的**中段行 +3~+8**，
且与基点、HEAD **两个版本都对不上**（本次移植对该文件只增不删，故「代码后来推移」的
善意假设不成立）→ 行号是估的，不是机械提取的。

---

## 9. 无法验证的部分

| 项 | 为什么验不了 | 该怎样才能验 |
|---|---|---|
| **外部真实 ckpt 加载** | 已尝试并失败：bucket 上只有 **v9** 权重（`checkpoint_20000`，2.84 GB），当前 config 默认 **v10**，`vlm.*` 全线 size mismatch（`[3840,1280]` vs `[3072,1024]` 等），**与 MemoryVLA 无关**。顺带**独立证实**移植方「v10 warm-start 在 http URL 后、本机无外网」属实（`config_holobrain_common.py:117`）——他们自造 state_dict 是合理的 | ① config 切回 v9 段后用 `checkpoint_20000`；或 ② 在有外网的机器上取 v10 checkpoint_60。**风险低**：key 集合差分已给出实质答案，且非循环 |
| **DDP / 多卡 unused-parameter** | 本机硬约束：任意两卡 gather 必崩 `ILLEGAL_ADDRESS`。且 P0-1 使该风险在真实路径上**必然发生**（`retrieval_blocks` 恒不参与计算 = 恒为 unused） | 换多卡可用的机器，或 AIDI 上起 2 卡 job。**修完 P0-1 后必须重验**——届时才知道真实数据分布下会不会撞上 |
| **A 的采样频率 / 降采样** | A repo 内只有消费端形参（`memory_vla.py:488`），定义端在 RLDS 管线外部；A 与宿主数据不同源，无法对跑 | 读 A 的 RLDS builder 的 step 定义或论文附录数据处理节 |
| **A 与宿主端到端数值可比性** | **原理上不可比**，非工程问题：A 的感知记忆记 LLM **之前**的视觉 patch，宿主记 VLM **之后**已被语言条件化的特征 | 不可验证。只能做宿主内「加/不加记忆库」的 A/B，不能做跨 repo 数值对齐 |
| **D 档墙钟时间** | 同一张卡上有同事进程（实测 8 张卡全部有占用），墙钟不可比 | 独占一张卡重测；或改用 CUDA event 计时并多次取中位数 |
| **`fifo` vs `tome` 的实际差异** | 移植方自陈 8 step 太短不可区分；本次未延长（P0-1 使该比较在真实路径上无意义） | 修完 P0-1 后跑到 episode 尺度再比 |

---

## 10. 给移植方的最短修复路径（按「修完能翻案」排序）

1. **接上 sampler**（P0-1）：`train.py` 构造 `DataLoader` 处按 `enable ∧ episode_stream_sampler`
   选 batch sampler；并把 `04-port-plan.md:78`（默认 False）与
   `config_holobrain_common.py:59`（实际 True）对齐。**这一条不修，其余都是空谈。**
2. **加一条静默不了的护栏**：`stream` 模式下检测到 batch 内 episode 全不相同即告警/报错。
   P0-1 的本质是「失效时没有任何外部症状」——**护栏比修复本身更值钱**，
   因为它让这一类错误将来不可能再无声发生。
3. **重跑 E 档，走真实训练入口**（禁止用 `run_gears.py` 的自建 sampler），
   并把「每 batch 不同 episode 数」「`retrieval_blocks` grad norm」两项写进验收输出。
   可直接复用 `$ROL_JFS/port/memoryvla/review/rev_decisive.py`。
4. **补 deploy 分支的 `uuid`**（P2-1），并把 `reset()` 接进 `robodojo_eval.py` 的 episode 循环
   ——目前推理/部署路径从未被实际执行过。
5. **修完后单独验一次 DDP**：P0-1 修好之前 `retrieval_blocks` 恒为 unused parameter，
   这个风险被 P0-1 掩盖着，修完才会真正暴露。

---

## 11. 给下次审查的提醒

- **最值钱的一条检查（成本 10 秒）**：把配置里出现的**每一个开关**拿去 grep 消费者。
  P0-1 就是一条 `grep -rn "episode_stream_sampler"` 找出来的，
  而它躲过了五档验证 + 三份文档 + 一次自评 PASS。
  **「声明了但没人读」是配置驱动系统里最廉价也最致命的 bug 类型。**
- **第二值钱**：读验证 harness 的源码，逐条问「它做了哪些被审代码没做的事」。
  凡是 harness 自己补上的集成，都是审查盲区。本次 `run_gears.py` 自建 sampler 即是。
  推论：**harness 与真实入口的每一处差异（sampler / `num_workers` / `lr` / optimizer 构造），
  都要单独列成一条待验项**——本次的「补三项空白」正是这么来的，
  其中 optimizer 分组这一项查完是**正确**的，也值得记：不是每条空白都藏着 bug，
  但每条都必须查，因为事先分不清哪条藏着。
- **设计实验时优先找「单次运行内的自证」**：本次决定性实验没有跨 run 比较，
  而是在同一次运行里包住模块量 `max|out − in|`。这绕开了「开启态构造顺序改变 RNG」
  这个混淆因子（P3-2）——若我用「开 vs 关」两次 run 比 loss，会被主干初始权重不同污染，
  得不出干净结论。**先写下事前预测再跑**，四项全中时结论才立得住。
- **给测试配阳性对照**。补空白③（worker 随机流）拿到 0.000e+00 时，
  这个 0 本身没有信息量——直到我证明「同一棵树 workers=0 vs 4 差 3.81e-02」，
  才知道这条流水线确实 worker 敏感、这个 0 是有牙的。**没有阳性对照的通过 = 未验证。**
- **最难查的一类**：本次是「分析对、实现对、验证过，唯独集成没做」。
  三份文档都**正确预警了**这个风险（`03:111`、`04:99`，`sampler.py:19-27` 的 docstring
  甚至精确写出了失效形态「silently degenerates to an identity-ish transform」），
  **文档质量高反而制造了安全感**。
  → 看到文档里写着某个风险，**不要认为它被处理了；去找那段处理它的代码。**
- **一个反直觉的观察**：本次移植在所有传统高危项上都干净（mask 极性、dtype、layout、
  detach 位置、原地操作、全局副作用、侵入度、optimizer 分组、许可证）。
  **质量越高的移植，剩下的缺陷越会集中在「接线」而不是「算法」**
  ——下次可以把更多预算直接投给「端到端到底跑没跑通」，而不是逐行比对算子。
