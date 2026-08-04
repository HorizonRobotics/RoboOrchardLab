# 06e — 侵入度与回归风险（逐 hunk）

审查者独立执行 · 日期 2026-08-04 · `git diff 3ce31c0c..port/memoryvla`

改动规模：**4 个宿主已有文件 + 8 个 hunk，+67 行，0 删除**；新增文件 6 个（4 代码 + 1 config spec + docs）。

---

## 1. 逐 hunk 判定

| # | 文件 | hunk | 内容 | 自述 | **实际** | 问题 |
|---|---|---|---|---|---|---|
| 1 | `models/holobrain/structure.py` | `@@ -123,6 +123,7` | `self.memoryvla = build(self.cfg.memoryvla)` | L1 | **L1** | 见 §2.1 |
| 2 | 同上 | `@@ -442,6 +443,11` | `if self.memoryvla is not None:` + 一次调用 | L1 | **L1** | — |
| 3 | 同上 | `@@ -560,6 +566,7` | Config 加 `memoryvla: MODULE_TYPE \| None = None` | L1 | **L1** | — |
| 4 | `models/holobrain/structure_qwen3_5.py` | `@@ -68,6 +68,7` | 同 hunk 1（该类跳过父类 `__init__`） | L1 | **L1** | — |
| 5 | `configs/data_configs/config_robodojo_dataset.py` | `@@ -285,6 +285,13` | 开关打开时给 `ItemSelection` 白名单加 `step_index` | L1 | **L1** | 见 §2.2 |
| 6 | `configs/config_holobrain_common.py` | `@@ -40,6 +40,24` | `memoryvla=dict(...)` 命名空间，默认 `enable=False` | L1 | **L1** | 见 §3 |
| 7 | 同上 | `@@ -114,6 +132,39` | 新增 `_build_memoryvla_cfg()` | L1 | **L1** | — |
| 8 | 同上 | `@@ -235,6 +286,7` | `memoryvla=_build_memoryvla_cfg(config)` 传给 `model_config` | L1 | **L1** | — |

**自述 L1 × 4 文件 = 实际 L1 × 4 文件，无虚报。** 全部为**纯增量**，无一处修改或删除已有语句
（`git diff --stat` 显示 0 删除，逐 hunk 目视确认属实）。

**形状检查**：hunk 2 是「一个开关判断 + 一次调用」，实现体全在 `robo_orchard_lab/models/memoryvla/`
子目录内，**没有把逻辑铺开写进 `_forward`**。符合协议要求的形状。

**无关改动检查**：8 个 hunk 全部与 MemoryVLA 相关，**没有混入**格式化、import 重排、
重命名或顺手重构。这一点重要——它让「原逻辑没被动过」保持可验证。

---

## 2. 两处需要单独说明的 hunk

### 2.1 hunk 1/4 —— 构造顺序被改变（开启态才有影响）

`self.memoryvla = build(...)` 被插在 `self.data_preprocessor` **之前**：

```
self.decoder = build(self.cfg.decoder)
self.spatial_enhancer = build(self.cfg.spatial_enhancer)
+self.memoryvla = build(self.cfg.memoryvla)          ← 插在这里
self.data_preprocessor = build(self.cfg.data_preprocessor)
```

**关闭态**：`cfg.memoryvla=None` → `build(None)` 返回 `None`，**不消耗全局 RNG** → 无影响。
这正是 A 档能过的机制，设计是对的。

**开启态**：构造 `MemoryVLAMemory` 会抽取全局 RNG，**其后所有模块的初始化随机数整体位移**。
后果不是「错」，而是「开启态与关闭态的主干初始权重不可比」——
比较「开/关」两次 run 的 loss 差时，差异里混着**主干初始化不同**这一项，
不全是记忆库带来的。移植方 B 档报的「开/关最大逐 step 差 6.203e-02」因此**不能**
解读为「记忆库的净效应」。

**定级 P3**（不影响正确性，影响的是对 B 档数字的解读）。
若要干净地量化净效应，应把 `build` 放在所有原有模块**之后**，或在构造前后存取 RNG 状态。

### 2.2 hunk 5 —— 白名单补丁覆盖不全（deploy 分支缺 `uuid`）

补丁放在 `if/elif` 链**之后**，对当前绑定的 `item_selection` 追加 `step_index`：

```python
if (config.get("memoryvla") or {}).get("enable", False):
    # `uuid` is already whitelisted; `step_index` is produced by the dataset
    # but dropped here, so add it back ...
    item_selection["keys"] = list(item_selection["keys"]) + ["step_index"]
```

放在链后是**聪明的**——一处代码覆盖 training/validation/deploy 三个分支，
`step_index` 三处都补上了。**但注释里那句「`uuid` is already whitelisted」只对两处成立**：

| 分支 | 定义处 | 含 `uuid`？ |
|---|---|---|
| `mode == "training"` | `config_robodojo_dataset.py:187-225` | ✅ 有（`:202`） |
| `mode == "validation"` | `:227-257` | ✅ 有（`:240`） |
| **`mode == "deploy"`** | **`:259-273`** | ❌ **没有** |

后果：`memoryvla.enable=True` 且 `mode="deploy"` 时，
`MemoryVLAMemory._episode_ids` 在 `wrapper.py:161` 抛 `KeyError`。

**定级 P2**（`06-review-report.md` 记为 P2-1）。**这是响亮的崩溃，不是静默错误**，
按协议「会崩的优先级天然低」。但它与移植方自陈的遗留 1（`reset()` 未接进评测循环）叠加，
说明**推理/部署路径从未被实际执行过**。

---

## 3. P0-1 —— `episode_stream_sampler` 是死开关，方法在真实训练路径上不生效

> **这一条是本次审查最重的发现。它不是「改坏了什么」，而是「开关打开后方法根本没运行」，
> 且不报错、loss 正常、日志干净。**

### 3.1 证据链（全部独立取证，命令与输出存于 `review/P0-1_dead_sampler_switch.txt`）

**① 配置键没有任何消费者**

```
$ grep -rn "episode_stream_sampler" . --exclude-dir=.git --exclude-dir=__pycache__
config_holobrain_common.py:50    # spans a batch. Needs episode_stream_sampler.   ← 注释
config_holobrain_common.py:59    episode_stream_sampler=True,                     ← 定义
docs_analysis/memoryvla/04-port-plan.md:78   ...
docs_analysis/memoryvla/PORT-STATUS.md:57    ...
docs_analysis/memoryvla/PORT-STATUS.md:99    ...
```

代码里只有**定义**和**注释**，**没有读取**。
唯一消费 `config["memoryvla"]` 的函数是 `_build_memoryvla_cfg()`，
它返回的 dict **不含** `episode_stream_sampler`（我逐字读过该函数全文），
而 `MemoryVLAMemory.__init__` 也**没有**这个形参。

**② sampler 类从未被宿主实例化**

```
$ grep -rn "MemoryVLAEpisodeStreamBatchSampler" . --exclude-dir=.git --exclude-dir=__pycache__
sampler.py:41    __all__ = [...]        ← 自己的导出
sampler.py:92    class ...              ← 自己的定义
sampler.py:143   logger.info(...)       ← 自己的日志字符串
memoryvla/__init__.py:34,46             ← 包的再导出
wrapper.py:70                           ← docstring 提及
02-host-seams.md:114                    ← 文档提及
```

**没有任何一处 `MemoryVLAEpisodeStreamBatchSampler(...)` 的调用。**

**③ 真实训练入口把宿主 sampler 写死了**

```python
# projects/holobrain_internal/common/train.py:117-131
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, num_workers=num_workers, ...
    batch_sampler=DistributedBatchFlagSampler(      # ← 硬编码，无分支
        train_dataset, config["batch_size"], drop_last=True,
        dataset_sample_weights=config.get("dataset_sample_weights"),
    ),
)
```

且 **`train.py` 不在本次改动的文件清单里**：
```
$ git diff --name-only 3ce31c0c..port/memoryvla -- projects/holobrain_internal/common/train.py | wc -l
0
```

**④ 宿主 sampler 是全局随机排列**

```python
# robo_orchard_lab/dataset/dataset_wrapper.py:132-134
generator = np.random.default_rng(seed=self.seed + self._epoch)
yield from generator.permutation(n)
```

**⑤ 移植方的 harness 自己补上了这个缺失的集成**

```python
# $ROL_JFS/port/memoryvla/tools/run_gears.py:138-145   （不在 git 内）
if args.sampler == "episode":
    from robo_orchard_lab.models.memoryvla import MemoryVLAEpisodeStreamBatchSampler
    batch_sampler = MemoryVLAEpisodeStreamBatchSampler(dataset, args.batch_size, seed=args.seed)
```

### 3.2 后果推演（为什么它是静默的）

真实训练路径下 `enable=True` 时：

1. 每个 batch 的 16 帧来自 ~16 个**随机不同**的 episode。
2. `CogMemBank.process_batch` 的 stream 分支（`memory_bank.py:378-381`）：
   `if episode_ids[i] != episode_ids[i-1]: clear_episode(episode_ids[i-1])`
   —— 随机顺序下相邻两样本几乎必然不同 episode，**上一条刚写进去就被清掉**。
3. 于是 `hist = self.bank.get(eid, [])` **恒空** → 走 `memory_bank.py:410-412` 的旁路：
   `retrieved_episode_mem = working_mem`。
4. 融合退化为 `fusion(working_mem, working_mem)`，而**两种融合方式在此都是精确恒等**：

   | 融合 | 公式 | x1=x2=w 时 |
   |---|---|---|
   | `gate` | `scale*x1 + (1-scale)*x2`，`scale=sigmoid(proj([x1;x2]))` | `scale*w + (1-scale)*w = w` **恒等，与 scale 无关** |
   | `add` | `(x1+x2)*0.5` | `(w+w)*0.5 = w` **恒等** |

5. ⇒ 感知与认知两个记忆库都是**精确恒等函数**，模型输出与 baseline 数值相同；
   `retrieval_blocks`（7.47 M 新参数的主体）**拿不到梯度**。

**没有异常、没有警告、没有 NaN、loss 曲线正常。**
唯一的外部症状是「加了记忆库但指标没提升」——会被归因为「这方法在我们数据上没用」，
而不是「这方法根本没运行」。

> 相邻位置恰好撞上同一 episode 的概率约 `15/600 ≈ 2.5%`/batch，
> 因此并非 100% 恒等，但 >99% 的样本走恒等旁路。这不改变结论。

### 3.3 为什么移植期没发现——这才是最该记住的部分

移植方**完整地识别了这个风险**，并写进了两份文档：

- `03-interface-diff.md:111`：「现有 sampler 是全局随机排列（cite: `dataset_wrapper.py:133`）
  → **需新 sampler**」——分析正确，cite 也对。
- `04-port-plan.md:99` 风险表：「`stream` 模式要求 episode 连续批 → 新 sampler + **E 档**冒烟」。
- `sampler.py:19-27` 自己的 docstring：「with shuffled frames the bank ...
  **silently degenerates to an identity-ish transform**」——**后果描述得完全正确**。

**分析对、实现对、验证也「过」了——唯独集成没做。**
而验证之所以会过，是因为 **`run_gears.py` 自己实例化了 sampler**。
验证装置补上了被审代码缺失的那块集成，于是缺口被验证过程本身掩盖。

两处硬证据表明这是**漏做**而非**有意为之**：

1. `04-port-plan.md:58` 承诺 `config_holobrain_common.py` 的改动含
   「+ **可选 batch sampler 选择**」——`git diff` 里没有这段代码。
2. `04-port-plan.md:78` 规定 `episode_stream_sampler` 默认 **False**（语义「用宿主原 sampler」），
   而实际 ship 的 `config_holobrain_common.py:59` 写的是 **True**。
   默认值与设计文档相反，且无论真假都没有消费者。

第 2 点使问题**从「缺失」升级为「误导」**：配置里明晃晃写着 `episode_stream_sampler=True`，
任何人读到都会认为流式顺序已生效。`PORT-STATUS.md:99` 更把它写成给使用者的行动项
（「训练时务必确认 `episode_stream_sampler=True`」）——**照做会得到虚假的安心**。

### 3.4 定级与建议

**P0**。判据：属于协议 P0 的「开关打开时方法失效」，且是纯静默。

**建议修法**（一句话，不给代码）：在 `train.py` 构造 `DataLoader` 处按
`config["memoryvla"]` 的 `enable ∧ episode_stream_sampler` 选择 batch sampler，
并让 `MemoryVLAMemory` 在 `dataloader_type="stream"` 却收到跨 episode 混排 batch 时
**主动报错或告警**（例如首个 batch 检测到 `len(set(episode_ids)) == batch_size` 即警告），
使这条路径**不可能再静默**。

---

## 4. 全局副作用扫描（关闭态也生效的东西，A 档抓不到）

```
$ git diff 3ce31c0c..HEAD | grep -nE "^\+.*(manual_seed|set_default_dtype|set_default_device|
                                     torch\.backends|register_(forward|full_backward)_hook|set_grad_enabled)"
→ （无匹配）
```

| 检查 | 结果 |
|---|---|
| 改种子 / 默认 dtype / 默认 device / `torch.backends` | ✅ **无** |
| 注册 forward / backward hook | ✅ **无** |
| 新子目录 `__init__.py` 在 import 期执行动作 | ✅ **无**（只有 docstring、`from ... import ...`、`__all__`） |
| 硬编码 `.cuda()` / `.cpu()` | ✅ **无** |
| 原地操作 `add_/mul_/div_/clamp_/copy_/scatter_/masked_fill_` | ✅ **无** |
| 就地改上游 batch dict / 共享张量 | ✅ **无**（先 `list()`/`dict()` 浅拷贝再写，见 06d §5） |
| 多搬 A 的基础设施（argparse/accelerate/deepspeed/wandb/FSDP/overwatch） | ✅ **无** |

**这一节全部通过。** 关闭态的行为不变有结构性保证（`build(None) → None`，模块不存在），
而不只是靠 A 档的经验数字。

---

## 5. 宿主兼容性四条（静态部分；数值部分见第二批 06f）

| # | 要求 | 结论 | 依据 |
|---|---|---|---|
| ① | 新增参数带前缀、**关闭时不构建** | ✅ **静态成立** | 参数全在 `self.memoryvla.*` 命名空间下；关闭时 `build(None)→None`，模块不存在。**参数量为 0 的数值证明留待 06f** |
| ② | 新参数进哪个 optimizer group 显式指定，现有 lr/wd 分组未变 | ⚠️ **静态存疑，待 06f 实测** | 宿主分组在 `config_holobrain_common.py:536-556`，按 `if "vlm." in name` / `p.dtype` 分三组。`memoryvla.*` 不含子串 `vlm.`（`memoryvla` 里是 `vla.`），故应落 `other_params`/`bit16_params`，得 `lr=base_lr, wd=5e-4`。**但移植方的 harness 用的是 `lr=0` 的扁平参数列表，这段真实分组代码在整个移植验证里一次都没被执行过** |
| ③ | 新 dataloader 字段只在开关打开时产出、collate 处理缺失 | ✅ **成立（但覆盖不全）** | 追加受 `if enable` 保护，关闭态白名单一字不变；`step_index` 的 int/np.int64 dtype 不一致已在 `wrapper.py:188-193` 显式两收。**deploy 分支缺 `uuid`，见 §2.2** |
| ④ | 新 metric 前缀化、不覆盖已有 key | ✅ **N/A 且成立** | 记忆库**不产生任何 metric 或 loss 分量**，无 key 可撞 |

---

## 6. 已移植方法的正交性

本仓库**首次移植**：`docs_analysis/MIGRATIONS.md` 只有 2 个 `## ` 小节
（「已经定下来的约定」+「1. MemoryVLA」），无第二个方法。
⇒ 协议的 **A' 档记 `N/A（首次移植）`**，不是跳过。

---

## 7. 小结

| 维度 | 结论 |
|---|---|
| 侵入度自述 vs 实际 | **完全一致**，4 文件 × L1，无虚报 |
| 形状 | ✅ 一个 if + 一次调用，实现体在子目录内 |
| 无关改动 | ✅ 零 |
| 全局副作用 | ✅ 零 |
| 宿主兼容性四条 | ①③④ 成立（③ 覆盖不全 → P2-1），② 待实测 |
| **方法是否真的接入** | ❌ **否 → P0-1** |

**侵入度这一维度，移植方做得干净且诚实。** 问题不在「动了不该动的」，
恰恰相反——**该动的一处（`train.py` 的 sampler 选择）没动**。
