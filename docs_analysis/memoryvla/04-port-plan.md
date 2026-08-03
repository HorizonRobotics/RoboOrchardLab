# 04 — 移植方案

基点 `3ce31c0c` · 分支 `port/memoryvla` · 依赖档位 **E0**（宿主主环境零改动，差异包清单为空）

## 侵入度：**L1**

自低向高逐档论证为什么停在 L1：

| 档 | 能不能做到 |
|---|---|
| **L0** 纯新增 | **做不到**。记忆库必须拿到 `_forward` 里 `_vlm_outputs_handler` 的输出，那是局部变量；宿主没有 hook / 注册点能在该处插入（cite: `robo_orchard_lab/models/holobrain/structure.py:443-458`） |
| **L1** 子类继承 / 一个 if + 一次调用 | **采用**。宿主已有同款可选子模块写法 `spatial_enhancer: MODULE_TYPE \| None = None`（cite: `structure.py:561`），照抄即可 |
| L2 多处开关分支 | 不需要 |
| L3 改基类签名 / 默认值 / ckpt 加载 / 分布式 | **不触发** → **不触发 Gate B** |

**Gate B 结论：无 L3 改动，无需停下等确认。**

## 改动清单

| 文件 | 新增/修改 | 侵入度 | 行数 | 回滚方式 |
|---|---|---|---|---|
| `robo_orchard_lab/models/memoryvla/memory_bank.py` | 新增 | L0 | 445 | 删目录 |
| `robo_orchard_lab/models/memoryvla/wrapper.py` | 新增 | L0 | 274 | 删目录 |
| `robo_orchard_lab/models/memoryvla/sampler.py` | 新增 | L0 | 179 | 删目录 |
| `robo_orchard_lab/models/memoryvla/__init__.py` | 新增 | L0 | 47 | 删目录 |
| `projects/holobrain_internal/common/configs/dataset_specs_memoryvla_robodojo_memory.py` | 新增 | L0 | ~40 | 删文件 |
| `docs_analysis/memoryvla/*.md` + `docs_analysis/MIGRATIONS.md` | 新增 | L0 | — | 删目录 |
| **`robo_orchard_lab/models/holobrain/structure.py`** | 修改 | **L1** | +6 | 见下 |
| **`robo_orchard_lab/models/holobrain/structure_qwen3_5.py`** | 修改 | **L1** | +1 | 见下 |
| **`projects/holobrain_internal/common/configs/data_configs/config_robodojo_dataset.py`** | 修改 | **L1** | +6 | 见下 |
| **`projects/holobrain_internal/common/configs/config_holobrain_common.py`** | 修改 | **L1** | +~25 | 见下 |

**触及的宿主已有文件共 4 个**（Phase 6 同步进 `MIGRATIONS.md`）。
比 `02-host-seams.md` 初判多一个 `config_holobrain_common.py`（新增 `cfg.memoryvla.*` 命名空间），
少一个 `collates.py`（实测无需改，见 `02-host-seams.md` §2.3）。

### 每处的确切形状

**`structure.py`** —— 三处、共 6 行：

1. Config 加一个字段（紧挨 `spatial_enhancer`）：`memoryvla: MODULE_TYPE | None = None`
2. `__init__` 加一行：`self.memoryvla = build(self.cfg.memoryvla)`
3. `_forward` 加一个 if：
   ```python
   if self.memoryvla is not None:
       feature_maps, text_dict = self.memoryvla(feature_maps, text_dict, inputs)
   ```

**`structure_qwen3_5.py`** —— 1 行，同第 2 条。
必须单独加，因为它的 `__init__` 用 `super(HoloBrain_Qwen2_5_VL, self).__init__(cfg)`
**跳过了父类 `__init__`**、自己重列了一遍 build（cite: `robo_orchard_lab/models/holobrain/structure_qwen3_5.py:69-75`）。
这是本次唯一需要改两遍的地方，也是最容易漏的一处。

**`config_robodojo_dataset.py`** —— 在 `build_transforms` 里，开关打开时把 `"step_index"`
加进 `ItemSelection` 白名单（training / validation / deploy 三处），**关闭时白名单一字不变**。

**`config_holobrain_common.py`** —— 新增 `cfg.memoryvla.*` 命名空间（默认全关）+
把它接到 `model_config(...)` 的 `memoryvla=` 参数 + 可选 batch sampler 选择。

**回滚**：四处改动都是「加一个 if / 加一行 / 加一个字段」，`git revert` 单个 commit 即可；
新增文件删目录即可。没有任何一处修改了已有语句。

## 新增 config 字段

| 名 | 类型 | 默认值 | 关闭语义 | 来源 |
|---|---|---|---|---|
| `memoryvla.enable` | bool | **False** | `False` → `memoryvla=None`，**模块根本不构建** | 本次新增 |
| `memoryvla.use_perceptual` | bool | True | 不建感知 bank | A 的 `per_mem_bank`（cite: `MemoryVLA@0eef5c3 vla/memory_vla.py:416`） |
| `memoryvla.use_cognitive` | bool | True | 不建认知 bank | A 的 `cog_mem_bank`（cite: `vla/memory_vla.py:404`） |
| `memoryvla.dataloader_type` | str | **`stream`** | — | A 默认 `group`（cite: `vla/memory_vla.py:369`），但 `group` 的记忆跨度只有一个 batch（实测，见 `01b`），论文语义对应 `stream` |
| `memoryvla.group_size` | int | 16 | 仅 `group` 用 | A 默认 16（cite: `vla/memory_vla.py:370`） |
| `memoryvla.mem_length` | int | 16 | — | A 默认 16（cite: `vla/memory_vla.py:372`） |
| `memoryvla.retrieval_layers` | int | 2 | — | A 默认 2（cite: `vla/memory_vla.py:373`） |
| `memoryvla.use_timestep_pe` | bool | True | 不建 `TimestepEmbedder`，也不需要 `step_index` | A 默认 True（cite: `vla/memory_vla.py:374`） |
| `memoryvla.fusion_type` | str | `gate` | `add` 时无 `GateFusion` 参数 | A 默认 `gate`（cite: `vla/memory_vla.py:375`） |
| `memoryvla.consolidate_type` | str | `tome` | — | A 默认 `tome`（cite: `vla/memory_vla.py:376`） |
| `memoryvla.update_fused` | bool | False | — | A 默认 False（cite: `vla/memory_vla.py:377`） |
| `memoryvla.episode_stream_sampler` | bool | False | 用宿主原 sampler | 本次新增；`stream` 模式下应设 True |

`token_size` 不做成字段：直接取 `config["embed_dims"]`，避免两处配置漂移。

## 与已移植方法的关系

本仓库**首次移植**，`docs_analysis/MIGRATIONS.md` 之前不存在 → 正交性检查记 `N/A（首次移植）`。

## 已经完成并验证的部分（写在这里以免与实际状态脱节）

新增的 4 个模块文件已落盘并通过 **C 档**：10 个靶子**全部逐位一致**（`max|diff| = 0.0`），
包括 `BottleneckSE` 改写后在 `hw=(8,8)` 下与原版逐位相同、且 `8×11` 非方形输入不再触发 assert。
详见 `06-verification.md`。**此时宿主已有文件尚未改动，改动从下一个 commit 开始。**

## 风险点与验证手段

| 风险 | 验证 |
|---|---|
| 新模块初始化吃掉全局 RNG → 关闭态数值漂移 | **A 档**：`enable=False` 时不构建模块，与 baseline 逐 step 比 |
| 感知特征语义不同（VLM 后 vs LLM 前） | 无法用数值证明，只能记录；写进最终汇报风险项 |
| 认知记忆只改 1 个 token，影响被稀释 | **B 档**打印记忆分量与 grad norm 确认确实有梯度流 |
| `stream` 模式要求 episode 连续批 | 新 sampler + **E 档**冒烟必须跨过 episode 边界 |
| 一批内全部样本无历史 → DDP unused parameter | **B 档**实测；本机单卡跑不了 DDP，记为未验证风险 |
| `step_index` dtype 不一致（int / np.int64） | wrapper 同时接受 Tensor 与 list，已在实现里处理 |
| batch 降到 1 时 `group` 失效 | 降档必须同时切 `stream`，写进 STATUS |
