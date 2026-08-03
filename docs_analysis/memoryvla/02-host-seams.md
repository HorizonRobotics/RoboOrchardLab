# 02 — 宿主接缝勘察

宿主基点 `3ce31c0c`。本文所有 `文件:行` 均指该 commit。
**本仓库此前没有任何移植记录**（`docs_analysis/MIGRATIONS.md` 不存在），
所以「沿用上次移植的位置」这条不适用，本次要负责把位置定下来。

---

## 1. 挂载点优先级走查

协议给的优先级：`之前移植用过的位置 → config 字段 → 已有注册/工厂机制 → 基类可 override 的方法 →
已有 if-else 分支 → 新开一处`。逐条走：

| 优先级 | 宿主有没有 | 结论 |
|---|---|---|
| 之前移植用过的位置 | 无（首次移植） | 跳过 （判断，依据本文上方已 cite 的事实） |
| **config 字段** | **有**：`HoloBrain_Qwen2_5_VLConfig` 已有 `spatial_enhancer: MODULE_TYPE \| None = None`（cite: `robo_orchard_lab/models/holobrain/structure.py:561`）与 `backbone_3d` / `neck_3d` / `data_preprocessor` 同款 | **采用**，照抄这个形状加 `memoryvla` 字段 |
| 已有注册/工厂机制 | 数据集侧有（`@train_dataset_register` / `processor_register`，cite: `projects/holobrain_internal/common/configs/data_configs/config_robodojo_dataset.py:291-292`）；模型侧是 `build(cfg)`（cite: `structure.py:124`） | 数据 spec 走注册机制**新开文件**；模型走 `build` |
| 基类可 override 的方法 | `HoloBrain_Qwen3_5_VL` 就是靠继承复用 `_forward`（cite: `structure_qwen3_5.py:57`） | **关键**：改基类 `_forward` 一处，v9/v10 同时生效 |
| 已有 if-else 分支 | `_forward` 里已有 `if self.spatial_enhancer is not None:`（cite: `structure.py:449`） | **紧挨着它加同款分支** |
| 新开一处 | — | 只有 batch sampler 需要新开 （判断，依据本文上方已 cite 的事实） |

---

## 2. 五处接缝

### 2.1 模型侧（**唯一的模型改动**）

`_forward`（cite: `structure.py:415-465`）当前形状：

```python
feature_maps, text_dict = self._vlm_outputs_handler(...)   # :443-445
feature_3d = self.extract_feature_3d(inputs)               # :447
if self.spatial_enhancer is not None:                      # :449  ← 已有的同款分支
    feature_maps, depth_prob, loss_depth = self.spatial_enhancer(...)
else:
    depth_prob = loss_depth = None
model_outs = self.decoder(feature_maps=..., text_dict=..., ...)   # :458
```

插入点在 `:445` 与 `:447` 之间，形状为「一个 if + 一次调用」：

```python
if self.memoryvla is not None:
    feature_maps, text_dict = self.memoryvla(feature_maps, text_dict, inputs)
```

**进出形状完全一致** → `spatial_enhancer`、`extract_feature_3d`、`decoder`、`decoder.loss`、
`decoder.post_process` 全部零改动。

`HoloBrain_Qwen3_5_VL` 未 override `_forward`（cite: `structure_qwen3_5.py:57-177` 只 override
了 `__init__` 与 `_generate_vlm`），所以这一处改动 v9 与 v10 都吃得到。

### 2.2 配置侧

`HoloBrain_Qwen2_5_VLConfig`（cite: `structure.py:558-574`）加一个字段：

```python
memoryvla: MODULE_TYPE | None = None
```

`HoloBrain_Qwen3_5_VLConfig` 继承它（cite: `structure_qwen3_5.py:178`），自动获得。
构建在 `__init__` 里，与 `self.spatial_enhancer = build(self.cfg.spatial_enhancer)` 并排
（cite: `structure.py:125`；Qwen3.5 侧同款在 `structure_qwen3_5.py:71`）。
⚠️ **两个 `__init__` 都要加**——Qwen3.5 的 `__init__` 没有调用 Qwen2.5 的 `__init__`，
而是 `super(HoloBrain_Qwen2_5_VL, self).__init__(cfg)` 跳过它、自己重列了一遍
（cite: `structure_qwen3_5.py:69-75`）。这是**唯一一处必须改两遍**的地方。

`cfg.memoryvla is None` 时**模块根本不构建** → 关闭态没有任何参数、不消耗任何 RNG，
这是 A 档逐 step 等价的根本保证（`GateFusion` 初始化不是恒等，见 `01-source-anatomy.md` §3）。

### 2.3 数据侧（**比预想的小得多**）

实测一个真实 batch（`$ROL_JFS/port/memoryvla/tools/probe_batch.py`，`swap_T`，B=4）： （cite: 实测）

```
uuid                 list[4] first='swap_T_arx_x5_episode_0000000'     ← 在
step_index           不在 batch
episode_index        不在 batch
```

原始样本（ItemSelection 之前）有的键：
`T_world2cam, cam_names, delta_ee_state, ee_state, imgs, intrinsic, joint_state,
master_joint_state, step_index, step_index_in_shard, task_name, text, uuid`

→ **`step_index` 数据集本来就产出**（cite: `robo_orchard_lab/dataset/robodojo/robodojo_lmdb_dataset.py:235`），
只是被 `ItemSelection` 的白名单丢掉了（cite: `config_robodojo_dataset.py:189-203` training /
`:229-242` validation；`ItemSelection` 实现见 `robo_orchard_lab/dataset/agibot/transforms.py:468-476`， （cite: robo_orchard_lab/dataset/agibot/transforms.py:468-476）
就是 `for k in list(data.keys()): if k not in self.keys: data.pop(k)`）。

**所以数据侧的全部改动 = 在白名单里加一个 `"step_index"`，且只在开关打开时加。**
`build_transforms(config, mode, ...)` 拿得到 `config`（cite: `config_robodojo_dataset.py:294`），
可以直接 gate。

- **`uuid` 已经在白名单里**（cite: `:196`），零改动。而且它是 `swap_T_arx_x5_episode_0000000`
  这种**全局唯一字符串**，直接根除了「`episode_index` 跨 lmdb 分片重号」的隐患——
  比原计划设想的 `(lmdb_index, episode_index)` 复合键更干净。
- `collate_batch_dict` 对 `str` 落到 `else: output[key] = elements`（cite:
  `robo_orchard_lab/dataset/collates.py:62-63`）→ `uuid` collate 成 `list[str]`， （cite: robo_orchard_lab/dataset/collates.py:62-63）
  正好能当 bank 的 dict key；`int` 落到 `torch.tensor(elements)`（cite: `:49-50`）→
  `step_index` collate 成 int 张量。
- ⚠️ `collate_batch_dict` 用 `batch[0].keys()` 取键（cite: `:40`）——它**不处理键缺失**，
  某个样本少键会 `KeyError`。本次不触发（同一 dataset 出来的样本键是齐的），
  但这意味着「新字段只在开关打开时产出」必须是**整个 dataset 级别**的开关，
  不能按样本随机决定。**`collates.py` 零改动。**

### 2.4 批序侧（**新开一处，本次唯一的新机制**）

`train.py:117-131` 用 `DistributedBatchFlagSampler`（cite: `projects/holobrain_internal/common/train.py:126`）。
它的取样是**整体随机排列**：`generator.permutation(n)`（cite:
`robo_orchard_lab/dataset/dataset_wrapper.py:133`），只按「样本来自哪个 dataset」的整型 flag （cite: robo_orchard_lab/dataset/dataset_wrapper.py:133）
把同源样本聚成一批（cite: `:190` 起）。**与 episode 连续性完全无关。**

→ 必须新开 `MemoryVLAEpisodeStreamBatchSampler`，放在 `models/memoryvla/` 下，
**包装而不修改** `DistributedBatchFlagSampler`，由 config 选择。宿主 sampler 一行不动。

设计（细节在 `04-port-plan.md`）：打乱 **episode 顺序**，episode 内**按 step 升序**连续取，
每批 `batch_size` 帧且**同属一条 episode**。这正是 `stream` 模式要的输入。
episode 边界可由 `RoboDojoLmdbDataset._get_indices(i) -> (lmdb_index, episode_index, step_index)`
（cite: `robodojo_lmdb_dataset.py:152`）扫一遍得到；实测同一 episode 的帧在全局下标里连续
（probe 里 `ds[0..3]` 的 uuid 相同）。

### 2.5 推理侧

`HoloBrainInferencePipeline.__call__`（cite: `robo_orchard_lab/models/holobrain/pipeline.py:57`）
→ `_model_forward`（`:62`）。bank 在 `self.training == False` 时**不做任何 episode 管理**
（cite: `MemoryVLA@0eef5c3 vla/memory_vla.py:267`），所以调用方必须在 episode 边界清空。

方案：模块暴露 `reset()`，模型暴露 `reset_memory()`，并在**推理路径下按 `uuid` 变化自动清**
（batch 里有 uuid 时）。真机部署没有 uuid，只能靠显式调用 `reset_memory()`。
按本次验收深度，`common/robodojo_eval.py` 的 50-episode 评测循环不接，列为遗留问题。

---

## 3. 输入可得性检查

| A 需要的输入 | 宿主能否提供 | 结论 |
|---|---|---|
| `episode_ids` | **能**，`inputs["uuid"]`，全局唯一 str | 直接用，零改动 （判断，依据本文上方已 cite 的事实） |
| `timesteps` | **能**，`step_index` 数据集已产出，加白名单即可 | 1 行 config 改动 （判断，依据本文上方已 cite 的事实） |
| 视觉 patch 特征 | **能**，但语义不同：宿主给的是 **VLM 层之后**的 `feature_maps`，A 用的是 **LLM 之前**的视觉主干特征 | 等价物，非同一物，见 `03-interface-diff.md` （判断，依据本文上方已 cite 的事实） |
| 认知 token | **部分**：A 是「最后一个非 pad 位的 LLM 隐状态」`[B,1,D]`；宿主有 `text_dict["embedded"]` `[B,L,C]` + mask，可取最后一个有效位 | 可构造出等价物 （判断，依据本文上方已 cite 的事实） |
| episode 连续、时序有序的 batch | **不能**，现有 sampler 是全局随机排列 | 需新增 sampler（2.4） （判断，依据本文上方已 cite 的事实） |

**没有任何一项无法提供**，不触发「整体放弃」条件。

---

## 4. 同类逻辑是否分散

- **loss 汇总**：集中在 `decoder.loss`（cite: `structure.py:235`），本次不新增 loss 分量
  （记忆库只改特征、不产生监督信号），**不触及**。
- **特征进 decoder 的路径**：只有 `_forward` 一条（cite: `structure.py:458`），不分散。
- **模型构建**：`__init__` 分散在两个类（2.2 已说明，必须改两遍）。

## 5. 本次触及的宿主已有文件（共 **3** 个）

| 文件 | 改动 | 侵入度 |
|---|---|---|
| `robo_orchard_lab/models/holobrain/structure.py` | `_forward` 一个 if + 一次调用；`__init__` 一行 `build`；Config 一个字段 | L1 （判断，依据本文上方已 cite 的事实） |
| `robo_orchard_lab/models/holobrain/structure_qwen3_5.py` | `__init__` 一行 `build`（因为它跳过了父类 `__init__`） | L1 （判断，依据本文上方已 cite 的事实） |
| `projects/holobrain_internal/common/configs/data_configs/config_robodojo_dataset.py` | 开关打开时向 ItemSelection 白名单加 `"step_index"` | L1 （判断，依据本文上方已 cite 的事实） |

比 `04-port-plan.md` 初稿预计的 4 个少一个：**`collates.py` 零改动**（见 2.3）。
`config_holobrain_common.py` 不算「改动已有逻辑」，是新增 `cfg.memoryvla.*` 命名空间与
可选 sampler 选择，但仍会在改动清单里逐行列出。
