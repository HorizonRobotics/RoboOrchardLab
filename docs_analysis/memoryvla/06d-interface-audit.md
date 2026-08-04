# 06d — 接口语义审查（静默错误主战场）

审查者独立执行 · 日期 2026-08-04

每一项都回到代码独立确认，**不接受 `03-interface-diff.md` 的表格自证**。
结论三选一：`已核对一致` / `已核对不一致` / `未确认`。协议那张表逐项给结论，不跳过。

---

## 1. 动作语义

**本次移植不进入动作路径**——记忆库只改 `feature_maps[0]` 与 `text_dict["embedded"]`，
动作监督由 `decoder.loss` 独立消费，`structure.py` 的 diff 不触及该路径（已逐 hunk 确认，见 06e）。
但协议要求逐项给结论，不能整段跳过，所以逐条核：

| 检查项 | 结论 | 依据 |
|---|---|---|
| delta vs 绝对位姿 | ✅ **已核对一致（不构成接口）** | 记忆库的输入输出都是特征张量，形状进出相同；动作字段 `hist_robot_state`/`pred_robot_state` 不经过 `MemoryVLAMemory.forward` |
| 是否相对首帧 / 累积方式 | ✅ 同上，不构成接口 | 同上 |
| 旋转表示、四元数正负号、角度 wrap | ✅ 同上，不构成接口 | 同上 |
| gripper 开合 0/1 语义、阈值 | ✅ 同上，不构成接口 | 同上 |
| action dim 顺序、单位（m/cm、rad/deg） | ✅ 同上，不构成接口 | 同上 |

> A 侧的动作头 `action_model.loss`（`memory_vla.py:559`）**整体不移植**，
> 所以「A 的动作语义 vs 宿主的动作语义」这道鸿沟本次**不需要弥合**——
> 它不是被忽略，是被绕开了。这个判断我独立复核过：`MemoryVLAMemory.forward` 的
> 签名只有 `(feature_maps, text_dict, inputs)`，返回同名两项，动作张量不在其中。

---

## 2. 归一化

| 检查项 | 结论 | 依据 |
|---|---|---|
| 用 A 的统计量还是宿主的 | ✅ **已核对一致** | 记忆库在**特征域**工作，不碰图像归一化。宿主的 `img_mean`/`img_std` 与 `channel_flip` 全在 `BaseDataPreprocessor`，位于 VLM 之前，移植点在 VLM 之后 |
| per-dim vs global / min-max vs mean-std | ✅ 不构成接口 | 同上 |
| 反归一化在哪层、是否做了两次 | ✅ 不构成接口 | 记忆库不做任何(反)归一化 |
| 图像 mean/std、RGB/BGR、`/255` 是否重复 | ✅ **已核对一致（未被触碰）** | `config_robodojo_dataset.py` 的 diff **只**给 `ItemSelection.keys` 追加 `step_index`，不改任何图像变换（逐 hunk 确认） |
| **模块内部的 LayerNorm 位置/类型** | ✅ **已核对一致** | `CrossTransformerBlock` 为 **post-norm**：`attn_norm(query+attn_out)` → `ffn_norm(x+ffn_out)`，`nn.LayerNorm` 默认 affine。与 A `memory_vla.py:96-101` 逐字一致 |

---

## 3. 时序

| 检查项 | 结论 | 依据 |
|---|---|---|
| action chunk 是否含当前帧 | ✅ **已核对一致** | A `:277` 与宿主 `memory_bank.py:384` 都是 `working_mem = tokens[i]`，即**当前帧本身**参与融合。无偏移 |
| chunk 长度 / stride | ✅ **已核对一致** | 宿主 `hist_steps=1`，每样本单帧，与 A 的逐帧语义匹配 |
| 时间轴在哪个 dim | ✅ **已核对一致** | 历史堆叠 `torch.stack(hist_feats,0).reshape(-1,D).unsqueeze(0)` → `(1, T*N, D)`；PE 走 `repeat_interleave(N, dim=1)` → `(1, T*N, D)`。**两者的展开顺序都是 t-major、n-minor，对齐正确**（逐位对照 A `:296-302`） |
| observation history 长度 | ✅ **已核对一致** | 由 `mem_length=16` 控制，与 A 默认值一致 |
| **`step_index` 是不是 episode 内步序** | ✅ **已核对一致（独立验证）** | `robodojo_lmdb_dataset.py:152` `lmdb_index, episode_index, step_index = self._get_indices(index)`；`:113-114` 用 `num_steps`（该 episode 的长度）夹取 `first_step/last_step` → **step_index 以 episode 为界，0 基** |
| 频率/降采样是否与 A 一致 | ⚠️ **未确认** | A 侧的采样频率未在 `memory_vla.py` 内定义（由 RLDS 管线决定），A repo 内未读到。**影响评估**：记忆库对绝对频率不敏感（PE 编的是整数 step 序号，不是物理时间），但若两侧每秒帧数差异大，`mem_length=16` 覆盖的**物理时长**不同 → 记忆跨度语义不同。属于「超参需要重调」而非「实现错误」 |

---

## 4. mask / padding —— 协议点名的「最高频静默错误」

### 4.1 `text_token_mask` 极性 —— ✅ **已核对一致**

这是本次移植里**最可能出静默错误的一处**：`wrapper.py:242-274` 用它挑「最后一个有效
text token」送进认知记忆库。极性反了的话，取到的是 pad 位，
认知记忆存的是垃圾，而 **loss 照降、不报错**。

**我不采信 `03-interface-diff.md` 的表格，独立走了两条互相印证的路：**

**路 1 — 构造端**（`structure.py:334-342`，基点/HEAD 均已核）：
```
text_feature_mask = ~img_feature_mask                 # True = 非图像
not_pad_mask = (vlm_input_ids != pad_token_id)        # True = 非 pad
text_feature_mask = text_feature_mask & not_pad_mask  # True = 非图像 ∧ 非 pad
```
→ **True = 有效**。

**路 2 — 消费端**（全仓 grep 所有消费者，每一处都**取反**后才当 padding mask 用）：
```
bip3d/grounding_decoder/bbox3d_decoder.py:262   key_padding_mask=~text_dict["text_token_mask"]
bip3d/grounding_decoder/bbox3d_decoder.py:547   cls.masked_fill_(~text_token_mask[:,None,:], float("-inf"))
sem_modules/action_decoder.py:409               key_padding_mask=~text_dict["text_token_mask"]
```
PyTorch 的 `key_padding_mask` 约定是 **True=屏蔽**，这些地方都传 `~mask`
→ 反证 `mask` 本身 **True=有效**。

两条路一致 ⇒ `wrapper.py:255` 的 docstring 断言**正确**，`_last_valid_index` 的实现
（`positions where valid else -1`，取 `max`）**取的确实是最后一个有效位**。

> ⚠️ 但该 docstring 的 cite `structure.py:344-352` **指错了地方**（真实证据在 334-342）。
> 结论对、出处错，计入 06a 的漂移统计。

### 4.2 其余 mask/padding 项

| 检查项 | 结论 | 依据 |
|---|---|---|
| padding 值是 0 还是 `-inf`，attention 里用哪个 | ✅ **已核对一致（已结构性绕开）** | A 的 `CrossTransformerBlock` **不接受 attn_mask**（`memory_vla.py:87-95`，cite 命中）。移植方的应对是**只喂无 padding 的输入**：认知侧 N=1（单个有效 token），感知侧 264 个视觉 token **全部有效**（8×11×3 定长网格，无 padding）。⇒ 不存在「padded history 污染检索」的路径 |
| loss 是否对 padding 位取均值 | ✅ **N/A** | 记忆库不产生 loss |
| 无有效 token 的行 | ✅ **已核对一致** | `wrapper.py:272-273` 显式兜底 `idx<0 → length-1`，不会取到负索引 |
| 左 padding 假设是否被依赖 | ✅ **已核对一致** | 宿主确为 `padding_side="left"`（真实位置 `structure.py:180`/HEAD `181`），但实现**不依赖**它——走的是 mask 而非「取最后一位」。docstring 明说「does not lean on that」，代码属实 |

---

## 5. 张量

| 检查项 | 结论 | 依据 |
|---|---|---|
| **layout / permute 往返是否可逆** | ✅ **已核对一致（逐维验算）** | `wrapper.py:228-240`：`[B,cams,C,h,w] --permute(0,1,3,4,2)--> [B,cams,h,w,C] --reshape--> [B,cams*h*w,C]`；回程 `reshape(b,cams,h,w,c).permute(0,1,4,2,3)`。**两者互为逆置换**，token 顺序为 (cam,h,w) major-to-minor，前后一致 |
| `view` vs `reshape` | ✅ **已核对一致** | 全部用 `reshape`（`wrapper.py:231,237`；`memory_bank.py:391,186`），permute 后非连续也安全。回程末尾显式 `.contiguous()` |
| **dtype 是否被隐式升降** | ✅ **已核对一致** | 宿主特征在 `structure.py:308/309` 已显式 `.to(torch.float32)`，记忆库全程 fp32。认知侧写回时显式 `fused.to(embedded.dtype)`（`wrapper.py:249`）。感知侧进出同为 fp32，无隐式转换 |
| **device：有无硬编码 `.cuda()`/`.cpu()`** | ✅ **已核对一致（零硬编码）** | `grep -rnE "\.cuda\(\)\|\.cpu\(\)\|device=[\"']cuda" robo_orchard_lab/models/memoryvla/` → **无匹配**。设备一律跟随输入（`memory_bank.py:397-399` `.to(working_mem.device)`、`wrapper.py:262` `device=` 参数传入） |
| **是否原地改了 batch dict 或共享张量** | ✅ **已核对一致（正确处理）** | `grep -rnE "\.(add\|mul\|div\|clamp\|copy\|scatter\|masked_fill)_\("` → **无匹配**。`wrapper.py:249` 用 `scatter`（非 `scatter_`）返回新张量；`wrapper.py:212` `feature_maps = list(feature_maps)`、`:218` `text_dict = dict(text_dict)` **先浅拷贝再写**，调用方的容器不被改 |

---

## 6. 常量与隐式 assert

| 检查项 | 结论 | 依据 |
|---|---|---|
| A 写死的 shape 假设 | ✅ **已核对一致（已识别并改写）** | A `BottleneckSE` 的 `assert _h*_h == _n`（`memory_vla.py:128`）假设方形网格；宿主是 8×11 必触发。移植方改写成显式收 `(h,w)`，方形路径原样保留 |
| 写死的 action dim / batch 维假设 | ✅ **已核对一致** | 移植代码内无写死维度；`token_size` 取自 `config["embed_dims"]`（`config_holobrain_common.py:_build_memoryvla_cfg`），不二次配置 |
| 隐式 assert | ✅ **已核对一致** | `memory_bank.py:246-248` 的三个 `assert` 是 A 原有的取值域检查，已移植；`wrapper.py:102-108` 新增的「两个流都关就报错」是**主动防御**（防止构造出一个不该存在的空模块），合理 |
| **`uuid` 是否真的 per-episode** | ✅ **已核对一致（独立验证）** | `robodojo_lmdb_dataset.py:152-156`：`lmdb_index, episode_index, step_index = self._get_indices(index)` → `index_data = self.idx_lmdbs[lmdb_index][episode_index]` → `uuid = index_data.uuid`。**uuid 按 episode_index 取，同 episode 所有帧共享**。若 uuid 是 per-frame，`hist` 将恒空、记忆库退化为恒等——已排除该可能 |

---

## 7. 本次额外核的三项（不在协议表内，但属于同类风险）

| 检查项 | 结论 | 依据 |
|---|---|---|
| **改过的 `text_dict` 是否真的到达 decoder** | ✅ **已核对一致** | `structure.py:446-449` 返回的 `feature_maps, text_dict` 随后**同时**喂给 `spatial_enhancer`（`:453-458`）与 `decoder`（`:461-467`）。修改沿两条路都传下去了 |
| **`feature_maps[0]` 是否漏了其他尺度** | ✅ **已核对一致（无遗漏）** | `structure.py:328-332` 处 `feature_maps = [ img_feature.reshape(...) ]` —— 在移植点上它**只有一个元素**。`feature_maps[0]` 即全部 |
| **sampler 的 episode 内时序单调性** | ✅ **已核对一致（实现正确）** | `sampler.py:174-178` `for b in range(start, end, batch_size)` 顺序步进，**不打乱 episode 内顺序**；只在 episode 之间 `rng.permutation`。DDP 按 episode 分片（`:138` `spans[rank::num_replicas]`），episode 不跨 rank。**实现是对的——问题在它没被接入，见 06e/P0-1** |

---

## 8. 逐类统计

| 类别 | 已核对一致 | 已核对不一致 | 未确认 |
|---|---:|---:|---:|
| 动作语义 | 5 | 0 | 0 |
| 归一化 | 5 | 0 | 0 |
| 时序 | 5 | 0 | **1** |
| mask / padding | 5 | 0 | 0 |
| 张量 | 5 | 0 | 0 |
| 常量 | 4 | 0 | 0 |
| 额外三项 | 3 | 0 | 0 |
| **合计** | **32** | **0** | **1** |

### 唯一的 `未确认` 及其影响评估

**A 的采样频率 / 降采样策略**（§3 末行）。

- **为什么验不了**：A 的帧率由其 RLDS 数据管线决定，`vla/memory_vla.py` 内不出现；
  A repo 中我只读到消费端（`timesteps: np.array` 形参），没读到定义端。
  跑 A 的训练也拿不到——它读的是 A 自己的数据集，与宿主 RoboDojo 不同源，比不了。
- **如果错了影响多大**：**中等偏低**。PE 编码的是整数 step 序号而非物理时间，
  所以实现层面不会错；受影响的是**超参语义**——`mem_length=16` 在两侧覆盖的物理时长不同。
  后果是「记忆跨度需要按宿主帧率重调」，属于调参，不是移植 bug。
  **不构成 P0/P1**，记为 P3 提示。
- **该怎样才能验**：读 A 的数据集配置（RLDS builder 的 `step` 定义）或其论文附录的数据处理节，
  与 RoboDojo 的 `num_steps` / 采集帧率对照。本次不在 A repo 内，需外部资料。

---

## 9. 结论

**接口语义层面没有发现任何不一致。** 32 项已核对一致，0 项不一致，1 项未确认且影响可控。

尤其是协议点名的「最高频静默错误」——mask 极性——我用构造端与消费端两条独立路径交叉验证，
**移植方的判断是对的**。`uuid` per-episode、`step_index` episode 内步序、permute 往返可逆、
scatter 非原地、无硬编码 device，这些都经得起独立检验。

> 需要强调的是：**接口语义正确，不等于方法生效。**
> 本节确认的是「喂进去的东西是对的」；而「有没有东西被喂进去」是 06e/P0-1 的问题——
> 在当前真实训练路径上，`hist` 恒空，上面这套正确的接口**根本没有机会发挥作用**。
