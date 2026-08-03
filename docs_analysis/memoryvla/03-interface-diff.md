# 03 — 接口差异表

A = MemoryVLA `0eef5c3` · 宿主 = robo_orchard_lab `3ce31c0c`（**v10 生效**）
形状全部实测，来源标在每行。实测脚本：`$ROL_JFS/port/memoryvla/tools/probe_batch.py` （cite: 实测）

**读法**：`(cite: …)` 表示该断言从代码或运行输出读出；`未确认` 表示没读到，不靠常识补。

---

## 0. 宿主当前的关键常量（实测，非推算） （cite: 实测）

| 量 | 值 | 来源 |
|---|---|---|
| `embed_dims`（= `feature_maps` 通道数） | **384** | 实测打印 config（cite: `configs/config_holobrain_common.py` v10 段 `config.update(embed_dims=384,…)`）；`feat_mapping` 输出维就是 `self.decoder.embed_dims`（cite: `structure.py:184-190`） |
| `vlm_pretrain` | `./ckpt/Qwen3.5-2B` | 实测打印 config （cite: 实测） |
| `dst_wh` | `(352, 256)` = (W, H) | 实测打印 config （cite: 实测） |
| VLM `patch_size` / `spatial_merge_size` | 16 / 2 | 实测读 `ckpt/Qwen3.5-2B/config.json` （cite: 实测） |
| `qwen_patch_size` | **32** = 16 × 2 | cite: `structure.py:204-207`；Qwen3.5 版 `structure_qwen3_5.py:50-54` |
| `num_cams` | **3**（`cam_left_wrist` / `cam_right_wrist` / `cam_head`） | 实测打印 `dataset_config["arx_x5a"]`（cite: `config_robodojo_dataset.py:72`） |
| 图像张量（模型内） | `[B, 3, 3, 256, 352]` NCHW | 数据集出的是 NHWC `[B,3,256,352,3]`（实测），由 `BaseDataPreprocessor` permute（cite: `robo_orchard_lab/models/layers/data_preprocessors.py:137`），模型按 `batch_size, num_cams, _, h, w` 解包（cite: `structure.py:416`） |
| 视觉 token 网格 | **h_=8, w_=11 → 88/相机，×3 = 264** | `h_,w_ = h//32, w//32`（cite: `structure.py:329`） |
| `tokenizer.padding_side` | **`left`** | cite: `structure.py:183` |

> **⚠️ 8×11 不是方形** —— 直接击中 A 的 `BottleneckSE` 里的
> `assert _h*_h == _n`（cite: `MemoryVLA@0eef5c3 vla/memory_vla.py:128`）。

---

## 1. 逐张量对照

### 1.1 感知侧（perceptual memory 的输入）

| 项 | A | 宿主 | 转换放哪 |
|---|---|---|---|
| 张量 | `vlm.vision_feats` 经 `per_compr` → `[B, N_patch, 256]`（cite: `memory_vla.py:534-535`） | `feature_maps[0]` = `[B, 3, 384, 8, 11]`（cite: `structure.py:335-341`） | 移植模块内 reshape |
| **语义** | **LLM 之前**的视觉主干 patch（DINO+SigLIP 拼接，`vision_dim`≈2176，cite: `:407-408`） | **VLM 层之后**、经 `feat_mapping` 逐层加权融合的特征（cite: `structure.py:276-291`） | — |
| layout | `[B, N, D]` | `[B, cams, C, h, w]` → 展平成 `[B, cams*h*w=264, 384]` | 模块内 permute/reshape，出去时原样还原 （cite: structure.py:335-341） |
| dtype | bf16（A 全程 bf16） | **float32**，`_vlm_outputs_handler` 显式 `.to(torch.float32)`（cite: `structure.py:311`） | bank 跑 float32 |
| 归一化 | 视觉主干自带（DINO/SigLIP 各自的 mean/std） | `BaseDataPreprocessor` 的 `img_mean`/`img_std`（cite: `data_preprocessors.py:139-140`），且 `channel_flip=True` BGR→RGB（cite: `config_holobrain_common.py:240-241`） | 不涉及，bank 在特征域工作 |

> **这是等价物，不是同一物。** A 的感知记忆记的是「原始视觉细节」，宿主这里记的是
> 「已经被语言条件化过的视觉特征」。方法的动机（保留跨时刻的细粒度视觉证据）仍然成立，
> 但**不能声称与 A 数值可比**。这一条写进最终汇报的风险项。

### 1.2 认知侧（cognitive memory 的输入）

| 项 | A | 宿主 |
|---|---|---|
| 张量 | `cog_tokens [B, 1, D]`，**单 token** | `text_dict["embedded"] [B, L, 384]`（cite: `structure.py:344-352`） |
| 取法 | 最后一个非 pad 位的 LLM 隐状态，用 `attention_mask.cumsum` 找（cite: `memory_vla.py:526-532`） | 用 `text_dict["text_token_mask"]` 找最后一个 True 位 |
| mask | **A 没有 mask 概念**，靠 `attention_mask` 现算 | 有 `text_token_mask [B, L]`，**True = 有效**（= 非图像 token **且** 非 pad，cite: `structure.py:344-351`） |
| L 的构成 | — | 只含非图像 token（`text_feature = vlm_outputs[~img_feature_mask]`，cite: `:345`） |

**采用**：`cog_source="last_valid"`（默认），取最后一个有效文本 token → `[B,1,384]`，
过 bank 后**写回原位**，`text_dict` 形状不变。理由：

- 与 A 的语义一致（A 就是单个 cognition token）。
- `CrossTransformerBlock` **没有 attn_mask 参数**（cite: `memory_vla.py:87-101`）。
  若改成对全部 L 个文本 token 做记忆，padding 位会污染检索，必须给 A 的算子加 mask ——
  那就不再是「搬过来」而是「改写」了。N=1 完全绕开这个问题。
- 顺带的便利：`padding_side="left"`（cite: `structure.py:183`）意味着最后一位天然非 pad，
  但实现仍走 mask，不依赖这个巧合。

⚠️ **已知语义削弱**：A 里这个 token 是 DiT 的全部条件输入，影响力 100%；
宿主的 decoder 同时吃 264 个图像 token 和 L 个文本 token，改 1 个 token 的影响被稀释。
记入已知问题；`cog_source="all_text"` 作为将来选项，但需要先给算子加 mask。

### 1.3 episode 标识

| 项 | A | 宿主 |
|---|---|---|
| 字段 | `episode_ids: np.array`，由训练脚本直接传进 `forward`（cite: `memory_vla.py:488`） | `inputs["uuid"]`，`list[str]`，长度 B（实测） |
| 取值 | 未确认（A 的 RLDS 管线自己编） | `swap_T_arx_x5_episode_0000000`（实测）——**全局唯一** |
| 用法 | 当 dict key：`self.bank.get(eid)`（cite: `:288`）、`!=` 比较（cite: `:283`） | `str` 同样可以当 key 与比较，**零适配** |

> 原计划担心的「`episode_index` 跨 lmdb 分片重号」**不存在**：`uuid` 本身全局唯一。
> （顺带实测：`_get_indices` 返回的 `episode_index` 其实是 **`str`**（`'0'`/`'1'`），不是 int。 （cite: 实测）
> 我们不用它，用 `uuid`。）

### 1.4 timestep

| 项 | A | 宿主 |
|---|---|---|
| 字段 | `timesteps: np.array`（cite: `memory_vla.py:487`） | `step_index`，数据集已产出（cite: `robodojo_lmdb_dataset.py:235`），**但被 `ItemSelection` 白名单丢掉**（cite: `config_robodojo_dataset.py:189-203`） |
| 语义 | 未确认（A 侧未读到定义） | **episode 内步序，0 基，逐帧 +1**（实测：global 0/1/2 → step 0/1/2；global 274 → 新 episode 的 step 1） |
| 用法 | `torch.tensor(hist_timesteps)` 喂 `TimestepEmbedder`（cite: `:285-286`） | 同 |

⚠️ **dtype 陷阱（实测）**：`step_index` 在第 0 条 episode 上是 Python `int`， （cite: 实测）
在之后的 episode 上是 `np.int64`。而 `collate_batch_dict` 是**按第一个样本的类型分派**的
（cite: `collates.py:40-63`）：`int` → `torch.tensor(...)`，`np.int64` 两个分支都不匹配 →
落到 `else` 变成**普通 list**。
→ 移植代码必须同时接受 `Tensor` 与 `list`，不能假设是张量。

### 1.5 动作 / 位姿（**本次不涉及，但按协议核过**）

记忆库只改特征、不碰动作监督，所以「动作是否 delta、旋转表示、坐标系」这些
**对本次移植不构成接口**。宿主侧动作相关字段为 `hist_robot_state [B,1,14,8]` /
`pred_robot_state [B,64,14,8]`（实测），由 `decoder.loss` 独立消费（cite: `structure.py:236`），
本次改动不进入该路径。A 侧 `actions` 走 `action_model.loss`（cite: `memory_vla.py:560`），
**该动作头不移植**。故此行无差异需要弥合。

---

## 2. 时序对齐

| 项 | A | 宿主 | 影响 |
|---|---|---|---|
| 当前帧是否在输入里 | 是，`tokens[i]` 即当前帧（cite: `:277`） | 同 | 无差异 |
| chunk 长度 | 无 chunk 概念，逐帧 | `hist_steps=1`（实测）→ 每个样本就是单帧 | **正好匹配**，无需改 `hist_steps` （cite: 实测） |
| 记忆跨度 | `stream` 才是 episode 级；`group` 只在 batch 内（见 `01-source-anatomy.md` §4.1，已由 `01b` 的 bank 长度实测坐实） | — | **必须用 `stream`** （cite: 实测） |
| batch 内顺序要求 | `stream` 要求同 episode 连续帧按时序进入（cite: `:270-273`、`:283-286`） | 现有 sampler 是全局随机排列（cite: `dataset_wrapper.py:133`） | **需新 sampler**，见 `02-host-seams.md` §2.4 |

---

## 3. mask / padding 语义汇总（最容易搞反的一栏）

| 名字 | 出处 | True 的含义 |
|---|---|---|
| `text_token_mask` | `structure.py:352` | **有效**（非图像 **且** 非 pad） （cite: structure.py:352） |
| `img_feature_mask` | `structure.py:320-328` | 该位置是图像 token （cite: structure.py:320-328） |
| `main_img_mask` | `structure.py:358` | 该 patch 属于主图像 （cite: structure.py:358） |
| A 的 `attention_mask` | `memory_vla.py:526` | 有效（HF 惯例） （cite: memory_vla.py:526） |

padding 值：宿主图像侧无 padding（定长 8×11）；文本侧 **左 padding**（cite: `structure.py:183`）。
A 的 `CrossTransformerBlock` 用 `F.scaled_dot_product_attention(..., is_causal=False)` 且
**不接受 attn_mask**（cite: `memory_vla.py:94`）——本次靠「认知侧只取 1 个有效 token、
感知侧全部有效」把 mask 需求消掉，不改 A 的算子。

---

## 4. 转换代码放哪

**全部放在 `robo_orchard_lab/models/memoryvla/` 内部**，宿主侧只出现一个 if + 一次调用。

```
feature_maps[0] [B,3,384,8,11] ──permute/reshape──> [B,264,384] ──per bank──> [B,264,384] ──还原──> [B,3,384,8,11]
text_dict.embedded [B,L,384] ──按 mask 取最后有效位──> [B,1,384] ──cog bank──> [B,1,384] ──写回原位──> [B,L,384]
inputs["uuid"]       list[str]  ──> episode_ids
inputs["step_index"] list|Tensor ──> timesteps
```

---

## 5. 未确认清单（不靠常识补）

- A 的 `timesteps` 具体语义（是否 episode 内 0 基）：**未确认**，A 的训练脚本未读到写入处。
  本次按宿主自己的 `step_index` 语义使用，不假设与 A 相同。
- A 的 `predict_action`（cite: `memory_vla.py:692`）在推理时如何管理 bank：**未确认**，未通读。
- A 的 `dataloader_type` 默认 `group`（cite: `:369`）与论文描述的 episode 级记忆不一致，
  **哪个是作者实际用于论文结果的配置：未确认**。本次按论文语义选 `stream`。
