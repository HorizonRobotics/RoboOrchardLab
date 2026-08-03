# 01 — MemoryVLA 源方法解剖

**源**：`~/git_repo/MemoryVLA` @ `0eef5c3`（`github.com/shihao1895/MemoryVLA`，MIT）
**论文**：MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models
for Robotic Manipulation，arXiv:2508.19236v2
**A 的环境状态**：`可运行`（`memvla_cu128`：py3.10.20 / torch 2.8.0+cu128 /
transformers 4.40.1，`prismatic`·`vla`·`action_model`·`MemoryVLA` 全部 import 成功，8 卡可见）

> 本文所有 `文件:行` 引用均指 A 仓库 `0eef5c3` 状态。

---

## 1. 论文侧 → 代码侧对照

| 论文概念 | 动机 | 代码落点 | 状态 |
|---|---|---|---|
| Perceptual Memory Bank | 保留细粒度视觉细节，跨时刻可回溯 | `PerMemBank`，`vla/memory_vla.py:335` | 有 （cite: vla/memory_vla.py:335） |
| Cognitive Memory Bank | 保留高层语义/意图 | `CogMemBank`，`MemoryVLA@0eef5c3 vla/memory_vla.py:158` | 有 （cite: vla/memory_vla.py:158） |
| Memory Consolidation | 容量受限时合并最相似的相邻两帧（ToMe） | `_consolidate_with_token_merge`，`MemoryVLA@0eef5c3 vla/memory_vla.py:211`；调度在 `_memory_consolidate`，`MemoryVLA@0eef5c3 vla/memory_vla.py:235` | 有 （cite: vla/memory_vla.py:211） |
| Memory Retrieval | 用当前帧做 query，对历史做 cross-attn | `CrossTransformerBlock`，`MemoryVLA@0eef5c3 vla/memory_vla.py:71`；调用在 `MemoryVLA@0eef5c3 vla/memory_vla.py:290-296` | 有 （cite: vla/memory_vla.py:71） |
| Adaptive Fusion | 门控融合「当前」与「检索出的历史」 | `GateFusion`，`MemoryVLA@0eef5c3 vla/memory_vla.py:139`；调用在 `MemoryVLA@0eef5c3 vla/memory_vla.py:303-306` | 有 （cite: vla/memory_vla.py:139） |
| 感知特征压缩 | 把 DINO+SigLIP 拼接的高维 patch 压到 256 | `BottleneckSE`，`MemoryVLA@0eef5c3 vla/memory_vla.py:105`；实例 `per_compr`，`MemoryVLA@0eef5c3 vla/memory_vla.py:406` | 有 （cite: vla/memory_vla.py:105） |
| 时序位置编码 | 给历史帧标注它来自哪个时刻 | `TimestepEmbedder`，`MemoryVLA@0eef5c3 vla/memory_vla.py:30`；调用在 `MemoryVLA@0eef5c3 vla/memory_vla.py:283-287` | 有 （cite: vla/memory_vla.py:30） |

论文有代码无：暂未发现。**代码有论文外行为**见第 4 节，是本次最需要注意的部分。

---

## 2. A 的完整前向数据流（`forward`，`vla/memory_vla.py:480-566`） （cite: vla/memory_vla.py:480-566）

```
vlm(...)                                            # :503
  ├─ output.hidden_states[-1]  →  去掉视觉 token 位  # :522-523  [B, L_text, D]
  │    └─ 按 attention_mask 取「最后一个非 pad 位」   # :526-531
  │         →  cog_tokens  [B, 1, D]                # ⚠ N == 1，单 token
  └─ self.vlm.vision_feats                          # :534  ⚠ side-effect 属性
       └─ per_compr(·)  BottleneckSE                # :535  [B, N_patch, 256]
            →  per_tokens

cog_tokens = cog_mem_bank.process_batch(cog_tokens, episode_ids, timesteps)   # :537
per_tokens = per_mem_bank.process_batch(per_tokens, episode_ids, timesteps)   # :543

action_model.loss(actions_repeated, cog_tokens_repeated, per_tokens_repeated) # :560
```

`episode_ids` / `timesteps` 是 `np.array`，由**训练脚本直接传进 forward**
（`forward` 签名 `MemoryVLA@0eef5c3 vla/memory_vla.py:487-488`），不是从 batch dict 里取的。 （cite: vla/memory_vla.py:487-488）

---

## 3. 组件表

| 组件 | 文件:行 | 消费的输入 | 产出 | 处置 | 耦合类型 | 硬编码常量 / 隐式假设 |
|---|---|---|---|---|---|---|
| `TimestepEmbedder` | `MemoryVLA@0eef5c3 vla/memory_vla.py:30-68` | `t [T]` int | `[T, D]` | **搬** | T1 | `max_period=10000`（`MemoryVLA@0eef5c3 vla/memory_vla.py:44`）；`frequency_embedding_size` 传入为 `token_size//4`（`MemoryVLA@0eef5c3 vla/memory_vla.py:194`） （cite: vla/memory_vla.py:30-68） |
| `CrossTransformerBlock` | `MemoryVLA@0eef5c3 vla/memory_vla.py:71-102` | `q [B,N,D]`, `k/v [B,M,D]` | `[B,N,D]` | **搬** | T1 | FFN 膨胀率写死 `*4`（`MemoryVLA@0eef5c3 vla/memory_vla.py:79`）；`is_causal=False`、`dropout_p=0.0`（`MemoryVLA@0eef5c3 vla/memory_vla.py:94`）；**无 attn_mask 参数** （cite: vla/memory_vla.py:71-102） |
| `GateFusion` | `MemoryVLA@0eef5c3 vla/memory_vla.py:139-155` | `x1,x2 [B,N,D]` | `[B,N,D]` | **搬** | T1 | 初始化 `normal(0, 1e-3)`（`MemoryVLA@0eef5c3 vla/memory_vla.py:143-144`）→ `sigmoid(≈0)≈0.5`，**不是恒等** （cite: vla/memory_vla.py:139-155） |
| `BottleneckSE` | `MemoryVLA@0eef5c3 vla/memory_vla.py:105-136` | `[B,N,C_in]` | `[B,N,C_out]` | **搬（须改写）** | T2 | ⚠ `assert _h*_h == _n`（`MemoryVLA@0eef5c3 vla/memory_vla.py:128`）**假设方形 token 网格**；SE 缩减比写死 `//16`（`MemoryVLA@0eef5c3 vla/memory_vla.py:117-118`） （cite: vla/memory_vla.py:105-136） |
| `CogMemBank` | `MemoryVLA@0eef5c3 vla/memory_vla.py:158-332` | `tokens [B,N,D]`, `episode_ids`, `timesteps` | `[B,N,D]` | **搬** | **T4** | 见第 4 节 （cite: vla/memory_vla.py:158-332） |
| `PerMemBank` | `MemoryVLA@0eef5c3 vla/memory_vla.py:335-357` | 同上 | 同上 | **搬** | **T4** | **当前只是 `CogMemBank` 的空壳子类**，`__init__` 逐参转发，无任何行为差异（`MemoryVLA@0eef5c3 vla/memory_vla.py:336-357`） （cite: vla/memory_vla.py:335-357） |
| `MemoryVLA` 壳类 | `MemoryVLA@0eef5c3 vla/memory_vla.py:360-873` | — | — | **不移植** | — | 绑死 `PrismaticVLM`，与宿主结构无关 （cite: vla/memory_vla.py:360-873） |
| `ActionModel` / `DiT` | `action_model/` | — | — | **不移植** | — | A 的动作头；宿主有自己的 `HoloBrainActionDecoder` （判断，依据本文上方已 cite 的事实） |
| FSDP 包裹策略 `get_fsdp_wrapping_policy` | `MemoryVLA@0eef5c3 vla/memory_vla.py:567-589` | — | — | **不移植** | — | 宿主用 accelerate，不用 FSDP 策略函数 （cite: vla/memory_vla.py:567-589） |
| `overwatch` logger / `from_pretrained` / `predict_action` | `MemoryVLA@0eef5c3 vla/memory_vla.py:597-873` | — | — | **不移植** | — | A 的基础设施与推理封装 （cite: vla/memory_vla.py:597-873） |

### 「不移植」的理由（逐条）

- **`MemoryVLA` 壳类**：它的职责是把 `PrismaticVLM` + `ActionModel` + 两个 bank 缝在一起。
  宿主的对应职责由 `HoloBrain_Qwen2_5_VL._forward` 承担，缝法完全不同。
- **`ActionModel` / `DiT`**：宿主的 `HoloBrainActionDecoder` 是另一套（flow/diffusion）动作头。
  换掉它等于换模型，不在本次范围（用户已确认「只移植两个记忆 bank」）。
- **FSDP / overwatch / CLI / trainer**：协议红线，一律接宿主的设施。
- **`self.vlm.vision_feats`（`MemoryVLA@0eef5c3 vla/memory_vla.py:534`）**：A 在 VLM 对象上挂了个 side-effect 属性来偷视觉特征。 （cite: vla/memory_vla.py:534）
  这是协议明令禁止的全局副作用，**不复制**——宿主侧直接用 `_vlm_outputs_handler` 的返回值。

---

## 4. 代码有而论文/直觉没有的行为（**本次最重要的两条**）

### 4.1 `group` 模式的记忆跨度只有一个 batch，不是一条 episode

`process_batch` 训练分支（`MemoryVLA@0eef5c3 vla/memory_vla.py:267-274`）： （cite: vla/memory_vla.py:267-274）

```python
if self.training:
    if self.dataloader_type == 'group':
        self.bank.clear()          # ← 每次 forward 都把整个 bank 清空
        self.eid_stream = None
    elif self.dataloader_type == 'stream':
        first_eid = episode_ids[0]
        if self.eid_stream is not None and self.eid_stream != first_eid:
            self.clear_episode(self.eid_stream)   # ← 只在换 episode 时清
        self.eid_stream = first_eid
```

**后果**：

| 模式 | 记忆跨度 | `mem_length` / ToMe 是否生效 |
|---|---|---|
| `group` | **单个 batch 内**，组内最后一个样本最多看到 `group_size-1` 帧历史 | `group_size=16` 且 `mem_length=16` 时 **15 < 16，ToMe 永不触发** （判断，依据本文上方已 cite 的事实） |
| `stream` | **跨 batch 持续**，直到 episode 变化才清 | 生效，长 episode 会反复触发 ToMe （判断，依据本文上方已 cite 的事实） |

所以**论文所说的「episode 级记忆」对应的是 `stream` 模式**，`group` 只是 batch 内近似。
A 的默认值是 `dataloader_type="group"`（`MemoryVLA@0eef5c3 vla/memory_vla.py:369`）——**默认配置跑的不是论文描述的那个东西**。 （cite: vla/memory_vla.py:369）

> 对本次移植的影响：宿主 episode 长 276–1374 帧（实测，见 `00-phase0-record.md`）， （cite: 实测）
> 要拿到论文语义**必须用 `stream`**，且要求连续 batch 承载同一 episode 的连续帧。

### 4.2 `eval` 时不做任何 episode 管理

`MemoryVLA@0eef5c3 vla/memory_vla.py:267` 的 `if self.training:` 意味着**推理路径下 `bank` 既不清空也不按 episode 隔离**。 （cite: vla/memory_vla.py:267）
调用方必须自己在 episode 边界调 `reset()`（`MemoryVLA@0eef5c3 vla/memory_vla.py:202`）或 `clear_episode()`（`MemoryVLA@0eef5c3 vla/memory_vla.py:207`）， （cite: vla/memory_vla.py:202）
否则跨 episode 串记忆。A 自己在 `predict_action`（`MemoryVLA@0eef5c3 vla/memory_vla.py:692`）里如何处理需单独确认（Phase 2）。 （cite: vla/memory_vla.py:692）

### 4.3 其他值得记的细节

- 历史特征以 `.detach().clone()` 存入（`MemoryVLA@0eef5c3 vla/memory_vla.py:243`）→ **历史不回传梯度**，只有当前帧的 query 路径有梯度。 （cite: vla/memory_vla.py:243）
- `_consolidate_with_token_merge`（`MemoryVLA@0eef5c3 vla/memory_vla.py:211`）用 `while len > mem_length` 驱动（`MemoryVLA@0eef5c3 vla/memory_vla.py:247-252`）， （cite: vla/memory_vla.py:211）
  每次调用只合并一对，靠循环收敛；`torch.no_grad()` 装饰。
- 相似度用 `F.cosine_similarity(f1, f2, dim=1).mean()`（`MemoryVLA@0eef5c3 vla/memory_vla.py:224`），在 `flatten(1)` 之后算， （cite: vla/memory_vla.py:224）
  **是 token 维展平后的余弦，不是逐 token 平均余弦**。
- `len(hist)==0` 时 `retrieved_episode_mem = working_mem`（`MemoryVLA@0eef5c3 vla/memory_vla.py:299-301`）， （cite: vla/memory_vla.py:299-301）
  再经 `GateFusion` → 输出 ≈ `0.5*(x+x) = x`（gate 初始 ≈0.5）。首帧近似恒等但**不精确恒等**。
- ⚠️ `retrieval_blocks` 在 `len(hist)==0` 分支下**完全不参与计算** → 若一个 batch 内所有样本
  都无历史，这些参数拿不到梯度，DDP 会报 unused parameter。

---

## 5. `BottleneckSE` 在本次移植中的处置（**结论：移植但默认不启用**）

A 用它把 DINO+SigLIP 拼接的 `vision_dim`（`MemoryVLA@0eef5c3 vla/memory_vla.py:407-408`，两个 backbone 的 embed 维之和，约 2176） （cite: vla/memory_vla.py:407-408）
压到 `per_token_size=256`，目的是给 DiT 一路独立的低维感知流。

宿主这边不需要压：`feature_maps` 的通道就是 `embed_dims=384`，而本次采用**形状保持**的接法
（bank 进什么形状就出什么形状，decoder 零改动）。插入 `BottleneckSE` 会把 384 压成 256，
**破坏与 `HoloBrainActionDecoder` 的形状契约**。

处置：改写（去掉方形网格 assert，显式收 `(h, w)`）后移植，作为 `cfg.memoryvla.per_compress_dim`
的可选项，**默认 `None` 即不启用**。仍然为它抓参考数值并跑 C 档——它便宜，且多一个对齐靶子。

---

## 6. 耦合类型判定

| 组件 | 类型 | 说明 |
|---|---|---|
| `CrossTransformerBlock` / `GateFusion` / `TimestepEmbedder` | **T1** | 纯模块，无外部耦合 （判断，依据本文上方已 cite 的事实） |
| `BottleneckSE` | **T2** | 改变通道数，属模块替换/新增分支 （判断，依据本文上方已 cite 的事实） |
| 两个 bank 本身 | **T3 + T4** | T3：需要 batch 提供 `episode_id`/`timestep` 两个新字段；T4：`self.bank` 跨 forward 持有状态，且 `stream` 模式下**训练循环的样本顺序成为语义的一部分** （判断，依据本文上方已 cite 的事实） |

**T4 是本次最贵的部分。** 低侵入替代方案（如用 detach 的自身副本当 teacher）在这里不适用——
记忆库的价值就在于跨时刻状态，去掉状态就没有方法了。代价已在 Phase 2/3 量化。

---

## 7. 参考数值的靶子（详见 `01b-reference-values.md`）

| 靶子 | 为什么选它 |
|---|---|
| `CrossTransformerBlock` | 检索的核心算子，纯函数，最容易严格对齐 （判断，依据本文上方已 cite 的事实） |
| `GateFusion` | 融合系数，初始化敏感 （判断，依据本文上方已 cite 的事实） |
| `TimestepEmbedder` | 纯函数，能独立锁死 （判断，依据本文上方已 cite 的事实） |
| `BottleneckSE` | 改写过，**必须**证明改写没改变数值（方形输入下与原版逐位一致） （判断，依据本文上方已 cite 的事实） |
| `CogMemBank.process_batch` 全流程 | 端到端，覆盖空历史 / 未满 / 触发 ToMe / 跨 episode 四种路径 （判断，依据本文上方已 cite 的事实） |
