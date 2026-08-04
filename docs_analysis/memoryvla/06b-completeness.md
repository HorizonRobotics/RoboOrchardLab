# 06b — 完整性：漏搬与多搬

审查者独立执行 · 日期 2026-08-04

**本节不从 `01-source-anatomy.md` 出发**——那是被审对象。
下面这张「本该搬什么」的清单是从论文 arXiv:2508.19236v2 的方法结构 +
`~/git_repo/MemoryVLA/vla/memory_vla.py`（`0eef5c3`，该文件在 A 的工作树中未被修改）
重新走一遍得到的，然后才与移植结果对照。

---

## 1. 方法要素 → A 的实现 → 宿主移植后

判定四选一：`已移植` / `声明不移植（理由成立）` / **`静默缺失`** / `多搬`。

| # | 方法要素（动机 → 结构） | A 的实现位置 | 宿主移植后位置 | 判定 |
|---|---|---|---|---|
| 1 | **感知记忆库**：逐帧存视觉 token，跨时刻保留细粒度视觉证据 | `PerMemBank` `:335-357` | `memory_bank.py:438` `PerMemBank` | ✅ 已移植（逐行一致） |
| 2 | **认知记忆库**：存单个认知 token，保留高层意图 | `CogMemBank` `:158-332` | `memory_bank.py:214` `CogMemBank` | ✅ 已移植（逐行一致） |
| 3 | **巩固 ToMe**：超 `mem_length` 时合并**相邻最相似**一对，`0.5*(f_i+f_j)` | `_consolidate_with_token_merge` `:211-232` | `memory_bank.py:289-318` | ✅ 已移植，系数 `0.5` 与 `argmax` 选对逐字一致 |
| 4 | **巩固 FIFO**（备选） | `:246-247` | `memory_bank.py:333-336` | ✅ 已移植 |
| 5 | **检索**：working memory 作 query，历史作 key/value，跨注意力 ×`retrieval_layers` | `CrossTransformerBlock` `:71-102`，循环 `:306-308` | `memory_bank.py:91`，循环 `:405-407` | ✅ 已移植 |
| 6 | **时序编码**：正弦 PE 加在**检索的 key 上，不加在 value 上** | `:301-302` + `:308` `block(query, episode_mem + pe, episode_mem)` | `memory_bank.py:400-401` + `:407` | ✅ 已移植，**key/value 非对称性保留** |
| 7 | PE 与历史 token 的对齐：`repeat_interleave(N, dim=1)` | `:302` | `memory_bank.py:401` | ✅ 已移植 |
| 8 | **自适应融合 gate**：`sigmoid(proj([x1;x2]))` 加权 | `GateFusion` `:139-155` | `memory_bank.py:191-210` | ✅ 已移植，含 `normal(0,1e-3)` 非恒等初始化 |
| 9 | **融合 add**（备选）：`(working+retrieved)*0.5` | `:318` | `memory_bank.py:416` | ✅ 已移植，系数 `0.5` 一致 |
| 10 | **stop-gradient**：历史以 `detach().clone()` 存，`@torch.no_grad()` 包住巩固 | `:210` `:234` `:243` `:231` | `memory_bank.py:288` `:320` `:330` `:317` | ✅ 已移植，**detach 位置逐处一致** |
| 11 | **episode 生命周期**：`stream` 换 episode 清上一条；`group` 每次 forward 清空 | `:267-288` | `memory_bank.py:360-381` | ✅ 已移植 |
| 12 | 「无历史 → 恒等旁路」：`retrieved = working_mem` | `:312-314` | `memory_bank.py:410-412` | ✅ 已移植 |
| 13 | **感知压缩 `BottleneckSE`**：2176 维视觉主干特征 → 256 | `per_compr` `:406-410`，类 `:105-136` | `memory_bank.py:135`，**已移植但未接入** | ⚠️ **声明不移植（理由成立）**，见 §2 |
| 14 | **DiT 动作头消费双记忆** | `action_model.loss(actions, cog, per)` `:559-563` | 宿主 `HoloBrainActionDecoder` | ✅ 声明不移植（理由成立，换掉等于换模型） |
| 15 | **episode 有序、时序单调的数据流**（`stream` 记忆的前提） | A 由训练脚本供给 `episode_ids`/`timesteps` `:486-489` | `sampler.py` 已实现，**但未接入任何宿主执行路径** | ❌ **静默缺失 → P0-1** |

---

## 2. 逐项说明（只写需要判断的）

### #13 `BottleneckSE` —— 声明不移植，理由成立

A 用它把 DINO+SigLIP 拼接的 `vision_dim≈2176` 压到 `per_token_size=256`
（`:402-410`）。宿主的 `feature_maps[0]` 本来就是 `embed_dims=384`，
且该宽度是与 decoder 的形状契约。压到 256 会当场破坏契约。

**理由成立**。且移植方做了一件超出要求的事：把它**搬过来、改写成显式收 `(h,w)`、
并做了数值对齐**（原方形路径逐位一致），只是不接入。
`01-source-anatomy.md` 与 `PORT-STATUS.md` 都明记了它是未接入的死代码。

> 顺带核实了改写的必要性：A 的 `assert _h*_h == _n`（`memory_vla.py:128`，**cite 精确命中**）
> 在宿主的 8×11 网格上必然触发。改写是**被迫的**，不是随意发挥，且保留了原路径。

### #15 episode 有序数据流 —— **静默缺失**

这是本节唯一的缺失项，也是全篇最重的一条。

方法要素 #11（`stream` 的 episode 生命周期）和 #1/#2 的「跨时刻」语义，
**前提是同一 episode 的连续帧按时序进入 `process_batch`**。
A 由它自己的训练脚本保证；宿主的默认 sampler 是全局随机排列
（`dataset_wrapper.py:133-134`，我已独立核对）。

移植方**正确识别了这个缺口**（`03-interface-diff.md:111`、`04-port-plan.md:99` 的风险表都写了），
**也正确实现了 sampler**（`sampler.py`，我逐行读过，时序单调性与 DDP 分片都是对的），
**但没有把它接进任何宿主执行路径**。

判定为 **`静默缺失`** 而不是 `已移植`：组件存在 ≠ 方法生效。
详细证据链与后果推演见 `06e-intrusion-audit.md` §3 / `06-review-report.md` P0-1。

---

## 3. 反向检查：多搬了 A 的基础设施吗

```
grep -rnE "argparse|accelerate|deepspeed|wandb|logging\.basicConfig|overwatch|fsdp|FSDP" \
     robo_orchard_lab/models/memoryvla/
→ （无匹配）
```

**没有多搬。** 逐项确认协议红线里点名的东西都没进来：

| A 的基础设施 | A 的位置 | 是否进入宿主 |
|---|---|---|
| `MemoryVLA` 壳类（绑死 `PrismaticVLM`） | `:360-873` | ❌ 未进入 ✅ |
| `ActionModel` / DiT 动作头 | `action_model.*` | ❌ 未进入 ✅ |
| `get_fsdp_wrapping_policy` | `:567-589` | ❌ 未进入 ✅ |
| `from_pretrained` / CLI / trainer / overwatch logger | `:597-873` | ❌ 未进入 ✅ |

新子目录 4 个文件全部是 `torch.nn` + 宿主自己的 `build()` 约定，
无第三方框架依赖——这也是依赖档位能停在 E0 的原因。

---

## 4. 「最容易静默丢掉」的六类，逐类结论

协议点名的六类，逐条给结论而不是整段跳过：

| 类别 | 结论 | 证据 |
|---|---|---|
| **公式系数/温度/缩放** | ✅ 全部一致 | ToMe 的 `0.5*(f_i+f_j)`、add 融合的 `*0.5`、`TimestepEmbedder` 的 `max_period=10000` 与 `half=dim//2`、`frequency_embedding_size=token_size//4` 逐字比对一致 |
| **stop-gradient / detach 位置** | ✅ 全部一致 | 见 #10。三处 `@torch.no_grad()` + 两处 `.detach().clone()` 位置与 A 完全对应 |
| **warmup / 分阶段启用** | ✅ N/A（A 无此逻辑） | A 的 `CogMemBank` 无 warmup、无 step 条件；唯一的条件是 `if self.training:`，已移植 |
| **归一化层位置与类型** | ✅ 一致 | `CrossTransformerBlock` 用 **post-norm**（`attn_norm(query+attn_out)`、`ffn_norm(x+ffn_out)`），`nn.LayerNorm` 默认 `elementwise_affine=True`，与 A 逐字一致 |
| **loss 归约方式** | ✅ N/A | 记忆库**不产生任何 loss 分量**，只改特征。宿主 loss 路径未被触碰（`structure.py` 的 diff 不进入 `decoder.loss`） |
| **A 在 `__init__`/模块顶层的隐式设置** | ✅ 已正确剥离，且**未连带丢功能** | A 的隐式设置都在 `MemoryVLA` 壳类里（`self.cur_timestep=0`、`vision_dim` 探测、FSDP 策略），属于「不移植」的壳；被移植的 6 个类的 `__init__` 里没有任何全局副作用（已用 R5 的 side-effect 扫描独立验证：无 `manual_seed`/`set_default_dtype`/`backends`/hook） |

---

## 5. 覆盖率结论

| | 数量 |
|---|---:|
| 论文/A 侧识别出的方法要素 | 15 |
| 已移植且与 A 逐行一致 | 12 |
| 声明不移植且理由成立 | 2（#13 #14） |
| **静默缺失** | **1（#15，P0-1）** |
| 多搬 | **0** |

**结论**：方法本体（存储/巩固/检索/融合/时序编码/生命周期）**搬得完整且忠实**，
系数、detach 位置、norm 位置、key/value 非对称性这些最容易静默丢的细节**一处没丢**。
唯一缺的是让这套机制真正跑起来的**数据流前提**——组件写好了，没接线。
