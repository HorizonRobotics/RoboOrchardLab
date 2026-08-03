# PORT-STATUS — MemoryVLA → HoloBrain

**日期**：2026-08-03（`date +%Y-%m-%d`）
**总判定**：**PASS**（无降级级别；见「降级说明」——降的是 batch 与训练时长，不是验收标准）

| 项 | 值 |
|---|---|
| A 的 repo | `github.com/shihao1895/MemoryVLA` @ `0eef5c3`，MIT （判断/方案，非实测） |
| A 的环境状态 | **可运行**（`memvla_cu128`，实测 import 全栈成功，8 卡可见）；全程只读，未装/升任何包 （判断/方案，非实测） |
| 宿主基点 | `3ce31c0c`（`feature/memory_dev1` 的 tip，tag `memory_dev1-stage1-20260803`） （判断/方案，非实测） |
| 分支 | `port/memoryvla` （判断/方案，非实测） |
| 依赖档位 | **E0** —— 宿主主环境零改动，**差异包清单为空**，未建 `.venv_memoryvla` （判断/方案，非实测） |

## 移植了什么 / 放弃了什么

| 组件 | 处置 | 一句话理由 | 耦合类型 |
|---|---|---|---|
| `CogMemBank` / `PerMemBank` | 移植 | 方法本体：按 episode 巩固+检索+融合 | **T3 + T4** （判断/方案，非实测） |
| `CrossTransformerBlock` | 移植 | 检索算子 | T1 （判断/方案，非实测） |
| `GateFusion` | 移植 | 自适应融合 | T1 （判断/方案，非实测） |
| `TimestepEmbedder` | 移植 | 历史帧的时序编码 | T1 （判断/方案，非实测） |
| `BottleneckSE` | 移植但**未接入** | 是方法的一部分且已数值验证，但它把通道压到 256 会破坏与 decoder 的形状契约（宿主特征本来就是 `embed_dims=384`） | T2 （判断/方案，非实测） |
| `MemoryVLA` 壳类 | 放弃 | 绑死 `PrismaticVLM`，宿主的对应职责由 `HoloBrain_Qwen2_5_VL._forward` 承担 | — （判断/方案，非实测） |
| `ActionModel` / DiT | 放弃 | A 的动作头；宿主有自己的 `HoloBrainActionDecoder`，换掉等于换模型 | — （判断/方案，非实测） |
| FSDP 策略 / overwatch / CLI / trainer | 放弃 | 协议红线：A 的基础设施一律接宿主的 | — （判断/方案，非实测） |

## 侵入度：**L1**，触及宿主已有文件 **4 个**，+67 行，**0 删除**

| 文件 | 档 | 改动 |
|---|---|---|
| `models/holobrain/structure.py` | L1 | 一个 config 字段 + 一行 `build` + 一个 `if` + 一次调用 （判断/方案，非实测） |
| `models/holobrain/structure_qwen3_5.py` | L1 | 一行 `build`（它跳过父类 `__init__`，必须单独加） （判断/方案，非实测） |
| `configs/data_configs/config_robodojo_dataset.py` | L1 | 开关打开时给 ItemSelection 白名单加 `step_index` （判断/方案，非实测） |
| `configs/config_holobrain_common.py` | L1 | `cfg.memoryvla.*` 命名空间 + `_build_memoryvla_cfg()` （判断/方案，非实测） |

新增文件（L0）：`models/memoryvla/{__init__,memory_bank,wrapper,sampler}.py`、
`configs/dataset_specs_memoryvla_robodojo_memory.py`、`docs_analysis/`。
**未触发 Gate B**（无 L3 改动）。**未触发 Gate E**（E0）。

## 验证结果（命令与证据见 `docs_analysis/memoryvla/06-verification.md`）

| 档 | 判据 | 结果 |
|---|---|---|
| 第 0 步 确定性 | 同 baseline 跑两遍 | **0.000e+00 → 逐位可复现**，A 档用严格判据 （cite: logs/baseline_run1.json vs run2） |
| **A 关闭态等价** | `atol < 1e-6` | **0.000e+00**（20 step 全同），参数量与移植前完全一致 → **PASS** （cite: logs/baseline_run1.json vs logs/gearA_off.json） |
| **B 开启态前向** | 走通一步 + 有梯度 | 68/68 张量有梯度，范数 8.39e-02，无 NaN；开/关差 6.20e-02 → **PASS** （cite: logs/gearB_on.json） |
| **C 数值对齐** | `atol < 1e-5` | **10/10 逐位一致（0.000e+00）** → **PASS** （cite: ref/manifest.json） |
| **D 资源** | 无 NaN、量级正常 | 参数 +0.66%、显存 +0.31 GiB、时间 +10% → **PASS** （cite: logs/gearA_off.json vs logs/gearB_on.json） |
| **E Memory 冒烟** | 跨过 episode 边界 | 2 条 episode，bank 峰值 16 后在 step 34 回落 → **PASS** （cite: logs/gearE_smoke.json） |
| 已有 ckpt 兼容 | 原样可加载 | 1000→1068 张量，新增 68 个全在 `memoryvla.*`，0 unexpected → **PASS** （cite: tools/check_ckpt_compat.py） |

## 新增 config 字段

`cfg.memoryvla.*`：`enable`(False) · `use_perceptual`(True) · `use_cognitive`(True) ·
`dataloader_type`("stream") · `group_size`(16) · `mem_length`(16) · `retrieval_layers`(2) ·
`use_timestep_pe`(True) · `fusion_type`("gate") · `consolidate_type`("tome") ·
`update_fused`(False) · `episode_stream_sampler`(True)。
**默认 `enable=False`，此时模块根本不构建。**

## 降级说明

| 项 | 用的档位 | 影响 |
|---|---|---|
| 卡数 | 单卡 | 本机任意两卡 gather 必崩；**DDP 行为未验证** （cite: 本机已知约束） |
| batch | 4（A/B/D）/ 8（E），非默认 16 | 8 张卡都有同事进程占 12–18 GiB。不影响结论：A 档比的是同 batch 的两棵树，C 档不过模型 （cite: 实测 nvidia-smi） |
| 训练时长 | 20–45 step，`lr=0` | 验收线不含收敛（用户已确认） （cite: 验收线） |

## 已知问题

1. **感知记忆的语义与 A 不同**：A 记的是 LLM **之前**的视觉主干 patch，宿主记的是 VLM
   **之后**、已被语言条件化的特征。角色等价，内容不等价 —— **不能声称端到端与 A 数值可比**。
   模块级的 C 档对齐不覆盖这一点。
2. **认知记忆影响被稀释**：A 里那个 token 是 DiT 的全部条件输入；宿主 decoder 同时吃 264 个
   图像 token 和 L 个文本 token，改 1 个 token 的影响小得多。
3. **DDP 未验证**：`retrieval_blocks` 在无历史分支下不参与计算，一批内全部样本都无历史时
   会触发 DDP unused parameter。单卡实测 68/68 全有梯度，多卡未验证。
4. **`BottleneckSE` 是未接入的死代码**：已验证、有出处，但当前不在任何执行路径上。
5. **`process_batch` 是逐样本 Python 循环**，B 从 4 涨到 16 时开销线性增长。
   协议要求移植期不做性能优化，**未优化**。

## 遗留问题（3-strike 格式）

本次**没有出现任何需要 3-strike 的报错**——所有开关都是一次跑通的。下面是主动留下的口子：

1. **推理路径的 `reset()` 未接进评测循环**。模块已提供 `reset()` 并在推理态按 episode
   变化自动清理，但 `common/robodojo_eval.py` 的 50-episode 循环没有接。
   按用户确认的验收深度（不跑仿真评测）本次不做。**真要跑 benchmark 前必须接**，
   否则跨 episode 串记忆。
2. **`fifo` vs `tome` 未真正比较**：8 step 太短，差异 4.030e-02 vs 4.029e-02 不可区分。
   要比较需要跑到 episode 尺度。
3. **多卡 / DDP 行为未验证**（见已知问题 3）。
4. **`cog_source="all_text"` 未实现**：需要先给 `CrossTransformerBlock` 加 attn_mask，
   那已经属于「改写」而非「搬运」。

## 下一步建议

1. 真要用它训练：先把 `reset()` 接进评测循环（遗留 1），再跑一次 Memory 六任务的完整训练，
   与 `07_results.md` 里 20k/100k 的 Memory 维度数字对比 —— 那才是这次移植值不值的答案。
2. 训练时务必确认 `episode_stream_sampler=True` 且 `dataloader_type="stream"`。
   两者不匹配会得到一个「跑得好好的但记忆库没生效」的结果，**不报错**。
3. 若上多卡，先单独验 DDP 的 unused-parameter 行为。

## 合规

- A 为 **MIT**，宿主为 **Apache-2.0**，兼容，可移植、可分发。
- ⚠️ A 的 `pyproject.toml:15` 写 `license={file="LICENSE"}`，但**仓库里没有 LICENSE 文件**；
  MIT 的判据来自 `pyproject.toml:21` 的 classifier。已在 `00-phase0-record.md` 记录。
- 搬运处逐段留出处：`# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L<a>-L<b>`，
  文件头保留 MemoryVLA 的出处与许可证声明。
- 第三方权重：本次**未引入任何新权重**。
