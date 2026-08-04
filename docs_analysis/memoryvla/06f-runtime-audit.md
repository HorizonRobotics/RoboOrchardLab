# 06f — 训练动力学与工程健壮性（含「补三项空白」）

审查者独立执行 · 日期 2026-08-04 · 单卡 `cuda:6`

---

## 1. 为什么必须补这三项

移植方的全部数值证据来自 `$ROL_JFS/port/memoryvla/tools/run_gears.py`。
该 harness **刻意不走 `train.py`**（文件头写明理由：accelerate/checkpoint/logging 会引入
这套比对承受不了的不确定性）。**这个理由是成立的**，但代价是三件事**结构性地验不到**：

| 空白 | harness 的做法 | 于是什么验不到 |
|---|---|---|
| ① 真实 optimizer 分组 | `torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)`（`run_gears.py:157-159`） | 宿主真实分组 `config_holobrain_common.py:528-556`（按 `"vlm." in name` / `p.dtype` 分三组、vlm 组 `lr*0.1`）**一次都没被执行过** |
| ② 参数是否真的在更新 | `lr=0`，权重按设计不动 | 「新参数真的在优化」**无法证明** |
| ③ worker 随机流 | `num_workers=0` | 「新 batch 字段扰动 worker 随机流」这一失败模式**根本不可能发生** |

三项全部由本次审查补齐，脚本 `$ROL_JFS/port/memoryvla/review/rev_decisive.py` 与 `rev_agear.py`。

---

## 2. 决定性实验：真实路径上记忆库到底做了什么

> 这是本次审查的核心实验，直接坐实 `06-review-report.md` 的 **P0-1**。

### 设计

在**同一次运行内**做三项互相独立的观测，从而**绕开** P3-2 的构造顺序 RNG 干扰
（不需要跨 run 比较，因此「开启态主干初始权重与关闭态不同」这件事不构成混淆）：

1. **episode 多样性**：每个 batch 里 `len(set(uuid))` vs `batch_size`
2. **恒等探针**：包住 `MemoryVLAMemory.forward`，直接量 `max|out − in|`
3. **梯度三态**：每个 memoryvla 张量的 grad 是 `None` / 恰好 0 / 非零

唯一变量是 sampler：`host` = `train.py:124` 实际构造的 `DistributedBatchFlagSampler`
（并按 train.py 的做法包 `ConcatDatasetWithFlag`）；`episode` = 移植方的 sampler（对照组）。
其余（seed / config / batch / 数据 / 设备 / **非零 lr**）完全相同。

### 事前预测（先写下来，再跑）

`hist` 恒空 ⇒ `retrieved = working_mem` ⇒ 融合退化为 `f(w, w)`：

- `gate`：`s·w + (1−s)·w = w`，**与 s 无关，精确恒等**
- `add`：`(w+w)·0.5 = w`，**精确恒等**
- `retrieval_blocks` / `timestep_encoder` 从未被调用 ⇒ grad 应为 **None**
- `GateFusion` 在图上但 `∂(s·w+(1−s)·w)/∂s = w − w = 0` ⇒ grad 应**存在且恰好为 0**

### 结果

| 观测 | **`host`（真实路径）** | `episode`（仅 harness 可选） |
|---|---|---|
| 每 batch 不同 episode 数 | **4 / 4**（步步如此） | **1 / 4**（步步如此） |
| 恒等差 感知 `max\|out−in\|` | **1.192093e-07** | 1.473746e+00 |
| 恒等差 认知 `max\|out−in\|` | **5.960464e-08** | 1.154852e+00 |
| grad = `None` | **64 / 68** | 0 / 68 |
| grad 存在但恰好 0 | **4 / 68** | 0 / 68 |
| **grad 非零** | **0 / 68** | **68 / 68** |
| bank 条目数 / 长度 | 恒为 1 个 episode、长度 **[1]** | 4 → 8 → 12 → **16 封顶** |
| **参数实际移动数（lr=1e-4，8 步）** | **0 / 68** | **66 / 68** |

**四项预测全部命中。** `5.96e-08 = 2⁻²⁴`、`1.19e-07 = 2⁻²³`——即 float32 的
1～2 ULP，是「先算 `s·w+(1−s)·w` 再舍入」而非代数化简的结果。
**数值意义上就是恒等。**

bank 长度那一列尤其说明问题：真实路径下 bank **永远只有 1 条**——
写进去一条，下一个样本因 episode 不同立刻把它清掉；
对照组里 bank 老老实实涨到 `mem_length=16` 并封顶（说明 ToMe 巩固路径被走到了）。

### 结论

**开启 `memoryvla.enable=True` 后在 `train.py` 下训练：**
两个记忆库都是**精确恒等函数**，7,467,264 个新参数中 **0 个** 会被更新，
模型等价于 baseline + 一坨死权重。**无异常、无告警、无 NaN、loss 曲线正常。**

同时这也**独立复现了移植方 B/E 档的数字**（对照组 68/68 有梯度、bank 封顶 16）——
他们测得没错，只是测的那条路宿主选不到。

---

## 3. 补空白① — 真实 `build_optimizer` 的分组 ✅ **结论正面**

直接调用宿主真实路径 `config_holobrain_common.build_optimizer(config, model)`：

| group | 张量数 | 参数量 | lr | weight_decay | 对应分支 |
|---:|---:|---:|---|---|---|
| 0 | 11 | 3,934,085 | 1e-7 | 5e-4 | `bit16_params` |
| **1** | **694** | **70,962,180** | 1e-7 | 5e-4 | `other_params` |
| 2 | 43 | 685,003,232 | 1e-8 | 5e-4 | `vlm_params`（`base_lr*0.1`） |

```
memoryvla -> group histogram : {'1': 68}      # 68 个张量全部落在 group 1
trainable tensors NOT in any optimizer group : 0
```

> lr 显示 1e-7 是因为 `ChainedScheduler` 的 `LinearLR(start_factor=0.001)` warmup，
> 我传的 `--lr 1e-4` 在 step 0 被缩到 1e-7。属预期。

**判定 ✅**：68 个新张量**全部**落进 `other_params`（group 1），拿到 `base_lr` 与
`weight_decay=5e-4`，与宿主其他非 VLM 模块**同等待遇**——这正是应有的归属。
且**没有任何可训练张量掉在 optimizer 之外**。

我在计划阶段担心的两个风险都**没有发生**：
- `"vlm." in name` 不会误命中 `memoryvla.*`（`memoryvla` 里的子串是 `vla.` 不是 `vlm.`）——已实测确认
- 现有 lr/wd 分组在开启前后结构不变

**这一项是移植方没验但结果正确的。** 记为正面结论，不计入 finding。

---

## 4. 补空白② — 新参数真的会被更新吗

用**非零 lr**（`--lr 1e-4`）跑 8 步，逐参数比对 before/after：

| sampler | 实际移动的 memoryvla 张量 |
|---|---|
| `episode`（对照） | **66 / 68** |
| **`host`（真实路径）** | **0 / 68** |

**判定**：优化器本身是通的——只要梯度进来，参数就会动（对照组 66/68）。
真实路径上 0/68 的原因**不是**优化器配错，而是 §2 的梯度根本不存在（P0-1）。

> 未动的 2/68 是梯度极小、在 warmup 后的 1e-7 lr 下更新量低于 fp32 分辨率所致，
> 与结论无关。

---

## 5. 补空白③ — worker 随机流 ✅ **PASS，且有阳性对照**

```
A-gear with num_workers=4   (12 steps)
  batch keys identical : True
  params diff          : +0
  MAX per-step |diff|  : 0.000000e+00        → PASS
```

**这个测试有牙**——先证明这条流水线确实 worker 敏感：

```
同一棵树, workers=0 vs workers=4 : MAX per-step |diff| = 3.811359e-02
→ 变换里存在 worker 播种的随机性
```

若移植改变了 worker 随机流，上面那个 0 就不可能是 0。**关闭态不扰动 worker 流，坐实。**

---

## 6. 协议 R6 其余各项

| 检查 | 结论 | 证据 |
|---|---|---|
| 新参数真的在优化 | ⚠️ **条件成立** | 见 §4。优化器通路正确；真实路径上因 P0-1 而全部冻结 |
| 梯度真的流到新模块 | ⚠️ **条件成立** | 对照组 68/68 非零；真实路径 0/68（P0-1） |
| **梯度没有流到不该流的地方** | ✅ **PASS** | 历史以 `.detach().clone()` 存于 `@torch.no_grad()` 内（`memory_bank.py:288,320,330`，与 A 逐处对应）。对照组下 `bank` 里的张量不在图上，梯度不回流历史 |
| **ckpt 兼容（新增 key 的范围）** | ✅ **PASS** | 关闭态 1,000 → 开启态 1,068 张量：**新增恰好 68，全部 `memoryvla.*`；removed 0；reshaped 0；unexpected 0**。前缀无泄漏。**非循环**：A 档已独立证明关闭态与移植前逐位相同，故「关闭态 state_dict == 移植前 state_dict」是证明而非假设 |
| ckpt 反向兼容 | ✅ **PASS** | 同上，把不含 `memoryvla.*` 的 state_dict 灌进开启态模型：`unexpected 0`、`missing 非 memoryvla 0`、`missing memoryvla.* 68`（预期） |
| **外部真实 ckpt 加载** | ❌ **无法验证** | 见 §7 |
| lr scheduler | ✅ **PASS（未被改变）** | `ChainedScheduler(LinearLR(start_factor=0.001, total_iters=warmup_step), …)` 构造于 `build_optimizer` 内，本次移植未触及该函数（diff 中 `config_holobrain_common.py` 的 3 个 hunk 均不在此函数内）。新参数组即 group 1，与既有非 VLM 参数共用同一条 lr 曲线 |
| EMA / 两阶段 | ✅ **N/A** | A 与宿主此路径均无 EMA/teacher-target 分支 |
| 就地操作 | ✅ **PASS** | `grep -rnE "\.(add\|mul\|div\|clamp\|copy\|scatter\|masked_fill)_\("` → 无匹配。`wrapper.py:249` 用 `scatter`（非原地）；`:212`/`:218` 先 `list()`/`dict()` 浅拷贝再写 |
| **开关全关时的构建** | ✅ **PASS** | 关闭态 `total_params` = 1,136,284,265，与基点**完全相等**；峰值显存亦相同（7.4683 GiB） |
| **数据增广 / collate 一致性** | ✅ **PASS** | 关闭态 batch key 集合 **14 个，与基点逐个相同**；`step_index` 仅在开启态出现（15 个）。白名单追加受 `if enable` 保护 |

---

## 7. 无法验证：外部真实 ckpt 加载

**做了什么**：尝试把 bucket 上的真实定版权重灌进移植后的模型——
这比移植方「自造 state_dict」的做法严格得多，是我在计划里点名要补的。

```
/horizon-bucket/robot_lab/users/kun01.wu/robo_orchard_lab/ckpts/
    holobrain_robodojo_posttrain_v9/checkpoint_20000/model.safetensors   (2.84 GB)
```

**为什么验不了**：架构不匹配，**与 MemoryVLA 无关**。该 ckpt 是 **v9**（Qwen2.5-VL），
当前 config 默认是 **v10**（Qwen3.5-2B）：

```
size mismatch for vlm.model.visual.blocks.23.attn.qkv.weight:
    checkpoint torch.Size([3840, 1280])  vs  model torch.Size([3072, 1024])
size mismatch for vlm.model.language_model.layers.0.mlp.gate_proj.weight:
    checkpoint torch.Size([11008, 2048]) vs  model torch.Size([6144, 2048])
    ...（`vlm.*` 全线不匹配，`memoryvla.*` 与 decoder 侧无一冲突）
```

**顺带独立证实了移植方的说法**：他们说「v10 warm-start 权重在一个 URL 后面，本机无外网」，
——属实。`config_holobrain_common.py:117` 的 v10 `checkpoint=` 确实是
`http://pfs-svcspawner.bcloud-bj-zone1.hobot.cc/...` 一个 http URL。
**他们退而自造 state_dict 是合理的，不是偷懒。**

**该怎样才能验**：① 把 config 切回 v9 段（`vlm_pretrain` 与 `embed_dims` 一并切）后
用 `checkpoint_20000` 加载；或 ② 在有外网的机器上取到 v10 的 checkpoint_60 再灌。
两者都超出本次审查的只读范围与算力预算。

**风险评估**：**低**。§6 已用 key 集合差分给出了这个问题的实质答案——
移植只增加 68 个 `memoryvla.*` 张量，不删不改形状，且该结论建立在 A 档
（关闭态 ≡ 移植前，逐位）之上，因此**不循环**。真实 ckpt 只会更换权重数值，
不会改变 key 集合的拓扑。

---

## 8. 小结

| 项 | 结论 |
|---|---|
| 空白① 真实 optimizer 分组 | ✅ **正确**（68 张量全进 group 1，0 张量掉队）——移植方没验，但结果对 |
| 空白② 参数真的更新 | ✅ 优化器通路正确；真实路径上因 P0-1 全部冻结（0/68） |
| 空白③ worker 随机流 | ✅ **PASS**，且有阳性对照证明测试有效 |
| ckpt 兼容 / 反向兼容 | ✅ **PASS**，且非循环 |
| 外部真实 ckpt | ❌ 无法验证（v9/v10 架构不匹配），风险低，已给出验证方法 |
| 就地操作 / 关闭态构建 / collate 一致性 / lr scheduler / 梯度不回流历史 | ✅ 全部 PASS |
| **真实训练路径上方法是否生效** | ❌ **否——精确恒等，0/68 参数被更新（P0-1）** |

**工程健壮性这一维度，除 P0-1 外全部通过，且有两项（optimizer 分组、worker 流）
是移植方结构上验不到而本次补上并确认正确的。**
