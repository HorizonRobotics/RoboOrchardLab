# PORT-STATUS — MemoryVLA → HoloBrain

**日期**：2026-08-03（`date +%Y-%m-%d`）
**日期（本轮修复）**：2026-08-04
**总判定**：**不自评**。上一轮自评 PASS，独立复审判 🔴 REJECT（P0-1：`episode_stream_sampler`
是死开关）。本轮是范围锁死的修复 + 数值证据，**裁决交给下一轮独立复审**。
逐条应答见 `08-review-response.md`。

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

## 侵入度：**L1**，触及宿主已有文件 **5 个**，**0 删除**

> **订正（2026-08-04）**：原记 4 个文件。移植当时漏掉了 `common/train.py`，
> 而那正是 P0-1 —— sampler 开关没有读取者。修复后 `train.py` 是第 5 个。

| 文件 | 档 | 改动 |
|---|---|---|
| `models/holobrain/structure.py` | L1 | 一个 config 字段 + 一行 `build` + 一个 `if` + 一次调用 （判断/方案，非实测） |
| `models/holobrain/structure_qwen3_5.py` | L1 | 一行 `build`（它跳过父类 `__init__`，必须单独加） （判断/方案，非实测） |
| `configs/data_configs/config_robodojo_dataset.py` | L1 | 开关打开时给 ItemSelection 白名单加 `step_index` （判断/方案，非实测） |
| `configs/config_holobrain_common.py` | L1 | `cfg.memoryvla.*` 命名空间 + `_build_memoryvla_cfg()` （判断/方案，非实测） |
| `common/train.py` | L1 | 一个开关判断选 batch sampler + 一条装配期护栏调用（2026-08-04 修复 P0-1）|

新增文件（L0）：`models/memoryvla/{__init__,memory_bank,wrapper,sampler}.py`、
`configs/dataset_specs_memoryvla_robodojo_memory.py`、`docs_analysis/`。
**未触发 Gate B**（无 L3 改动）。**未触发 Gate E**（E0）。

## 验证结果（2026-08-04 全部重测；命令与证据见 `08-review-response.md`）

> **旧的五档数字全部作废。** 上一轮全部产自 `run_gears.py`：A/B/D 档与 5 个消融跑的是
> `--sampler sequential`（一个仓库里不存在的手写连续索引列表），E 档跑的是自建的
> `MemoryVLAEpisodeStreamBatchSampler`。**宿主没有任何路径能到达那两种装配。**
> 本轮所有档位一律从 `train.py` 真实入口进，观测装置只注入不构造。

| 档 | 判据 | 结果 |
|---|---|---|
| 第 0 步 确定性 | 同配置跑两遍自比 | **step 0 精确 `0.000000e+00`；20 步内峰值 `2.899e-04`**（step 11）。真实入口**不逐位可复现**，见下节 |
| **A 关闭态等价** | ① step 0 严格 0 ② 全程 ≤ 实测地板 | commit `701679a9`：step0 **`0.000000e+00`**，峰值 `1.249e-04`；commit `166b8756`：step0 **`0.000000e+00`**，峰值 `1.554e-04`。**两者都低于两次同配置 baseline 之间的 `2.899e-04`**。参数量 `1,136,284,265` 与移植前一致；关闭态 sampler 仍是 `DistributedBatchFlagSampler`；`memoryvla.*` 张量 **0 个**（模块根本不构建）→ **PASS** |
| **B 开启态** | 走真实入口，grad 与参数移动 | sampler 链实测 `['MemoryVLAEpisodeStreamBatchSampler']`；每 batch **1** 个 episode（原 4/4）；grad **0 None / 0 零 / 68 非零**（原 64/4/0）；参数移动 **62→65 / 68**（原 0/68）；恒等间隙 per `1.297` / cog `1.124`（原 `1.19e-07` / `5.96e-08`）；68 张量全在 optimizer group 1，0 个游离 → **PASS** |
| **C 数值对齐** | `atol < 1e-5` | **10/10 逐位一致（`0.000e+00`）**，修复前后各跑一次，结果相同 → 改动未溢出范围 → **PASS** |
| **D 资源** | 量级正常 | 参数 `1,136,284,265 → 1,143,751,529`（**+7,467,264 / +0.657%**）；峰值显存 `8.9767 → 9.3024 GiB`（**+0.3257 GiB**）；**墙钟不下结论**，见下 |
| **E Memory 冒烟** | 跨过 episode 边界 | 见 `08-review-response.md` |
| 护栏自验 | 故意配错必须 raise | 见 `08-review-response.md` |
| 静态判据 preflight | 三条红一起变绿 | `18106b05`：`ORPHAN` + `UNUSED` + `DRIFT` → **FAILED**；`166b8756`：0 finding → **PASSED**。同一版本工具、同一组豁免 → **PASS** |

### 确定性：真实入口不是逐位可复现的，判据因此改成两档

上一轮记「地板恰为 0，故 A 档用严格判据」。**那是 harness 的性质，不是宿主的性质**——
`run_gears.py` 用 `lr=0`，权重不动，逐 step 值是「数据 + seed」的纯函数，误差没有累积的机会。

走真实入口（真 optimizer、真 lr、`num_workers=4`）实测：

```
step 0   0.000000e+00     ← 前向逐位一致
step 1   0.000000e+00     （单分量 1.788e-07）
step 11  2.899170e-04     ← 20 步内峰值
```

误差从**反向/optimizer 的 float32 非确定性归约**进来，前向本身精确。
开 `cudnn.deterministic` + `use_deterministic_algorithms(warn_only=True)` 只把峰值压到
`1.564e-04`，**压不到 0**（有算子没有确定性实现，warn_only 下继续走非确定性路径）。

**所以 A 档判据是两档**：

1. **严格档**：step 0 的 7 个分量与 total 必须**精确** `0.000000e+00`。
   这才是 A 档真正要回答的问题——接线有没有改动关闭态的前向。
2. **地板档**：其余步 ≤ 实测地板 `2.899e-04`。

**阳性对照**（没有阳性对照的通过 = 未验证）：开启态与关闭态的恒等间隙相差 **7 个数量级**
（`1.19e-07` → `1.297`），远在地板之上；也就是说这套判据能分辨的变化，比「把开关打开」
小得多。

### D 档：墙钟不用来下结论

两次**完全同配置**（都是关闭态）的 baseline，墙钟 `260.9 s` vs `203.6 s`，**差 22%**。
卡是共享的（本次同卡上有同事进程，另有本人的 `collect_data` 作业占着别的卡）。
所以 D 档只报参数量与显存这两个可信量，**墙钟只记录不解释** ——
这也证实了复审 P3-1：上一轮「开启 +10% 时间」落在噪声内，结论不成立。

要真测时间需独占卡，或改用 CUDA event + 多次取中位数。

### 一条新的运行期硬要求：`ulimit -n`

默认软限 **1024**，这套数据会击穿它：6 个 RoboDojo 任务 × 3 个 LMDB env（meta/image/depth）
× (4 worker + 父进程)。**接上 episode sampler 后更紧**——`_episode_spans` 要走遍全部
328,975 帧，于是**父进程也初始化 LMDB**，而宿主 sampler 从不这么做。

症状极具欺骗性：worker 里是 `OSError: [Errno 24] Too many open files`，
浮到上层变成 `RuntimeError: Pin memory thread exited unexpectedly`，
**看起来像 dataloader 偶发抖动，不像资源限制**。本次 A 档头两次尝试、B 档三次尝试全折在这上面。

→ **跑训练前 `ulimit -n 65536`**（硬限 1048576，普通用户可自行提升）。

## 新增 config 字段

`cfg.memoryvla.*`：`enable`(False) · `use_perceptual`(True) · `use_cognitive`(True) ·
`dataloader_type`("stream") · `group_size`(16) · `mem_length`(16) · `retrieval_layers`(2) ·
`use_timestep_pe`(True) · `fusion_type`("gate") · `consolidate_type`("tome") ·
`update_fused`(False) · `episode_stream_sampler`(True)。
**默认 `enable=False`，此时模块根本不构建。**

> **订正（2026-08-04）**：`episode_stream_sampler` 的读取者是
> `common/train.py`（DataLoader 构造处）。判据是 **`enable ∧ episode_stream_sampler`** ——
> 该键 ship 值为 `True` 但**挂在 `enable=False` 之下**，只读它会让全关配置也换掉 sampler。
> 装配期护栏 `assert_episode_stream_wired()` 会在开启态校验实际迭代的 sampler 类型。

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
5. **训练动力学变了，且原 A 档论证覆盖不到**（2026-08-04 新增）。接上 sampler 后每 batch
   从 4 个 episode 变成 **1 个**：梯度方差、epoch 内样本相关性、归一化层统计全都与关闭态不同。
   A 档证明的是「关闭态没变」，**不是**「开启态的训练行为已被验证」。
   这是新的遗留风险，不是「接完线就回到已验证状态」。
6. **DDP 多了一层新风险**（2026-08-04 新增，**复审也没记这条**）。
   `MemoryVLAEpisodeStreamBatchSampler` 按 episode 分片（`spans[rank::num_replicas]`），
   而 episode 长度差异极大（中位 276 → 1203 帧），所以**各 rank 的 `__len__` 不相等**、
   收尾不齐。本机任意两卡 gather 必崩 `ILLEGAL_ADDRESS`，无法本地验证。
7. **外部真实 ckpt 加载仍未验证**：bucket 上只有 v9，config 是 v10，`vlm.*` 全线 size
   mismatch，且 v10 warm-start 在 http URL 后面而本机无外网。本轮所有档位一律
   `checkpoint=null`（随机初始化 + 本地 `vlm_pretrain`），与移植当时同口径。
8. **`ulimit -n` 必须提到 65536**（2026-08-04 新增）。默认 1024 会被这套数据击穿，
   症状伪装成 dataloader 偶发抖动。详见「验证结果」末节。

## 下一步建议

1. 真要用它训练：先把 `reset()` 接进评测循环（遗留 1），再跑一次 Memory 六任务的完整训练，
   与 `07_results.md` 里 20k/100k 的 Memory 维度数字对比 —— 那才是这次移植值不值的答案。
2. 训练时确认 `episode_stream_sampler=True` 且 `dataloader_type="stream"`。
   **订正（2026-08-04）**：这条建议在写下时是**无法执行**的 —— 该键当时没有读取者，
   设成什么都一样。现在它有读取者了，而且两者不匹配**会直接 raise**，不再是静默 no-op。
3. 若上多卡，先单独验 DDP 的 unused-parameter 行为。

## 合规

- A 为 **MIT**，宿主为 **Apache-2.0**，兼容，可移植、可分发。
- ⚠️ A 的 `pyproject.toml:15` 写 `license={file="LICENSE"}`，但**仓库里没有 LICENSE 文件**；
  MIT 的判据来自 `pyproject.toml:21` 的 classifier。已在 `00-phase0-record.md` 记录。
- 搬运处逐段留出处：`# [port:memoryvla] from MemoryVLA@0eef5c3 vla/memory_vla.py:L<a>-L<b>`，
  文件头保留 MemoryVLA 的出处与许可证声明。
- 第三方权重：本次**未引入任何新权重**。
