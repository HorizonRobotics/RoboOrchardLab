# 调查:推理侧 `step_index` 比训练侧小 32 倍(2026-08-13)

**状态**:缺陷已确认、已修、单元验证通过;**AIDI 真跑的行为确认待做**(见 §8)。
**本文档的结论与直觉相反**:修复**不预期**提高成功率,在 stride1 权重上**预期持平或更差**。
理由在 §5,那不是免责声明,是一个可被证伪的预测,写在跑之前。

上级文档:`docs_analysis/memoryvla/HANDOFF-20260812.md`(它的 §"记忆步长"一节原本写
「timestep 单位是对齐的」,已订正并链到本文)。

本文所有路径以两个仓根为前缀,不再重复:

- `L = /home/users/kun01.wu-labs/git_repo/robo_orchard_lab`
- `R = /home/users/kun01.wu-labs/git_repo/RoboDojo`

行号对应 2026-08-13 的工作树(`L` 侧含本次修复 commit `8c1b324e`)。

---

## 1. 一句话

评测时喂给模型的 `step_index` 是**前向次数**(一条 800 帧的 episode 上取值 `0…24`),
而训练时喂的是**数据集里的真实帧号**(`0…800`)。`use_timestep_pe=True`,
这个数会被正弦编码后加进记忆检索的注意力 ⇒ **推理与训练的时间位置编码差 32 倍尺度**。

---

## 2. 改动前:评测时 `step_index` 是怎么变成 0…24 的

### 2.1 环境侧的循环确实每帧都调 `update_obs`

`L/projects/holobrain_internal/common/holobrain_robodojo_policy/deploy.py:70-90`

```python
def eval_one_episode(TASK_ENV, model_client):        # :70
    model_client.call(func_name="reset")             # :71
    pending = []
    while not TASK_ENV.is_episode_end():             # :74
        obs = TASK_ENV.get_obs()                     # :75
        ...
        model_client.call(func_name="update_obs", obs=obs)      # :79
        actions = model_client.call(func_name="get_action")     # :80
        pending = []

        for action_idx, action in enumerate(actions):           # :83
            TASK_ENV.take_action(action)                        # :84
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break                                          # :85-86
            next_obs = TASK_ENV.get_obs()                       # :87
            ...
            model_client.call(func_name="update_obs", obs=next_obs)  # :90
```

一条 800 帧、`valid_action_step=32` 的 episode 里,这段代码调 `update_obs`
**约 800 次**(外层 25 次 + 内层 25×31 次)。所以"每帧都更新观测"这个意图**在环境侧是成立的**。

### 2.2 但这些调用**在客户端就被吞掉了**

`R/XPolicyLab/client_server/ws/model_client.py:60-62`

```python
if func_name == "update_obs":
    self._latest_obs = obs
    return None
```

`update_obs` **根本不发包**,只把 obs 存进**客户端**的 `self._latest_obs`。
(`update_obs_batch` 同理,`:81`。)

### 2.3 ws 协议里没有 obs-only 消息类型 —— 所以这不是配置问题

`R/XPolicyLab/client_server/ws/protocol/messages.py:8-22`

```python
class MessageType(str, Enum):        # :8
    HELLO / HELLO_ACK               # :9-10
    PREPARE_CASE / PREPARE_CASE_ACK # :11-12
    RESET / RESET_RESULT            # :13-14
    INFER / INFER_RESULT            # :15-16
    TRIAL_END / TRIAL_END_ACK       # :17-18
    HEARTBEAT / HEARTBEAT_ACK       # :19-20
    CLOSE / ERROR                   # :21-22
```

**十四种消息里没有一种能"只送一帧观测、不要动作"。** 唯一能把 obs 送到策略进程的是 `INFER`。

### 2.4 `INFER` 把 `update_obs` 与 `get_action` 绑成一次

`R/XPolicyLab/client_server/ws/model_server.py:186, 230-252`

```python
if frame.message_type == MessageType.INFER:      # :186
    ...
    update_obs = getattr(self.model, "update_obs", None)   # :230
    get_action = getattr(self.model, "get_action", None)   # :231
    if callable(update_obs) and callable(get_action):      # :232
        ...
        update_result = update_obs(observation)            # :236 / :244
        ...
        result = get_action()                              # :239 / :249
```

⇒ **策略侧的 `update_obs` 每次 `INFER` 被调一次,也就是每次前向一次。**
内层那 25×31 次调用是纯浪费(唯一副作用是 `_STREAM` 那条 piggyback 攒缩略图,
见 `deploy.py:23-32` —— 那段注释本来就写明了这个吞掉行为,只是没有人把它追到
`step_index` 上)。

### 2.5 于是 `_env_step` 计的是前向数,却被当成帧号用

`L/.../holobrain_robodojo_policy/deploy_policy.py`(修复前)

```python
self._env_step = 0                                  # 注释声称这是 "the exact env frame index"

def update_obs(self, obs):
    self._obs = obs
    self._env_step += 1                             # ← 每次前向 +1,不是每帧

...
model_input["step_index"] = [max(0, self._env_step - 1)]
```

### 2.6 这个数唯一的下游

- `L/robo_orchard_lab/models/memoryvla/wrapper.py:117` `timestep_key: str = "step_index"`
- 训练侧同名量来自 `L/robo_orchard_lab/dataset/robodojo/robodojo_lmdb_dataset.py:235`
  `"step_index": step_index` —— 由 `_get_indices(index)`(`:152`)给出,
  是 **episode 内的绝对帧号**。
- `L/projects/holobrain_internal/common/configs/config_holobrain_common.py:55`
  `use_timestep_pe=True`。

**结论:训练喂 0…800,推理喂 0…24,同一个 `use_timestep_pe` 通路。**

---

## 3. 改动后

`L/.../holobrain_robodojo_policy/deploy_policy.py`(commit `8c1b324e`)

| 行 | 改动 |
|---|---|
| `:387-391` | 新增 `self._step_index_stride = 1 if os.environ.get("HOLOBRAIN_STEP_INDEX_MODE") == "forward" else int(cfg.valid_action_step)` |
| `:604` | `update_obs`:`+= 1` → `+= self._step_index_stride` |
| `:611` | `update_obs_batch`:同上 |
| `:565-566` | `step_index = [max(0, self._env_step - self._step_index_stride)]` |
| `:368-382` | 把那段声称"exact env frame index"的注释换成事实与实测数字 |

`cfg.valid_action_step` 一定存在:dataclass 默认 32(`:64`)、`deploy.yml:25` 也是 32、
`robodojo_eval.py:210/946-948` 通过 `HOLOBRAIN_VALID_ACTION_STEP` 覆盖、
`:342-343` 断言为正 ⇒ **不存在 `AttributeError` 风险**。

**精确性**:每次前向下发正好 `valid_action_step` 个动作
(`deploy_policy.py:582-588` 截断并断言长度足够),所以按步长累加是精确的 ——
**唯一例外是 episode 提前结束的最后一个不完整 chunk**,而那时 episode 已经结束了。

**保留旧行为的开关**:`HOLOBRAIN_STEP_INDEX_MODE=forward` 退回 `+= 1`。
理由是已有 17 个评测格子全部在旧编号下测得,**复现它们必须可行**。默认值是修复后的行为。

---

## 4. 证据(不是推理)

**① 真跑日志。** `deploy_policy.py:665-666` 每条 episode 结束时打
`logger.info("policy reset: %s", self.memory_stats())`,其中含 `env_step`(`:662`)与
`eval_forwards`。一个 50 条 cover_blocks(`step_lim` 800)的跑里,
**每一条**的 `env_step` 都等于 `eval_forwards`:14/14、16/16、17/17、18/18,
其中 **40 条是 `25 == 25`**。若 `_env_step` 真是帧号,800 帧的 episode 不可能是 25。

**② 单元验证**(纯 python,`object.__new__` 造 stub,不起 Isaac Sim、不占 GPU):

| # | 检验 | 结果 |
|---|---|---|
| 1 | 默认(stride=32)前 4 次 `step_index` | `[0, 32, 64, 96]` ✅ |
| 2 | `HOLOBRAIN_STEP_INDEX_MODE=forward` | `[0, 1, 2, 3]` ✅ |
| 3 | 25 次前向后 `_env_step` | `800`(修复前 25)✅ |
| 4 | 源码里确有该开关与 `int(cfg.valid_action_step)` 默认 | ✅ |
| 5 | batch 路径用同一步长 | ✅ |

---

## 5. 后果:**为什么坏成这样,cover_blocks 还有成功率?**

三层原因,越往后越关键。

### 5.1 第一层:这个量走的是旁路,碰不到动作

**`L/robo_orchard_lab/models/holobrain/structure.py` 里 `step_index` 与 `timestep`
零次出现。** ⇒ 3D 特征提取(`:469`)、spatial enhancer(`:471`)、10 步扩散
decoder(`:480`)**完全不吃这个量**。唯一消费者是 `MemoryVLAMemory`。

### 5.2 第二层:在记忆库里,它也只影响"注意力偏向哪一格历史"

`L/robo_orchard_lab/models/memoryvla/memory_bank.py:400-407`

```python
pe = self.timestep_encoder(hist_timesteps).unsqueeze(0)   # :400
pe = pe.repeat_interleave(N, dim=1)                       # :401
...
query = block(query, episode_mem + pe, episode_mem)       # :407
                     ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^
                     key(带 PE)        value(不带 PE)
```

三个性质:

1. **PE 只加在 key 上,不加在 value 上** ⇒ 取回的内容永远是干净的历史特征,
   时间码错了也污染不到内容;
2. **当前帧自己的 query 完全没有 PE** ⇒ 错的绝对偏移不会进入 query;
3. 正弦编码(`memory_bank.py:52-90`,`TimestepEmbedder.timestep_embedding`,
   `max_period=10000`)是**连续函数** ⇒ 任何 `t` 都合法,**不会越界报错、不会产生 NaN**。
   这解释了为什么这个缺陷能潜伏 —— 它没有任何响亮的失败信号。

⇒ 缺陷能造成的最坏情况是「检索时挑错了历史格子」,而**不是**「动作被破坏」。

### 5.3 第三层(最关键):错的编号**在真正要紧的那一维上恰好是分布内的**

看训练时库里到底装的是什么数。`dataloader_type="stream"`、原 `frame_stride=1`
⇒ 同一 episode 的样本按帧号连续进库,`mem_length=16`
⇒ **库里是一个宽约 16 的紧簇,相邻两条的帧号差 1**,簇的绝对偏移落在 0…800 任意处。

| | 相邻间距 Δt | 簇宽 | 绝对偏移 |
|---|---|---|---|
| 训练(stride1) | **1** | **~16** | 0…800 任意 |
| **旧(错)推理** | **1** ✅ | **~16** ✅ | 恒 ~0 ❌ |
| 修复后推理 | 32 ❌ | 768 ❌ | 0…768 ✅ |

**旧行为在「相邻间距」与「簇宽」这两维上与训练完全一致,只有绝对偏移错了**
—— 它等于让模型永远以为"现在还在 episode 开头"。而检索注意力需要区分的正是
「≤16 格历史里哪一格更相关」,**这个信息在旧编号下完好无损**。

⇒ **cover_blocks 上 23/100 的成功率不是巧合,也不是"碰巧躲过"** ——
是因为这个缺陷破坏的是绝对时间,而记忆检索在这个规模上主要依赖相对顺序。

### 5.4 一个反而让原结论更稳的推论

消融臂(base,`MemoryVLAMemory=False`)**没有记忆库**,`step_index` 连评测包都进不去:
`L/projects/holobrain_internal/common/export.py:71-101` 只给**接了记忆的**数据集配置
把 `step_index` 加进 `ItemSelection` 白名单(`config_robodojo_dataset.py:290-293`)。

⇒ **base 臂完全不受本缺陷影响,记忆臂受影响。**
那么 cover_blocks 上 **23/100 vs 1/100(p<0.0001)是记忆臂带着这个 handicap 赢的**
⇒ 修复只可能让该结论被**低估**,不会推翻它。

---

## 6. **改完会期望成功率更高吗?——不会。在 stride1 权重上预期持平或更差。**

这是本文档最反直觉的一节。修复让**绝对值**对了,代价是把**相邻间距**从 1 变成 32、
**簇宽**从 16 变成 768 —— 而这两维原本是对的(§5.3)。

定量算一遍。`token_size = 384`(评测包 `model.config.json` 实测),
`frequency_embedding_size = token_size // 4 = 96`,`half = 48`。
第 k 维的角频率是 `ω_k = 10000^(-k/48)` rad / 单位 index,k = 0…47:

| | ω | Δt=1 的相位变化 | Δt=32 的相位变化 |
|---|---|---|---|
| k=0(最高频) | 1.000 | 1.00 rad(平滑可分) | **32 rad ≈ 5.1 整圈 ⇒ 彻底 aliasing** |
| k=47(最低频) | 1.21e-4 | 1.2e-4 rad | 3.9e-3 rad |

**Nyquist 判据**:第 k 维能无歧义分辨间距 Δt 的条件是 `ω_k · Δt < π`。

- Δt=1:最高频 ω=1.0 < π ⇒ **48 维全部可分**。stride1 模型学会读顺序,靠的就是这些高频维。
- Δt=32:需要 `ω_k < π/32 = 0.0982` ⇒ `k > 12.1` ⇒ **k=0…12 共 13 维(48 维的 27%)
  发生 aliasing**,而那正是 Δt=1 下唯一能分辨相邻格子的那批维。

另一侧:最低频维在 t∈[0,24] 上总共只动 **0.0029 rad**(近似常数),在 t∈[0,768] 上动
**0.093 rad** ⇒ 修复确实补回了低频/绝对时间的信息,**但代价是废掉 27% 高频维的顺序信号**。

⇒ **`step_index` 修复与 stride32 训练是一对**:只有当训练侧写入间距也是 32(簇宽 512)时,
推理侧的 Δt=32 才是分布内的。**单独上任何一个,都只是换了一种错配。**

**预测(写在跑之前,以便被证伪)**

| | 旧 step_index(0…24) | 修复后(0,32,…) |
|---|---|---|
| **stride1 权重** | 已测 18%(seed0)/ 28%(seed1) | **预测持平或更差** |
| **stride32 权重** | 预测中等(绝对值错) | **预测最好(两维都匹配)** |

判据是**对角线(匹配)优于反对角线(错配)**。若 2×2 四格都落在 ±2 成功/50 的噪声底以内,
那说明 **timestep PE 在这个规模上根本不重要**,也是一个干净结论 ——
并且会顺带削弱"VAS 单调性由记忆合并次数解释"这条机制假设的一部分。

---

## 7. 对已有结果可比性的影响

- **17 个评测格子全部在旧编号下测得** ⇒ 与修复后的数**不可直接比较**,基线要重测。
  这是修这个缺陷的成本,必须一并说清。
- 复现旧格子:`HOLOBRAIN_STEP_INDEX_MODE=forward`。
- **不受影响的结论**:所有 base 臂的数字(§5.4),以及"记忆臂优于 base 臂"这个方向。
- **可能被重新解释的结论**:VAS 单调性(32 > 16 > 8)。VAS 越小,旧编号的绝对尺度错得
  越少(VAS=8 时是 0…100,更接近真值),所以"VAS 越大越好"与"尺度错得越多越好"在旧数据里
  **是混淆的**。2×2 正好能分开这两者。

---

## 8. 待验清单

| # | 判据 | 状态 |
|---|---|---|
| 1 | AIDI 真跑里 `policy reset` 的 `env_step` 从 ~25 变成 ~800 | ⏳ 待 job E1 |
| 2 | 对照格 `HOLOBRAIN_STEP_INDEX_MODE=forward` 仍打 ~25 | ⏳ 待 job E1 |
| 3 | 2×2 的对角线优于反对角线(§6 预测) | ⏳ 待 stride32 训练完(~08-14 18:50Z) |

**只改了环境变量不算生效** —— 和 VAS 那次一样,provenance 必须落在日志里。

---

## 9. 这次为什么能潜伏这么久(过程教训)

1. **注释被当成了事实。** 原注释写 "counting the former is the exact env frame index",
   前提是"`deploy.py` 每帧调一次 `update_obs`" —— 这个前提**在环境侧成立**(§2.1),
   只是调用没穿过客户端。**跨仓的调用链断点不在任何一侧的注释里。**
2. **没有响亮的失败。** 正弦编码接受任何 `t`(§5.2.3),不越界、不 NaN、不报错。
3. **同一个吞掉行为已经被写在代码里了**(`deploy.py:23-32`),却只被用来解释
   `HOLOBRAIN_STREAM_HISTORY` 为什么要 piggyback,没有人回头问"那 `step_index` 呢"。
4. **发现它靠的是产物**:比对 `policy reset` 里的 `env_step` 与 `eval_forwards`,
   发现两个本该差 32 倍的数**恒等**。和存储那边"只有产物不会骗人"是同一条原则。

## 10. 后记(2026-08-14):对齐了标签,却砍掉了内容

第 5 节预测「修完不会更好」,这一点被 13/100 vs 23/100(p=0.097)支持。但**为什么**
所有与 timestep 有关的操作都推不动成功率,要到 E4 才看清,而看清它靠的是一个此前
没人算过的量。

### 10.1 四次操作,一次都没动

cover_blocks · seed 0 · mem 包 · num_envs=1,每格 50 条:

| 操作 | 成功 | 对照 | p |
|---|---|---|---|
| step_index 由 0…24 改为 0,32,…(两 seed 合计) | 13/100 | 23/100 | 0.097 |
| `mem_length` 16 → 32(a0) | 7/50 | 9/50 | 0.79 |
| `consolidate_type` tome → fifo(a1) | 8/50 | 9/50 | 1.00 |
| 每帧前向 + ACT temporal ensemble(E4) | 6/50 | 5/50 | 1.00 |

**四个记忆库旋钮,没有一个把成功率推出噪声。**

### 10.2 被漏掉的那个量:记忆库能回看多远

`step_index` 的间距不只是编码尺度,它同时决定了库的**时间跨度**——
库长 16 条,每条之间隔多少帧,乘起来就是回看范围:

| | 两次写库间隔 | reach = 16 × 间隔 |
|---|---|---|
| 推理 · chunk(32 帧一次前向) | 32 帧 | **512 帧 = 20.5 s** |
| 推理 · perstep/ensemble(每帧一次) | 1 帧 | **16 帧 = 0.64 s** |
| 训练 · `stream_frame_stride=1`(现有权重) | 1 帧 | **16 帧 = 0.64 s** |
| 训练 · `stream_frame_stride=32`(进行中) | 32 帧 | **512 帧 = 20.5 s** |

两条推论,都不在原计划里:

1. **每帧前向"和训练完全对齐"的代价,是把回看范围从 20.5 秒压到 0.64 秒。**
   对一个 Memory 维度的 benchmark,0.64 秒近乎没有记忆。E4 因此不是一次失败的
   尝试,而是一个**结构上注定**的结果——它对齐了标签,砍掉了内容。
2. **现有权重是在 0.64 秒窗口下训练的,而 chunk 模式推理一直给它 20.5 秒。**
   此前所有拿到成绩的格子都在"超发"记忆:时间戳标错,但内容比训练时多得多。
   记忆臂 13/100 vs 消融臂 1/100 这个结论,是在这种超发状态下取得的。

⇒ **`stream_frame_stride=32` 是唯一在两个轴上都与 chunk 模式推理对齐的配置**
(间距 32 ✓,reach 512 帧 ✓),也就是 2×2 的右下角。第 5 节说「step_index 修复与
stride32 训练是一对」,现在可以说得更准确:**它们是同一件事的两半,而缺的那半是
reach,不是编码尺度。**

### 10.3 E4 唯一显著的差异,在成功率之外

「拿到任何分」(成功 + 部分得分)ensemble **23/50** vs chunk+修复编号 **39/50**,
**p=0.0018**。与新增观测量一致:cover_blocks 上 `action_path` 均值 69.8(49 条),
而 chunk 模式的参考值是 103.8(仅 1 条)。**每帧只执行 chunk 的第 0 个动作会让
机械臂走得更少**——`a[0]` 是最保守的一个,chunk 模式靠执行后面幅度更大的动作
绕开了它;ACT 的融合把大部分拉了回来(perstep 44.7 → ensemble 69.8),但不足以
补平,且最终成功次数不变。

⚠️ 「恢复到 86%」那个说法出自 1 条 vs 1 条,**不成立**;49 条的均值是 69.8,而
chunk 那侧至今只有 1 条参考。

## 11. 2×2 结算(2026-08-15,E8 四格跑完)

cover_blocks · 两 seed 合并 · 每格 100 条 · `num_envs=1` · chunk 模式:

| | 旧编号(0,1,2,…) | 修复编号(0,32,…) |
|---|---|---|
| **stride1** | **23/100**(匹配) | 13/100 |
| **stride32** | 1/100 | **5/100**(匹配) |

四格 provenance 全 PASS(`fixed` 格 `env_step=800`、`forward` 格 `=25`,`bank_len=16`)。

### 11.1 预登记的判据成立,但它是交互项

第 2 节写的判据是「**对角线(匹配)优于反对角线(错配)**」。这个claim **不是**任意两格
相比 —— 两行基线相差一个量级,跨行合并「匹配」与「错配」等于拿 stride1 的格去比
stride32 的格。它是 2×2 的**交互项**,必须按交互项检验。

Woolf 齐性检验(Haldane-Anscombe 0.5 修正,因为有一格只有 1 次成功;该修正使检验偏保守):

```
stride1  OR(旧/修复) = 1.97   (>1:旧编号更好)
stride32 OR(旧/修复) = 0.26   (<1:修复编号更好)
chi2 = 4.02   p = 0.045
```

**两行各自偏向自己的匹配格,方向相反** —— 正是「库内相邻条目的间距要与训练一致」
所预测的形状。行内单独看都不显著(stride1 p=0.097,stride32 p=0.212);
**交互项才是判据**。

### 11.2 但训练臂的主效应压倒一切

```
训练臂  stride1 36/200 vs stride32 6/200   p = 8.3e-07
编号    旧 24/200 vs 修复 18/200           p = 0.415
```

**stride32 最好的格(5/100)远低于 stride1 最差的格(13/100)。** 匹配带来的好处被完全淹没。

编号没有主效应,与 11.1 自洽:**要紧的是"对齐"而不是绝对数值**。这也是为什么单独修
`step_index` 会让成绩下降 —— 它把 stride1 权重从匹配推到了错配(23/100 → 13/100)。

### 11.3 2×2 这个设计救了这次实验

单臂实验会得出两个**都对但都不完整**的结论:只看 stride1 行 ⇒「修复无用甚至有害」;
只看跨行 ⇒「stride32 更差」。真相是两个因子**交互**,而单臂测不出交互。

### 11.4 必须与结论一起引用的脆弱性

1. **p=0.045 压线**,且交互估计**压在 stride32 那两格(1 和 5 次成功)上** ——
   那里几乎没有统计功效,多一次成功就会明显移动结论。
2. **没有 stride32-base 控制臂** ⇒ **11.2 的主效应无法归因**:分不开
   「跨步采样伤害了记忆机制」与「跨步采样单纯是更差的训练数据」。
3. 训练随机性每臂 **n=1**;±2/50 噪声底;同一 layout 两次跑测出过 0.05 与 1.0。

### 11.5 这与前四个零结果如何自洽

`mem_length` 16→32(p=0.79)、tome→fifo(p=1.00)、per-step+ensemble(p=1.00)
都在改**记忆库的容量/合并/写入频率**,而 11.2 说主导因素是**训练数据的采样方式**,
11.1 说 PE 只在"对齐与否"这一个二值维度上起作用。⇒ **时间维度上能调的都调过了**,
且第 10 节的算式显示它已经饱和(16 槽 × 32 帧 = 512,覆盖 800 帧 episode 的 64%,
25 槽即可全覆盖)。

⚠️ 唯一的例外:**`mem_length=32` 从未被真正测过** —— 800 帧只有 25 次前向,
库最多 25 条、**从不淘汰**(a0 实测 `bank_len max=25`)。那个 p=0.79 测的是
「库不用淘汰」,不是「库更大」。
