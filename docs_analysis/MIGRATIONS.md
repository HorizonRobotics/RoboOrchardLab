# 移植记录（累积）

**这份文件是写给「下一个做移植的自己」的。** 每次往 `robo_orchard_lab` 里搬一个外部方法，
就在末尾追加一节。读它比读 diff 快，因为它记的是**为什么这么放**，以及**哪些坑不用再踩一遍**。

## 已经定下来的约定（第 1 次移植时确立，后续沿用）

| 事项 | 约定 |
|---|---|
| 新代码放哪 | `robo_orchard_lab/models/<method>/`，一个方法一个目录，不往宿主目录里掺 |
| 数据集 spec | **新开** `configs/dataset_specs_<method>_*.py`，自包含、不 import 别的 spec；用 `--kwargs '{"dataset_specs": "..."}'` 选择 |
| 命名 | 类名带方法前缀（`MemoryVLAMemory`）；config 用嵌套命名空间 `cfg.<method>.*` |
| 开关语义 | **关闭 = 不构建**，不是「构建了但不用」。见下面「最重要的一条」 |
| 分析文档 | `docs_analysis/<method>/`，事实必须带 `cite:`，读不到的写 `未确认` |
| 参考数值 | `$ROL_JFS/port/<method>/ref/*.npz`，**连 state_dict 一起存** |
| 抽象 | **等到第二次**。本文件末尾点名了哪几处将来该提炼，但现在一处都不提炼 |

### 最重要的一条：关闭态必须是「不构建」

`cfg.<method>` 为 `None` → `build()` 返回 `None` → 模块根本不存在。
**不要**做成「构建一个 `enable=False` 的模块然后早退」。两个原因：

1. 构造任何 `nn.Module` 都会消耗全局 RNG，后续所有随机数错位，
   「关闭态与移植前逐 step 一致」当场失败，而且症状看起来像「移植改坏了原逻辑」。
2. MemoryVLA 的 `GateFusion` 初始化是 `normal(0,1e-3)` 而不是恒等，
   所以「构建但不用」这条路本身也不是恒等。

这条已经用实测验证：移植前 `3ce31c0c` 与移植后 `111981a5`（`enable=False`）
跑 20 step，**逐 step 差值 0.000e+00，参数量完全相同**。

> **订正（2026-08-04）**：`0.000e+00` 这个数产自 harness 路径（`run_gears.py`，`lr=0` 权重不动），
> **不是宿主真实入口的性质**——真实入口不逐位可复现，见教训 9。
> 但**这条约定本身仍然成立**，而且现在有更强的证据：真实入口下关闭态 5 个 run、10 组两两比较，
> **逐样本 id 序列全部一致、参数量严格相等、`sys.modules` 里根本没有 port 包**。
> 换句话说，站住的是「不构建」这条设计，不是那个 `0`。

---

## 1. MemoryVLA — 感知/认知双记忆库（2026-08-03）

| 项 | 值 |
|---|---|
| 论文 | MemoryVLA: Perceptual-Cognitive Memory in VLA Models for Robotic Manipulation，arXiv:2508.19236v2 |
| A 的 repo | `github.com/shihao1895/MemoryVLA` @ `0eef5c3`，**MIT**（⚠️ `pyproject.toml:15` 指向 LICENSE 文件，但仓库里没有该文件） |
| 宿主基点 | `3ce31c0c`，分支 `port/memoryvla` |
| 依赖档位 | **E0**，宿主主环境零改动，差异包清单**为空** |
| 移植了什么 | `TimestepEmbedder` / `CrossTransformerBlock` / `GateFusion` / `BottleneckSE` / `CogMemBank` / `PerMemBank` |
| 没移植什么 | `MemoryVLA` 壳类、`ActionModel`+DiT 动作头、`prismatic` 相关、FSDP 策略、overwatch logger（协议红线：不移植 A 的基础设施） |

### 改了宿主哪几处（共 5 个文件；`train.py` +38/−6，`sampler.py` +94/−1）

> **订正（2026-08-04）**：原记 4 个文件。缺的第 5 个是 `common/train.py`，
> 也就是 P0-1：`episode_stream_sampler` 这个键当时没有任何读取者。

> **订正 2（2026-08-04）**：本标题原写「**0 删除**」，不成立。实测
> `train.py` **+38/−6**、`sampler.py` **+94/−1**、`wrapper.py` +118/−0。
> 那 6 行是**代码位移**：`DistributedBatchFlagSampler(...)` 从 `DataLoader` 的实参提为局部变量，
> 构造参数一字未动。**但「没动过」不能靠读 diff 判**——已用精确判据实测
> （逐样本 id 序列、batch key 集合、参数量、峰值显存、sampler 链，五项全同）。
> 详见 `memoryvla/PORT-STATUS.md` 侵入度节。

| 文件:位置 | 改成什么形状 | 能拿到什么上下文 |
|---|---|---|
| `models/holobrain/structure.py:_forward` | 一个 `if self.memoryvla is not None:` + 一次调用 | `feature_maps`、`text_dict`、完整 `inputs` batch |
| `models/holobrain/structure.py:__init__` + `Config` | `self.memoryvla = build(self.cfg.memoryvla)` + `memoryvla: MODULE_TYPE \| None = None` | — |
| `models/holobrain/structure_qwen3_5.py:__init__` | 同上一行 | — |
| `configs/data_configs/config_robodojo_dataset.py:build_transforms` | 开关打开时给 `ItemSelection` 白名单加 `"step_index"` | `config` dict |
| `configs/config_holobrain_common.py` | `cfg.memoryvla.*` 命名空间 + `_build_memoryvla_cfg()` + 传给 `model_config(...)` | — |
| `common/train.py:DataLoader` | 一个 `if enable and episode_stream_sampler:` 选 batch sampler；关闭态那支构造参数一字不动 | `config` dict、`train_dataset` |
| `common/train.py:SimpleTrainer` 之后 | 一条 `assert_episode_stream_wired(config, trainer.dataloader)`（查 post-`prepare()` 的对象）| `config`、trainer |

### ⚠️ 下次一定会再踩的坑

1. **`structure_qwen3_5.py` 必须单独改一遍。** 它的 `__init__` 走
   `super(HoloBrain_Qwen2_5_VL, self).__init__(cfg)`，**跳过父类 `__init__`**、自己重列了一遍
   `build(...)`。只改 `structure.py` 的话，v9 有该属性而 v10 没有，
   报错会是 `AttributeError`，出现在你以为早就跑通的地方。
   → **任何给 HoloBrain 加子模块的移植都要改两处。**
2. **`ItemSelection` 是白名单。** 数据集 `__getitem__` 产出的字段会被
   `config_<dataset>_dataset.py` 里的 `keys=[...]` 静默丢掉。
   本次要的 `step_index` 数据集**本来就有**，只是没进白名单；`uuid` 恰好在白名单里。
   → 先看白名单，再考虑改数据集。
3. **`collate_batch_dict` 按第一个样本的类型分派。** `step_index` 在第 0 条 episode 上是
   Python `int`、之后是 `np.int64`；前者 collate 成张量，后者两个分支都不匹配、
   落到 `else` 变成普通 list。**消费方要同时接受 Tensor 和 list。**
4. **`uuid` 是全局唯一的**（`swap_T_arx_x5_episode_0000000`），可以直接当 episode key。
   不要费劲去拼 `(lmdb_index, episode_index)`——顺带一提，`_get_indices` 返回的
   `episode_index` 其实是 **`str`** 不是 int。
5. **宿主的图像张量在数据集出口是 NHWC，在模型里是 NCHW**，中间由
   `BaseDataPreprocessor` permute。看 `inputs["imgs"].shape` 时先确认自己在哪一侧。
6. **符号链接在 `projects/holobrain_internal/common/`**（`ckpt` / `data` / `urdf` / `workspace`）。
   任何要跑训练的脚本 **cwd 必须是那个目录**，否则 `./urdf/...` 解析失败。
   开 git worktree 做移植前基线时，这 4 个链接要手工补。

7. **config 键必须有真实读取者 —— 写进文档、导出类、给了默认值，都不算接上。**
   本次 `episode_stream_sampler` 定义在 `config_holobrain_common.py:59`（ship 值 `True`）、
   配套 sampler 类实现完整并进了 `__all__`、3 份文档写了用法，**而全仓没有一行读它**：
   `train.py:124` 硬编码 `DistributedBatchFlagSampler`，`train.py` 压根不在移植的改动文件里。
   开关打开什么都不会发生 —— bank 恒为 `[1]`、gate 与 add 两种融合都是**精确恒等**
   （`s·w+(1−s)·w` 与 `(w+w)·0.5`，实测差 1~2 ULP）、7.47M 参数 grad 全 None 或精确零、
   loss 正常、无告警、日志干净。**这是最坏的失败形态：不是错，是白跑。**
   → **每加一个 config 键，落地前 grep 一次它的读取者；只命中「定义 + 表格 + 注释」就是没接。**
   `$ROL_JFS/port/_shared/orphan_switch_check.py` 已把这步做成判据 K/C/D：10 秒、零误报，
   键无人读 / 类无人构造 / 文档默认值与 ship 不符，**三个互相独立的角度同时指向同一个洞**。

8. **验证装置不许自建宿主装配 —— 它补上的每一块集成，都是复审看不见的地方。**
   `run_gears.py:138-145` 自己 import 并实例化了那个 sampler，而宿主没有任何路径能做到；
   更广的一层是 A/B/D 档与全部 5 个消融实际跑的是 `--sampler sequential`，
   一个**仓库里根本不存在**的手写连续索引列表，它碰巧产出 episode 连续批 ——
   所以「68/68 张量有梯度」是这条假路径的产物，真实路径上是 64 None / 4 精确零 / 0 非零。
   最阴的地方在于：harness 不走 `train.py` 的**理由本身成立**
   （accelerate / checkpoint / logging 会引入这套比对承受不了的不确定性），
   于是这个偏离在 code review 里读起来像谨慎，不像问题。
   → **harness 只允许注入输入、抓取张量；sampler / dataloader / optimizer / model builder
   一个都不许自己 new。harness 需要而宿主没有的东西，先补进宿主再验。**
   本次的做法：JFS 侧一个只 wrap 不 new 的 runner，
   `runpy.run_path("train.py", run_name="__main__")` 原样跑真实入口，
   靠包住 `SimpleTrainer.__init__` 去读宿主自己建好的那些对象。
   顺带一条：**护栏要查 `accelerator.prepare()` 之后的那个 dataloader** ——
   prepare 会把 batch_sampler 重新包一层，「构造出来的」不等于「被迭代的」。

9. **真实训练入口下浮点不逐位可复现，所以关闭态等价性必须用精确量判定，不能用「差异很小」。**
   本次实测：同一份 `train.py` 跑两遍，逐 step loss 最大差 `1.159e-04`；
   而 base 与 head 两份不同 `train.py` 之间最大差 `9.108e-05`——**跨代码组的区间完整落在
   同代码组区间之内**，10 组两两比较里最大的那个差恰恰出现在共用同一份代码的两个 run 之间。
   也就是说这个量级**与「是否同代码」不相关**，拿它当通过判据等于没判。
   误差从反向/optimizer 的 float32 非确定性归约进来；开 `cudnn.deterministic` +
   `use_deterministic_algorithms(warn_only=True)` 只压到 `1.564e-04`，压不到 0。
   （第一轮之所以量到恰好 `0`，是因为 harness 用 `lr=0`、权重不动——那是 harness 的性质。）
   → **每轮自测噪声地板，并额外给出至少一条不受浮点噪声影响的精确判据。**
   最便宜也最有力的一条是**逐样本 id 序列**（每 batch 的原始样本 key）：
   改动如果碰了选 batch 的那段代码，它一定变；没碰，它一定不变；无噪声、无阈值、无 GPU 成本。
   其余可用的精确量：batch key 集合 · 参数量 · 峰值显存 · sampler 链的类型与嵌套。
   顺带一条：**观测装置本身会不会污染被观测量要单列一条查** ——
   本次的探针每个 forward clone 两个张量，直接把 D 档显存读数抬高了 6 MiB 量级。

10. **任何「无差异」结论，都要先用阳性对照证明判据有牙 —— 没有阳性对照的通过 = 未验证。**
    判据失灵和被测对象没问题，在报告上长得**一模一样**：都是一片绿。
    区分它们的唯一办法是先构造一个**已知会改变行为**的扰动，确认判据真的会报警。
    本次的对照：把 `num_workers` 从 4 改成 0，同一套测量得到 `1.028e-01`，
    **比噪声地板高 3 个数量级**——于是「跨代码组只有 `6e-05`」才是有分辨力支撑的结论，
    而不是「差异很小」。同一条也适用于静态判据（在基点的树上跑一遍，必须报出那几条红）
    和护栏（人为构造退化场景，必须触发）：**一个从没被看见响过的探针不算证据。**
    → **每条判据都成对交付：一个阴性用例（正常配置不响）+ 一个阳性用例（已知坏配置必响）。**

11. **护栏挂在生产端会被整体绕过 —— 要挂在消费端，判据用可观测的后果，不用配置项的名字。**
    本次三道护栏全挂在 sampler 的构造处及其下游，所以只要走一条**不经过 sampler 构造**的
    路径，它们就一起沉默：`dataloader_type="group"` + `episode_stream_sampler=False` 下，
    装配期检查因两个键「相符」放行、batch 组成检查因 `dataloader_type != "stream"` 提前
    return、恒等探针因「没有历史」永不 arm ——**P0-1 的失效签名在护栏齐全的 commit 上原样复现**
    （bank 恒为 1、grad `64 None / 4 零 / 0 非零`、参数移动 `0/68`、护栏日志 0 行）。
    护栏没坏，是挂错了地方。

    两条判据上的病因，比落点更值得记：
    - **判据写成了配置项的名字**（`dataloader_type == "stream"`），
      于是只覆盖想到的那几种组合。**后果判据**（bank 长度有没有涨过 1 / batch 内 episode
      唯一数 / 输出与输入的差是否大于噪声）覆盖所有组合，包括将来新增的取值。
    - **判据依赖「前提已经成立」才能启动**：恒等探针要等历史存在才 arm，
      而失效形态恰恰是「历史永远不存在」。**要抓一个「X 从未发生」的失效，
      判据不能以 X 已发生为前提。** 换成「跑满 K 步后 X 有没有发生过」就没有这个死角。

    → **护栏挂在真正依赖那个前提的地方，问「我拿到的输入满足我的前提吗」，
    不要在生产端问「我被正确构造了吗」。判据用后果，不用配置项名字。**

    顺带两条，都是本次的具体形态：
    - **报错文案本身会制造事故。** 那条 raise 写着「the episode sampler is only meaningful
      for `stream`」，等于指引使用者去关掉 sampler —— 而那正是唯一没有护栏的格子。
      **护栏的文案要说真实原因和唯一正解，不要给出一个更省事的错误出口。**
      现在有断言禁止那几句话回来：文案也是可以回归测试的。
    - **护栏的阴性用例要单独构造，不能只等它在真实 run 里不响。**
      本次用「按 sampler 自己的 span 表重排它自己的输出」做故障注入：
      sampler 对象不动，所以装配期判据照样放行，**只有消费端能发现**。
      其中 batch=1 那一档是决定性的——batch 组成检查判不了、恒等探针不 arm，
      **只剩看门狗能响**，它响了。

### A 与论文/直觉不一致处（**实测，不是读代码读出来的**）

- **`dataloader_type` 的两个取值差别远大于名字暗示的。**
  `group` 会在**每次训练 forward 开头** `self.bank.clear()`，所以记忆跨度只有一个 batch，
  `group_size <= mem_length` 时巩固逻辑永不触发；只有 `stream` 才是论文说的 episode 级记忆。
  **A 的默认值是 `group`。** 实测三批之后的 bank 长度：`group [3,3,3]` vs `stream [3,4,4]`。
  → 本次用 `stream`，并因此需要一个 episode 有序的 batch sampler。
- **`PerMemBank` 只是 `CogMemBank` 的空壳子类**，`__init__` 逐参转发，无任何行为差异。
  它存在的意义是让感知流有自己的一套参数。
- **`GateFusion` 初始化不是恒等**（`normal(0,1e-3)` → `sigmoid≈0.5`）。
- **`BottleneckSE` 假设方形 token 网格**（`assert _h*_h == _n`）。宿主是 8×11，必然触发。
  → 改写成显式收 `(h, w)`；方形路径原样保留，实测与原版逐位一致。
- **A 的 `CrossTransformerBlock` 不接受 attn_mask**。任何要喂它变长/带 padding 历史的用法
  都得先给算子加 mask，那就不再是「搬」而是「改」了。

### 宿主 docs 与代码不符处

- 未发现实质性不符。`projects/holobrain_internal/docs/` 里两条 pipeline 的记录描述的是 v9，
  而仓库默认已是 v10——但这一点 `docs/04_config_system.md` 顶部已有显式勘误，不算不符。

### 本次的参考数值与验证

- `ref/`：`$ROL_JFS/port/memoryvla/ref/`，10 个 `.npz` + `manifest.json`，
  由 `tools/gen_reference.py` 在 `memvla_cu128` 生成（torch 2.8.0+cu128，与宿主同版本）。
- **C 档：10/10 逐位一致**（`max|diff| = 0.0`），含 `BottleneckSE` 改写前后与非方形输入。
  **这一档不受 P1-1 影响**（不经 sampler），且已在 `2b739226` 上由复审独立重跑，结论不变。
- ~~**A 档：0.000e+00**，参数量一致。~~
  **订正（2026-08-04）**：那个 `0` 产自 harness 路径，真实入口不逐位可复现（见教训 9）。
  参数量一致仍成立且是精确判据。现行 A 档判据见 `memoryvla/PORT-STATUS.md`
  「关闭态等价性：改用精确判据」。
- ~~开启态参考：`logs/gearB_on.json`（memoryvla 7,467,264 参数，68/68 张量有梯度）。~~
  **订正（2026-08-04）**：`68/68 有梯度` 是假路径产物（见教训 8），真实路径当时是
  `64 None / 4 精确零 / 0 非零`。接线修复后真实入口实测 **`0 None / 0 零 / 68 非零`**，
  证据在 `$ROL_JFS/port/memoryvla/fix/runs/2026-08-04/gearB_on.json`。
  参数量 7,467,264 这个数**两条路径上相同**（它不依赖批的组织方式）。

### 依赖差异包（下次判断冲突用）

**空。** 本次 E0，宿主主环境一个包都没动。
值得记一笔：A 的**完整栈**其实是 E2（A 的 env 是 transformers 4.40.1，宿主是 5.10.2，跨大版本）。
把本次拉回 E0 的唯一原因是「要搬的那几个类是纯 `torch.nn`」。
**换一个组件就未必了——先确认要搬的代码触不触 transformers/prismatic，再定档位。**

### 将来若再被撞上就该提炼的位置（**现在不提炼**）

只有一个样本，提前设计的扩展点大概率插错地方。下面这两处**点名**，等第二次移植真撞上再动：

1. **`structure.py:_forward` 里那个 `if self.memoryvla is not None:`。**
   第二个方法要挂在同一位置时，把它换成一个「特征后处理器列表」的分发点，
   顺序与互斥关系在那时才有足够信息定义。**换成分发是机械替换**，
   前提是现在这一处保持「一个 if + 一次调用」的形状——所以不要把逻辑铺开写进 `_forward`。
2. **`config_robodojo_dataset.py` 里按开关加白名单字段那一处。**
   第二个方法也要额外 batch 字段时，提炼成「按需附加样本字段」的通用机制
   （比如让每个方法声明自己需要的 key，由一处汇总）。现在是硬编码 `"step_index"`。

另外，`structure.py` / `structure_qwen3_5.py` 两个 `__init__` 重复列 `build(...)` 是宿主自身的
重复，**不是本次移植该收敛的东西**（协议：不顺手重构宿主）。但它每次都会咬人，
值得单独提一个 issue 给宿主维护者。
