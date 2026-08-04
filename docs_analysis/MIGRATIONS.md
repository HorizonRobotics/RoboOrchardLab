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

### 改了宿主哪几处（共 5 个文件，0 删除）

> **订正（2026-08-04）**：原记 4 个文件。缺的第 5 个是 `common/train.py`，
> 也就是 P0-1：`episode_stream_sampler` 这个键当时没有任何读取者。

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
- **A 档：0.000e+00**，参数量一致。
- 开启态参考：`logs/gearB_on.json`（memoryvla 7,467,264 参数，68/68 张量有梯度）。

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
