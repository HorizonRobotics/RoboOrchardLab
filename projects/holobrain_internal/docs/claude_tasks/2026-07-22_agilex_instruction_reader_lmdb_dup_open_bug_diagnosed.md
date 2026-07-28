# Claude Session Handover — HoloBrain common `agilex` InstructionReader 重复 open LMDB Bug

> **日期**：2026-07-22
> **本 session 状态**：诊断完成 —— 已定位为**仓库当前默认配置下必现的 P0 bug**，非环境/版本问题；**用户要求先不改代码**，本文档记录完整证据链、多角度审查与修复方案，待下一位 Claude 或作者拿到批准后实施。
> **交接给**：下一个 Claude session 或维护者。
>
> 本文档目的：让下一个接手者不需要重复调查，一次读完就能拍板"改哪里、怎么改、要不要改"。

---

## 0. TL;DR — 下次接手直接看这里

### 现象

单卡训练命令：
```bash
cd projects/holobrain_internal/common
python3 train.py --config configs/config_holobrain_common.py
```
在 dataset 构造阶段（构造完 `behavior_manipulation` 后、构造下一个 agilex-family dataset 时）报：

```
lmdb.Error: The environment './data/instructions_v2/agilex' is already open in this process.
```

### 根因（一句话）

`dataset_specs.py` 里 **8 条 dataset_type=agilex 的 spec**（`dataset_name` 各不相同）都填了同一份共享的 `instruction_paths=[".../instructions_v2/agilex"]`，但 `config_agilex_dataset.py::build_dataset` **对每条 spec 都 new 一个新的 `InstructionReader`**，每个 reader 内部又都 `lmdb.open(...)` 同一目录 → py-lmdb "同进程内禁止同一 env 二次 open" 保护抛错。

### 判定

- **是 bug**（不是环境/CUDA/py-lmdb 版本问题，也不是用户配置错误）。
- **仓库当前 `filter_list` 默认就命中**，跟着 README 单卡命令跑必然复现。
- 同事那台 4090 能跑 = 他们本地配置绕开了（filter_list 裁剪 / config 白名单 / 未 push 的 patch），**不是他们的代码修好了**。

### 修复方向

**推荐方案 A**：在 `dataset_factory._build_typed_datasets` 里按 `tuple(sorted(instruction_paths))` 做 `InstructionReader` 缓存复用。改动集中、语义清晰、对未来其它共享 lmdb 场景自动生效。（细节见 §5）

### 下一步（下 session 若要动手）

1. 让用户/同事跑 `projects/holobrain_internal/common/diagnose_agilex_instruction_dup.py`（本 session 已写好），确认同事那边的 "启用后仍 open agilex instr lmdb 的数量" = 0/1，进一步确认 root cause 是配置差异而非其他。
2. 用户拍板 → 按 §5 方案 A 打 patch。改动点估计：`dataset_factory.py` +~15 行、`config_agilex_dataset.py` 改 ~5 行、`InstructionReader/Lmdb/base_lmdb_dataset.py` 不动。
3. 补 §7 里的 7 项回归检查。

---

## 1. 环境与相关文件

- 仓库：`/home/users/kun01.wu-labs/git_repo/robo_orchard_lab`（分支 `feature/memory_dev1`）
- 工作目录：`projects/holobrain_internal/common/`
- Conda env：`holobrain_internal`（py3.11、py-lmdb 2.3.0、LMDB C 0.9.35）
- 单卡启动命令（README 教程原文）：
  ```bash
  cd projects/holobrain_internal/common
  python3 train.py --config configs/config_holobrain_common.py
  ```

**关键涉及文件（file:line 引用，Ctrl+点直达）**：

| 角色 | 文件 | 行 |
|---|---|---|
| 训练入口 | `projects/holobrain_internal/common/train.py` | 114 (`build_dataset(config)`), 252 (`main`) |
| Config 顶层 | `projects/holobrain_internal/common/configs/config_holobrain_common.py` | 36 (`dataset_specs=...`), 515 (`build_training_dataset`) |
| 训练集入口 | `projects/holobrain_internal/common/configs/dataset_factory.py` | 199-233 (`build_training_dataset`) |
| **循环调 build_func 的地方** | `projects/holobrain_internal/common/configs/dataset_factory.py` | **158-192**（`_build_typed_datasets`，每条 spec 调一次） |
| **每条 spec new reader 的地方** | `projects/holobrain_internal/common/configs/data_configs/config_agilex_dataset.py` | **820-821** |
| **reader 内 new Lmdb 的地方** | `robo_orchard_lab/dataset/lmdb/base_lmdb_dataset.py` | **126-137**（`InstructionReader.init_lmdb`） |
| dataset 构造立即调 init_lmdb | `robo_orchard_lab/dataset/lmdb/base_lmdb_dataset.py` | 274-276, 290-294 |
| **Lmdb ctor 立即 open** | `robo_orchard_lab/dataset/lmdb/lmdb_wrapper.py` | **41-73**, **127-129** (`lmdb.open`) |
| Spec 白名单 | `projects/holobrain_internal/common/configs/dataset_specs.py` | 43-604 (TRAINING_DATASETS), 612-659 (`filter_list`), 708-710（过滤应用） |
| 诊断脚本 | `projects/holobrain_internal/common/diagnose_agilex_instruction_dup.py` | 全文件 |

---

## 2. 错误 traceback 原文（用户提供）

```
Rank[0/1] 07/21/2026 05:27:59 INFO base_lmdb_dataset.py:369 |
behavior_manipulation dataset length: 64172023, number of episode: 156323, ...
Traceback (most recent call last):
  File ".../projects/holobrain_internal/common/train.py", line 252, in <module>
    main(args, accelerator)
  File ".../projects/holobrain_internal/common/train.py", line 114, in main
    train_dataset = build_dataset(config)
  File ".../configs/config_holobrain_common.py", line 515, in build_training_dataset
    return build(config, lazy_init)
  File ".../configs/dataset_factory.py", line 219, in build_training_dataset
    datasets, dataset_names = _build_typed_datasets(...)
  File ".../configs/dataset_factory.py", line 187, in _build_typed_datasets
    datasets[dataset_name] = build_func(...)
  File ".../configs/data_configs/config_agilex_dataset.py", line 826, in build_dataset
    return HorizonManipulationLmdbDataset(...)
  File ".../robo_orchard_lab/dataset/horizon_manipulation/horizon_manipulation_dataset.py", line 96, in __init__
    super().__init__(...)
  File ".../robo_orchard_lab/dataset/lmdb/base_lmdb_dataset.py", line 276, in __init__
    self._init_lmdb()
  File ".../robo_orchard_lab/dataset/lmdb/base_lmdb_dataset.py", line 294, in _init_lmdb
    self.instruction_reader.init_lmdb()
  File ".../robo_orchard_lab/dataset/lmdb/base_lmdb_dataset.py", line 129, in init_lmdb
    self.lmdbs = [ Lmdb(uri=path, writable=False, ...) for path in self.paths ]
  File ".../robo_orchard_lab/dataset/lmdb/lmdb_wrapper.py", line 73, in __init__
    self.open()
  File ".../robo_orchard_lab/dataset/lmdb/lmdb_wrapper.py", line 135, in open
    self.env = self.open_lmdb()
  File ".../timeout_decorator/timeout_decorator.py", line 82, in new_function
    return function(*args, **kwargs)
  File ".../robo_orchard_lab/dataset/lmdb/lmdb_wrapper.py", line 129, in open_lmdb
    return lmdb.open(self.uri, **self.kwargs)
lmdb.Error: The environment './data/instructions_v2/agilex' is already open in this process.
```

对应到代码路径 = §1 表格里的调用链，两次都会经过 `lmdb_wrapper.py:129` 的 `lmdb.open(...)`。

---

## 3. 从多个角度证明这是 bug（不是环境问题）

### 3.1 直接复现（隔离 py-lmdb）

在同一 conda env 里跑 20 行纯 py-lmdb 复现（**不 import 仓库任何代码**）：

```python
import lmdb, tempfile
d = tempfile.mkdtemp()
lmdb.open(d, readonly=False, map_size=1024*1024).close()  # 先建 env
kw = dict(readonly=True, lock=False, meminit=False, map_async=True, sync=False, map_size=10485760)
e1 = lmdb.open(d, **kw); print("first open ok")
e2 = lmdb.open(d, **kw)                                     # 第二次
```
输出：
```
first open ok
SECOND open error: The environment '/tmp/tmpXXXX' is already open in this process.
```

**结论**：py-lmdb 本身对"同进程重复 open 同一 env 目录"就是硬性拒绝。仓库代码只是"引爆"了这条保护。

### 3.2 排除的候选原因

| 候选原因 | 判定 | 依据 |
|---|---|---|
| CUDA / torch / 显卡版本 | ❌ 无关 | 错误在导入 CUDA 之前抛；纯 py-lmdb 复现命中 |
| py-lmdb 版本差异 | ❌ 无关 | 同进程 dup-open 保护 py-lmdb 至少 10 年不变 |
| 数据目录被别的进程占用（文件锁）| ❌ 无关 | 提示 **"in this process"**，不是"another process" |
| DataLoader worker fork 引发 | ❌ 无关 | 错误在 dataset **构造期**就抛，DataLoader 都还没建 |
| 磁盘/挂载/权限 | ❌ 无关 | 第 1 次 open 成功、第 2 次才失败 |
| 环境变量差异（`HOLOBRAIN_DATA_BASE`）| ❌ 无关 | 只改 `DATA_BASE` 前缀，不改冲突关系 |
| 同事的 py-lmdb 被本地 patch | ❌ 极不可能 | conda 官方 wheel，改这个成本远高于改仓库 |

### 3.3 同事那台能跑通的 4 种"合理解释"

**都指向配置差异，不是代码修好**：

1. 他们 `filter_list` 里 agilex-family 只留 ≤1 个（其它注释掉）；
2. 他们 `config_holobrain_common.py` 有 `training_datasets = [...]` 白名单，只列了 1 个 agilex；
3. 他们本地对 `config_agilex_dataset.py` 或 `base_lmdb_dataset.py` 打了 patch 但没 push；
4. 他们跑的是 `projects/holobrain_internal/common/workspace/configs/dataset_specs.py`（仓库里另一份并存的 spec，路径可能不同）。

诊断脚本 `diagnose_agilex_instruction_dup.py` 就是**同时在你和同事两边跑一下**，比较"启用且 open agilex instr lmdb"这一行数字，就能确认是这 4 条里的哪一种。

### 3.4 "设计意图 vs. 实现"的裂缝

- 8 条 spec **都填了同一个** `instruction_paths=[".../instructions_v2/agilex"]` → 说明这是被设计成**共享词典**的（"任务名 → 文本指令"，天然被所有 agilex 家族数据集共用）。
- 但实现里每条 spec 各 new 一个 `InstructionReader` → **意图共享的对象没有被共享**。
- 意图与实现不一致 = bug。

### 3.5 是新引入的回归（非历史遗留）

从 git log：`ed751fbc chore(holobrain): Move load_extrinsic to data_spec` 表明 spec 层最近有重构。`instruction_paths` 由"每 dataset 独立一份" → "合并到共享的 `instructions_v2/agilex`" 这种重构，正是最容易漏掉"reader 也需要跟着单例化"的时机。**典型"共享化重构漏了单例化"缺陷模式**。

### 3.6 影响面

- **触发条件**：训练态 + 启用 ≥2 个共享同一 `instruction_paths` 的 dataset。
- **当前受影响**：`filter_list` 默认启用 8 条 agilex-family（见 §4 明细表），主进程构造第 2 条时就炸。
- **潜在定时炸弹**：`instructions_v2/horizon_piper_grasp_anything_2025`（当前只 1 处用；一旦有第 2 处就复发）。任何未来 dataset-family 想共享一份 instruction lmdb，都会撞同样的坑。

### 3.7 是"配置 pitfall"还是"代码 bug"？

我考虑过"这不是 bug，只是用户不该开 8 个"——**不成立**，理由：
- `filter_list` 默认打开 8 个，README 也没提到限制；
- 8 条 spec 显式填了共享路径，说明**共享是设计要求**，不是"用户越界组合"；
- 错误信息不面向用户（"in this process"是给库开发者看的），普通用户看到不知道怎么办。

---

## 4. 冲突数据集清单（filter_list 命中的 agilex-family）

**`dataset_specs.py` 里 9 条 `dataset_type == "agilex"`，其中 8 条在 `filter_list` 里**（按 `dataset_name` 字母序）：

| # | dataset_name | setting_type | 在 filter_list？ | instruction_paths 含 agilex？ |
|---|---|---|---|---|
| 1 | `agilex` | `agilex` | ✅ | ✅ |
| 2 | `challenge` | `challenge` | ✅ | ✅ |
| 3 | `challenge_finetune` | `challenge_finetune` | ✅ | ✅ |
| 4 | `challenge_self_collect` | `challenge_self_collect` | ✅ | ✅ |
| 5 | `horizon_beijing` | `horizon_piper_435_low_beijing` | ✅ | ✅ |
| 6 | `horizon_beijing_piper_x` | `horizon_piper_x_435` | ✅ | ✅ |
| 7 | `horizon_grasp_anything` | `horizon_piper_435_low_beijing` | ✅ | ✅（+ `horizon_piper_grasp_anything_2025`）|
| 8 | `horizon_shanghai` | `horizon_piper_435_low_shanghai` | ✅ | ✅ |
| 9 | `horizon_shanghai_fold_clothes` | `horizon_piper_435_high` | ❌ | ✅ |

**8 个不同 dataset_name / setting_type，共用同一份 `instructions_v2/agilex`**。这份 lmdb 是"任务名 → 文本指令"字典，共享是合理的；不合理的是**每 dataset 各 new 一个 reader**。

字母序里第一个撞车点：`agilex` 先跑成功打开 → `behavior_manipulation` 无 instruction_paths 跑过 → `challenge` 第二次 open → 炸。**跟 traceback 里 "构造完 behavior_manipulation 后炸"完美吻合**。

---

## 5. 修复方案（5 选 1，含推荐）

### 方案 A（推荐）：`dataset_factory` 层按 path 缓存复用 InstructionReader

**动机**：factory 是"看得见全局 spec 列表"的唯一层，最适合做 dedup。

**实现草图**（不动代码，仅描述）：

- `dataset_factory.py::_build_typed_datasets` 或 `build_training_dataset` 入口：
  1. 先扫一遍 `dataset_specs`，对每条 spec 提取 `instruction_paths`（允许 `list`/`str`/`None`）。
  2. 对每个非空 paths 做 `tuple(sorted(paths))` 做 key，实例化**一次** `InstructionReader`，存入 factory 内部局部 dict `shared_readers: dict[tuple, InstructionReader]`。
  3. 循环里，从 `dataset_spec` 中 `pop("instruction_paths")`，改为传 `instruction_reader=shared_readers[key]` kwarg 给 `build_func`。

- `config_agilex_dataset.py::build_dataset` 签名修改：
  - 优先接受 `instruction_reader` kwarg（factory 传的）；
  - 若未传但有 `instruction_paths`（旧兼容路径，比如单元测试或 tools 直接调），fallback 到原逻辑自己 new。

**优点**：
- 变更集中（1 个文件为主），几十行；
- 对未来其它 `dataset_type`（rh20t、horizon_piper_grasp_anything_2025 等）自动生效；
- 不动 `InstructionReader`/`Lmdb` 的语义，pickle/fork 行为不变；
- 局部作用域 dict，多次调用 `build_training_dataset` 相互独立。

**缺点**：
- 需要在 factory 和 build_func 之间统一 kwarg 契约；目前只有 agilex 传 `instruction_paths`，改动可控。

**风险 / 关注点**：
- DataLoader worker pickle：`Lmdb.__getstate__` 会 close env、`__setstate__` 会重新 open。共享 reader 后，多个 dataset **持有对同一 reader 的引用**——pickle 时会各自序列化一份，worker 端各自恢复一份 `Lmdb`。**这跟改前"多 dataset 各自持有独立 reader"在 worker 端表现一致**（每 worker 一份 env），主进程侧不再 dup-open，正是我们要的。
- `_mem_manager` 的 read_times 逻辑不变（它管的是 img/depth，不管 instruction）。

**评级**：**首选**。

### 方案 B：`InstructionReader` 内部维护类级 path→instance 缓存

改 `base_lmdb_dataset.py`，让 `InstructionReader.__init__` 或 `init_lmdb` 检查类变量 `_instances`。

- 优点：agilex config / factory 都不用动，"从此以后 new 多少次都没事"。
- 缺点：
  - lib 级全局 mutable 状态；
  - pickle 后 dict 里的 lmdb env 生命周期语义诡异；
  - evict 谁负责？测试怎么隔离？
- **不推荐**。违反最小惊讶原则（"new 出来的对象和别的实例共享底层资源" 反直觉）。

### 方案 C：`Lmdb` wrapper 层做进程级 env 复用

改 `lmdb_wrapper.py`，对 `(uri, readonly=True)` 建立进程级缓存。

- 优点：最底层最通用。
- 缺点：
  - `Lmdb` 既写又读，`writable=True` 的复用带来严重语义混乱（并发 txn/commit/reset）；
  - `close()` 变成引用计数或 no-op，影响所有其他调用点（`_mem_manager` 在内存压力下 close img/depth lmdb 的机制会失效）；
  - 侵入面大，回归风险高。
- **不推荐**。

### 方案 D：只在 `config_agilex_dataset.py` 里做模块级 dedup

`_reader_cache = {}`，`build_dataset` 里查一下再决定 new。

- 优点：改动最小，不改契约；影响面最小。
- 缺点：
  - 只治 agilex；rh20t 等家族要照抄一遍；
  - 模块级 mutable 状态污染，多次调 `build_training_dataset` 会跨调用共享缓存（可能不是想要的）。
- **备选**（如果只想最小侵入、立刻绿灯）。

### 方案 E：改 spec 层——去除"共享意图"

`dataset_specs.py` 里把 8 条 spec 拆成"1 条 spec 打开 reader、其余引用它"的双层结构（或者提出一个新的顶层 `SHARED_INSTRUCTION_READERS` 字段）。

- 优点：语义上把"共享"显式化。
- 缺点：改 schema，扩散到所有 spec；未来加 dataset 容易漏；改动量大。
- **不推荐**。

---

## 6. 推荐方案 A 的伪代码（不动仓库，仅示意）

**`dataset_factory.py`（伪码）**：

```python
def _build_typed_datasets(config, dataset_specs, registry, mode, lazy_init=False):
    from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import InstructionReader

    # 1) 预扫，按 sorted-tuple(paths) 缓存 reader
    shared_readers: dict[tuple[str, ...], InstructionReader] = {}
    for spec in dataset_specs:
        paths = spec.get("instruction_paths")
        if paths is None:
            continue
        if isinstance(paths, str):
            paths = [paths]
        key = tuple(sorted(paths))
        if key not in shared_readers:
            shared_readers[key] = InstructionReader(paths=list(key))

    datasets, dataset_names = {}, []
    for dataset_spec in dataset_specs:
        # ... 原本的 filter 逻辑不变 ...
        dataset_spec = dataset_spec.copy()
        dataset_type = dataset_spec.pop("dataset_type")
        dataset_name = dataset_spec["dataset_name"]
        # ... skip 分支不变 ...

        # 2) instruction_paths → 换成 shared reader
        paths = dataset_spec.pop("instruction_paths", None)
        if paths is not None:
            if isinstance(paths, str):
                paths = [paths]
            key = tuple(sorted(paths))
            dataset_spec["instruction_reader"] = shared_readers[key]

        # 3) data_paths callable 展开 + 调 build_func（保持原逻辑）
        if "data_paths" in dataset_spec:
            dataset_spec["data_paths"] = _resolve_data_paths(dataset_spec["data_paths"])
        build_func = registry.get(dataset_type)
        if build_func is None:
            raise KeyError(f"Dataset type `{dataset_type}` has not been registered.")
        datasets[dataset_name] = build_func(config, mode=mode, lazy_init=lazy_init, **dataset_spec)
        # ...
    return datasets, dataset_names
```

**`config_agilex_dataset.py`（伪码）** —— 签名新增 `instruction_reader=None`，优先用它：

```python
@train_dataset_register(DATA_TYPE)
@validation_dataset_register(DATA_TYPE)
def build_dataset(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode,
    instruction_paths=None,        # 旧兼容
    instruction_reader=None,       # 新，factory 传
    lazy_init=True,
    ...
):
    from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import InstructionReader
    # 优先 factory 传入
    if instruction_reader is None and instruction_paths is not None:
        instruction_reader = InstructionReader(paths=instruction_paths)
    # ... 后续不变 ...
```

`InstructionReader` / `Lmdb` / `base_lmdb_dataset.py` **不改**。

---

## 7. 回归检查清单（改前定，改后逐条验）

1. **单 GPU 训练冒烟**：`python3 train.py --config configs/config_holobrain_common.py`，进入训练循环 ≥1 step 无报错。
2. **DataLoader `num_workers > 0`**：确认 worker fork/spawn 两种模式都能读到 instruction。`Lmdb.__getstate__` close + `__setstate__` open 的循环在 worker 侧仍正确工作。
3. **验证态 `lazy_init=True`**：`build_validation_dataset` 分支下 `HorizonManipulationLmdbDataset(lazy_init=True)`，`_init_lmdb` 不在构造期触发；共享 reader 也应懒开。
4. **单 dataset 场景**：临时把 `filter_list` 只留 1 个 agilex，跑 1 step；行为应与改动前完全一致（reader 仍只 open 一次）。
5. **非 agilex-family**：agibot / droid / libero / behavior / robocasa 等无 `instruction_paths` 的 dataset，构造与改前完全一致。
6. **多份不同 instruction paths**：`horizon_grasp_anything` 那条同时有 `instructions_v2/agilex` **和** `instructions_v2/horizon_piper_grasp_anything_2025` 两个路径。key 使用 `tuple(sorted(...))`——如果与其它 dataset paths 列表**完全相同**才复用；否则各自 new。这是保守正确的策略。
7. **DDP 多进程**：每 rank 独立 Python 进程，进程内共享；rank 之间本就独立 open，改动不影响。

## 8. 相关但未修的顺手 TODO（不阻塞本次修复）

- `InstructionReader` 缺 `close()` / `__del__`，评估是否补上以便测试 tear-down。
- 加一条 warning：`InstructionReader.__init__` 如果检测到同 process 内相同 `paths` 被再次 new，就 log 一行（作为未来同类问题的早期预警）。
- 单元测试：写个 test 同时构造 2 个共享 `instruction_paths` 的 dataset，断言不抛 `lmdb.Error`。
- `common/workspace/configs/dataset_specs.py` 这份并存副本要不要清理？和顶层的关系目前不清晰。

## 9. 本 session 已产出的文件

- **`projects/holobrain_internal/common/diagnose_agilex_instruction_dup.py`**（诊断脚本，纯 Python 无 GPU 依赖）
  - 用法：`cd projects/holobrain_internal/common && python diagnose_agilex_instruction_dup.py`
  - 输出：
    - 启用的 agilex-family 明细；
    - **"启用 AND open agilex instr lmdb"** 的数量；
    - VERDICT：≥2 → 会命中 lmdb.Error；≤1 → 安全。
  - **用途**：对比用户机 vs. 同事机的输出，若同事机 ≤1 而用户机 ≥2 → 100% 确认根因是 filter_list 差异，非环境。

## 10. 用户明确表达的约束（本 session）

1. **先不要改动代码**：只做诊断和方案设计，动手改需要拿到显式批准。
2. **多角度审查是否 bug**：需要给出"不是环境/版本问题"的证明链，不能只说"我看代码是这样"。
3. **命名对齐已有 handover 文档**：本文件遵循 `YYYY-MM-DD_<主题>_<状态>.md` 格式，见 `README.md`。

## 11. 参考交叉

- 相邻 handover：`2026-07-22_robotwin_eval_env_ready_blocked_curobo.md`（RoboTwin 评估流程，与本 bug 无关，但共享同一 `holobrain_internal` conda env）。
- README 命名规则：`README.md` §"命名规则"。
- 本 session 尚未落成 memory 索引，若下 session 认为值得，可在 `~/.claude/projects/-home-users-kun01-wu-labs-git-repo-robo-orchard-lab/memory/` 加一条 `feedback` 或 `project` 记录本 bug 的存在。

---

## 附录 A：调用链一图流

```
train.py:114
  └─ build_training_dataset (config_holobrain_common.py:515 → dataset_factory.py:199)
       └─ _build_typed_datasets (dataset_factory.py:158-192)
            └─ for each spec (agilex family, 8 条):
                 └─ build_func = TRAIN_DATASET_BUILD_FUNCS["agilex"]
                 └─ config_agilex_dataset.py:820-821
                      └─ InstructionReader(paths=["./data/instructions_v2/agilex"])   ★ new 8 次
                 └─ HorizonManipulationLmdbDataset(..., lazy_init=False)              # training 模式
                      └─ BaseLmdbManipulationDataset.__init__
                           └─ base_lmdb_dataset.py:276  → _init_lmdb()
                                └─ base_lmdb_dataset.py:294  → instruction_reader.init_lmdb()
                                     └─ base_lmdb_dataset.py:130  → Lmdb(uri=...)
                                          └─ lmdb_wrapper.py:73   → self.open()
                                               └─ lmdb_wrapper.py:135 → open_lmdb()
                                                    └─ lmdb_wrapper.py:129
                                                         → lmdb.open("./data/instructions_v2/agilex", readonly=True, lock=False, ...)
                                                            ★ 第 2 次调用抛 lmdb.Error
```

## 附录 B：为什么 py-lmdb 拒绝二次 open

py-lmdb 内部维护一个 process-wide 的 env 表，`lmdb.open(path)` 会先做 `LMDB_MAYBE_DUPOPEN` 检查：如果该 path 已经在本进程 open 过（无论 readonly/writable、lock/no-lock）就抛 `already open in this process.`。这个行为至少从 py-lmdb 0.9x 就存在，与 py-lmdb / LMDB C 版本无关。

**回避方式只有两种**：
1. 复用同一个 `Environment` 对象（方案 A / B / C 都是变种）；
2. 或者把第一个 close 掉再 open 第二个（不适用——训练主循环需要两者同时可读）。
