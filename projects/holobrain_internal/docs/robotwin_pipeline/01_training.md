# 01 — RoboTwin 后训练

> **⚠️ 2026-08-03 合并 `feature/sem_internal` 之后**，仓库默认配置已从 v9 切到 v10
> （VLM 换成 Qwen3.5-2B、`patch_size` 28→32），本文所有 `config_holobrain_common.py:<行号>`
> 引用都已漂移。对照表见 [`../04_config_system.md`](../04_config_system.md) 顶部。
> **本文记录的仍是 v9 那次的实况，没有改写。**

## 1. 怎么跑的

**本机单卡，不是集群 job。** 这是与 [`../robodojo_pipeline/01_training.md`](../robodojo_pipeline/01_training.md)
最大的差别 —— RoboDojo 那条线是 AIDI 2 pod × 8 卡 accelerate，这条线就是一条前台命令：

```bash
cd /home/users/kun01.wu-labs/git_repo/robo_orchard_lab/projects/holobrain_internal/common
python3 train.py --config configs/config_holobrain_common.py
```

| 项 | 值 |
|---|---|
| 进程 | PID 1540974，GPU 0 独占 ~24 GB |
| 并行 | **无** —— 单进程单卡，没走 `accelerate launch`，没有梯度累积 |
| 起止 | 2026-07-21 05:19 → 07-22 12:16 |
| 步数 | 跑满 `max_step=1e5` |
| `--workspace` | 默认 `./workspace`，是指向 JFS 的**绝对路径 symlink** |

`aidi_submit_config/submit_cfg.json` 那份**是集群训练的模板，本次没用到**
（它的 `python_launcher` 是 `accelerate`、`--workspace /job_data`）。

> 单卡 `batch_size=16` ⇒ **全局 batch 就是 16**。RoboDojo 那边是 16×2×8 = 256。
> 两条线的 step 数不可直接类比。

---

## 2. 起点权重：这是"后训练"不是"从零训"

`config_holobrain_common.py:85-94` 的 v9 段：

```python
config.update(
    num_vlm_layers=4,
    embed_dims=384,
    decoder_layers=10,
    checkpoint="http://pfs-svcspawner.bcloud-bj-zone1.hobot.cc/.../holobrain_v9_newinit_.../output/checkpoints/checkpoint_50/model.safetensors",
    multi_modal_attn=True,
)
```

`checkpoint` 字段由 `train.py:153` 的 `load_checkpoint(model, config.get("checkpoint"), accelerator)`
消费 —— **只载权重，不载 optimizer/scheduler**，所以 step 从 0 开始计。

起点是同事 xuewu.lin 名下 AIDI job `bcloud-bj-zone1-23a35623c35d`（`holobrain_v9_newinit`）
的 `checkpoint_50`。**这是一个 PFS HTTP URL，不是 bucket 路径** —— AIDI 归档有留存期，
将来这条 URL 可能失效，要复现得先确认它还在。

---

## 3. ⚠️ 数据集口径：仓库现状 ≠ 当时

**这是本文档最重要的一节。**

`train.py:61-68` 每次启动会把整个 `configs/` copytree 到 workspace：

```python
if accelerator.is_main_process:
    shutil.copytree("configs", os.path.join(args.workspace, "configs"), dirs_exist_ok=True)
```

所以**权威快照**是：

```
/jfs-public/users/kun01.wu/robo_orchard_lab/workspace/configs/
```

（mtime 2026-07-21 07:21–07:25，即本次训练启动那一刻。）

### 当时的 `filter_list`（`workspace/configs/dataset_specs.py:612-660`）

```python
filter_list = [
    "robotwin1_0",
    "robotwin2_0",
#     "robotwin2_0_ur5_wsg",
#     "robotwin2_0_arx_x5a",
#     ...  以下 40 余项（abc130k / agilex / agibot / droid / egodex /
#          libero / table30v2 / behavior / robocasa / robodojo …）全部注释掉
]
```

**仓库里现在这些行都是打开的**（`configs/dataset_specs.py`，07-28 为 RoboDojo 改回全量）。
只看仓库会得出"用全量数据后训练"这个**错误结论**。

### 采样权重当时没生效

`workspace/configs/dataset_specs.py:706`：

```python
use_dataset_sample_weights = False

training_datasets = [
    x for x in training_datasets if x["dataset_name"] in filter_list
]
if use_dataset_sample_weights:          # <- False，整块跳过
    ...
    x["sample_weight"] = dataset_sample_weights[x["dataset_name"]]
```

所以 `dataset_sample_weights` 里写的 `robotwin1_0=0.8` / `robotwin2_0=3`
（`:661` 起那个 dict）**完全没有生效**，两个数据集按各自的自然规模混采。

### 实际读的数据

`DATA_BASE`（`:23`）= `os.environ.get("HOLOBRAIN_DATA_BASE", "./data")`，
而 `common/data` 是指向 `/horizon-bucket/robot_lab2/datasets/all_data` 的 symlink。展开后：

| dataset_name | setting_type | 路径 |
|---|---|---|
| `robotwin1_0` | `aloha_v1` | `.../all_data/robotwin1.0` |
| `robotwin2_0` | `aloha_v2` | `.../all_data/robotwin2.0/aloha_agilex_demo_clean`<br>`.../all_data/robotwin2.0/agilex_demo_randomized_500_part{1..10}` |

即 **1 份 clean + 10 份 randomized 分片**，LMDB 布局（`depth/ image/ index/ meta/`）。

`VALIDATION_DATASETS = None`（`:607`）⇒ `train.py:155-172` 里 `val_dataloader = None`、
`metric = None`，**本次训练全程没有验证集、没有验证指标**。

---

## 4. 超参

`config_holobrain_common.py:19-44` 基础 + `:85-94` v9 覆盖：

| 项 | 值 | 行 |
|---|---|---|
| `batch_size` | 16 | `:29` |
| `max_step` | `int(1e5)` | `:30` |
| `save_step_freq` | 5000 | `:32` |
| `step_log_freq` | 50 | `:31` |
| `lr` | `1e-4`（VLM 参数组 ×0.1） | `:34`；`build_optimizer :483-486` |
| optimizer | AdamW，`weight_decay=5e-4` | `:487-491` |
| lr schedule | LinearLR warmup 500 step（`start_factor=0.001`）→ MultiStepLR 在 90% 处 ×0.1 | `:492-509` |
| grad clip | norm，`max_norm=10` | `train.py:180-181` |
| `pred_steps` / `chunk_size` | 64 / 4 → 16 chunks | `:21-22` |
| `hist_steps` | 1 | `:20` |
| VLM | Qwen2.5-VL-3B-Instruct，前 4 层，**`freeze_vlm=False`** | `:39,42,86` |
| decoder | 10 层，`embed_dims=384`，`multi_modal_attn=True` | `:87-90` |
| 深度 | `with_depth=True` + `with_depth_loss=True`，SwinTransformer 分支 | `:24-28` |
| diffusion | 训练 DDPM 1000 步 / 推理 DPMSolver++ 10 步 | `:380-393` |

---

## 5. Checkpoint 落点与编号 → step 映射

`train.py:228-241` 用 `ProjectConfiguration(automatic_checkpoint_naming=True, total_limit=3)`，
配合 `SaveCheckpointConfig(save_step_freq=5000)`（`:190-193`）。

**编号从 `checkpoint_0` 开始，`checkpoint_N` ⇔ step `(N+1)×5000`。** 实测核对：

| 目录 | `scheduler.bin` `last_epoch` | `custom_checkpoint_0.pkl` `global_step_id` |
|---|---|---|
| `checkpoint_11`（冷备） | 60000 | 59999 |
| `checkpoint_18` | 95000 | 94999 |
| `checkpoint_19` | 100000 | 99999 |
| `checkpoint_20` | 100000 | 100000（`epoch_id=1, step_id=0`，训练结束时的收尾保存） |

现存：

```
/jfs-public/users/kun01.wu/robo_orchard_lab/workspace/
    checkpoints/checkpoint_{18,19,20}/          <- rolling total_limit=3 的存活窗口
    checkpoints_backup/checkpoint_11_step60000/ <- 07-22 01:16 手工冷备，躲过滚动删除
```

`checkpoint_{0..17}` 已被 accelerate 的滚动策略删除。

> **训练输出目录永远不能放 bucket** —— rolling `total_limit` 要 delete 最旧的一份，
> 而 bucket 在 POSIX 层拒绝 delete/rename。见
> [`../robodojo_pipeline/00_storage_layout.md`](../robodojo_pipeline/00_storage_layout.md)。

### ⚠️ 评测用的是 checkpoint_11，不是终版

07-22 01:16 备份 `checkpoint_11` 时训练还在跑（当时它是滚动窗口里最新的一份），
之后一路跑到 100000。**所有 RoboTwin 评测数字都是 step 60000 的**，
终版 `checkpoint_19` 从未评测。见 [07_results.md](07_results.md) §6。

---

## 6. ⚠️ loss 曲线已经不存在了

`train.py:223-242`：

```python
if args.logging_dir is None:
    args.logging_dir = os.path.join(args.workspace, "logs")   # :223-224
accelerator = Accelerator(log_with="tensorboard", ...)         # :229
accelerator.init_trackers("tensorboard")                       # :242
```

配置是对的，但产物没了。三处都查过：

1. `/jfs-public/users/kun01.wu/robo_orchard_lab/workspace/logs/` —— **空目录**，
   mtime 停在 2026-07-21 05:19（建目录那一刻，此后再没写过）
2. `/home/users/kun01.wu-labs` 与 `/jfs-public/users/kun01.wu` 全盘
   `find -name "events.out.tfevents*"` —— **零结果**
3. `StatsMonitorConfig` + `LossMovingAverageTrackerConfig`（`train.py:184-189`）
   每 `step_log_freq=50` 步把 loss 打到 **stdout**，
   而本次是**前台跑的**、没重定向 —— `.bash_history` 里三条 `python3 train.py ...`
   后面都没有 `> log` 也没有 `nohup`

→ **这轮后训练的 loss 曲线不可恢复。** 唯一还能间接反映训练状态的是 ckpt 里的
`optimizer.bin` / `scheduler.bin`。

**下次的教训**：本机长训必须 `nohup ... > train.log 2>&1 &` 或 `tee`，
不能只靠 tensorboard —— tensorboard 目录空掉这件事不会有任何报错提示。
（集群 job 没有这个问题，AIDI 自动把 stdout 收进 `log/`。）
