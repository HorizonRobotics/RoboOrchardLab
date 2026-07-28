# 03 · 环境与快速开始

> **阅读前置**：[01_overview](./01_overview.md)、[02_repo_structure](./02_repo_structure.md)
>
> **本章目标**：把三个必要软链建好，用一条命令跑起来单卡或多卡训练；能导出 processor + safetensors。

---

## 3.1 Docker 镜像

来自 `projects/holobrain_internal/common/README.md`。

| 版本 | 镜像 tag | 说明 |
|------|----------|------|
| 新版（推荐，支持 Qwen3） | `docker.hobot.cc/imagesys/robot_lab:ubuntu22.04-gcc11.4-py3.10-cuda11.8-torch260-robotwin2-transformer4571-20251030` | Torch 2.6 + transformers 4.57.1 |
| 旧版（deprecated） | `docker.hobot.cc/imagesys/robot_lab:ubuntu22.04-gcc11.4-py3.10-cuda11.8-torch241-robotwin2-20250918` | Torch 2.4.1 |

评估用的额外镜像：Isaac Lab、Behavior-1K、LIBERO-Plus 各自有独立 Docker，见 `aidi_submit_config/submit_cfg_*_eval.json` 的 `image` 字段。

## 3.2 建立三个软链（内网用户）

```bash
cd projects/holobrain_internal/common

ln -s /horizon-bucket/robot_lab2/datasets/all_data                            data
ln -s /horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711        urdf
ln -s /horizon-bucket/robot_lab/users/xuewu.lin/ckpt                           ckpt
```

其中：
- `data/` 是所有数据集 LMDB 的根目录。`dataset_specs.py:23` 里 `DATA_BASE = os.environ.get("HOLOBRAIN_DATA_BASE", "./data")` 会读它；也可以在启动前 `export HOLOBRAIN_DATA_BASE=/其他/路径` 覆盖。
- `urdf/` 是所有 URDF + 网格文件。config 里写死了相对路径 `./urdf/<family>/...`，因此**这个软链名不能改**。
- `ckpt/` 包含预训练 VLM（如 `Qwen2.5-VL-3B-Instruct/`）与 HoloBrain 已有的 safetensors。

## 3.3 单机单卡：最小可跑通命令

```bash
cd projects/holobrain_internal/common

python3 train.py \
    --config configs/config_holobrain_common.py \
    --workspace ./workspace_debug
```

会做的事：
1. 主进程把 `configs/` 整体拷到 `workspace_debug/configs/`（可复现）。
2. 加载 config，构建模型（`build_model`）。
3. **主进程**先根据 `deploy_specs.py` 导出每个数据集的 `<name>_processor.json` 与 `<name>_inference.config.json`。
4. 构建 dataloader（`DistributedBatchFlagSampler` 保证一 batch 一个 embodiment）。
5. 构建 AdamW + `ChainedScheduler(LinearLR warmup + MultiStepLR)`。
6. `register_save_state_pre_hook` + `load_checkpoint`（默认从 `./ckpt/HoloBrain_v0.0_Qwen/model.safetensors` 加载预训练权重）。
7. `SimpleTrainer(...).__call__` 进入训练循环。

## 3.4 单机多卡：Accelerate 启动

```bash
accelerate launch \
    --multi-gpu \
    --num-processes 4 \
    --gpu-ids 0,1,2,3 \
    train.py \
    --config configs/config_holobrain_common.py \
    --workspace ./workspace_4gpu
```

要点：
- `train.py` 里只 `Accelerator(...)` 实例化一次，所有 DDP / mixed precision / gradient accumulation 都由 `accelerate` 的 CLI 参数或 `~/.cache/huggingface/accelerate/default_config.yaml` 决定。
- 因为 dataloader 用的是 `batch_sampler=DistributedBatchFlagSampler(...)`，rank 分片是**在 sampler 内部**做的，不依赖 accelerate 的 `DistributedSampler`。
- `DataLoaderConfiguration(use_seedable_sampler=True, non_blocking=True)` 已在 `train.py:237-240` 打开。

第一次多卡运行前建议先跑一次：

```bash
accelerate config
```

用问答式生成默认配置。生产环境上 AIDI 直接用 `aidi_submit_config/submit_cfg.json` 里 `"python_launcher": "accelerate"`。

## 3.5 CLI 参数（`train.py`）

来源：`projects/holobrain_internal/common/train.py:214-221`。

| 参数 | 默认 | 作用 |
|------|------|------|
| `--config` | 无 | 必填，Python 配置文件路径 |
| `--workspace` | `./workspace` | checkpoint、导出物、拷贝的 configs 都落在这里 |
| `--logging_dir` | `<workspace>/logs` | TensorBoard 目录，云端会覆盖为 `/job_tboard` |
| `--eval_only` | `False` | 只跑验证不训练；要求 `VALIDATION_DATASETS` 非空 |
| `--kwargs` | `None` | JSON 字符串或 JSON 文件路径，用于**在启动时覆盖 config 顶层字段**；未知 key 会报错 |

### `--kwargs` 用法举例

覆盖 batch size 与最大步数，直接命令行：

```bash
python3 train.py --config configs/config_holobrain_common.py \
    --kwargs '{"batch_size": 8, "max_step": 20000}'
```

或者写成文件：

```bash
cat > /tmp/override.json <<'JSON'
{"batch_size": 8, "max_step": 20000, "num_workers": 4}
JSON

python3 train.py --config configs/config_holobrain_common.py \
    --kwargs /tmp/override.json
```

## 3.6 导出（打包成推理用）

在训练结束后，用 `export.py` 把 config + processor + safetensors + inference pipeline JSON 一起打包：

```bash
cd projects/holobrain_internal/common

python3 export.py \
    --config configs/config_holobrain_common.py \
    --workspace ./exported_model \
    --reload_test \
    --dataset_names libero_goal,robotwin2_0_aloha_v2
```

产物结构（`export.py:37-115`）：

```
exported_model/
├── configs/                                    # 完整 config 复制
├── libero_goal_processor.json                  # 每个 deploy dataset 一份 processor json
├── robotwin2_0_aloha_v2_processor.json
└── model/
    ├── model.safetensors                       # ★ 权重
    ├── model.config.json                       # ★ 模型 config 序列化
    ├── libero_goal_inference.config.json       # 每个 deploy dataset 一份 pipeline json
    ├── robotwin2_0_aloha_v2_inference.config.json
    └── urdf/                                   # 处理器引用到的 URDF 会被复制进来
```

`--reload_test` 会立刻用 `ModelMixin.load_model(model_path, load_impl="native")` 和 `HoloBrainInferencePipeline.load_pipeline(...)` 反序列化一次，验证导出是否正确。

`--dataset_names` 用逗号切分白名单；不写则导出 `deploy_specs.py` 中所有条目。

## 3.7 AIDI 集群一条命令

```bash
RoboOrchardJob-AIDISubmit submit_from_config \
    --config projects/holobrain_internal/common/aidi_submit_config/submit_cfg.json
```

`submit_cfg.json` 内会：
- 上传 `robo_orchard_lab / configs / train.py / holobrain_utils.py`；
- 用 `python_launcher = "accelerate"` 启动；
- 挂载三个 bucket 作 `data / urdf / ckpt`；
- 8 worker × 8 GPU；
- 日志写 `/job_tboard`、checkpoint 落 `/job_data/checkpoints/checkpoint_<n>/`；
- checkpoint 自动只保留最后 3 个（`ProjectConfiguration(total_limit=3)`）。

其他评估作业同目录下每个 json 一个，见 [09 章](./09_export_and_eval.md)。

## 3.8 常见环境问题一览

| 症状 | 可能原因 | 修复 |
|------|----------|------|
| `FileNotFoundError: ./data/...` | 没建 data 软链，且未设 `HOLOBRAIN_DATA_BASE` | 见 3.2 |
| `pytorch_kinematics` 报无法解析 URDF | urdf 软链失效或 URDF 路径打错 | 见 3.2；核对 `config_*_dataset.py` 中 `kinematics_config["urdf"]` |
| `safetensors` 加载 `Missing keys: ['vlm.***']` 一大堆 | 你重训了 VLM 但 checkpoint 是"冻结 VLM"版；`freeze_vlm=True/False` 与 checkpoint 需匹配 | 改 config 里 `freeze_vlm` 或换 checkpoint URL |
| `Unknown config keys in kwargs: {...}` | `--kwargs` 里的 key 不在 `config` 里 | 只覆盖 `config = dict(...)` 已声明的 key |
| accelerate CUDA OOM | batch_size 太大 | `--kwargs '{"batch_size": 4}'` 或缩小 `pred_steps` |
| 训练卡住无输出 | worker 0 在 build LMDB index，可能耗时长 | 首次运行等 1–3 分钟；看看 `--num_workers` 是否过大 |

## 3.9 快速自测：5 分钟内确认能训练

```bash
# 1) 只跑 100 步 + batch_size=2，最小烟囱测试
python3 train.py --config configs/config_holobrain_common.py \
    --workspace /tmp/hb_smoke \
    --kwargs '{"batch_size": 2, "max_step": 100, "step_log_freq": 10, "save_step_freq": 100, "num_workers": 2}'

# 2) 看 workspace 下有没有 checkpoints/checkpoint_0/
ls -la /tmp/hb_smoke/checkpoints
# 应看到 checkpoint_0/model.safetensors 与配套 config
```

若这一步能跑通、看到 loss 从大到小，说明数据 + 模型 + 训练器都装好了。剩下就是按你的实验目标改 config、加数据集、改网络，都在后续章节。

---

**下一篇 →** [04_config_system.md](./04_config_system.md)
