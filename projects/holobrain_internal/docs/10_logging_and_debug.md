# 10 · 日志、可视化与常见坑

> **阅读前置**：[08_loss_and_training](./08_loss_and_training.md)
>
> **本章目标**：知道去哪看训练曲线；能用 `data_visualize` 快速核对新加的数据 transform；碰到报错时能定位到具体模块。

---

## 10.1 TensorBoard 日志

### 10.1.1 打日志的地方

`train.py:228-242` 里 `Accelerator(log_with="tensorboard", ...)` + `accelerator.init_trackers("tensorboard")` 打开日志。之后：

- **`LossMovingAverageTrackerConfig`** hook：每 `step_log_freq` 步把每个 loss key 的滑动均值调 `accelerator.log({...}, step=global_step)`。
- **`StatsMonitorConfig`** hook：打印时间 / throughput / 显存到 log。

### 10.1.2 日志目录

| 场景 | 目录 |
|------|------|
| 本地 | `<workspace>/logs/tensorboard/` |
| 云端 | `/job_tboard/tensorboard/`（AIDI 上 `--logging_dir /job_tboard`） |

### 10.1.3 查看命令

```bash
# 本地
tensorboard --logdir ./workspace/logs

# 云端
# 用 aidictl 拉：
aidictl tboardlog <job_name>            # 视你团队的 aidictl 版本而定
```

### 10.1.4 关注的曲线

| Scalar | 含义 | 期望走势 |
|--------|------|----------|
| `loss_angle` | 关节角度 loss | 单调下降 |
| `loss_xyz` | EE 位置 loss（米²） | 迅速降到 1e-4 量级 |
| `loss_rot` | 姿态 loss（rot mat 元素差²） | 稳定下降但偶尔有 spike，不 nan 就行 |
| `loss_angle_fk / loss_xyz_fk / loss_rot_fk` | FK 版 | 与 `_fk` 主 loss 大致同步 |
| `loss_depth` | 深度分类 CE | 前期陡降，后期低位平稳 |
| `learning_rate` | 学习率 | 前 `warmup_step` 线性升，`0.9*max_step` 处 ×0.1 |
| `train/step_time_ms` | 单步耗时 | 关注是否有异常波动 |

## 10.2 数据可视化

来源：`projects/holobrain_internal/common/data_visualize/`。

### 10.2.1 离线渲染 MP4 (`video.py`)

```bash
cd projects/holobrain_internal/common

python3 data_visualize/video.py \
    --config configs/config_holobrain_common.py \
    --dataset_names robotwin2_0 libero_goal \
    --workspace ./vis_out \
    --episode_interval 5 \
    --max_episode 20 \
    --vis_mode auto
```

参数含义：

| 参数 | 作用 |
|------|------|
| `--config` | 用哪份 config 构建数据集 |
| `--dataset_names` | 要可视化的 dataset name（可多个） |
| `--workspace` | 输出目录 |
| `--vis_validation` | 使用 validation 数据集而非 training |
| `--manual` | 交互式选 episode index |
| `--episode_interval` | 每几个 episode 出一个视频 |
| `--max_episode` | 最多渲染多少 episode |
| `--vis_mode` | `auto / holobrain / dataset` — 模型侧渲染 vs 原始数据侧渲染 |

实际渲染由 `holobrain_utils.py::HolobrainVideoVisualizer` 完成。

### 10.2.2 交互式 Web 浏览 (`app.py`)

```bash
python3 data_visualize/app.py \
    --config configs/config_holobrain_common.py \
    --host 0.0.0.0 --port 13333
```

浏览器打开 `http://<host>:13333`，可以：
- 按 dataset / episode 选样本；
- 逐帧看 RGB / depth / projected 3D；
- 播放序列并盯 `hist_robot_state / pred_robot_state` 曲线；
- `FrameCache` (LRU) + `FramePrefetcher` (ThreadPool) 保证滑条流畅。

用途：**新加 transform 或改 URDF 后一定跑一次**，肉眼看深度、投影、关节骨架是否正确。

## 10.3 二次开发切入点

下面列 5 类最常见的改动，每类给出"改哪几个文件"。

### 10.3.1 加一个新数据集家族

1. 在 `robo_orchard_lab/dataset/<new>/` 加：
   - `<new>_lmdb_dataset.py`：继承 `BaseLmdbManipulationDataset`，实现 `__getitem__`；
   - `transforms.py`：家族专用 transform（如果通用 transform 已经够用可以不加）。
2. 在 `projects/holobrain_internal/common/configs/data_configs/config_<new>_dataset.py`：
   - `build_transforms(config, mode)`；
   - `@train_dataset_register("<new>") build_datasets(config, dataset_name, data_paths, mode, lazy_init)`；
   - `@processor_register("<new>") build_processors(config, dataset_name, **kwargs)`。
3. `projects/holobrain_internal/common/configs/data_configs/__init__.py` 加 `from .config_<new>_dataset import *`。
4. `dataset_specs.py`：`TRAINING_DATASETS` 加 spec dict + `filter_list` 加 dataset_name。
5. 若要导出，`deploy_specs.py::DEPLOY_DATASETS` 也加一条。
6. 用 `data_visualize/video.py` 抽样核对，尤其 `hist_robot_state / pred_robot_state / projection_mat` 是否合理。

### 10.3.2 换一个 VLM（比如 Qwen3-VL）

1. `config["vlm_pretrain"] = "./ckpt/Qwen3-VL-2B"`。
2. 如果路径不含 `qwen3.5 / qwen3_5`，需要**自己写一份 config**（`config_holobrain_common.py:149-156` 目前只判断这两个关键字）。示例：

```python
# 新增一个 config 文件 config_holobrain_qwen3vl.py 里
if "qwen3vl" in vlm_pretrain:
    patch_size = 32
    model_class = HoloBrain_Qwen3VL
    model_config = HoloBrain_Qwen3VLConfig
```

3. Qwen3-VL 要求 `transformers >= 4.57.1`；确保 docker 匹配。
4. `feat_mapping` 层数会不同（Qwen3VL 是 `num_layers`，Qwen2.5 是 `num_layers+1`）——**checkpoint 迁移时会有 `unexpected_keys` 一栏"weight"多一维**，注意。

### 10.3.3 改 Decoder 结构

- **op 顺序**：改 `config_holobrain_common.py:193-215` 的 `decoder_operation_order`。
- **多加一层 attention**：在 `HoloBrainDecoderTransformerConfig` 里加一个字段 + 在 `operation_order` 里插入相应 op 名 + 在 `action_decoder.py::forward_layers` 里加 dispatch 分支。
- **换 head**：改 `head = dict(type=UpsampleHead, ...)` 为你自己的 head 类。head 类只要接受 `x [B, num_joint, num_chunk, C_in]` 输出 `[B, num_joint, pred_steps, out_dim]` 即可。

### 10.3.4 加一个 loss 项

1. 在 `HoloBrainActionLoss.forward`（`loss.py:57-111`）加一段：

```python
if inputs.get("my_target") is not None:
    output["loss_myloss"] = self._loss_func(
        model_outs["my_pred"],
        inputs["my_target"],
        weight=None,
        pred_mask=pred_mask,
        timestep=model_outs["timesteps"],
        num_parallel=model_outs["num_parallel"],
    )
```

2. 在 `HoloBrainActionDecoder.forward_layers` 里让模型也吐出 `my_pred`，塞进返回 dict。
3. 因为 `MyBatchProcessor.forward` 是 `sum(v.mean() for k, v in output.items() if "loss" in k)`，只要 key 含 `loss` 就会自动加总。

### 10.3.5 加一个新的推理协议

参考 `realworld_eval.py`（HTTP）或 `geniesim3_inference_server.py`（WebSocket）复制一份改写：
1. 用 `HoloBrainProcessor.load(...)` 恢复处理器；
2. 用 `ModelMixin.load_model(model_dir, load_impl="native")` 恢复模型（或 `HoloBrainInferencePipeline.load_pipeline(...)`）；
3. 在服务器 handler 里把请求转成 `MultiArmManipulationInput` 或 batch dict 直接扔给模型；
4. 把 `model.decoder.async_inference_plugin = RTCInferencePlugin(...)` 挂上（如果需要 RTC 融合）。

## 10.4 常见坑与排查

### 10.4.1 数据侧

| 症状 | 定位 | 修复 |
|------|------|------|
| `FileNotFoundError: ./data/...` | `HOLOBRAIN_DATA_BASE` 环境变量未设，或 data 软链未建 | 见 [03 章 3.2](./03_env_and_quickstart.md#32-建立三个软链内网用户) |
| `assert num_shards > 0` 或 LMDB 为空 | `data_paths` 里的 lambda glob 不到目录 | 打开 config 里的 `data_paths` lambda，用 `_glob_sorted` 手跑一遍 |
| `pytorch_kinematics: cannot parse urdf` | urdf 软链 broken，或 URDF 路径写死错了 | 检查 `configs/data_configs/config_<x>_dataset.py::kinematics_config.urdf` |
| batch 内 shape 不一致导致 `stack_batch` padding 太多 | `DistributedBatchFlagSampler` 里 flag 未生效（新数据集忘了给 `flag=`） | 打印 `dataset.flag`；每个 sub-dataset 都应有唯一 int flag |
| `pred_mask` 全 False | `SimpleStateSampling` 的 `static_threshold` 太严，`pred_state` 全被判静止 | 调 `static_threshold` 或看 `step_index / num_steps` 分布 |

### 10.4.2 模型侧

| 症状 | 定位 | 修复 |
|------|------|------|
| `Missing keys: ['vlm.***', 'feat_mapping.4.***']` 一大堆 | checkpoint 与当前 `num_vlm_layers / freeze_vlm` 不匹配 | 换 checkpoint 或改 config 里对应字段 |
| `shape mismatch in weight` | `embed_dims` 与 checkpoint 不匹配（v0=256 vs v9=384） | 别混用 |
| `image_token_id not in vocab` | HF `AutoProcessor` 版本与模型不匹配 | 换 transformers 版本 |
| `RuntimeError: h % patch_size != 0` | Resize 后的 H/W 不整除 `qwen_patch_size` | 调 `dst_wh` 使得 `H % patch_size == 0 and W % patch_size == 0` |
| `torch.compile` OOM 或"CUDA graphs conflict" | 用 `torch.compile(mode="reduce-overhead")` 与 KV cache 冲突 | 用 `default` mode 或不启用 compile（`action_decoder.py:719-771` 里也有强制断言） |
| `loss = 0` 或 grad = 0 | `pred_mask` 全 False → 走到 `_fake_loss(pred) = pred.sum() * 0` | 参考数据侧修复 |
| decoder 推理慢 | `_set_attn_cache(True)` 未启用 | 检查 `action_decoder.py:656` 附近；或 `model.eval()` 前是否走了错误分支 |

### 10.4.3 训练侧

| 症状 | 定位 | 修复 |
|------|------|------|
| worker deadlock（`persistent_workers=True` 时更常见） | fork + CUDA 混用 | `set_start_method("spawn", force=True)` 已经在 `train.py:251`；若还有问题设 `num_workers=0` 定位 |
| step 慢，DataLoader 是瓶颈 | `num_workers` 太小 / `prefetch_factor=2` 不够 | 加 `--kwargs '{"num_workers": 16, "prefetch_factor": 4}'` |
| VLM 占显存太多 | `num_vlm_layers=None` 保留了所有 32 层 | 改为 `num_vlm_layers=4` 或更小 |
| accelerate 报 `RuntimeError: unable to open shared memory object` | 单机开太多 worker | 调小 `num_workers`；或 `docker run --shm-size=32g` |
| 多机跑到某个 step 挂 | `DistributedBatchFlagSampler` 某个 rank 数据不够 | 检查 `drop_last=True`；或让每个 rank 都够 1 个 batch 的样本量 |

### 10.4.4 推理 / 部署侧

| 症状 | 定位 | 修复 |
|------|------|------|
| `realworld_eval` 客户端一 POST 就 500 | `MultiArmManipulationInput` 字段缺失 | 检查 [09 章 9.8](./09_export_and_eval.md#98-realworld_evalpyflask-推理服务) 的必需字段列表 |
| 输出动作大跳变 | 没设 `--max_action_delta` 或 `--interpolation` | 加 `--max_action_delta 2 --interpolation 6.67` |
| `q_score.final` 极低（Behavior-1K） | `--num_trials_per_task` 太小、`Xvfb` 挂了、模型没加载 | 从 log 找 "Loaded model" 字样验证 |
| Isaac subprocess 失败但主进程无输出 | subprocess 的 stderr 未 tee | 直接跑单 task shell 命令观察 |

## 10.5 定位问题的一般套路

1. **先跑烟囱**：用 [03 章 3.9](./03_env_and_quickstart.md#39-快速自测5-分钟内确认能训练) 的最小命令跑 100 step，看看能不能出 checkpoint。
2. **打印 batch dict shape**：在 `MyBatchProcessor.forward` 一开始加：

   ```python
   for k, v in batch.items():
       print(k, type(v), getattr(v, "shape", None), getattr(v, "dtype", None))
   ```

   立刻能看到是不是 transform 少了、shape 不对。

3. **打印 loss dict**：在同一位置加 `print(output.keys())`；缺少某个 loss 项就往上追。

4. **可视化**：用 `data_visualize/video.py --dataset_names <你的新集>` 看一遍原始数据。

5. **单元测试**：`tests/test_robo_orchard_lab/models/holobrain/` 有测试用例；添加新层时可以按同风格加一个 smoke test。

## 10.6 编辑习惯建议

- **改 config 优先，改代码次之**。绝大多数实验都可以只改 `config_holobrain_common.py` 或数据 config。
- **不要直接改 `data_configs/config_libero_dataset.py` 上传**：如果只是实验，先在 `--kwargs` 里覆盖；确认有效再回到源文件。
- **checkpoint 命名**：本地训练时 `--workspace` 给一个明确的 branch 名（如 `workspace/exp_20260721_pred_steps_128`），便于回溯。
- **提交前测 reload**：`export.py --reload_test` 一定要过；否则线上会缺文件 / URDF。

---

**下一篇 →** [11_glossary.md](./11_glossary.md)
