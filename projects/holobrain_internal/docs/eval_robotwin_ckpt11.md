# RoboTwin 2.0 评估：checkpoint_11 (step ≈ 60000)

> **本文档是本机路线的操作说明，而本机路线最终没跑通**（卡在 curobo，见 §7）。
> 实际出结果的是 AIDI 集群路线。**结论与结果请看
> [`robotwin_pipeline/`](robotwin_pipeline/)**：
> [README](robotwin_pipeline/README.md) · [01_training](robotwin_pipeline/01_training.md) ·
> [03_eval](robotwin_pipeline/03_eval.md) · [07_results](robotwin_pipeline/07_results.md)
>
> 本文档的 §2（benchmark 判定）、§4（seed 规则）、§6（regex）、§8（路径速查）仍然准确；
> §5.2 描述的产物结构**只适用于本机**，集群下是扁平的 `/job_data/<task>/<task_config>/`
> （`script/eval_policy.py:126-128`）。

本文档配套脚本：[`common/scripts/eval_robotwin_ckpt11.sh`](../common/scripts/eval_robotwin_ckpt11.sh)

## 1. 快速上手

> **不是 client-server 架构**。`holobrain_robotwin_policy/deploy_policy.py` 里
> `HoloBrainPolicy` 直接 `ModelMixin.load_model(...)` 并 `self.model(data)` —
> HoloBrain 模型和 RoboTwin sapien env 在**同一个 Python 进程、同一张卡、
> 同一个 conda env** 里运行。因此需要一个"融合环境"：既能 import
> `sapien / mplib / gymnasium`（RoboTwin 侧），又能 import
> `transformers / accelerate / diffusers / robo_orchard_lab`（HoloBrain 侧）。

```bash
# 前置：先在一个独立的 conda 环境里装好 RoboTwin 依赖 + HoloBrain 侧依赖
conda create -n robotwin python=3.10 -y
conda activate robotwin

# RoboTwin 官方 requirements（源自 script/requirements.txt）
pip install torch==2.4.1 torchvision transforms3d==0.4.2 \
            sapien==3.0.0b1 scipy==1.10.1 mplib==0.2.1 \
            gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 \
            imageio==2.34.2 pydantic zarr h5py 'pyglet<2' wandb moviepy \
            termcolor av matplotlib huggingface_hub==0.25.0

# ★ HoloBrain 侧额外依赖（装进同一个 env）
pip install \
    'transformers>=4.49,<4.58' \
    accelerate safetensors diffusers \
    filelock requests einops \
    pytorch_kinematics
# pytorch3d 与 flash-attn 视 torch/cuda 版本单独装 pre-built wheel

# 系统级依赖
sudo apt install -y ffmpeg xvfb

# 一键跑评估（脚本会自动完成备份 → 组装 EVAL_MODEL_DIR → 拷贝 policy → 启动）
bash projects/holobrain_internal/common/scripts/eval_robotwin_ckpt11.sh
```

脚本启动前会做一次 **`python3 -c "import sapien; import robo_orchard_lab..."`
冒烟测试**，缺哪个包会立刻打印出来并 abort，不用等 sapien 起来才发现。

默认使用 **`CUDA_VISIBLE_DEVICES=2,3`**（GPU 0 上还在跑训练，GPU 1 也留出），任务 `place_empty_cup,stack_blocks_three`，每任务 100 trial。想改用其他 GPU / 任务：

```bash
CUDA_DEVICES=4,5 \
TASK_NAMES=place_empty_cup,stack_blocks_three,beat_block_hammer,lift_pot \
TEST_NUM=50 \
bash projects/holobrain_internal/common/scripts/eval_robotwin_ckpt11.sh
```

## 2. Benchmark 判定：RoboTwin 2.0

判据（全部指向 v2.0）：

| 证据 | 位置 |
|------|------|
| `demo_clean.yml` 的 `embodiment: [aloha-agilex]` 是 v2.0 引入的双臂形态 | `$ROBOTWIN_DIR/task_config/demo_clean.yml` |
| `envs/` 下含 `blocks_ranking_rgb / blocks_ranking_size / dump_bin_bigbin / place_dual_shoes` 等 v2.0 独有任务 | `$ROBOTWIN_DIR/envs/` |
| HoloBrain 侧配套的 dataset config 名为 `robotwin2_0_*` | `configs/data_configs/config_robotwin_dataset.py` |
| AIDI 提交模板 `job_name = "eval_robotwin_holobrain"`、任务列表覆盖 16 个 v2.0 任务 | `common/aidi_submit_config/submit_cfg_robotwin_eval.json` |
| 依赖里 `sapien==3.0.0b1 + mplib==0.2.1` 属于 v2.0 生态 | `$ROBOTWIN_DIR/script/requirements.txt` |

RoboTwin v1.0 用的是单臂 6-arm 配置；本次跑的 aloha-agilex 双臂 + 双 D435 深度相机是 v2.0 独有布局。

## 3. 本次评估的完整参数表

| 项 | 取值 | 来源 |
|---|---|---|
| 任务列表 | `place_empty_cup, stack_blocks_three` | 用户指定（对齐 README） |
| `--task_config` | `demo_clean` | README 示例 |
| `--test_num` | `100` | README 示例 |
| CUDA | `CUDA_VISIBLE_DEVICES=2,3` | 用户指定（避开训练用的 GPU 0） |
| checkpoint | `workspace/checkpoints_backup/checkpoint_11_step60000/model.safetensors` | 备份自训练进程的 `checkpoint_11` |
| 模型配置 | `checkpoint_11_eval/model.config.json` | 由 accelerate save_state 生成 |
| processor | `robotwin2_0_processor.json`（默认） | `--model_processor`；`robotwin_eval.py:104` |
| model_prefix | `model`（默认） | `--model_prefix`；`robotwin_eval.py:106` |
| vlm_ckpt_dir | `/horizon-bucket/robot_lab/users/xuewu.lin/ckpt` | README |
| urdf_dir | `/horizon-bucket/robot_lab2/datasets/all_data/urdf/urdf_v20260711` | README |
| **seed 起点** | `st_seed = 100000 * (1 + 0) = 100000` | `deploy_policy.yml:seed=0` + `script/eval_policy.py:167` |
| instruction_type | `unseen` | `deploy_policy.yml` |
| episode_num（task config 里的） | 100 | `demo_clean.yml:episode_num` |
| language_num | 100 | `demo_clean.yml:language_num` |
| embodiment | `aloha-agilex`（左右双臂） | `demo_clean.yml:embodiment` |
| head_camera / wrist_camera | `D435` | `demo_clean.yml:camera` |
| domain_randomization | 全部关闭（`clean_background_rate: 1`, `random_light: false`, ...） | `demo_clean.yml:domain_randomization` |
| eval_video_log | `true` | `demo_clean.yml`（每 episode 生成 mp4） |
| render_freq | `0`（headless） | `demo_clean.yml` |
| valid_action_step | `32`（每次 policy.get_action 只用前 32 步） | `holobrain_robotwin_policy/deploy_policy.py:156` |

## 4. seed 选取规则

`script/eval_policy.py:225-277` 里 `while succ_seed < test_num`：

1. 用 expert (`TASK_ENV.play_once()`) 尝试 seed，若不合法（`UnStableError` 或异常），`now_seed += 1` 直接跳过；
2. 合法 → 用 policy 跑一次；无论成功失败 `succ_seed += 1, now_seed += 1`。

因此**真实执行的 100 个 seed 不是连号 100000..100099**，而是过滤过的 100 个可行 seed。

> ⚠️ **勘误（2026-07-31）**：本节原来写的是"在同一 (task, task_config, seed=0, test_num)
> 组合下，不同 checkpoint 面对的是完全相同的 seed，可以直接横向对比"。
> **该结论与实测不符。**
>
> 2026-07-23 与 07-24 用**同一份 checkpoint_11、同一套参数**跑了两次完整评测，
> 两次的末位 seed 并不相同（`place_dual_shoes` 100070 vs 100081、
> `rotate_qrcode` 100064 vs 100067），说明 **expert 过滤这一步本身就不是确定性的**，
> 两次面对的 episode 集合并不完全一样。
>
> 后果：两次跑逐任务差 0–12 个百分点，均值 42.625% vs 43.875%。
> 按 50 次伯努利试验、`p≈0.4` 估算，两次测量之差的标准误约 9.8 pp，
> 95% 区间约 ±19 pp —— **±5–10 pp 的差异属噪声，不能当作真实差异**。
>
> **要做 checkpoint 之间的横向对比，50 trial 不够**，需要加大 `--test_num`
> 或用多个 seed 各跑一轮再合并。
>
> 完整对照表与解读见 [`robotwin_pipeline/07_results.md`](robotwin_pipeline/07_results.md) §4。

## 5. 输出文件去向

### 5.1 每 task 子进程 stdout
`robotwin_eval.py:36-40, 67-70` 把子进程 stdout 定向到：

```
$ROBOTWIN_DIR/eval_result/<task_name>/<task_config>/log.txt
```

例如 `eval_result/place_empty_cup/demo_clean/log.txt`。

### 5.2 `script/eval_policy.py` 自己的汇总文件
`eval_policy.py:126, 185`：

```
$ROBOTWIN_DIR/eval_result/<task_name>/holobrain_robotwin_policy/<task_config>/<ckpt_setting>/<timestamp>/_result.txt
```

内容是 `每次采样的成功率`（数值列表）。同目录下还有：
- `episode<N>.mp4` × N（每一个 test episode 一份视频）；`demo_clean.yml:eval_video_log: true`。

### 5.3 顶层 driver JSON 汇总
`robotwin_eval.py:152-157` 会打印：

```json
{
    "place_empty_cup": 42.0,
    "stack_blocks_three": 27.0,
    "num_tasks": 2,
    "mean": 34.5,
    "test_num_per_task": 100
}
```

脚本用 `tee` 把这段写到：

```
workspace/checkpoints_backup/eval_ckpt11_<yyyymmdd_HHMMSS>.log
```

## 6. 提取 Success rate 的 regex

`robotwin_eval.py:73-79`：从每 task 的 log 末尾往回扫描找 `Success rate`，用 `re.findall(r"\d+\.?\d+%", out)` 抓百分比。这个数字对应 `eval_policy.py:347` 打印的：

```
Success rate: {suc}/{test_num} => {round(suc/test_num*100, 1)}%, current seed: {now_seed}
```

## 7. 常见坑

- **conda env 必须切到 RoboTwin 侧、并额外补装 HoloBrain 依赖**：不是 client-server，模型和 sim 在**同一个 Python 进程**里；所以要用**一个"融合 env"**（robotwin 那套 + `transformers/accelerate/diffusers/robo_orchard_lab` 那套）。`holobrain_internal` env 里没有 `sapien / mplib / gymnasium`，反过来纯 `robotwin` env 里没有 `transformers/accelerate/diffusers`——两个都不能直接跑，必须在 robotwin env 里 `pip install` HoloBrain 侧依赖（详见 §1）。脚本会检测并 abort：
  - 若检测到 `CONDA_DEFAULT_ENV=holobrain_internal` → 提示切 env；
  - 若 `import sapien / robo_orchard_lab.*` 失败 → 打印具体缺哪个模块并给 `pip install` 命令。
- **`PYTHONPATH` 需要同时含两处**：`$ROBOTWIN_DIR`（让 `eval_policy.py` 能 `import envs / holobrain_robotwin_policy`）与仓库根目录（让 `deploy_policy.py` 能 `from robo_orchard_lab.models.holobrain... import`）。脚本已经处理。
- **`ROBOTWIN_DIR` 是 bucket 直挂，只读概率大**：脚本自动 `rsync -a --delete --exclude data/ --exclude eval_result/` 到 `$HOME/robotwin_eval_run/` 后再跑。第二次跑复用同名目录、速度快。
- **Xvfb**：sapien 3 需要 GL / vulkan；无显示器场景先 `Xvfb :99 -screen 0 1920x1200x24 & export DISPLAY=:99`。
- **不要碰 GPU 0**：训练还在 `python3 train.py --config configs/config_holobrain_common.py` 里，`nvidia-smi` 应仍显示 GPU 0 ≈ 24 GB 显存占用；本脚本已在启动前检查 GPU 2/3 显存 < 1 GB 才继续。
- **HTTP checkpoint vs 本地目录**：`holobrain_robotwin_policy/deploy_policy.py:84-92` 只在 `--model_config` 以 `http` 开头时才走 `download_file`；本地目录路径直接进 `HoloBrainProcessor.load(...)` + `ModelMixin.load_model(...)`。
- **URDF 双重来源**：脚本已经把 `workspace/urdf/` 拷进 `EVAL_MODEL_DIR/urdf/`，`deploy_policy.py:99-101` 里 `os.symlink` 的 `--urdf_dir` 就不会再建软链（已存在），是幂等的，`--urdf_dir` 参数仍然可以传，用作 processor 里如果有绝对路径引用时的 fallback。

## 8. 备份与产物路径速查

| 用途 | 路径 |
|------|------|
| 冷备 checkpoint_11（accelerate 目录格式，含 optimizer/scheduler/random_states） | `workspace/checkpoints_backup/checkpoint_11_step60000/` |
| EVAL 用组装目录（model + processor + urdf） | `workspace/checkpoints_backup/checkpoint_11_eval/` |
| Driver 主日志 | `workspace/checkpoints_backup/eval_ckpt11_<timestamp>.log` |
| RoboTwin 侧每 task 详细日志 | `$ROBOTWIN_DIR/eval_result/<task>/<task_config>/log.txt` |
| 每 episode 视频 | `$ROBOTWIN_DIR/eval_result/<task>/holobrain_robotwin_policy/<task_config>/<ckpt>/<ts>/episode<N>.mp4` |

## 9. 与训练进程共存的保证

- 训练主进程 PID 1540974 使用 GPU 0，共享文件系统上仅访问 `workspace/checkpoints/*`；本脚本只读 `workspace/checkpoints/checkpoint_11`（一次 `cp -a`）并写在 `workspace/checkpoints_backup/`，**不会与训练进程写入冲突**。
- 训练进程用 `accelerate.ProjectConfiguration(total_limit=3)` 滚动删除更老的 `checkpoint_9/10/11`，但**只会删 `checkpoints/` 目录下的**，`checkpoints_backup/` 不在滚动范围内。
- 训练 GPU（0）与评估 GPU（2/3）物理隔离，无 CUDA context 干扰。评估中 `nvidia-smi` 应仍观察到 GPU 0 显存占用 ≈ 24 GB、GPU 2/3 逐渐上升到 ≈ 15–20 GB（视 batch 而定）。
