# GenieSim 3.0 Benchmark Evaluation Guide

This guide walks you through running the [GenieSim 3.0](https://agibot-world.com/challenge2026/reasoning2action) benchmark with a HoloBrain policy — from data preparation, fine-tuning, model export, to launching the WebSocket inference server for closed-loop evaluation.

> [!NOTE]
> All commands below assume you are at the **repository root** (`/path/to/robo_orchard_lab`) unless otherwise noted.

---

## Prerequisites

| Item | Description |
|------|------------|
| **HoloBrain environment** | Follow [Installation](../README.md#1-installation) in the main README. |
| **GenieSim 3.0 dataset** | Download the fine-tuning dataset from the `01 Dataset` section of the official [Reasoning2Action Quick Start](https://agibot-world.com/challenge2026/reasoning2action/quick-start). See the Hugging Face [Dataset Structure](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2026#dataset-structure) for the expected layout after decompression. |
| **G2 omnipicker URDF** | Download from the GenieSim repository: [`G2_omnipicker.urdf`](https://github.com/AgibotTech/genie_sim/blob/v3.0.3/source/geniesim/app/robot_cfg/G2_omnipicker/G2_omnipicker.urdf). Place or symlink it to `projects/holobrain/urdf/G2_omnipicker.urdf`. |
| **GenieSim 3.0 simulation server** | Required for closed-loop evaluation (step 5). See the official [Sim-Evaluation Guide](https://agibot-world.com/sim-evaluation/docs/#/v3?id=_352-run-icra-tasks) for setup and Docker images. |

---

## 1. Prepare Data

Pack the raw GenieSim 3.0 fine-tuning data into RoboOrchard Arrow shards. Make sure `ffmpeg` and `ffprobe` are available in `PATH`.

### Quick start (single task, single shard)

```bash
python3 robo_orchard_lab/dataset/agibot_geniesim/packer/arrow_pack_geniesim3.py \
    --dataset_name AgiBotWorldChallenge-2026 \
    --input_dir /path/to/GenieSim3.0-Dataset/Reasoning2Action-Sim/dataset_without_depth \
    --urdf_path /path/to/G2_omnipicker.urdf \
    --robot_name G2_omnipicker \
    --task_name hold_pot \
    --output_dir projects/holobrain/data/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/hold_pot \
    --writer_batch_size 500 \
    --num_jobs 8 \
    --job_idx 0 \
    --force_overwrite
```

For each task, run every `job_idx` in `[0, num_jobs)` (e.g. `hold_pot` with `num_jobs=8` requires 8 runs, `--job_idx 0` through `--job_idx 7`). Each job writes one zero-padded shard directory, e.g. `hold_pot/part-00000-of-00008`.

<details>
<summary><b>Task reference (no-depth split) — click to expand</b></summary>

`num_jobs` is the recommended number of shards to split each task's data into for parallel packing. Larger tasks have more episodes and benefit from more shards. For each task, run the packer once per shard with `--job_idx` from `0` to `num_jobs - 1`.

| Task name | Recommended `num_jobs` |
|-----------|----------:|
| `clean_the_desktop_addition` | 4 |
| `clean_the_desktop_part_1` | 2 |
| `clean_the_desktop_part_2` | 2 |
| `hold_pot` | 8 |
| `open_door` | 8 |
| `place_block_into_box` | 11 |
| `pour_workpiece` | 8 |
| `scoop_popcorn` | 4 |
| `scoop_popcorn_part_2` | 4 |
| `sorting_packages_part_1` | 3 |
| `sorting_packages_part_2` | 3 |
| `sorting_packages_part_3` | 3 |
| `stock_and_straighten_shelf` | 4 |
| `stock_and_straighten_shelf_part_2` | 4 |
| `take_wrong_item_shelf` | 9 |

</details>

The packer auto-detects depth videos — with `dataset_without_depth` input only RGB features are written. Run `python3 robo_orchard_lab/dataset/agibot_geniesim/packer/arrow_pack_geniesim3.py --help` for the full option list.

> **Custom output path?** Update `data_paths` in `configs/config_agibot_geniesim_dataset.py` accordingly before training.
>
> **Gripper units:** GenieSim 3.0 stores gripper observations in the raw actuator range, while gripper actions are already normalized. The dataset loader divides observation grippers by `gripper_divisor` but keeps action grippers unchanged so `joint_state` and `master_joint_state` remain in the same normalized training range after sampling.

### Verify packed data

After packing, visualize the results to sanity-check images, joint states, and action labels:

```bash
cd projects/holobrain

python3 scripts/data_visualize.py \
    --config configs/config_holobrain_qwen_common.py \
    --dataset_names agibot_geniesim3_challenge \
    --kwargs '{"interval": 3, "ee_indices": [7,15], "fps": 30}'
```

---

## 2. Train

First, make sure `configs/config_holobrain_qwen_common.py` uses the GenieSim 3.0 dataset. Add or verify the following `config.update` block at the end of the config overrides:

```python
config.update(
    training_datasets=[
        "agibot_geniesim3_challenge",
    ],
    deploy_datasets=["agibot_geniesim3_challenge"],
)
```

This tells the training pipeline to load the Arrow shards packed in step 1 via the `agibot_geniesim3_challenge` entry defined in `configs/config_agibot_geniesim_dataset.py`.

Also confirm that pretrained checkpoint paths (`vlm_pretrain` and `checkpoint` in the same config file) point to valid local paths.

### Single-GPU

```bash
cd projects/holobrain

python3 scripts/train.py \
    --config configs/config_holobrain_qwen_common.py
```

### Multi-GPU / multi-machine (example: 2 machines × 8 GPUs)

```bash
cd projects/holobrain

accelerate launch \
    --num_machines 2 \
    --num-processes 16 \
    --multi-gpu \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --machine_rank ${CURRENT_RANK} \
    --main_process_ip ${MAIN_PROCESS_IP} \
    --main_process_port 1227 \
    scripts/train.py \
    --workspace ./workspace \
    --config configs/config_holobrain_qwen_common.py
```

For additional training options (`--workspace`, `--eval_only`, `--kwargs`, etc.), see the [main README — Run Training](../README.md#3-run-training).

---

## 3. Export Model

Export bundles the trained checkpoint, processor configs, and pipeline definition into a self-contained artifact ready for inference.

```bash
cd projects/holobrain

python3 scripts/export.py \
    --config configs/config_holobrain_qwen_common.py \
    --workspace ./model_export_path
```

The exported `./model_export_path` directory is used as `MODEL_DIR` in the next step.

---

## 4. Launch Inference Server

Start the GenieSim 3.0 WebSocket policy server. The server waits for connections from the GenieSim simulation client.

```bash
cd projects/holobrain

python3 scripts/geniesim3_inference_server.py \
    --model_dir ./model_export_path \
    --inference_prefix agibot_geniesim3_challenge \
    --host 0.0.0.0 \
    --port 8999
```

On startup, the server prints one or more `ws://` URLs — pass the appropriate URL to the GenieSim 3.0 benchmark client as `--infer-host`.

<details>
<summary><b>Server options — click to expand</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | `./model` | Directory of the exported model (from step 3). |
| `--inference_prefix` | `agibot_geniesim3_challenge` | Prefix of the saved inference pipeline config files. |
| `--model_prefix` | `model` | Prefix of the exported model files inside `model_dir`. |
| `--load_weights` | `true` | Whether to load model weights. |
| `--load_impl` | `native` | Model loading backend (`native` or `accelerate`). |
| `--host` | `0.0.0.0` | WebSocket bind address. |
| `--port` | `8999` | WebSocket port. |
| `--valid_action_step` | `32` | Number of action steps sent to the simulator per inference call. |
| `--sampling_ratio` | `1.0` | Resampling ratio for model outputs before truncation. `1` keeps raw values; integer > 1 uses stride sampling; other positive values use linear interpolation. |
| `--gripper_limit` | `1.0` | Scale applied after raw payload gripper observations are divided by `120.0`; keep `1.0` for the normalized training scale. |
| `--use_depth` | `false` | Set `true` only when both the exported model and the benchmark payload include depth data. |

</details>

The deploy policy uses the configured task-name instruction map as the canonical prompt source for supported GenieSim tasks. A payload `prompt` is used only when the incoming `task_name` has no configured default instruction.

---

## 5. Run Closed-Loop Evaluation

Once the inference server (step 4) is running, launch the GenieSim 3.0 simulation environment and connect it to the server's WebSocket URL. The simulation client drives the closed-loop: it sends observations to the inference server and receives action chunks at each step.

For detailed instructions on setting up and running the simulation evaluation, refer to the official documentation:

- **[GenieSim 3.0 Sim-Evaluation Guide — Run ICRA Tasks](https://agibot-world.com/sim-evaluation/docs/#/v3?id=_352-run-icra-tasks)** — covers environment setup, Docker images, task configuration, and how to point the benchmark client at your inference server (`--infer-host ws://<your_ip>:8999`).

> [!TIP]
> The inference server (step 4) prints available `ws://` URLs on startup. Use one of those URLs as the `--infer-host` argument when launching the GenieSim benchmark client.

---

## Quick Reference

Minimal end-to-end run (assuming prerequisites are ready):

```bash
# 0. Pack data (from repo root, repeat for each task and each job_idx in [0, num_jobs))
#    e.g. for hold_pot with num_jobs=8, run 8 times with --job_idx 0..7
python3 robo_orchard_lab/dataset/agibot_geniesim/packer/arrow_pack_geniesim3.py \
    --dataset_name AgiBotWorldChallenge-2026 \
    --input_dir /path/to/dataset_without_depth \
    --urdf_path /path/to/G2_omnipicker.urdf \
    --task_name hold_pot \
    --output_dir projects/holobrain/data/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/hold_pot \
    --num_jobs 8 --job_idx ${JOB_IDX} --force_overwrite

# 1. Train (single GPU)
cd projects/holobrain
python3 scripts/train.py --config configs/config_holobrain_qwen_common.py

# 2. Export
python3 scripts/export.py \
    --config configs/config_holobrain_qwen_common.py \
    --workspace ./model_export_path

# 3. Serve
python3 scripts/geniesim3_inference_server.py \
    --model_dir ./model_export_path \
    --inference_prefix agibot_geniesim3_challenge \
    --port 8999
```
