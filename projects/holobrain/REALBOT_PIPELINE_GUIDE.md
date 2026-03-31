# Real Robot Pipeline Guide of HoloBrain-0

This guide walks you through the complete real-robot pipeline using the **Grasp Anything** task as a running example. By the end, you will have recorded (or downloaded) demonstration data, packaged it for training, trained a HoloBrain model, and prepared it for deployment on a physical robot.

**What is Grasp Anything?** A dual-arm Agilex Piper robot picks arbitrary objects from a table and places them into a basket. The setup uses three Intel RealSense cameras (left, right, middle) to observe the workspace. In the codebase this task is called `place_objects_to_basket`.

## Prerequisites

- **Installation:** Follow the [main README](README.md#-quick-start) to install the project.
  ```bash
  cd /path/to/robo_orchard_lab
  make version
  pip install ".[holobrain_0]"
  ```
- **Hardware (for recording & deployment):** Dual Agilex Piper arms + three Intel RealSense cameras. See the [Real Robot Deployment Guide](REALBOT_DEPLOY_GUIDE.md) for full hardware details.
- **Working directory:** All commands below assume you are in the `projects/holobrain` directory:
  ```bash
  cd projects/holobrain
  ```

---

## Pipeline Overview

The pipeline consists of five sequential stages. Each stage depends on the output of the previous one:

| Stage | What You Do | What You Get |
| :--- | :--- | :--- |
| **1.&nbsp;Data&nbsp;Recording** | Teleoperate the robot to demonstrate the task while the system records all camera and joint data. | Raw `.mcap` recording files — one per demonstration. See [What are `.mcap` files?](#what-are-mcap-files) below. |
| **2.&nbsp;Data&nbsp;Packaging** | Convert raw recordings into a training-ready dataset that supports efficient random access. | A structured dataset: sharded `.arrow` files, plus metadata for the training pipeline. See [Expected Output Structure](#expected-output-structure). |
| **3.&nbsp;Data&nbsp;Checking** | Visually verify that the packaged data is correct and the training inputs look reasonable. | Review videos (`.mp4`) and reconstructed `.mcap` files you can inspect in [Foxglove](https://foxglove.dev/). |
| **4.&nbsp;Model&nbsp;Training** | Train the HoloBrain model on your packaged dataset using a config file. | Training checkpoints — containing model weights (`.safetensors`) and model config (`.config.json`) — plus a processor config (`*_processor.json`) that defines the data pre/post-processing pipeline. Ready for export. |
| **5.&nbsp;Deployment** | Export the trained model and launch an inference server connected to the physical robot. | A running inference server that sends real-time action commands to the robot. |

### Where Should I Start?

Not everyone starts from scratch. Use this guide to find your entry point:

```
Have nothing?
  └─➔ Start at Step 1 (Data Recording)

Have raw .mcap files (e.g., downloaded from HuggingFace)?
  └─➔ Start at Step 2 (Data Packaging)

Have packaged .arrow datasets?
  └─➔ Start at Step 4 (Model Training)

Have trained checkpoints (or exported model)?
  └─➔ Start at Step 5 (Deployment)
```

### Shortcut: Download Data from HuggingFace

If you don't want to record your own data, you can download pre-recorded datasets from HuggingFace. The download commands below use the `hf` CLI install it with: `pip install -U "huggingface_hub[cli]"`
#### Option A: Download raw `.mcap` files → skip to Step 2 (Data Packaging)

```bash
hf download HorizonRobotics/Real-World-Dataset \
    --repo-type dataset \
    --include "raw_data/place_objects_to_basket/*.mcap" \
    --local-dir ./data
```

After downloading, your directory should look like:
```
data/raw_data/place_objects_to_basket/
  data0.mcap
  data1.mcap
  ...
```

Continue to **Step 2 (Data Packaging)**.

#### Option B: Download packaged `.arrow` datasets → skip to Step 4 (Model Training)

```bash
hf download HorizonRobotics/Real-World-Dataset \
    --repo-type dataset \
    --include "arrow_dataset/place_objects_to_basket/*" \
    --local-dir ./data
```

After downloading, your directory should look like:
```
data/arrow_dataset/place_objects_to_basket/
  part-00000/
    data-00000-of-00070.arrow
    ...
    dataset_info.json
    meta_db.duckdb
    state.json
```

Skip ahead to **Step 4 (Model Training)**.

You can also browse the [Real-World Dataset collection](https://huggingface.co/datasets/HorizonRobotics/Real-World-Dataset/tree/main) for other available tasks.

---

## 1. Data Recording

This step generates the **raw trajectory data** used for training. If you downloaded data from HuggingFace, you can skip this step entirely.

### What are `.mcap` files?

`.mcap` files are ROS message bags — they record synchronized sensor streams and robot states into a single file. For the Grasp Anything task, each recording captures:
- **Camera data:** RGB and depth images from three RealSense cameras (left, right, middle), plus their calibration parameters.
- **Robot data:** Joint states and action commands from both Piper arms.

### Visualizing `.mcap` Files

You can inspect `.mcap` files using [Foxglove](https://foxglove.dev/). To get a pre-configured view for this project, download the example layout from HuggingFace: [Example Foxglove Layout](https://huggingface.co/datasets/HorizonRobotics/Real-World-Dataset/blob/main/visualization/arrow_foxglove_layout.json)

In Foxglove, go to **Layout → Import layout** and load the downloaded `.json` file, then open your `.mcap` file.

### Recording Setup

A typical recording session involves:
1. Mounting the three cameras to observe the workspace.
2. Connecting both Piper arms via CAN bus.
3. Using teleoperation to demonstrate the pick-and-place task while the system records all streams.

For detailed recording instructions, see the [data collection guide](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain).

### When to Skip This Step

- You downloaded `.mcap` files from HuggingFace or another source.
- You already have raw recordings from a previous session.

Example raw data layout:
```
data/raw_data/
    data0.mcap
    data1.mcap
    ...
```

---

## 2. Data Packaging

This step converts raw `.mcap` recordings into the **Arrow columnar format** that the training pipeline expects. The conversion is necessary because the training dataloader needs a standardized, indexed format for efficient random access — raw `.mcap` files are sequential streams that can't be efficiently sampled during training.

### URDF: What Is It and Where to Find It

The packer needs a URDF (Unified Robot Description Format) file that describes your robot's kinematic chain — joint limits, link lengths, etc. For the Grasp Anything dual-arm Piper setup, use:

```
./urdf/piper_description_dualarm.urdf
```

If you don't have the URDF file locally, download it from HuggingFace:

```bash
hf download HorizonRobotics/Real-World-Dataset \
    --repo-type dataset \
    --include "urdf/piper_description_dualarm.urdf" \
    --local-dir .
```

> [!NOTE]
> If you're using a different robot, you'll need to provide your own URDF file.

### Packaging Command (Grasp Anything)

```bash
python3 -m robo_orchard_lab.dataset.horizon_manipulation.packer.mcap_arrow_packer \
    --input_path "./data/raw_data/place_objects_to_basket/*.mcap" \
    --output_path "./data/arrow_dataset/place_objects_to_basket" \
    --urdf_path "./urdf/piper_description_dualarm.urdf"
```

> [!TIP]
> If the output directory already exists, add `--force_overwrite` to overwrite it.

During packaging, the tool:
- Extracts ROS messages directly from `.mcap` files
- Synchronizes multi-modal sensor streams while discarding static frames
- Transforms trajectories into the standardized RO dataset format
- Compiles essential metadata (saved as `.duckdb`) for the training pipeline

### Expected Output Structure

```
data/arrow_dataset/
  place_objects_to_basket/
    data-00000-of-00070.arrow
    data-00001-of-00070.arrow
    ...
    state.json
    dataset_info.json
    meta_db.duckdb
```

Each directory is one recording session. The sharded `.arrow` files contain all episodes. The `state.json` file is used by the data loader to discover valid datasets — this is important for troubleshooting (see FAQ).

---

## 3. Data Checking

This step verifies the correctness and quality of your packaged dataset. It is **optional but strongly recommended** when working with a new dataset or after modifying the data pipeline. For re-runs on known-good data, you can safely skip to Step 4.

### 3.1 Packaging Result Checking

**Purpose:** Ensures the `.arrow` packaging process accurately preserved all sensor streams, actions, and timestamps without data loss.

**Output:** Reconstructed `.mcap` files for visual inspection in tools like [Foxglove](https://foxglove.dev/).

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py
python3 data_convert_mcap.py --config ${CONFIG}
```

Output `.mcap` files are saved to `./workspace/` by default. Use `--workspace <path>` to change the output directory.

**What to look for in Foxglove:**
- **Timestamp alignment:** Camera frames and joint states should be synchronized. Large gaps or jitter indicate packaging issues.
- **Missing streams:** All three cameras (left, right, middle) should have continuous image data. A missing stream usually means the camera topic name in the config doesn't match the recording.
- **Trajectory continuity:** Joint state plots should be smooth. Sudden jumps may indicate corrupted data points.

### 3.2 Training Data Checking

**Purpose:** Validates that the `.arrow` dataset loads correctly into the training pipeline and that all data transforms (e.g., resizing, normalization, and noise augmentations) behave as expected.

**Output:** Visualization videos (`.mp4`) showing exactly what the model will "see" as input tensors.

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py
python3 scripts/data_visualize.py --config ${CONFIG}
```

Output videos are saved to `./workspace/` by default. Use `--workspace <path>` to change the output directory.

**What to look for in the output videos:**
- **Correct images:** Camera views should show the workspace clearly, not black/corrupted frames.
- **Reasonable augmentations:** Random crops and noise should look natural — extreme distortions indicate misconfigured transform parameters.
- **Action overlay:** If visualized, predicted actions should roughly follow the demonstrated trajectory.

---

## 4. Model Training

This step trains the HoloBrain model using your packaged `.arrow` datasets. Training is configured through two Python files that work together.

### 4.1 Configuration Structure

| Config Type | Purpose | Example |
| :--- | :--- | :--- |
| **Dataset Config** | Defines data locations, camera names, URDF, and the preprocessing pipeline. | [`config_agilex_ro_dataset.py`](configs/config_agilex_ro_dataset.py) |
| **Training Config** | Defines model architecture, hyperparameters, and references which datasets to use. | [`config_holobrain_qwen_common.py`](configs/config_holobrain_qwen_common.py) |

The training config imports and references the dataset config, so you typically edit both.

### 4.2 Creating Your Dataset Config

Use [`config_agilex_ro_dataset.py`](configs/config_agilex_ro_dataset.py) as a template. The key section is the `dataset_config` dictionary. For Grasp Anything, the values are already filled in:

```python
dataset_config = dict(
    grasp_anything_ro=dict(
        data_paths=[
            "./data/arrow_dataset/place_objects_to_basket/part*",
        ],
        urdf="./urdf/piper_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
    ),
)
```

The three fields you need to customize for your own task:
- `data_paths`: Path(s) to your packaged `.arrow` dataset directories. The loader looks for `state.json` inside each directory to discover valid datasets.
- `urdf`: Path to your robot's URDF file.
- `cam_names`: Camera names matching the topics used during recording.

### 4.3 Registering the Dataset in the Training Config

Open your training config (e.g., [`config_holobrain_qwen_common.py`](configs/config_holobrain_qwen_common.py)) and add your dataset identifier to both `training_datasets` and `deploy_datasets`:

```python
config.update(
    training_datasets=[
        "grasp_anything_ro",  # must match the key in dataset_config
    ],
    deploy_datasets=[
        "grasp_anything_ro",
    ],
)
```

### 4.4 Key Hyperparameters

The training config contains several hyperparameters. Here are the most important ones for beginners:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `batch_size` | `16` | Number of samples per training step. Reduce if you run out of GPU memory. |
| `pred_steps` | `64` | Number of future action steps the model predicts at each timestep. |
| `lr` | `1e-4` | Learning rate. The VLM backbone uses `lr * 0.1` automatically. |
| `max_step` | `100000` | Total training iterations. |
| `save_step_freq` | `5000` | Save a checkpoint every N steps. |
| `step_log_freq` | `100` | Print training metrics (loss, learning rate, etc.) to the console every N steps. |
| `num_workers` | `16` | Dataloader workers. Reduce if you hit CPU/memory limits. For the Grasp Anything task, `4` is usually sufficient. |

> [!TIP]
> With `batch_size=16`, training requires ~16 GB GPU memory. If you hit OOM, reduce `batch_size` to `8` or `4`.

### 4.5 Pretrained Checkpoint

The default training config loads a pretrained HoloBrain checkpoint to fine-tune from:

```python
checkpoint="hf://model/HorizonRobotics/HoloBrain_v0.0_Qwen/pretrain/model.safetensors"
```

This is downloaded automatically on the first training run. Starting from this checkpoint significantly speeds up convergence compared to training from scratch.

### 4.6 Launching the Training

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py

# Single-GPU training
python3 scripts/train.py --config ${CONFIG}

# Multi-GPU / multi-machine training (example: 2 machines × 8 GPUs)
accelerate launch  \
    --num_machines 2 \
    --num-processes 16  \
    --multi-gpu \
    --gpu-ids 0,1,2,3,4,5,6,7  \
    --machine_rank ${current_rank} \
    --main_process_ip ${main_process_ip} \
    --main_process_port 1227 \
    scripts/train.py \
    --workspace ./workspace \
    --config ${CONFIG}
```

Checkpoints are saved to the `./workspace` directory (or the path specified by `--workspace`).

**Training output:**
- **Checkpoints:** `{workspace}/checkpoints/` (default `./workspace/checkpoints/`). Only the latest 3 are kept. Each checkpoint directory contains:
  - `model.safetensors` — trained model weights.
  - `model.config.json` — model architecture and inference configuration.
- **Processor config:** `{workspace}/*_processor.json`. Defines the data pre- and post-processing pipeline for inference (camera extrinsics, image transforms, coordinate-frame conversions, robot kinematics, etc.).
- **TensorBoard logs:** `{workspace}/logs/`. View with `tensorboard --logdir ./workspace/logs`.

---

## 5. Deployment

After training, you need to **export** the checkpoint into a deployment-ready directory before launching inference. The export step produces three artifacts that the inference server requires:

| Artifact | Description |
| :--- | :--- |
| `model/model.safetensors` | Trained model weights. |
| `model/model.config.json` | Model architecture and inference configuration. |
| `*_processor.json` | Data processor config — defines the pre- and post-processing pipeline for inference, including camera extrinsics, image transforms, coordinate-frame conversions, and robot kinematics. |

### 5.1 Export Command

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py
python3 scripts/export.py --config ${CONFIG} --workspace ./workspace
```

The exported directory (`./workspace`) will contain the `model/` subdirectory and processor JSON files listed above.

### 5.2 Deploy to Robot

Deployment involves:
1. **Hardware setup:** Agilex Piper dual arms + Intel RealSense cameras, connected via CAN bus and USB.
2. **Camera calibration:** Use the provided extrinsics or run hand-eye calibration for custom setups.
3. **Inference server:** Launch the HoloBrain model server to serve predictions over the network.
4. **Robot app:** Connect the [ROS2 deploy node](https://github.com/HorizonRobotics/RoboOrchard/tree/master/ros2_package/robo_orchard_deploy_ros2) to the inference server and execute predicted actions.

For the complete walkthrough — including hardware setup, hand-eye calibration, CAN/camera ID configuration, and sync/async inference modes — see the **[Real Robot Deployment Guide](REALBOT_DEPLOY_GUIDE.md)**.

---

## Troubleshooting / FAQ

**Q: My arrow dataset has 0 episodes.**
The data loader discovers episodes by looking for `state.json` inside directories specified by your `data_paths`. Double-check that:
1. Your `data_paths` (e.g., `./data/arrow_dataset/place_objects_to_basket`) point to existing directories.
2. Each directory contains a `state.json` file.

**Q: Visualization shows black images.**
This usually means the camera topic names in your dataset config (`cam_names`) don't match the topics used during recording. Verify the camera names by inspecting the raw `.mcap` file in Foxglove, then update `cam_names` in your dataset config accordingly.

**Q: Training loss doesn't decrease.**
Common causes:
1. **Dataset paths don't resolve:** The training script may silently load zero episodes. Run `python3 scripts/data_visualize.py` first to confirm data loads correctly.
2. **Wrong URDF:** If the URDF doesn't match the robot that recorded the data, kinematics transforms will produce garbage. Verify you're using the correct URDF file.
3. **Corrupted data:** Run the data checking step (Step 3) to inspect the actual training inputs visually.
