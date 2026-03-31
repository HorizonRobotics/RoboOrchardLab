# Real Robot Deployment Guide of HoloBrain-0

This guide covers deploying a trained HoloBrain model on a physical robot — from hardware setup (including camera calibration) through to launching inference on the real robot.

> [!NOTE]
> This guide focuses on **deployment only**. If you need to record data, package datasets, or train a model first, see the **[Real Robot Pipeline Guide](REALBOT_PIPELINE_GUIDE.md)**.

## Prerequisites

Before following this guide, make sure you have completed the following:

- **Trained and exported a HoloBrain model** — see [Export Model](README.md#5-export-model-and-processors-and-pipeline) in the main README
- **Installed the Real Robot App** — see [HoloBrain Real Robot App](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain) for setup instructions

## Deployment Overview

The deployment process has three stages:

| Stage | Section | What You'll Do |
| --- | --- | --- |
| 1 | [Hardware Setup](#1-hardware-setup) | Assemble robot arms, cameras, brackets, and calibrate cameras |
| 2 | [Launch Inference Server](#2-launch-inference-server) | Start the model server on your GPU machine |
| 3 | [Configure & Launch Robot App](#3-configure-and-launch-real-robot-app) | Wire up the robot app and start autonomous operation |

> [!NOTE]
> This guide involves two repositories:
> - **RoboOrchardLab** (this repo) — model training, export, and inference server (`projects/holobrain/`)
> - **[Real Robot App](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain)** — robot control, camera, and launch configuration (`projects/HoloBrain/`)
>
> All file paths below are prefixed with their repository name for clarity.

## 1. Hardware Setup

Before you begin, make sure you have the following hardware components ready.

| Component | Specification | Quantity |
| --- | --- | --- |
| Robot Arm | Agilex Piper | 2 |
| Camera | Intel RealSense D435i | 3 |

### 1.1 Hardware Setup for Grasp Anything Task

The table below shows the reference configuration used in our Grasp Anything experiments. If your setup differs, you'll need to perform your own camera calibration (see [Camera Calibration](#12-camera-calibration) below).

| Parameter | Value | Description |
| --- | --- | --- |
| Arm Spacing | 60 cm | Distance between the base-link centers of the two Piper arms |
| Arm Mounting Height | 0 cm (table level) | Both arms are mounted on the table surface |
| Middle Env Camera Height | 50 cm | Height of the middle environment camera optical center above the table surface |
| Wrist Camera Bracket | See [bracket files](./assets/) | 3D-printable bracket for mounting D435i on each Piper wrist |

> [!NOTE]
> All height values are measured relative to the table surface. You can also purchase a ready-made camera bracket from Agilex directly: [link](https://item.taobao.com/item.htm?id=974751867024&mi_id=0000hTzpIhXxYygNl_eTQu9bu2vBIAj8rpzN26HxSygPPCo&spm=a21xtw.29178619.0.0&xxc=shop).

### 1.2 Camera Calibration

Accurate camera extrinsics — the position and orientation of each camera relative to the robot — are essential for the model to correctly map visual observations to robot actions.

If your hardware is set up **strictly following the configuration above**, you can directly use the camera extrinsics provided in our [HuggingFace processor](https://huggingface.co/HorizonRobotics/HoloBrain_v0.0_GD/blob/main/real_world_agilex_grasp_anything_processor.json) — no additional calibration is needed.

If you have a **customized hardware setup** (e.g., different camera positions, arm spacing, or mounting angles), you will need to perform hand-eye calibration. Please refer to:

👉 [Hand-Eye Calibration Tool](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain/handeye_calib)

## 2. Launch Inference Server

Once your hardware is set up and cameras are calibrated, the first step is to start the HoloBrain inference server. This server loads your trained model and serves action predictions over the network — the real robot app (Section 3) connects to it to get real-time commands.

```bash
cd projects/holobrain
# RoboOrchardLab · projects/holobrain/scripts/inference_server.py
python3 scripts/inference_server.py \
    --model_dir "/your/model_dir" \
    --port 2000 \
    --server_name holobrain \
    --num_joints 7 \
    --valid_action_step 64
```

> [!TIP]
> The `model_dir` should point to an exported model directory (see [Export Model](README.md#5-export-model-and-processors-and-pipeline) in the main README). You can also use a HuggingFace model path directly (e.g., `hf://HorizonRobotics/HoloBrain_v0.0_Qwen`).

## 3. Configure and Launch Real Robot App

The real robot application handles camera capture, robot arm control, and communication with the inference server. It is hosted in a separate repository — see [HoloBrain Real Robot App](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain) for full setup instructions.

Before launching, you need to configure the app to match your specific hardware. The subsections below walk through each configuration item.

### 3.1 Configure Robot Arm CAN IDs

Each Piper arm communicates over CAN (Controller Area Network) bus. You need to identify the correct CAN ports and map them to the left/right arms.

1. Discover all available CAN ports:

   ```bash
   # Real Robot App · projects/HoloBrain/teleop/find-all-can-port.sh
   bash teleop/find-all-can-port.sh
   ```

2. Based on the output, update the CAN port mappings in `teleop/templates/rename-can.sh` (Real Robot App) to match your robot arms.

### 3.2 Configure Camera IDs

Each Intel RealSense camera has a unique serial number. Edit `launch/templates/launch.yaml` (Real Robot App) and update the camera serial numbers in the `environment` section to match your three D435i cameras (left, right, middle).

> [!TIP]
> You can find camera serial numbers by running `rs-enumerate-devices`, a CLI tool included in the [Intel RealSense SDK (librealsense)](https://github.com/IntelRealSense/librealsense).

### 3.3 Add Inference Tmux Session

Add the following tmux session to `launch/templates/launch.yaml` (Real Robot App) so the inference client starts automatically:

```yaml
  - window_name: inference
    layout: tiled
    shell_command_before:
      - cd $DOCKER_ROBO_ORCHARD_PATH/projects/holobrain
    panes:
      - shell_command:
        - CMD="bash inference/launch_async_infer.sh"
        - history -s "$CMD"
        - eval "$CMD"
```

### 3.4 Configure Inference Settings

HoloBrain supports two inference modes. Choose the one that fits your needs:

| Mode | Behavior | Pros | Cons |
| --- | --- | --- | --- |
| **Sync** | Robot waits for each prediction before executing | Simpler to set up | Slower, may cause pauses between actions |
| **Async** | Robot continues executing while the next prediction is computed | Smoother, faster motion | Requires additional RTC plugin configuration |

> [!TIP]
> **Async mode is recommended for real deployments** — it produces smoother motion and better task performance. Use sync mode for initial testing or debugging.

#### Sync Inference

In sync mode, the robot waits for each prediction before executing the next action.

- Modify `inference/gen_sync_config.py` (Real Robot App) with your inference server address and model settings. Key fields: `server_url` (inference server address) and `infer_frequency` (prediction rate in Hz).

#### Async Inference

In async mode, the robot continues executing while the next prediction is being computed, resulting in smoother and faster motion.

- Modify `inference/gen_async_config.py` (Real Robot App) with your inference server address and model settings. Key fields: `server_url` (inference server address), `infer_frequency`, and `delay_horizon`.
- Add the RTC (Real-Time Correction) plugin to your exported model's config file `model.config.json` (RoboOrchardLab, inside your model export directory, e.g., `model_export_path/model/model.config.json`):
    ```json
    "async_inference_plugin": {
        "type": "robo_orchard_lab.models.rtc_plugin.rtc_plugin:RTCInferencePlugin"
    },
    ```

> [!NOTE]
> The RTC plugin is only needed at inference time — it is not used during training.

### 3.5 Start Inference

Once everything is configured, start the Real Robot App using the launch script:

```bash
# Real Robot App · projects/HoloBrain/launch/start.sh
bash launch/start.sh
```

Then:

1. Open the **control panel** in the app UI.
2. Set **Control Mode** to `Auto`.
3. Under **Inference Control**, click `Start` to begin autonomous operation.

> [!TIP]
> If the robot does not move after clicking Start, verify that the inference server (Section 2) is running and reachable from the robot app machine.
