# HoloBrain 0.5

HoloBrain 0.5 is the action-only release of the architecture: Qwen3.5-2B
with four language blocks, a 384-dimensional action decoder, RGB-D input and
64-step action prediction. The model and processor live in
`robo_orchard_lab.models.holobrain_0_5`, separately from earlier HoloBrain models.
Only RoboTwin 2.0 (ALOHA and UR5-WSG) and standard LIBERO are supported.

## Install

Use a separate Python 3.10+ environment for this version:

```bash
pip install -e '.[holobrain_0_5]'
# Optional accelerated attention; choose a wheel matching PyTorch/CUDA.
pip install flash-attn --no-build-isolation
```

The validated model environment uses PyTorch 2.8 / CUDA 12.8 and Transformers
5.10.2. Install PyTorch3D matching your PyTorch build. Without FlashAttention,
set `"attn_implementation": "sdpa"` in the training overrides. Install the
upstream [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) or
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulator and its assets
for data replay and closed-loop evaluation. LIBERO uses robosuite 1.4 and an EGL
renderer; RoboTwin requires its matching SAPIEN, mplib and CuRobo environment.

## Docker

A prebuilt environment is available as
`docker.io/horizonrobotics/holobrain:v0.5-ubuntu2204-cu128-torch280-tf5102`.
Use it for training and evaluation with PyTorch 2.8, CUDA 12.8 and Transformers
5.10.2. The host needs an NVIDIA driver and NVIDIA Container Toolkit. The commands
below mount the repository and runtime resources from the host, so run them from
the repository root and replace each `/path/to/...` placeholder with a host path:

```bash
docker pull docker.io/horizonrobotics/holobrain:v0.5-ubuntu2204-cu128-torch280-tf5102

docker run --rm -it --gpus all --ipc=host \
  -v "$PWD:/workspace/robo_orchard_lab" \
  -v /path/to/data:/workspace/data \
  -v /path/to/ckpt:/workspace/ckpt \
  -v /path/to/urdf:/workspace/urdf \
  -v /path/to/third_party:/workspace/third_party \
  -v /path/to/outputs:/workspace/outputs \
  -v /path/to/hf_cache:/workspace/hf_cache \
  -e HF_HOME=/workspace/hf_cache \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --entrypoint /bin/bash \
  -w /workspace/robo_orchard_lab \
  docker.io/horizonrobotics/holobrain:v0.5-ubuntu2204-cu128-torch280-tf5102
```

Inside the container, set the resource roots and run the same training or
evaluation commands as below. The default config downloads the action
checkpoint from [HoloBrain_v0.5](https://huggingface.co/HorizonRobotics/HoloBrain_v0.5)
through the `hf://` URI; the container therefore needs outbound access to the
Hugging Face Hub. `Qwen3.5-2B` remains a local resource and should be available
under `/workspace/ckpt/Qwen3.5-2B`.

```bash
export HOLOBRAIN_DATA_BASE=/workspace/data
export HOLOBRAIN_CKPT_BASE=/workspace/ckpt
export HOLOBRAIN_URDF_BASE=/workspace/urdf
pip install -e '.[holobrain_0_5]' --no-deps

# Multi-GPU training. Use `python -m` for a single-GPU run.
accelerate launch --multi_gpu -m projects.holobrain_0_5.train \
  --workspace /workspace/outputs/holobrain_0_5 \
  --kwargs '{"training_datasets":["libero_goal"]}'

# LIBERO evaluation.
MODEL_DIR=/workspace/outputs/holobrain_0_5 \
LIBERO_ROOT=/workspace/third_party/LIBERO \
bash projects/holobrain_0_5/eval_libero.sh

# LIBERO-Plus evaluation.
MODEL_DIR=/workspace/outputs/holobrain_0_5 \
LIBERO_PLUS_ROOT=/workspace/third_party/LIBERO-plus \
bash projects/holobrain_0_5/eval_libero_plus.sh

# RoboTwin evaluation.
MODEL_DIR=/workspace/outputs/holobrain_0_5 \
ROBOTWIN_ROOT=/workspace/third_party/RoboTwin \
bash projects/holobrain_0_5/eval_robotwin.sh
```

For a single-GPU smoke run, replace `accelerate launch --multi_gpu -m` with
`python -m` and pass smaller `batch_size` and `max_step` overrides. Mount the
Hugging Face cache to avoid downloading the action checkpoint again when a new
container is started.

To evaluate the published checkpoint without training, build an export with the
default `hf://` checkpoint and use that directory as `MODEL_DIR` in the evaluation
commands above:

```bash
python -m projects.holobrain_0_5.export \
  --workspace /workspace/outputs/holobrain_0_5_pretrained
```

Place Qwen3.5-2B weights, tokenizer and processor files in a local directory.
By default that is `./ckpt/Qwen3.5-2B`; see `HOLOBRAIN_CKPT_BASE` below to keep
them elsewhere. Training and evaluation use these local data, simulator and
Qwen resources; the action checkpoint is downloaded from Hugging Face through
the `hf://` URI above.

## Prepare data

Existing LMDB datasets must contain `index`, `meta`, `image`, and `depth`.
RoboTwin preparation reuses the existing packer:

```bash
python -m robo_orchard_lab.dataset.robotwin.robotwin_packer \
  --input_path ./raw/robotwin --output_path ./data/robotwin2.0/aloha_agilex_demo_clean \
  --config_name demo_clean
```

Replay original LIBERO HDF5 demonstrations to capture metric depth, camera
calibration, and absolute controller targets:

```bash
MUJOCO_GL=egl python -m projects.holobrain_0_5.prepare_libero \
  --libero-root ./third_party/LIBERO --input ./raw/libero_goal \
  --output ./data/libero/lmdb_goal_abs
```

Use `--max-episodes 1` for a small preparation check. Repeat for `libero_object`,
`libero_spatial`, and `libero_10`. Keep the original suite directory names.
The packer requires original `states`, delta `actions`, and `model_file`
metadata; RGB-only datasets are insufficient to reconstruct this model's input.

Set `HOLOBRAIN_DATA_BASE` to the directory containing `robotwin2.0/` and
`libero/`. Set `HOLOBRAIN_CKPT_BASE` to the directory containing `Qwen3.5-2B/`;
it defaults to `./ckpt`. For RoboTwin, also set `HOLOBRAIN_URDF_BASE` to the body
resource root
used by [robot_specs.py](configs/robot_specs.py); e.g. the ALOHA body expects
`robotwin/aloha/arx5_description_isaac.pruned.urdf`. Preserve joint order and the
body/camera calibration corresponding to your demonstrations.

## Train

Run commands from the repository root. A single GPU smoke run:

```bash
export HOLOBRAIN_DATA_BASE=./data
export HOLOBRAIN_URDF_BASE=./urdf
export HOLOBRAIN_CKPT_BASE=./ckpt
python -m projects.holobrain_0_5.train --workspace ./outputs/holobrain_0_5_robotwin \
  --kwargs '{"vlm_pretrain":"Qwen3.5-2B","training_datasets":["robotwin2_0"],"batch_size":1,"max_step":20,"warmup_step":2,"step_log_freq":1}'
```

For LIBERO, use `"training_datasets":["libero_goal"]` and a different workspace.
Available selections are in [dataset_specs.py](configs/dataset_specs.py).
The two RoboTwin selections are `robotwin2_0` (ALOHA) and
`robotwin2_0_ur5_wsg` (UR5-WSG); other RoboTwin bodies and RoboTwin 1.0
are not included.
`data_paths` accepts a mapping from selection name to local path(s), and
`task_names` can restrict the run to named tasks. `--kwargs` also accepts a JSON
file. Use `accelerate launch -m projects.holobrain_0_5.train ...` for multiple GPUs.

For the supplied Panda Droid simulation data, select `panda_droid_rigid`,
`panda_droid_articulated`, or both. These datasets require an image and depth
resize of **448 pixels wide by 256 pixels high**. Explicitly set
`"dst_wh": [448, 256]` in the training configuration; the public default
`(352, 256)` is not suitable for this data. The two selections share the
`panda_droid` deployment processor:

```bash
python -m projects.holobrain_0_5.train --workspace ./outputs/holobrain_0_5_panda \
  --kwargs '{"training_datasets":["panda_droid_rigid","panda_droid_articulated"],"dst_wh":[448,256]}'
```

Use `"dst_wh": [448, 256]` for validation and standalone export as well. The
exported processor and inference configuration retain the resolved size;
loading an existing export does not use the project default. Within one run,
this override applies to every selected dataset and every exported processor.

`vlm_pretrain` is resolved in one of two ways. A bare name such as
`Qwen3.5-2B` is looked up under `HOLOBRAIN_CKPT_BASE` (default `./ckpt`); it is
not searched for anywhere else, so a missing directory fails with the tried
path rather than silently loading other weights. Anything containing a path
separator, such as `./ckpt/Qwen3.5-2B` or an absolute path, is used verbatim
relative to the current directory and ignores the variable.

`checkpoint` optionally names a pretrained action `.safetensors` file, either as
a full local path or as an `hf://` URI such as
`hf://model/<org>/<repo>/pretrain/model.safetensors`, which is downloaded to the
Hugging Face cache before loading. It is not affected by `HOLOBRAIN_CKPT_BASE`.
Loading is strict: incompatible architectures fail instead of silently dropping
parameters. The default loads the published checkpoint at
`hf://model/HorizonRobotics/HoloBrain_v0.5/model.safetensors`. To train the action
head from scratch while initializing Qwen from its local pretrained directory,
pass `--kwargs '{"checkpoint":null}'` (or add it to your other overrides).

The final export contains `model.safetensors`, `model.config.json`, one
`<dataset>_processor.json` and `<dataset>_inference.config.json` per entry in
[deploy_specs.py](configs/deploy_specs.py), plus required URDF copies. The
processors and pipeline configs are written *before* training starts, so a run
that is cut short still leaves a usable export next to its checkpoints; only
`model.safetensors` waits for the end. Deployment is deliberately decoupled from
training: every body in `deploy_specs.py` is exported regardless of
`training_datasets`, which is why `HOLOBRAIN_URDF_BASE` must resolve even for a
LIBERO-only run. It must contain the RoboTwin URDFs and
`isaac/panda_droid/franka_droid.pruned.urdf`. Panda exports use
`panda_droid_processor.json` and `panda_droid_inference.config.json`.

By default `ckpt/Qwen3.5-2B` is a symlink to
the supplied local resource; preserve or replace that link when moving exports.
That reference is resolved relative to the export directory, so loading an
export never depends on `HOLOBRAIN_CKPT_BASE`.
This is a small training entry with one optimizer update per batch, final export,
loss logging, and a `training_result.json` recording completed optimizer
updates. Before training starts it logs the resolved config and copies it into
the workspace: the config directory as `<workspace>/configs/` and the resolved
values, overrides included, as `training_config.json`.

## Export without training

`export.py` builds the same layout from a config alone, reloading every
artifact to verify it. Use it to package an existing checkpoint, or to refresh
processors after changing `deploy_specs.py`:

```bash
python -m projects.holobrain_0_5.export --workspace ./outputs/holobrain_0_5_export \
  --kwargs '{"checkpoint":"./ckpt/action.safetensors"}'
```

`save_step_freq` (default 5000) writes a resumable Accelerate checkpoint every
that many committed optimizer updates, under `<workspace>/checkpoints/checkpoint_<n>`;
the final export stays at the workspace root. `total_limit` (default 3) caps how
many are kept. Each holds the model plus optimizer state and measured about
7.6 GB for the shipped config, so the default retention needs roughly 23 GB.
Checkpoints are also written at the end of each epoch and of the training loop.

To resume, point `resume_from` at a checkpoint directory:

```bash
python -m projects.holobrain_0_5.train --workspace ./outputs/holobrain_0_5_robotwin \
  --kwargs '{"resume_from":"./outputs/holobrain_0_5_robotwin/checkpoints/checkpoint_2"}'
```

This restores the optimizer, scheduler and step counters, unlike `checkpoint`,
which only loads weights.

## Evaluate the exported policy

One script per benchmark runs the full default sweep across every visible GPU.
Each reads its paths from the environment and forwards any extra arguments to
the underlying entry point:

```bash
MODEL_DIR=./outputs/holobrain_0_5_robotwin ROBOTWIN_ROOT=./third_party/RoboTwin \
  bash projects/holobrain_0_5/eval_robotwin.sh

MODEL_DIR=./outputs/holobrain_0_5_libero LIBERO_ROOT=./third_party/LIBERO \
  bash projects/holobrain_0_5/eval_libero.sh

MODEL_DIR=./outputs/holobrain_0_5_libero LIBERO_PLUS_ROOT=./third_party/LIBERO-plus \
  bash projects/holobrain_0_5/eval_libero_plus.sh
```

`MODEL_DIR` and the benchmark root are required; `OUTPUT`, `PROCESSOR`,
`PROCESSES_PER_GPU` and (for RoboTwin) `TASK_CONFIG` and `EPISODES` override the
defaults. The LIBERO scripts export the EGL renderer variables. The
`PROCESSES_PER_GPU` defaults of 5 for LIBERO and 2 for RoboTwin match the
validated sweeps; lower them if a GPU runs out of memory. For the UR5-WSG body
use `PROCESSOR=robotwin2_0_ur5_wsg_processor TASK_CONFIG=demo_clean_ur5-wsg`.

To narrow a sweep, pass the entry point's own flags through:

```bash
... bash projects/holobrain_0_5/eval_robotwin.sh --tasks place_empty_cup --episodes 2
... bash projects/holobrain_0_5/eval_libero.sh --task-suites libero_goal --num-trials-per-task 2
```

Or call the modules directly:

```bash
python -m projects.holobrain_0_5.robotwin_eval \
  --robotwin-root ./third_party/RoboTwin \
  --model-dir ./outputs/holobrain_0_5_robotwin \
  --tasks place_empty_cup stack_blocks_three --episodes 2 \
  --output ./outputs/robotwin_eval

MUJOCO_GL=egl python -m projects.holobrain_0_5.libero_eval \
  --libero-root ./third_party/LIBERO \
  --model-dir ./outputs/holobrain_0_5_libero \
  --output ./outputs/libero_eval
```

RoboTwin evaluates every task its checkout defines under `envs/` unless
`--tasks` names a subset. Tasks whose expert planner cannot reliably solve a
given embodiment are skipped with a warning rather than burning their seed
budget for a rate measured over a handful of episodes.

By default LIBERO evaluates every task of `libero_10`, `libero_goal`,
`libero_object` and `libero_spatial` with 50 trials each: 40 tasks, 2000
episodes. Those `(suite, task)` pairs are dealt round-robin across every visible
GPU, one subprocess per worker, each loading the policy once and sweeping its
share. Narrow the sweep with `--task-suites`, `--task-ids` and
`--num-trials-per-task`, size the fan-out with `--num-gpus` and
`--processes-per-gpu`, and print the selection without running it using
`--dry-run`.

Each task writes `<output>/<suite>/task_<n>/results.json` and the launcher folds
them into `<output>/summary.json`. That per-task file is also the resume record:
a worker only evaluates tasks that lack one. A MuJoCo segfault kills its process
uncatchably, so the worker relaunches on whatever is still missing and gives up
on a task after three relaunches with no progress, rather than losing the sweep.

Pass `--benchmark libero_plus --libero-plus-root <checkout>` for LIBERO-Plus. It
reuses the same four suite names from its own checkout and defaults to one trial
per task, since the benchmark already varies each task.

Select a matching `--processor` when using another body or suite. RoboTwin
executes up to 32 predicted joint actions per call; LIBERO executes eight
absolute end-effector targets with the original frame and gripper conversions.

`--model-dir` accepts either an export root or an intermediate
`<workspace>/checkpoints/checkpoint_<n>` directory. Weights always come from the
given directory; the processor is looked up beside it and then two levels up, so
a checkpoint picks up the `<dataset>_processor.json` and `urdf/` that training
wrote at the workspace root. The directory actually used is logged. Evaluating a
checkpoint directory also needs `HOLOBRAIN_CKPT_BASE` set to the directory
containing `Qwen3.5-2B`, because a checkpoint records the VLM by bare name while
a final export carries its own `ckpt/` link.
Result JSON files include episode and success counts. A short training run
validates the execution path; it is not a benchmark-quality trained policy.
