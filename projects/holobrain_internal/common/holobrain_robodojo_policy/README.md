# HoloBrain RoboDojo policy

This directory is an XPolicyLab policy adapter for RoboDojo's dual ARX X5
configuration. It supports HoloBrain joint-space checkpoints and keeps the
policy runtime separate from the RoboDojo/Isaac Sim runtime.

## Install into a RoboDojo checkout

From the RoboOrchard Lab repository root:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export ROBODOJO_ROOT="$REPO_ROOT/RoboDojo"

cp -r projects/holobrain_internal/common/holobrain_robodojo_policy \
  "$ROBODOJO_ROOT/XPolicyLab/policy/"
```

The adapter does not maintain a copy of RoboDojo's environment config.
`robodojo_eval.py` selects a config with `--env_config` (default `arx_x5`),
copies it from RoboDojo's `env_cfg/` into the writable runtime config
directory, and enables the camera intrinsics and extrinsics required by the
exported HoloBrain processor. RoboDojo's source config remains unchanged.

## Model configuration

Export the RoboDojo processor with the model, then expose the model and
RoboOrchard Lab source to the policy environment:

```bash
cd projects/holobrain_internal/common
python3 export.py \
  --config configs/config_holobrain_common.py \
  --workspace ./workspace \
  --dataset_names robodojo

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HOLOBRAIN_MODEL_DIR=/path/to/exported/model
export HOLOBRAIN_VLM_CKPT_DIR=/path/to/ckpt
export HOLOBRAIN_URDF_DIR=/path/to/urdf
```

`HOLOBRAIN_MODEL_DIR` can also be an HTTP checkpoint URL. Optional overrides
are `HOLOBRAIN_MODEL_PROCESSOR`, `HOLOBRAIN_MODEL_PREFIX`,
`HOLOBRAIN_LOAD_IMPL`, and `HOLOBRAIN_VALID_ACTION_STEP`.

## Evaluation

Use the repository launcher so the policy, writable environment config, GPU
workers, and result paths are prepared consistently:

```bash
python projects/holobrain_internal/common/robodojo_eval.py \
  --policy_source projects/holobrain_internal/common/holobrain_robodojo_policy \
  --model_dir /path/to/exported/model \
  --model_processor robodojo_arx_x5a_processor \
  --robodojo_root "$ROBODOJO_ROOT" \
  --policy_env /path/to/holobrain/venv \
  --tasks stack_bowls \
  --eval_num 1
```

`--policy_env` accepts either a Conda environment name or an absolute path to
a virtualenv. The simulator continues to run in the separate `RoboDojo` Conda
environment.

This adapter defaults to non-batched evaluation. It accepts RGB camera data,
creates the zero-depth inputs used by RoboDojo training, converts Isaac/USD
camera-to-world matrices to OpenCV world-to-camera matrices, and emits 14-D
dual-arm joint action chunks.
