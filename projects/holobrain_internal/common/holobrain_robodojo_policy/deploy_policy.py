# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import fcntl
import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robo_orchard_lab.dataset.robodojo.robodojo_lmdb_packer import (
    cam2world_usd_to_world2cam_cv,
)
from robo_orchard_lab.models.holobrain.processor import (
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin

logger = logging.getLogger(__name__)

ROBODOJO_CAMERAS = (
    "cam_left_wrist",
    "cam_right_wrist",
    "cam_head",
)
ROBODOJO_ACTION_DIM = 14
ROBODOJO_GRIPPER_INDICES = (6, 13)
DEFAULT_MODEL_CACHE_DIR = "./workspace/robodojo_model"
EXTRINSIC_KEYS = ("extrinsic_matrix", "extrinsics_matrix")


@dataclass
class HoloBrainRoboDojoPolicyCfg:
    """Configuration for the RoboDojo HoloBrain policy adapter."""

    model_dir: str | None = None
    model_processor: str = "robodojo_processor"
    model_prefix: str = "model"
    load_impl: str = "native"
    vlm_ckpt_dir: str | None = None
    urdf_dir: str | None = None
    valid_action_step: int = 32
    use_depth: bool = False
    camera_names: tuple[str, ...] = ROBODOJO_CAMERAS
    extrinsic_type: str = "cam2world_usd"
    task_name: str = ""


def _env_or_config(
    env_name: str,
    model_cfg: dict[str, Any],
    config_name: str,
    default: Any = None,
) -> Any:
    value = os.environ.get(env_name)
    if value is not None and value != "":
        return value
    return model_cfg.get(config_name, default)


def _resolve_model_dir(model_cfg: dict[str, Any]) -> str:
    configured = _env_or_config(
        "HOLOBRAIN_MODEL_DIR",
        model_cfg,
        "model_dir",
    )
    if configured:
        return str(configured)

    ckpt_name = model_cfg.get("ckpt_name")
    if not ckpt_name:
        raise ValueError(
            "HoloBrain requires `model_dir`, HOLOBRAIN_MODEL_DIR, or "
            "ckpt_name."
        )
    ckpt_name = str(ckpt_name)
    if ckpt_name.startswith(("http://", "https://")):
        return ckpt_name
    checkpoint_path = Path(ckpt_name).expanduser()
    if checkpoint_path.exists() or checkpoint_path.is_absolute():
        return str(checkpoint_path)
    return str(Path(__file__).resolve().parent / "checkpoints" / ckpt_name)


def _policy_cfg_from_model_cfg(
    model_cfg: dict[str, Any],
) -> HoloBrainRoboDojoPolicyCfg:
    action_type = model_cfg.get("action_type", "joint")
    if action_type != "joint":
        raise ValueError(
            "HoloBrain RoboDojo evaluation only supports action_type=joint."
        )
    action_dim = model_cfg.get("action_dim")
    if action_dim is not None and int(action_dim) != ROBODOJO_ACTION_DIM:
        raise ValueError(
            "HoloBrain RoboDojo action_dim must be "
            f"{ROBODOJO_ACTION_DIM}, got {action_dim}."
        )

    return HoloBrainRoboDojoPolicyCfg(
        model_dir=_resolve_model_dir(model_cfg),
        model_processor=str(
            _env_or_config(
                "HOLOBRAIN_MODEL_PROCESSOR",
                model_cfg,
                "model_processor",
                "robodojo_processor",
            )
        ),
        model_prefix=str(
            _env_or_config(
                "HOLOBRAIN_MODEL_PREFIX",
                model_cfg,
                "model_prefix",
                "model",
            )
        ),
        load_impl=str(
            _env_or_config(
                "HOLOBRAIN_LOAD_IMPL",
                model_cfg,
                "load_impl",
                "native",
            )
        ),
        vlm_ckpt_dir=_env_or_config(
            "HOLOBRAIN_VLM_CKPT_DIR",
            model_cfg,
            "vlm_ckpt_dir",
        ),
        urdf_dir=_env_or_config(
            "HOLOBRAIN_URDF_DIR",
            model_cfg,
            "urdf_dir",
        ),
        valid_action_step=int(
            _env_or_config(
                "HOLOBRAIN_VALID_ACTION_STEP",
                model_cfg,
                "valid_action_step",
                32,
            )
        ),
        use_depth=bool(model_cfg.get("use_depth", False)),
        extrinsic_type=str(model_cfg.get("extrinsic_type", "cam2world_usd")),
        task_name=str(model_cfg.get("task_name") or ""),
    )


def _is_http_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def _download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        logger.info("Using cached file: %s", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            try:
                shutil.copyfileobj(response, temporary_file)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                os.replace(temporary_path, output_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise


def prepare_model_dir(
    model_dir: str,
    processor_name: str,
    model_prefix: str,
    output_dir: str = DEFAULT_MODEL_CACHE_DIR,
) -> str:
    """Download a remote HoloBrain export or return a local directory."""
    if not _is_http_url(model_dir):
        return model_dir

    model_url = model_dir.rstrip("/")
    processor_url = "/".join(
        model_url.split("/")[:-2] + [f"{processor_name}.json"]
    )
    cache_key = hashlib.sha256(model_url.encode("utf-8")).hexdigest()[:16]
    output_path = Path(output_dir) / cache_key
    files = {
        f"{model_prefix}.safetensors": f"{model_url}/model.safetensors",
        f"{model_prefix}.config.json": (
            f"{model_url}/{model_prefix}.config.json"
        ),
        f"{processor_name}.json": processor_url,
    }
    lock_path = (
        Path(tempfile.gettempdir()) / f"holobrain-model-{cache_key}.lock"
    )
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        for filename, url in files.items():
            logger.info("Downloading %s", url)
            _download_file(url, output_path / filename)
    return str(output_path)


def link_model_resource(link_path: Path, source_dir: str | None) -> None:
    if source_dir is None or os.path.lexists(link_path):
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_path.symlink_to(Path(source_dir).expanduser().resolve())
    except FileExistsError:
        if not os.path.lexists(link_path):
            raise


def link_model_resources(
    model_dir: str,
    vlm_ckpt_dir: str | None,
    urdf_dir: str | None,
) -> None:
    model_path = Path(model_dir)
    link_model_resource(model_path / "ckpt", vlm_ckpt_dir)
    link_model_resource(model_path / "urdf", urdf_dir)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_matrix4(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.shape == (3, 3):
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = matrix
        return result
    raise ValueError(
        f"{name} must have shape (3, 3) or (4, 4), got {matrix.shape}."
    )


def _require_vector(
    mapping: dict[str, Any],
    key: str,
    size: int,
) -> np.ndarray:
    if key not in mapping:
        raise ValueError(f"RoboDojo observation is missing state key `{key}`.")
    value = np.asarray(mapping[key], dtype=np.float32)
    if value.shape != (size,):
        raise ValueError(
            f"RoboDojo state `{key}` must have shape ({size},), "
            f"got {value.shape}."
        )
    return value


def robodojo_obs_to_joint_state(obs: dict[str, Any]) -> np.ndarray:
    """Pack a RoboDojo dual-arm observation in training-data order."""
    state = obs.get("state")
    if not isinstance(state, dict):
        raise ValueError(
            "RoboDojo observation is missing the `state` mapping."
        )
    return np.concatenate(
        [
            _require_vector(state, "left_arm_joint_state", 6),
            _require_vector(state, "left_ee_joint_state", 1),
            _require_vector(state, "right_arm_joint_state", 6),
            _require_vector(state, "right_ee_joint_state", 1),
        ]
    ).astype(np.float32)


def _extract_instruction(obs: dict[str, Any], fallback: str) -> str:
    instruction = obs.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction
    instructions = obs.get("instructions")
    if isinstance(instructions, (list, tuple)):
        for value in instructions:
            if isinstance(value, str) and value.strip():
                return value
    return fallback


def _extract_extrinsic(camera_data: dict[str, Any]) -> tuple[str, Any]:
    for key in EXTRINSIC_KEYS:
        if key in camera_data:
            return key, camera_data[key]
    raise ValueError(
        "RoboDojo camera observation is missing `extrinsic_matrix` "
        "(or the documented alias `extrinsics_matrix`)."
    )


class HoloBrainRoboDojoPolicy:
    """Translate RoboDojo observations and HoloBrain joint predictions."""

    def __init__(
        self,
        cfg: HoloBrainRoboDojoPolicyCfg | dict[str, Any],
        *,
        processor: HoloBrainProcessor | None = None,
        model: Any | None = None,
        pipeline: Any | None = None,
    ) -> None:
        if isinstance(cfg, dict):
            cfg = _policy_cfg_from_model_cfg(cfg)
        if cfg.valid_action_step <= 0:
            raise ValueError("valid_action_step must be positive.")
        if cfg.extrinsic_type not in {"cam2world_usd", "world2cam_cv"}:
            raise ValueError(
                "extrinsic_type must be `cam2world_usd` or `world2cam_cv`."
            )

        self.cfg = cfg
        self.processor = processor
        self.model = model
        self.pipeline = pipeline
        # The policy server has no logging setup of its own -- train.py has
        # log_basic_config, this process has nothing -- so the root logger
        # sits at WARNING and every logger.info in the model is discarded.
        # That silently disables all the eval-side instrumentation, which is
        # the only place it was ever meant to run. Only configure if nobody
        # else has, so a caller that set logging up keeps its own format.
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s | %(message)s",
            )

        self._obs: dict[str, Any] | None = None
        self._batch_obs: dict[int, dict[str, Any]] = {}
        # How many observations this episode has produced. `deploy.py`
        # calls update_obs once per env step and get_action once per
        # `valid_action_step` of them, so counting the former is the exact
        # env frame index -- including for episodes that end early. See
        # _run_holobrain for why the processor's own value cannot be used.
        self._env_step = 0

        if pipeline is None and model is None:
            if cfg.model_dir is None:
                raise ValueError("model_dir must be provided.")
            model_dir = prepare_model_dir(
                cfg.model_dir,
                cfg.model_processor,
                cfg.model_prefix,
            )
            link_model_resources(model_dir, cfg.vlm_ckpt_dir, cfg.urdf_dir)
            self.processor = self._load_processor(model_dir)
            self.model = ModelMixin.load_model(
                model_dir,
                model_prefix=cfg.model_prefix,
                load_impl=cfg.load_impl,
            )
            self.model.eval()
            self.model.requires_grad_(False)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model.to(device)

        if pipeline is not None:
            self.processor = getattr(pipeline, "processor", processor)
            pipeline_model = getattr(pipeline, "model", None)
            if pipeline_model is not None:
                pipeline_model.eval()

    def _load_processor(self, model_dir: str) -> HoloBrainProcessor:
        processor_file = f"{self.cfg.model_processor}.json"
        model_path = Path(model_dir)
        candidates = [
            model_path / processor_file,
            model_path.parent / processor_file,
            model_path.parent.parent / processor_file,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return HoloBrainProcessor.load(
                    str(candidate.parent),
                    candidate.name,
                )
        raise FileNotFoundError(
            f"Cannot find processor `{processor_file}` under {candidates}."
        )

    def data_preprocess(
        self,
        obs: dict[str, Any],
    ) -> MultiArmManipulationInput:
        vision = obs.get("vision")
        if not isinstance(vision, dict):
            raise ValueError(
                "RoboDojo observation is missing the `vision` mapping."
            )

        images: dict[str, list[np.ndarray]] = {}
        depths: dict[str, list[np.ndarray]] = {}
        intrinsics: dict[str, np.ndarray] = {}
        t_world2cam: dict[str, np.ndarray] = {}
        for camera_name in self.cfg.camera_names:
            camera_data = vision.get(camera_name)
            if not isinstance(camera_data, dict):
                raise ValueError(
                    f"RoboDojo observation is missing camera `{camera_name}`."
                )
            if "color" not in camera_data:
                raise ValueError(
                    f"RoboDojo camera `{camera_name}` is missing `color`."
                )
            color = np.asarray(camera_data["color"])
            if color.ndim != 3 or color.shape[-1] != 3:
                raise ValueError(
                    f"RoboDojo camera `{camera_name}` color must have shape "
                    f"(H, W, 3), got {color.shape}."
                )
            # The deploy transform mirrors cv2-loaded training images (BGR).
            images[camera_name] = [np.ascontiguousarray(color[..., [2, 1, 0]])]

            if self.cfg.use_depth:
                if "depth" not in camera_data:
                    raise ValueError(
                        f"RoboDojo camera `{camera_name}` is missing `depth`."
                    )
                depth = np.asarray(camera_data["depth"], dtype=np.float32)
                depth = np.squeeze(depth)
                if depth.shape != color.shape[:2]:
                    raise ValueError(
                        f"RoboDojo camera `{camera_name}` depth has shape "
                        f"{depth.shape}, expected {color.shape[:2]}."
                    )
            else:
                depth = np.zeros(color.shape[:2], dtype=np.float32)
            depths[camera_name] = [depth]

            if "intrinsic_matrix" not in camera_data:
                raise ValueError(
                    f"RoboDojo camera `{camera_name}` is missing "
                    "`intrinsic_matrix`; enable it in the RoboDojo env "
                    "config."
                )
            intrinsics[camera_name] = _to_matrix4(
                camera_data["intrinsic_matrix"],
                name=f"{camera_name}.intrinsic_matrix",
            )

            extrinsic_key, extrinsic = _extract_extrinsic(camera_data)
            extrinsic_matrix = _to_matrix4(
                extrinsic,
                name=f"{camera_name}.{extrinsic_key}",
            )
            if self.cfg.extrinsic_type == "cam2world_usd":
                extrinsic_matrix = cam2world_usd_to_world2cam_cv(
                    extrinsic_matrix
                )
            t_world2cam[camera_name] = extrinsic_matrix

        joint_state = robodojo_obs_to_joint_state(obs)
        return MultiArmManipulationInput(
            image=images,
            depth=depths,
            intrinsic=intrinsics,
            t_world2cam=t_world2cam,
            history_joint_state=[joint_state],
            instruction=_extract_instruction(obs, self.cfg.task_name),
        )

    def _has_memory(self) -> bool:
        """Whether the loaded model carries episode-scoped memory.

        `structure.py` sets `memoryvla` to None when the port is switched
        off, so this is also the switch: everything below it is inert on a
        baseline model.
        """
        target = self.model
        if target is None and self.pipeline is not None:
            target = getattr(self.pipeline, "model", None)
        return getattr(target, "memoryvla", None) is not None

    def _run_holobrain(self, data: MultiArmManipulationInput) -> Any:
        with torch.inference_mode():
            if self.pipeline is not None:
                if self._has_memory():
                    raise RuntimeError(
                        "This model has episode-scoped memory, and the "
                        "pipeline path cannot supply it with a frame index: "
                        "the correction below happens between pre_process "
                        "and the model, and HoloBrainInferencePipeline does "
                        "both in one call. Memory would silently see frame 0 "
                        "for the entire episode. Load the model directly "
                        "(pipeline=None), which is what robodojo_eval.py "
                        "does, or teach the pipeline to take a step index."
                    )
                return self.pipeline(data)
            if self.processor is None or self.model is None:
                raise RuntimeError("Policy is missing processor or model.")
            model_input = self.processor.pre_process(data)
            if "step_index" in model_input:
                # processor.py:158 derives step_index from
                # len(history_joint_state) - 1, and data_preprocess above
                # always builds that list with exactly one entry -- so on
                # this path the processor's value is 0 on every frame of
                # every episode. Training feeds the real frame index from
                # the dataset, and TimestepEmbedder encodes it, so leaving
                # it at 0 makes the whole episode share one positional
                # encoding: the memory keeps its contents but loses all
                # sense of when anything happened.
                #
                # step_index only survives ItemSelection when the memory is
                # switched on (config_robodojo_dataset.py:288-293), so this
                # key is itself the switch and a baseline package is
                # untouched.
                model_input["step_index"] = [max(0, self._env_step - 1)]
            model_outputs = self.model(model_input)
            return self.processor.post_process(model_outputs, model_input)

    def predict_actions(self, obs: dict[str, Any]) -> np.ndarray:
        output = self._run_holobrain(self.data_preprocess(obs))
        action_value = getattr(output, "action", output)
        actions = _to_numpy(action_value)
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions[:, 0]
        if actions.ndim != 2 or actions.shape[1] != ROBODOJO_ACTION_DIM:
            raise ValueError(
                "HoloBrain RoboDojo action must have shape (T, 14), got "
                f"{actions.shape}."
            )
        if actions.shape[0] < self.cfg.valid_action_step:
            raise ValueError(
                "Predicted action length is shorter than valid_action_step: "
                f"{actions.shape[0]} < {self.cfg.valid_action_step}."
            )
        actions = np.asarray(
            actions[: self.cfg.valid_action_step],
            dtype=np.float32,
        ).copy()
        if not np.all(np.isfinite(actions)):
            raise ValueError(
                "HoloBrain RoboDojo action contains non-finite values."
            )
        actions[:, ROBODOJO_GRIPPER_INDICES] = np.clip(
            actions[:, ROBODOJO_GRIPPER_INDICES],
            0.0,
            1.0,
        )
        return actions

    def update_obs(self, obs: dict[str, Any]) -> None:
        self._obs = obs
        self._env_step += 1

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        self._batch_obs = {
            int(obs.get("env_idx", index)): obs
            for index, obs in enumerate(obs_list)
        }
        self._env_step += 1

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._obs is None:
            raise RuntimeError("update_obs must be called before get_action.")
        return action_chunk_to_dicts(self.predict_actions(self._obs))

    def get_action_batch(
        self,
        env_idx_list: list[int] | None = None,
    ) -> list[list[dict[str, np.ndarray]]]:
        if env_idx_list is None:
            env_idx_list = sorted(self._batch_obs)
        missing = [idx for idx in env_idx_list if idx not in self._batch_obs]
        if missing:
            raise KeyError(
                f"Missing RoboDojo observations for env indices: {missing}."
            )
        if len(env_idx_list) > 1 and self._has_memory():
            raise RuntimeError(
                "Batched evaluation across {} envs is not supported by a "
                "model with episode-scoped memory. The loop below runs one "
                "forward per env, but they share this policy's single "
                "episode identity, so all {} envs would read and write one "
                "memory bank -- each acting on the others' history, with no "
                "error and only the score to show for it. `deploy.yml` sets "
                "eval_batch: false and main.py:313 forces num_envs to 1 on "
                "the strength of it; that setting is now a correctness "
                "requirement, not a throughput choice.".format(
                    len(env_idx_list), len(env_idx_list)
                )
            )
        return [
            action_chunk_to_dicts(
                self.predict_actions(self._batch_obs[env_idx])
            )
            for env_idx in env_idx_list
        ]

    def memory_stats(self) -> dict[str, Any]:
        """Counters from the episode-scoped memory, or {} without one.

        A pull-based path on purpose: a log line can be swallowed by a
        logging level nobody set, a return value cannot.
        """
        target = self.model
        if target is None and self.pipeline is not None:
            target = getattr(self.pipeline, "model", None)
        memory = getattr(target, "memoryvla", None)
        stats = getattr(memory, "memory_stats", None)
        out = stats() if callable(stats) else {}
        out["env_step"] = self._env_step
        return out

    def reset(self) -> None:
        logger.info("policy reset: %s", self.memory_stats())
        self._obs = None
        self._batch_obs.clear()
        self._env_step = 0
        target = self.pipeline if self.pipeline is not None else self.model
        reset = getattr(target, "reset", None)
        if callable(reset):
            reset()


def action_chunk_to_dicts(
    actions: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ROBODOJO_ACTION_DIM:
        raise ValueError(
            "actions must have shape "
            f"(T, {ROBODOJO_ACTION_DIM}), got {actions.shape}."
        )
    return [
        {
            "left_arm_joint_state": action[:6].copy(),
            "left_ee_joint_state": action[6:7].copy(),
            "right_arm_joint_state": action[7:13].copy(),
            "right_ee_joint_state": action[13:14].copy(),
        }
        for action in actions
    ]
