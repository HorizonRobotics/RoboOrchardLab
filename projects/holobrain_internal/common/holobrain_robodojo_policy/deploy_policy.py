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
        # Bumped by reset(), so an episode key carries the two properties the
        # dataset's uuid has -- constant within an episode, distinct across
        # them -- and, with the env index, distinct across envs as well.
        self._reset_count = 0
        # The env frame index of the observation currently held, used as
        # `step_index` (see _run_holobrain for why the processor's own value
        # cannot be used).
        #
        # `deploy.py` does call update_obs once per env step, but the
        # inner-loop calls never leave the client: model_client.py:60-62
        # stores the obs and returns, and the ws protocol has no obs-only
        # message (protocol/messages.py:8-22) -- INFER carries update_obs and
        # get_action together (ws/model_server.py:230-256). So this method is
        # reached once per *forward*, and counting calls gave 0..24 for an
        # 800-frame episode while training fed the dataset's real 0..800.
        # Measured: every `policy reset` line of a 50-episode cover_blocks run
        # had env_step == eval_forwards exactly (25 == 25 for 40 of them).
        #
        # So advance by however many actions the previous forward dispatched.
        # Exact except in the last partial chunk of an episode ending early.
        #
        # Per env, not a single scalar. The plan argued a scalar was right
        # because every env advances in lockstep within a round -- true, and
        # still the wrong conclusion, because this counter is bumped once per
        # *env* per round, not once per round. Measured: num_envs=2 reported
        # env_step=1600 for an 800-frame episode. A scalar therefore made
        # step_index wrong by a factor of num_envs, which is the same defect
        # class as the 32x one it was introduced to fix.
        # The 17 evaluation cells measured before 2026-08-13 all ran with the
        # old per-forward numbering; keep it reachable so they can be
        # reproduced. Default is the correct one -- the old is a bug.
        # How the predicted chunk reaches the environment.
        #
        #   chunk    -- one forward per valid_action_step frames, all of them
        #               executed open loop. Every cell measured before
        #               2026-08-13 ran this.
        #   perstep  -- one forward per frame, only the first action executed.
        #   ensemble -- perstep, plus ACT temporal ensembling across the
        #               predictions earlier forwards made for this same frame.
        #
        # perstep and ensemble also align the memory with training: at
        # stream_frame_stride=1 training writes a bank entry every frame, and
        # so does a per-step forward.
        self._action_mode, self._te_m, _stride = self._resolve_modes(
            int(cfg.valid_action_step)
        )
        # Per env, like everything else that survives across forwards --
        # sharing one buffer across envs is the defect this file spent
        # 2026-08-13 removing.
        self._init_runtime_state()

        self._step_index_stride = _stride

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

    def _run_holobrain(
        self, data: MultiArmManipulationInput, env_idx: int | None = None
    ) -> Any:
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
                if env_idx is None:
                    raise RuntimeError(
                        "The memory keys its history per env, but this "
                        "observation carried no `env_idx`, so there is "
                        "nothing to key it by. Refusing to guess: the "
                        "previous code defaulted to 0, which silently maps "
                        "every env onto one bank -- shapes stay right, "
                        "nothing raises, and only the success rate reflects "
                        "it. RoboDojo stamps env_idx in "
                        "eval_env.get_obs_batch (eval_env.py:285) on both "
                        "the single-env and batch paths; another harness "
                        "must do the same."
                    )
                model_input["step_index"] = [
                    max(
                        0,
                        self._env_step.get(env_idx, 0)
                        - self._step_index_stride,
                    )
                ]
                # Same gate as step_index above: that key is itself the
                # memory switch, so a baseline package gets neither. Without
                # a uuid the memory keys every env's history into one bank.
                model_input["uuid"] = [self._episode_key(env_idx)]
            model_outputs = self.model(model_input)
            return self.processor.post_process(model_outputs, model_input)

    def _episode_key(self, env_idx: int) -> str:
        """The memory's per-episode, per-env bank key.

        Without this the module falls back to a constant for the whole batch
        (wrapper.py:303-315), which is one bank shared by every env.
        """
        return f"eval-env{env_idx}-ep{self._reset_count}"

    def predict_actions(
        self, obs: dict[str, Any], env_idx: int | None = None
    ) -> np.ndarray:
        output = self._run_holobrain(
            self.data_preprocess(obs), env_idx=env_idx
        )
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
        return self._deliver(actions, env_idx)

    @staticmethod
    def _resolve_modes(valid_action_step: int) -> tuple[str, float, int]:
        """Read the environment: (action_mode, te_m, step_index_stride).

        Separate from __init__ so it can be exercised without loading a 2.4 GB
        checkpoint. A test that re-derives this expression instead would be
        checking its own copy -- the failure mode that let two dead patches
        ship this week.

        ``te_m`` is ACT's m from w_i = exp(-m*i). NOTE the direction, which
        reads backwards and which someone will eventually "fix": i = 0 is the
        OLDEST surviving prediction, so older predictions carry MORE weight
        and a LARGER m incorporates new observations more slowly. That is what
        the paper states and what the reference implementation does -- its
        exp_weights run over the populated entries, which are ordered oldest
        first. Reversing it is a different algorithm, not a bug fix.

        One forward is one frame under perstep/ensemble, so the step index
        advances by 1 there -- which is what the pre-fix code did for the
        wrong reason. HOLOBRAIN_STEP_INDEX_MODE=forward still reproduces the
        old chunk-mode numbering, for the 17 cells measured under it.
        """
        mode = (
            os.environ.get("HOLOBRAIN_ACTION_MODE", "chunk").strip().lower()
        )
        if mode not in ("chunk", "perstep", "ensemble"):
            raise ValueError(
                "HOLOBRAIN_ACTION_MODE must be chunk, perstep or ensemble, "
                f"got {mode!r}"
            )
        te_m = float(os.environ.get("HOLOBRAIN_TE_M", "0.01"))
        stride = (
            1
            if (
                mode != "chunk"
                or os.environ.get("HOLOBRAIN_STEP_INDEX_MODE") == "forward"
            )
            else int(valid_action_step)
        )
        return mode, te_m, stride

    def _deliver(self, chunk: np.ndarray, env_idx: int | None) -> np.ndarray:
        """Turn a full predicted chunk into what the env executes now.

        Returning fewer actions than were predicted is what makes per-step
        forwarding work without touching RoboDojo: its inner loop breaks once
        it has executed the last action it was handed.
        """
        if self._action_mode == "chunk":
            out = chunk
        elif self._action_mode == "perstep":
            out = chunk[:1].copy()
        else:
            out = self._ensemble(chunk, env_idx)
        self._record_motion(out, env_idx)
        return out

    def _ensemble(self, chunk: np.ndarray, env_idx: int | None) -> np.ndarray:
        """ACT temporal ensembling for the frame about to be executed.

        Each past forward at frame ``t0`` predicted the whole window
        ``t0 .. t0+H-1``; the ones whose window still covers the current frame
        each contribute their prediction for it.
        """
        horizon = int(chunk.shape[0])
        # _env_step was advanced by update_obs before this forward, so the
        # frame being acted on is one stride back -- the same expression the
        # step_index injection uses, and for the same reason.
        now = max(0, self._env_step.get(env_idx, 0) - self._step_index_stride)

        buf = self._te_buf.setdefault(env_idx, [])
        buf.append((now, chunk))
        # Nothing older than the horizon can still cover the current frame.
        if len(buf) > horizon:
            del buf[: len(buf) - horizon]

        # Oldest first, which is the order ACT's exponential weights assume.
        picks = [c[now - t0] for t0, c in buf if 0 <= now - t0 < horizon]
        if not picks:  # pragma: no cover - buf always holds this forward
            return chunk[:1].copy()
        weights = np.exp(
            -self._te_m * np.arange(len(picks), dtype=np.float32)
        )
        weights /= weights.sum()
        blended = (np.stack(picks, axis=0) * weights[:, None]).sum(axis=0)
        # A convex combination of clipped actions is still clipped, so the
        # gripper bounds survive without re-clipping.
        return np.asarray(blended, dtype=np.float32)[None, :]

    def _record_motion(self, out: np.ndarray, env_idx: int | None) -> None:
        """How far this env is actually being commanded to move.

        A score of 0.0 cannot distinguish an arm that barely moves from one
        that moves and is wrong; those are different bugs. This can, and it
        costs two array subtractions per forward.
        """
        st = self._act_stats.setdefault(
            env_idx, {"path": 0.0, "jump": 0.0, "gap": 0.0, "forwards": 0}
        )
        st["forwards"] += 1
        # How far the chunk starts from the pose it was computed from. A chunk
        # normally begins near the current state; computing it from another
        # env's observation makes it begin far away.
        js = self._last_js.get(env_idx)
        if js is not None:
            st["gap"] = max(st["gap"], float(np.abs(out[0] - js).sum()))
        if out.shape[0] > 1:
            st["path"] += float(np.abs(np.diff(out, axis=0)).sum())
        prev = self._last_cmd.get(env_idx)
        if prev is not None:
            step = float(np.abs(out[0] - prev).sum())
            st["path"] += step
            st["jump"] = max(st["jump"], step)
        self._last_cmd[env_idx] = out[-1].copy()

    @staticmethod
    def _obs_env_idx(obs: dict[str, Any]) -> int | None:
        """Which env this observation came from, or None if it does not say.

        Never falls back to a positional index. Position within a batch is
        not identity: over ws the batch is split into one INFER per
        observation, so every list this side ever sees has length 1 and the
        position is always 0.
        """
        value = obs.get("env_idx")
        return None if value is None else int(value)

    def update_obs(self, obs: dict[str, Any]) -> None:
        """The per-env entry point -- including when num_envs > 1.

        update_obs_batch below is unreachable over the websocket transport:
        model_client.py:81-83 stores the batch client-side and returns, then
        get_action_batch (:85-101) sends one INFER per observation, and
        model_server._handle_infer binds update_obs + get_action, never the
        batch pair. So this method runs once per env per round, which is why
        the frame counter has to be per env, and why the env identity has to
        be recovered here rather than from a batch argument.
        """
        self._obs = obs
        self._cur_env_idx = self._obs_env_idx(obs)
        self._advance(self._cur_env_idx)
        self._record_obs(obs, self._cur_env_idx)

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        """Kept for transports that really do hand over a batch.

        Not reached over ws -- see update_obs. Each observation carries its
        own env_idx, so a missing one is an error rather than a position.
        """
        self._batch_obs = {}
        for obs in obs_list:
            env_idx = self._obs_env_idx(obs)
            self._batch_obs[env_idx] = obs
            self._advance(env_idx)

    def _init_runtime_state(self) -> None:
        """Every per-run, per-env dict, in one place.

        Test stubs cannot call __init__ (it loads a 2.4 GB checkpoint) so they
        build the object with object.__new__ and fill state in. Listing the
        fields by hand in each stub went stale three times, once per new
        reading added; calling this instead means adding state touches one
        place. reset() uses it too, so "cleared on reset" and "present at
        construction" cannot drift apart either.
        """
        # Per-env memory-bank keys and frame counters.
        self._env_step: dict[int | None, int] = {}
        self._cur_env_idx: int | None = None
        # ACT temporal-ensemble prediction buffers.
        self._te_buf: dict[int | None, list] = {}
        # What the policy commands: motion, chunk-boundary jumps, and how far
        # a chunk starts from the state it was computed from.
        self._act_stats: dict[int | None, dict] = {}
        self._last_cmd: dict[int | None, Any] = {}
        # What each env is FED, as opposed to what the policy does with it.
        # See _record_obs for why the output-side reading was not enough.
        self._obs_stats: dict[int | None, dict] = {}
        self._last_js: dict[int | None, Any] = {}
        self._obs_sig: dict[int | None, tuple] = {}

    @staticmethod
    def _obs_signature(obs: dict[str, Any], js) -> tuple:
        """Cheap identity for an observation.

        Includes image bytes rather than only the joint vector: every robot
        starts an episode at the same home pose, so a joints-only signature
        would report a cross-env duplicate on the first frame of every
        episode and the reading would be useless exactly when it matters.
        """
        state_sig = hash(np.asarray(js, dtype=np.float64).tobytes())
        image_sig = None
        vision = obs.get("vision")
        if isinstance(vision, dict):
            for name in sorted(vision)[:1]:
                cam = vision.get(name)
                if isinstance(cam, dict) and "color" in cam:
                    arr = np.asarray(cam["color"])
                    # A strided sample: enough to separate two scenes, cheap
                    # enough to run every forward.
                    image_sig = hash(arr[::16, ::16].tobytes())
        return state_sig, image_sig

    def _record_obs(self, obs: dict[str, Any], env_idx: int | None) -> None:
        """Is this env's observation stream coherent, and is it its own?

        E5 measured commanded motion per env and found, inside one batch of
        four, two envs thrashing at 7-9x the single-env reference and two
        below its minimum. That is an output-side symptom of an input-side
        fault, and this makes the input directly readable instead.
        """
        try:
            js = robodojo_obs_to_joint_state(obs)
        except Exception:
            return  # not a RoboDojo observation; nothing to measure
        st = self._obs_stats.setdefault(
            env_idx,
            {
                "jump": 0.0,
                "jump_max": 0.0,
                "dup": 0,
                "dup_image_only": 0,
                "dup_state_only": 0,
                "n": 0,
            },
        )
        st["n"] += 1
        prev = self._last_js.get(env_idx)
        if prev is not None:
            step = float(np.abs(js - prev).sum())
            st["jump"] += step
            st["jump_max"] = max(st["jump_max"], step)
        self._last_js[env_idx] = js

        state_sig, image_sig = self._obs_signature(obs, js)
        for other, (o_state, o_image) in self._obs_sig.items():
            if other == env_idx:
                continue
            same_state = o_state == state_sig
            same_image = image_sig is not None and o_image == image_sig
            if same_state and same_image:
                # The whole observation, byte for byte. Direct evidence of
                # misrouting rather than an inference from the score.
                st["dup"] += 1
            elif same_image:
                # Another env's images with this env's proprioception. A
                # policy that conditions mostly on images would then predict a
                # pose far from the joint state it was handed -- which is what
                # act_gap 12.4 looks like against a clean 0.3-1.1.
                st["dup_image_only"] += 1
            elif same_state:
                # Expected at the start of an episode: every robot resets to
                # the same home pose. Counted separately so it cannot be
                # mistaken for the line above.
                st["dup_state_only"] += 1
        self._obs_sig[env_idx] = (state_sig, image_sig)

    def _advance(self, env_idx: int | None) -> None:
        """One forward's worth of frames, for this env alone."""
        self._env_step[env_idx] = (
            self._env_step.get(env_idx, 0) + self._step_index_stride
        )

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._obs is None:
            raise RuntimeError("update_obs must be called before get_action.")
        return action_chunk_to_dicts(
            self.predict_actions(self._obs, self._cur_env_idx)
        )

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
        # This used to refuse len(env_idx_list) > 1 outright with a memory,
        # because every env shared one episode identity and so one bank. Each
        # forward now carries its own key via _episode_key, and
        # _autoreset_for_eval no longer clears the envs that are not in the
        # current single-element batch. Both halves are needed: either one
        # alone still corrupts, and neither raises when it does.
        #
        # Still one forward per env rather than a real batch, so only the
        # simulation parallelises -- about 91% of eval wall-clock.
        return [
            action_chunk_to_dicts(
                self.predict_actions(self._batch_obs[env_idx], env_idx)
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
        # Scalar max first, so the provenance parsers of E1/E2 keep working
        # unchanged; the per-env map is the new observable. Both are needed:
        # the max alone cannot show the counters are separate, and a run
        # where they are not is exactly what this is watching for.
        out["env_step"] = max(self._env_step.values(), default=0)
        out["action_mode"] = self._action_mode
        out["action_path_by_env"] = {
            str(k): round(v["path"], 2)
            for k, v in sorted(
                self._act_stats.items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            )
        }
        out["obs_jump_by_env"] = {
            str(k): round(v["jump"] / max(1, v["n"] - 1), 3)
            for k, v in sorted(
                self._obs_stats.items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            )
        }
        out["obs_dup_by_env"] = {
            str(k): v["dup"]
            for k, v in sorted(
                self._obs_stats.items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            )
        }
        out["obs_dup_image_only_by_env"] = {
            str(k): v["dup_image_only"]
            for k, v in sorted(
                self._obs_stats.items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            )
        }
        out["obs_dup_state_only_by_env"] = {
            str(k): v["dup_state_only"]
            for k, v in sorted(
                self._obs_stats.items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            )
        }
        out["act_gap_by_env"] = {
            str(k): round(v["gap"], 3)
            for k, v in sorted(
                self._act_stats.items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            )
        }
        out["action_jump_by_env"] = {
            str(k): round(v["jump"], 3)
            for k, v in sorted(
                self._act_stats.items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            )
        }
        out["env_step_by_env"] = {
            str(k): v for k, v in sorted(
                self._env_step.items(), key=lambda kv: (kv[0] is None, kv[0])
            )
        }
        return out

    def reset(self) -> None:
        logger.info("policy reset: %s", self.memory_stats())
        self._obs = None
        self._batch_obs.clear()
        self._init_runtime_state()
        self._reset_count += 1
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
