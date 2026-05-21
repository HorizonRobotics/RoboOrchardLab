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
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests
import torch
from filelock import FileLock, Timeout

from robo_orchard_lab.dataset.agibot_geniesim.transforms import (
    GenieSim3CalibrationToExtrinsic,
)
from robo_orchard_lab.models.holobrain.processor import (
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin

logger = logging.getLogger(__name__)

GENIESIM_CAMERAS = ("hand_left", "hand_right", "top_head")
GENIESIM_ACTION_DIM = 22
GENIESIM_STATE_DIM = 24
GRIPPER_ENCODE_OFFSET = 0.0
GRIPPER_ENCODE_RANGE = 120.0
DEFAULT_MODEL_CACHE_DIR = "./workspace/model"

TASK_NAME_TO_HEAD_STATE = {
    "clean_the_desktop": (0.0, 0.0, 0.11464),
    "hold_pot": (0.0, 0.0, 0.11464),
    "open_door": (0.0, 0.0, 0.11464),
    "place_block_into_box": (0.0, 0.0, 0.11464),
    "pour_workpiece": (0.0, 0.0, 0.11464),
    "scoop_popcorn": (0.0, 0.0, 0.0),
    "sorting_packages": (0.0, 0.0, 0.11464),
    "sorting_packages_continuous": (0.0, 0.0, 0.11464),
    "stock_and_straighten_shelf": (0.0, 0.0, 0.11464),
    "take_wrong_item_shelf": (0.0, 0.0, 0.1745),
}

DEFAULT_CAMERA_INTRINSICS = {
    "hand_left": [
        [486.13733, 0.0, 614.31964, 0.0],
        [0.0, 485.94153, 529.99976, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "hand_right": [
        [465.1793, 0.0, 630.648, 0.0],
        [0.0, 465.0162, 527.8828, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "top_head": [
        [306.6911, 0.0, 319.90094, 0.0],
        [0.0, 306.55075, 201.29141, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
}

DEFAULT_CAMERA_CALIBRATION = {
    "hand_left": {
        "position": [-0.089796559162, -0.001158827139, 0.060707139061],
        "orientation": [
            0.25628239463,
            -0.25628239463,
            0.659029084489,
            -0.659029084489,
        ],
    },
    "hand_right": {
        "position": [0.0898, 0.00116, 0.060707139061],
        "orientation": [
            -0.25628239463,
            -0.25628239463,
            0.659029084489,
            0.659029084489,
        ],
    },
    "top_head": {
        "position": [0.10237, 0.02375, 0.10256],
        "orientation": [
            -0.594510412187,
            0.594544262709,
            -0.378729147283,
            0.386831646171,
        ],
    },
}

DEFAULT_TASK_NAME_TO_INSTRUCTION = {
    "hold_pot": (
        "Grasp both handles of the pot with left and right hands, Move the "
        "pot to the stove and put it down"
    ),
    "clean_the_desktop": (
        "Pick up the pen on the left side and place it into the pen holder, "
        "close the laptop, pick up the tissue on the table and place it into "
        "the trash bin on the right size. Then, pick up the mouse and place "
        "it on the right side of the laptop. Finally, straighten the colored "
        "pencil box."
    ),
    "open_door": "Turn the doorknob with the right arm, Push the door",
    "place_block_into_box": (
        "The robot is in front of the table, where 5-10 building blocks and "
        "a block box are placed. The block box has multiple holes of "
        "different shapes."
    ),
    "pour_workpiece": "Pour the workpiece into the box with the right arm.",
    "scoop_popcorn": (
        "Scoop the popcorn with the right arm and pour it into the popcorn "
        "bucket, Scoop the popcorn with the right arm and pour it into the "
        "popcorn bucket, Scoop the popcorn with the right arm and pour it "
        "into the popcorn bucket"
    ),
    "sorting_packages": (
        "Grab the package on the table with right arm black, Turn the waist "
        "right to face the barcode scanner, Place the package on the scanning "
        "table with the barcode facing up, The right arm grabs the package, "
        "Rotate the waist with the right arm, Place the package in the blue "
        "bin, Both arms coordinate and the waist returns to the initial "
        "posture"
    ),
    "sorting_packages_continuous": (
        "Grab the package on the table with right arm black, Turn the waist "
        "right to face the barcode scanner, Place the package on the scanning "
        "table with the barcode facing up, The right arm grabs the package, "
        "Rotate the waist with the right arm, Place the package in the blue "
        "bin, Both arms coordinate and the waist returns to the initial "
        "posture"
    ),
    "stock_and_straighten_shelf": (
        "Pick up the wei-chuan orange juice in the shopping basket and place "
        "it on the shelf with right arm, Straighten the overturned "
        "wei-chuan grape juice with right arm"
    ),
    "take_wrong_item_shelf": (
        "The right arm picks up the incorrectly placed item from the shelf, "
        "Place the misplaced items from the shelf into the shopping basket"
    ),
}


@dataclass
class HoloBrainGenieSim3PolicyCfg:
    model_dir: str | None = None
    model_processor: str = "agibot_geniesim3_challenge_processor"
    model_prefix: str = "model"
    load_impl: str = "native"
    valid_action_step: int = 32
    sampling_ratio: float = 1.0
    gripper_limit: float = 1.0
    use_depth: bool = False
    camera_intrinsics: dict[str, Any] | None = None
    camera_calibration: dict[str, Any] | None = None
    task_name_to_instruction: dict[str, str] = field(
        default_factory=lambda: DEFAULT_TASK_NAME_TO_INSTRUCTION.copy()
    )


def _is_http_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def download_file(url: str, file_name: str, timeout: int = 180) -> None:
    if os.path.exists(file_name):
        logger.info("File existed: %s", file_name)
        return

    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
    lock_path = file_name + ".lock"
    lock = FileLock(lock_path, timeout=timeout)

    try:
        with lock:
            if os.path.exists(file_name):
                logger.info("File existed: %s", file_name)
                return

            temp_dir = os.path.dirname(file_name) or "."
            with tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir
            ) as tmp_file:
                tmp_path = tmp_file.name
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())

                    os.rename(tmp_path, file_name)
                    logger.info("Download success: %s", file_name)
                except Exception:
                    logger.exception("Download fail: %s", file_name)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
    except Timeout:
        logger.exception("Download timeout: %s", file_name)
        raise


def download_job_ckpt_processor(
    ckpt_url: str,
    processor_name: str,
    output_dir: str = DEFAULT_MODEL_CACHE_DIR,
    model_prefix: str = "model",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    ckpt_url = ckpt_url.rstrip("/")
    model_url = f"{ckpt_url}/model.safetensors"
    model_config_url = f"{ckpt_url}/{model_prefix}.config.json"
    processor_url = "/".join(
        ckpt_url.split("/")[:-2] + [f"{processor_name}.json"]
    )
    logger.info(
        "model_ckpt: %s\nmodel_config: %s\nprocessor: %s",
        model_url,
        model_config_url,
        processor_url,
    )
    for url in [model_url, model_config_url, processor_url]:
        file_name = os.path.join(output_dir, url.split("/")[-1])
        download_file(url, file_name)


def _ensure_resource_link(model_dir: str, name: str, source: str) -> None:
    target = os.path.join(model_dir, name)
    if os.path.lexists(target):
        return

    source_path = os.path.abspath(source)
    if not os.path.exists(source_path):
        logger.warning(
            "Skip linking %s into %s because %s does not exist.",
            name,
            model_dir,
            source_path,
        )
        return

    os.symlink(source_path, target)
    logger.info("Linked %s -> %s", target, source_path)


def prepare_model_dir(
    model_dir: str,
    processor_name: str,
    model_prefix: str = "model",
    output_dir: str | None = None,
) -> str:
    if not _is_http_url(model_dir):
        return model_dir

    if output_dir is None:
        output_dir = DEFAULT_MODEL_CACHE_DIR
    download_job_ckpt_processor(
        ckpt_url=model_dir,
        processor_name=processor_name,
        output_dir=output_dir,
        model_prefix=model_prefix,
    )
    _ensure_resource_link(output_dir, "ckpt", "./ckpt")
    _ensure_resource_link(output_dir, "urdf", "./urdf")
    return output_dir


def _to_matrix4(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr.copy()
    if arr.shape == (3, 3):
        ret = np.eye(4, dtype=np.float64)
        ret[:3, :3] = arr
        return ret
    raise ValueError(f"Unsupported matrix shape: {arr.shape}")


def _decode_image(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return np.asarray(arr)[:, :, ::-1]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if (
        hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "numpy")
    ):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _decode_depth(camera_name: str, depth: Any) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if camera_name == "top_head":
        arr /= 1000.0
    else:
        arr /= 10000.0
    return arr


def _decode_gripper_state(
    encoded_value: float, gripper_limit: float = 1.0
) -> float:
    scaled = (
        float(encoded_value) - GRIPPER_ENCODE_OFFSET
    ) / GRIPPER_ENCODE_RANGE
    return scaled * gripper_limit


def build_joint_state_from_payload(
    payload_state: Any,
    task_name: str,
    gripper_limit: float = 1.0,
) -> np.ndarray:
    state = np.asarray(payload_state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 21:
        raise ValueError(
            "GenieSim payload state must have at least 21 dims, "
            f"got {state.shape[0]}"
        )
    if task_name not in TASK_NAME_TO_HEAD_STATE:
        raise ValueError(
            f"Unknown GenieSim3 task_name `{task_name}` for head state."
        )

    joint_state = np.zeros(GENIESIM_STATE_DIM, dtype=np.float32)
    joint_state[:7] = state[:7]
    joint_state[7] = _decode_gripper_state(state[14], gripper_limit)
    joint_state[8:15] = state[7:14]
    joint_state[15] = _decode_gripper_state(state[15], gripper_limit)
    joint_state[16:19] = np.asarray(
        TASK_NAME_TO_HEAD_STATE[task_name], dtype=np.float32
    )
    joint_state[19:24] = state[16:21]
    return joint_state


def convert_actions_to_geniesim(
    actions: Any,
    valid_action_step: int,
    sampling_ratio: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != GENIESIM_STATE_DIM:
        raise ValueError(
            f"Expected action array of shape [T, {GENIESIM_STATE_DIM}], "
            f"got {arr.shape}"
        )

    valid_action_step = int(valid_action_step)
    if valid_action_step <= 0:
        raise ValueError(
            f"valid_action_step must be > 0, got {valid_action_step}"
        )

    sampling_ratio = float(sampling_ratio)
    if sampling_ratio <= 0:
        raise ValueError(f"sampling_ratio must be > 0, got {sampling_ratio}")

    raw_len = arr.shape[0]
    if raw_len == 0:
        raise ValueError("actions is empty")

    if np.isclose(sampling_ratio, 1.0):
        sampled = arr
    else:
        rounded_ratio = int(round(sampling_ratio))
        if sampling_ratio > 1.0 and np.isclose(sampling_ratio, rounded_ratio):
            sampled = arr[::rounded_ratio]
        else:
            target_len = int(raw_len / sampling_ratio)
            if target_len <= 0:
                raise ValueError(
                    "No actions available after resampling with "
                    f"sampling_ratio={sampling_ratio}, input_steps={raw_len}"
                )
            if raw_len == 1:
                sampled = np.repeat(arr, target_len, axis=0)
            else:
                x_ori = np.linspace(0, raw_len - 1, num=raw_len)
                x_tgt = np.linspace(0, raw_len - 1, num=target_len)
                sampled = np.stack(
                    [
                        np.interp(x_tgt, x_ori, arr[:, dim])
                        for dim in range(arr.shape[1])
                    ],
                    axis=1,
                ).astype(np.float32)

    if sampled.shape[0] < valid_action_step:
        raise ValueError(
            "Resampled action length is shorter than valid_action_step: "
            f"len(sampled)={sampled.shape[0]}, "
            f"valid_action_step={valid_action_step}"
        )

    valid_action = sampled[:valid_action_step]
    ret = np.zeros((valid_action_step, GENIESIM_ACTION_DIM), dtype=np.float32)
    ret[:, :7] = valid_action[:, :7]
    ret[:, 7:14] = valid_action[:, 8:15]
    ret[:, 14] = valid_action[:, 7]
    ret[:, 15] = valid_action[:, 15]
    ret[:, 20] = valid_action[:, 23]
    return ret


class HoloBrainGenieSim3Policy:
    """Deployment policy for running HoloBrain on GenieSim3 payloads."""

    def __init__(
        self,
        cfg: HoloBrainGenieSim3PolicyCfg,
        *,
        processor: HoloBrainProcessor | None = None,
        model: Any | None = None,
        pipeline: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.processor = processor
        self.model = model

        if self.pipeline is None and self.model is None:
            if cfg.model_dir is None:
                raise ValueError("model_dir must be provided.")
            cfg.model_dir = prepare_model_dir(
                cfg.model_dir,
                processor_name=cfg.model_processor,
                model_prefix=cfg.model_prefix,
            )
            self.processor = self._load_processor(cfg.model_dir)
            self.model = ModelMixin.load_model(
                cfg.model_dir,
                model_prefix=cfg.model_prefix,
                load_impl=cfg.load_impl,
            )
            self.model.eval()
            self.model.requires_grad_(False)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model.to(device)

        if self.pipeline is not None:
            self.processor = getattr(self.pipeline, "processor", None)
            model_obj = getattr(self.pipeline, "model", None)
            if model_obj is not None:
                model_obj.eval()

        camera_intrinsics = (
            DEFAULT_CAMERA_INTRINSICS
            if cfg.camera_intrinsics is None
            else cfg.camera_intrinsics
        )
        camera_calibration = (
            DEFAULT_CAMERA_CALIBRATION
            if cfg.camera_calibration is None
            else cfg.camera_calibration
        )
        self._camera_intrinsics = {
            cam: _to_matrix4(camera_intrinsics[cam])
            for cam in GENIESIM_CAMERAS
        }
        self._setup_calibration(camera_calibration)

    def _load_processor(self, model_dir: str) -> HoloBrainProcessor:
        processor_file = f"{self.cfg.model_processor}.json"
        candidates = [
            os.path.join(model_dir, processor_file),
            os.path.join(os.path.dirname(model_dir), processor_file),
            os.path.join(
                os.path.dirname(os.path.dirname(model_dir)),
                processor_file,
            ),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return HoloBrainProcessor.load(
                    os.path.dirname(candidate),
                    os.path.basename(candidate),
                )
        raise FileNotFoundError(
            f"Cannot find processor `{processor_file}` under {candidates}."
        )

    def _setup_calibration(
        self, camera_calibration: dict[str, Any] | None
    ) -> None:
        if camera_calibration is None or self.processor is None:
            return
        for transform in getattr(self.processor, "transforms", []):
            if isinstance(transform, GenieSim3CalibrationToExtrinsic):
                transform.calibration = transform.calibration_handler(
                    camera_calibration
                )
                break

    def _resolve_instruction(self, payload: dict[str, Any]) -> str:
        prompt = payload.get("prompt", "")
        task_name = payload.get("task_name", "")
        return self.cfg.task_name_to_instruction.get(task_name, prompt)

    def data_preprocess(
        self, payload: dict[str, Any]
    ) -> MultiArmManipulationInput:
        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("Payload `images` must be a dict.")

        task_name = payload.get("task_name", "")
        joint_state = build_joint_state_from_payload(
            payload_state=payload.get("state", []),
            task_name=task_name,
            gripper_limit=self.cfg.gripper_limit,
        )

        image_data: dict[str, list[np.ndarray]] = {}
        depth_data: dict[str, list[np.ndarray]] = {}
        payload_depths = payload.get("depth") or {}

        for cam_name in GENIESIM_CAMERAS:
            if cam_name not in images:
                raise ValueError(f"Payload is missing image for `{cam_name}`.")

            decoded_image = _decode_image(images[cam_name])
            image_data[cam_name] = [decoded_image]

            payload_depth = payload_depths.get(cam_name)
            if self.cfg.use_depth:
                if payload_depth is None:
                    raise ValueError(
                        f"Payload is missing depth for `{cam_name}`."
                    )
                depth_data[cam_name] = [_decode_depth(cam_name, payload_depth)]
            else:
                if payload_depth is not None:
                    black_depth = np.zeros_like(
                        _decode_depth(cam_name, payload_depth),
                        dtype=np.float32,
                    )
                else:
                    black_depth = np.zeros(
                        decoded_image.shape[:2],
                        dtype=np.float32,
                    )
                depth_data[cam_name] = [black_depth]

        return MultiArmManipulationInput(
            image=image_data,
            depth=depth_data,
            intrinsic=self._camera_intrinsics,
            history_joint_state=[joint_state],
            instruction=self._resolve_instruction(payload),
        )

    def _run_holobrain(self, data: MultiArmManipulationInput) -> Any:
        if self.pipeline is not None:
            return self.pipeline(data)
        if self.processor is None or self.model is None:
            raise RuntimeError("Policy is missing processor or model.")
        model_input = self.processor.pre_process(data)
        model_outs = self.model(model_input)
        return self.processor.post_process(model_input, model_outs)

    def get_actions(self, payload: dict[str, Any]) -> np.ndarray:
        output = self._run_holobrain(self.data_preprocess(payload))
        return convert_actions_to_geniesim(
            _to_numpy(output.action),
            self.cfg.valid_action_step,
            sampling_ratio=self.cfg.sampling_ratio,
        )

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        return self.get_actions(obs)


def build_policy_from_deploy_config(
    deploy_config: dict[str, Any],
) -> HoloBrainGenieSim3Policy:
    cfg = HoloBrainGenieSim3PolicyCfg(
        model_dir=deploy_config["model_dir"],
        model_processor=deploy_config.get(
            "model_processor",
            "agibot_geniesim3_challenge_processor",
        ),
        model_prefix=deploy_config.get("model_prefix", "model"),
        load_impl=deploy_config.get("load_impl", "native"),
        valid_action_step=deploy_config.get("valid_action_step", 32),
        sampling_ratio=deploy_config.get("sampling_ratio", 1.0),
        gripper_limit=deploy_config.get("gripper_limit", 1.0),
        use_depth=deploy_config.get("use_depth", False),
        task_name_to_instruction=deploy_config.get(
            "task_name_to_instruction",
            DEFAULT_TASK_NAME_TO_INSTRUCTION.copy(),
        ),
    )
    return HoloBrainGenieSim3Policy(cfg=cfg)
