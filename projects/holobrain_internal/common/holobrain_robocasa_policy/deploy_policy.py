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
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from holobrain_utils import download_file
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.robocasa.utils import (
    DEFAULT_CAMERA_CONFIGS,
    DEFAULT_EEF_TO_HAND,
    GRIPPER_WIDTH,
    STATE_SLICES,
    get_camera_fovy,
    get_gripper_openness,
    intrinsic_from_fovy,
    state_base_pose,
    state_camera_world_pose,
    t_base2cam_from_world_pose,
)
from robo_orchard_lab.models.holobrain.processor import (
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin

logger = logging.getLogger(__name__)

ROBOCASA_CAMERAS = (
    "robot0_eye_in_hand",
    "robot0_agentview_left",
    "robot0_agentview_right",
)
ROBOCASA_STATE_DIM = 8
ROBOCASA_ACTION_DIM = 12
ROBOCASA_T_EEF_TO_HAND = DEFAULT_EEF_TO_HAND
DEFAULT_MODEL_CACHE_DIR = "./workspace/model"


@dataclass
class HoloBrainRoboCasaPolicyCfg:
    model_dir: str | None = None
    model_processor: str = "robocasa_processor"
    model_prefix: str = "model"
    load_impl: str = "native"
    vlm_ckpt_dir: str | None = None
    urdf_dir: str | None = None
    valid_action_step: int = 8
    use_depth: bool = False
    camera_names: tuple[str, ...] = ROBOCASA_CAMERAS
    gripper_width: float = GRIPPER_WIDTH
    control_mode: float = 1.0
    use_env_calibration: bool = True
    default_intrinsics: dict[str, Any] | None = None
    default_t_base2cam: dict[str, Any] | None = None


def _is_http_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


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
    return output_dir


def link_model_resources(
    model_dir: str,
    vlm_ckpt_dir: str | None,
    urdf_dir: str | None,
) -> None:
    link_model_resource(os.path.join(model_dir, "ckpt"), vlm_ckpt_dir)
    link_model_resource(os.path.join(model_dir, "urdf"), urdf_dir)


def link_model_resource(link_path: str, source_dir: str | None) -> None:
    if source_dir is not None and not os.path.exists(link_path):
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        os.symlink(source_dir, link_path)


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


def _to_matrix4(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr.copy()
    if arr.shape == (3, 3):
        ret = np.eye(4, dtype=np.float64)
        ret[:3, :3] = arr
        return ret
    raise ValueError(f"Unsupported matrix shape: {arr.shape}")


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norm, 1e-12, None)


def _apply_inverse_embodiment(
    robot_state: torch.Tensor,
    embodiedment_mat: torch.Tensor,
) -> torch.Tensor:
    transform = torch.linalg.inv(embodiedment_mat)
    state_flat = robot_state.reshape(-1, ROBOCASA_STATE_DIM)
    joint_val = state_flat[:, :1]
    pos = state_flat[:, 1:4]
    quat = state_flat[:, 4:]
    r_mats = quaternion_to_matrix(quat)
    t_mats = torch.eye(
        4,
        device=robot_state.device,
        dtype=robot_state.dtype,
    ).repeat(state_flat.shape[0], 1, 1)
    t_mats[:, :3, :3] = r_mats
    t_mats[:, :3, 3] = pos
    t_new = transform.to(robot_state.device, robot_state.dtype) @ t_mats
    pos_new = t_new[:, :3, 3]
    quat_new = matrix_to_quaternion(t_new[:, :3, :3])
    ret = torch.cat([joint_val, pos_new, quat_new], dim=-1)
    return ret.reshape(robot_state.shape)


def _raw_camera_name(name: str) -> str:
    if name.startswith("video."):
        return name[len("video.") :]
    return name


def _video_key(camera_name: str) -> str:
    return f"video.{_raw_camera_name(camera_name)}"


def _first_available(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise ValueError(f"Observation is missing all keys: {keys}.")


def robocasa_obs_to_robot_state(
    obs: dict[str, Any],
    gripper_width: float = GRIPPER_WIDTH,
) -> np.ndarray:
    gripper_state = np.asarray(
        _first_available(
            obs,
            (
                "state.gripper_qpos",
                "state.hand.gripper_qpos",
                "hand.gripper_qpos",
                "robot0_gripper_qpos",
            ),
        ),
        dtype=np.float32,
    ).reshape(1, -1)
    gripper_openness = get_gripper_openness(gripper_state)
    if not np.isclose(gripper_width, GRIPPER_WIDTH):
        gripper_openness *= GRIPPER_WIDTH / gripper_width

    ee_pos = np.asarray(
        _first_available(
            obs,
            (
                "state.end_effector_position_relative",
                "state.body.end_effector_position_relative",
                "body.end_effector_position_relative",
                "robot0_base_to_eef_pos",
            ),
        ),
        dtype=np.float32,
    ).reshape(1, 3)
    ee_quat_xyzw = np.asarray(
        _first_available(
            obs,
            (
                "state.end_effector_rotation_relative",
                "state.body.end_effector_rotation_relative",
                "body.end_effector_rotation_relative",
                "robot0_base_to_eef_quat",
            ),
        ),
        dtype=np.float32,
    ).reshape(1, 4)
    ee_quat_wxyz = np.concatenate(
        [ee_quat_xyzw[:, 3:4], ee_quat_xyzw[:, :3]],
        axis=1,
    )
    ee_quat_wxyz = _normalize_quaternion(ee_quat_wxyz)
    return np.concatenate(
        [gripper_openness.astype(np.float32), ee_pos, ee_quat_wxyz],
        axis=1,
    ).astype(np.float32)


def convert_ee_poses_to_robocasa_actions(
    current_robot_state: np.ndarray,
    target_robot_state: np.ndarray,
    *,
    valid_action_step: int,
    control_mode: float = 1.0,
    eef_body_to_site_rot: np.ndarray | None = None,
) -> np.ndarray:
    current_robot_state = np.asarray(current_robot_state, dtype=np.float64)
    target_robot_state = np.asarray(target_robot_state, dtype=np.float64)
    if current_robot_state.shape[-1] != ROBOCASA_STATE_DIM:
        raise ValueError(
            "current_robot_state must end with dim "
            f"{ROBOCASA_STATE_DIM}, got {current_robot_state.shape}."
        )
    if target_robot_state.ndim == 3 and target_robot_state.shape[1] == 1:
        target_robot_state = target_robot_state[:, 0]
    if target_robot_state.ndim != 2:
        raise ValueError(
            "target_robot_state must have shape [T, 8] or [T, 1, 8], "
            f"got {target_robot_state.shape}."
        )
    if target_robot_state.shape[1] != ROBOCASA_STATE_DIM:
        raise ValueError(
            "target_robot_state must end with dim "
            f"{ROBOCASA_STATE_DIM}, got {target_robot_state.shape}."
        )

    valid_action_step = int(valid_action_step)
    if valid_action_step <= 0:
        raise ValueError(
            f"valid_action_step must be > 0, got {valid_action_step}."
        )
    if target_robot_state.shape[0] < valid_action_step:
        raise ValueError(
            "Predicted action length is shorter than valid_action_step: "
            f"{target_robot_state.shape[0]} < {valid_action_step}."
        )

    target_robot_state = target_robot_state[:valid_action_step]
    target_quat_wxyz = _normalize_quaternion(target_robot_state[:, 4:])
    target_quat_xyzw = target_quat_wxyz[:, [1, 2, 3, 0]]
    target_rot = Rotation.from_quat(target_quat_xyzw).as_matrix()
    if eef_body_to_site_rot is not None:
        eef_body_to_site_rot = np.asarray(
            eef_body_to_site_rot,
            dtype=np.float64,
        )
        if eef_body_to_site_rot.shape != (3, 3):
            raise ValueError(
                "eef_body_to_site_rot must have shape (3, 3), got "
                f"{eef_body_to_site_rot.shape}."
            )
        target_rot = target_rot @ eef_body_to_site_rot
    target_rotvec = Rotation.from_matrix(target_rot).as_rotvec()

    actions = np.zeros(
        (valid_action_step, ROBOCASA_ACTION_DIM),
        dtype=np.float32,
    )
    actions[:, :3] = target_robot_state[:, 1:4].astype(np.float32)
    actions[:, 3:6] = target_rotvec.astype(np.float32)
    actions[:, 6:7] = np.clip(
        target_robot_state[:, :1],
        0.0,
        1.0,
    ).astype(np.float32)
    actions[:, 11] = np.float32(control_mode)
    return actions


def robocasa_action_to_dict(action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(ROBOCASA_ACTION_DIM)
    return {
        "action.end_effector_position": action[0:3],
        "action.end_effector_rotation": action[3:6],
        "action.gripper_close": action[6:7],
        "action.base_motion": action[7:11],
        "action.control_mode": action[11:12],
    }


def extract_eef_body_to_site_rot(env: Any) -> np.ndarray:
    """Return fixed right_hand-body to grip-site rotation."""
    inner_env = _unwrap_robocasa_env(env)
    robot = inner_env.robots[0]
    sim = inner_env.sim
    body_name = robot.robot_model.eef_name["right"]
    site_name = robot.gripper["right"].important_sites["grip_site"]
    body_rot = sim.data.get_body_xmat(body_name).reshape(3, 3)
    site_rot = sim.data.get_site_xmat(site_name).reshape(3, 3)
    return body_rot.T @ site_rot


class HoloBrainRoboCasaPolicy:
    """Deployment policy for RoboCasa gym observations."""

    def __init__(
        self,
        cfg: HoloBrainRoboCasaPolicyCfg,
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
            link_model_resources(
                cfg.model_dir,
                cfg.vlm_ckpt_dir,
                cfg.urdf_dir,
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

        self._default_intrinsics = self._build_default_mats(
            cfg.default_intrinsics
        )
        self._default_t_base2cam = self._build_default_mats(
            cfg.default_t_base2cam
        )

    def _build_default_mats(
        self,
        values: dict[str, Any] | None,
    ) -> dict[str, np.ndarray]:
        if values is None:
            return {
                cam_name: np.eye(4, dtype=np.float64)
                for cam_name in self.cfg.camera_names
            }
        return {
            cam_name: _to_matrix4(values[cam_name])
            for cam_name in self.cfg.camera_names
        }

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

    def data_preprocess(
        self,
        obs: dict[str, Any],
        *,
        env: Any | None = None,
    ) -> MultiArmManipulationInput:
        image: dict[str, list[np.ndarray]] = {}
        depth: dict[str, list[np.ndarray]] = {}
        for camera_name in self.cfg.camera_names:
            key = _video_key(camera_name)
            if key not in obs:
                raise ValueError(f"Observation is missing image key `{key}`.")
            image[camera_name] = [np.asarray(obs[key])]

            depth_key = f"{key}_depth"
            if self.cfg.use_depth:
                if depth_key not in obs:
                    raise ValueError(
                        f"Observation is missing depth key `{depth_key}`."
                    )
                depth[camera_name] = [
                    np.asarray(obs[depth_key], dtype=np.float32)
                ]
            else:
                depth[camera_name] = [
                    np.zeros(
                        np.asarray(obs[key]).shape[:2],
                        dtype=np.float32,
                    )
                ]

        intrinsic, t_base2cam = self.get_camera_calibration(obs=obs, env=env)
        robot_state = robocasa_obs_to_robot_state(
            obs,
            gripper_width=self.cfg.gripper_width,
        )
        return MultiArmManipulationInput(
            image=image,
            depth=depth,
            intrinsic=intrinsic,
            t_base2cam=t_base2cam,
            history_joint_state=[robot_state],
            instruction=obs.get("annotation.human.task_description", ""),
        )

    def get_camera_calibration(
        self,
        *,
        obs: dict[str, Any],
        env: Any | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if not self.cfg.use_env_calibration or env is None:
            return self._default_intrinsics, self._default_t_base2cam
        try:
            return extract_robocasa_calibration(
                env=env,
                obs=obs,
                camera_names=self.cfg.camera_names,
            )
        except Exception:
            logger.exception(
                "Failed to extract RoboCasa camera calibration; "
                "falling back to configured defaults."
            )
            return self._default_intrinsics, self._default_t_base2cam

    def _run_holobrain(self, data: MultiArmManipulationInput) -> Any:
        if self.pipeline is not None:
            return self.pipeline(data)
        if self.processor is None or self.model is None:
            raise RuntimeError("Policy is missing processor or model.")
        model_input = self.processor.pre_process(data)
        model_outs = self.model(model_input)
        output = self.processor.post_process(model_outs, model_input)
        pose = output.pose

        # # vis DEBUG
        # import cv2
        # from holobrain_utils import HolobrainVideoVisualizer
        # vis_img = HolobrainVideoVisualizer.get_vis_imgs(
        #     model_input["imgs"][0].cpu().numpy(),
        #     model_input["projection_mat"][0].cpu().numpy(),
        #     model_input["hist_robot_state"][0, -1].cpu().numpy(), [0],
        #     pose.cpu().numpy()
        # )
        # cv2.imwrite("vis_img.png", vis_img)

        pose[..., 0] = 1 - pose[..., 0] * 2
        if pose.ndim == 3 and pose.shape[1] == 1:
            pose = pose.squeeze(1)
        target_state = _apply_inverse_embodiment(
            pose,
            model_input["embodiedment_mat"][0],
        )
        output.pose = target_state
        return output

    def get_actions(
        self,
        obs: dict[str, Any],
        *,
        env: Any | None = None,
    ) -> np.ndarray:
        data = self.data_preprocess(obs, env=env)
        output = self._run_holobrain(data)
        target_robot_state = _to_numpy(output.pose)
        if target_robot_state.ndim == 3 and target_robot_state.shape[1] == 1:
            target_robot_state = target_robot_state[:, 0]
        if target_robot_state.shape[-1] == ROBOCASA_STATE_DIM:
            target_robot_state[..., 4:] = _normalize_quaternion(
                target_robot_state[..., 4:]
            )
        current_robot_state = data.history_joint_state[-1]
        return convert_ee_poses_to_robocasa_actions(
            current_robot_state=current_robot_state,
            target_robot_state=target_robot_state,
            valid_action_step=self.cfg.valid_action_step,
            control_mode=self.cfg.control_mode,
            eef_body_to_site_rot=extract_eef_body_to_site_rot(env)
            if env is not None
            else None,
        )

    def get_action_dicts(
        self,
        obs: dict[str, Any],
        *,
        env: Any | None = None,
    ) -> list[dict[str, np.ndarray]]:
        return [
            robocasa_action_to_dict(x) for x in self.get_actions(obs, env=env)
        ]


def extract_robocasa_calibration(
    *,
    env: Any,
    obs: dict[str, Any] | None = None,
    camera_names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    inner_env = _unwrap_robocasa_env(env)
    sim = inner_env.sim

    camera_height = _first_scalar_env_attr(env, "camera_heights", 256)
    camera_width = _first_scalar_env_attr(env, "camera_widths", 256)

    base_pos, base_quat_xyzw = _extract_base_pose(obs)
    eef_rel_pos, eef_rel_quat_xyzw = _extract_eef_pose(obs)
    state = np.zeros(16, dtype=np.float64)
    state[STATE_SLICES["base_position"]] = base_pos
    state[STATE_SLICES["base_rotation_xyzw"]] = base_quat_xyzw
    state[STATE_SLICES["eef_position_relative"]] = eef_rel_pos
    state[STATE_SLICES["eef_rotation_relative_xyzw"]] = eef_rel_quat_xyzw
    base_pos, base_rot = state_base_pose(state)

    intrinsic: dict[str, np.ndarray] = {}
    t_base2cam: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        camera_name = _raw_camera_name(camera_name)
        k4 = np.eye(4, dtype=np.float64)
        fovy = _camera_fovy(sim=sim, camera_name=camera_name)
        k4[:3, :3] = intrinsic_from_fovy(fovy, camera_height, camera_width)
        intrinsic[camera_name] = k4
        cam_pos, cam_rot = _camera_world_pose(
            sim=sim,
            camera_name=camera_name,
            state=state,
        )
        t_base2cam[camera_name] = t_base2cam_from_world_pose(
            cam_pos,
            cam_rot,
            base_pos,
            base_rot,
        )
    return intrinsic, t_base2cam


def _unwrap_robocasa_env(env: Any) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    return getattr(unwrapped, "env", unwrapped)


def _first_scalar_env_attr(env: Any, name: str, default: int) -> int:
    candidates = [getattr(env, "unwrapped", None), env]
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, name, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0]
        return int(value)
    return int(default)


def _camera_fovy(*, sim: Any, camera_name: str) -> float:
    camera_cfg = DEFAULT_CAMERA_CONFIGS.get(camera_name)
    if camera_cfg is not None:
        return get_camera_fovy(camera_cfg, camera_name)
    cam_id = sim.model.camera_name2id(camera_name)
    return float(sim.model.cam_fovy[cam_id])


def _camera_world_pose(
    *,
    sim: Any,
    camera_name: str,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_cfg = DEFAULT_CAMERA_CONFIGS.get(camera_name)
    if camera_cfg is not None:
        return state_camera_world_pose(
            state=state,
            camera_name=camera_name,
            cam_cfg=camera_cfg,
        )
    cam_id = sim.model.camera_name2id(camera_name)
    return (
        np.asarray(sim.data.cam_xpos[cam_id], dtype=np.float64),
        np.asarray(sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3),
    )


def _extract_base_pose(
    obs: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if obs is None:
        raise ValueError("RoboCasa observation is required for base pose.")
    base_pos = np.asarray(
        _first_available(
            obs,
            (
                "state.base_position",
                "state.body.base_position",
                "body.base_position",
                "robot0_base_pos",
            ),
        ),
        dtype=np.float64,
    ).reshape(3)
    base_quat_xyzw = np.asarray(
        _first_available(
            obs,
            (
                "state.base_rotation",
                "state.body.base_rotation",
                "body.base_rotation",
                "robot0_base_quat",
            ),
        ),
        dtype=np.float64,
    ).reshape(4)
    return base_pos, base_quat_xyzw


def _extract_eef_pose(
    obs: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if obs is None:
        raise ValueError("RoboCasa observation is required for eef pose.")
    eef_pos = np.asarray(
        _first_available(
            obs,
            (
                "state.end_effector_position_relative",
                "state.body.end_effector_position_relative",
                "body.end_effector_position_relative",
                "robot0_base_to_eef_pos",
            ),
        ),
        dtype=np.float64,
    ).reshape(3)
    eef_quat_xyzw = np.asarray(
        _first_available(
            obs,
            (
                "state.end_effector_rotation_relative",
                "state.body.end_effector_rotation_relative",
                "body.end_effector_rotation_relative",
                "robot0_base_to_eef_quat",
            ),
        ),
        dtype=np.float64,
    ).reshape(4)
    return eef_pos, eef_quat_xyzw
