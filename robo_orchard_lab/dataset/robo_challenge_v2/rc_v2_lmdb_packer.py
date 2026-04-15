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

"""Pack the RoboChallenge v2 table30 manipulation dataset into LMDB.

Source layout (per task / per episode)::

    {input_path}/{task}/task_desc.json
    {input_path}/{task}/data/episode_{NNNNNN}/
        meta/episode_meta.json          # intrinsics/extrinsics, robot_id, ...
        states/states.jsonl             # single-arm (ARX5)
        states/left_states.jsonl        # dual-arm (ALOHA)
        states/right_states.jsonl       # dual-arm (ALOHA)
        videos/cam_{view}_rgb.mp4       # 2-3 cameras, 30 fps H264

Output: four LMDBs under ``{output_path}/`` (``index``, ``meta``, ``image``,
``depth``) following the same key layout as ``horizon_manipulation dataset`` so
the existing ``BaseLmdbManipulationDataset`` infra can consume it without
changes. Depth is unused (no depth modality in this dataset). Camera metadata
is packed with ``intrinsic`` (3x3), ``extrinsic`` (base-to-camera; wrist
cameras are per-frame ``[T, 4, 4]``), and ``calibration`` (environment
``camera_in_base`` or wrist ``camera_in_ee``).
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as iio
import numpy as np
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Single-arm fields that can vary in dimensionality are stored as
# numpy arrays with shape ``(N, D)`` or ``(N,)`` depending on the source.
SINGLE_ARM_STATE_FIELDS = (
    "joint_positions",
    "joint_velocities",
    "effort",
    "ee_positions",
    "gripper_width",
    "gripper_velocity",
)

# Dual-arm per-arm fields. `master_qpos` is the teleop leader signal
# and is typically used as the action target for IL training.
DUAL_ARM_STATE_FIELDS = (
    "joint_positions",
    "joint_velocities",
    "effort",
    "ee_positions",
    "gripper_width",
    "qpos",
    "master_qpos",
    "master_effort",
)

EPISODE_DIR_RE = re.compile(r"^episode_(\d+)$")
VIDEO_FILE_RE = re.compile(r"^cam_(.+)_rgb\.mp4$")


def _detect_embodiment(
    robot_id: Optional[str],
    has_left: bool,
    has_right: bool,
) -> str:
    """Infer robot family-style embodiment from ``robot_id`` or layout."""

    robot_id = str(robot_id or "").strip().lower()
    if robot_id.startswith("rc_arx5"):
        return "arx5"
    if robot_id.startswith("rc_ur5"):
        return "ur5"
    if robot_id.startswith("rc_aloha"):
        return "aloha"
    if robot_id.startswith("rc_w1"):
        return "dos-w1"
    if has_left and has_right:
        return "aloha"
    return "arx5"


def _canonical_embodiment(robot_type: Optional[str]) -> str:
    """Normalize embodiment aliases across metadata and calibration files."""

    normalized = str(robot_type or "").strip().lower()
    if normalized == "w1":
        return "dos-w1"
    return normalized


def _normalize_task_tags(task_desc: Dict[str, Any]) -> List[str]:
    """Normalize task tags from ``task_desc.json`` for layout decisions."""

    raw_tags = task_desc.get("task_tag")
    if raw_tags is None:
        raw_tags = task_desc.get("tags")
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    return [str(tag).strip().lower() for tag in raw_tags]


def _is_dual_arm_task(task_desc: Dict[str, Any]) -> bool:
    """Return whether the task uses dual-arm state layout."""

    tags = set(_normalize_task_tags(task_desc))
    has_dual_arm = "dual-arm" in tags
    has_single_arm = "single-arm" in tags
    if has_dual_arm == has_single_arm:
        raise ValueError(
            "task_desc.json must contain exactly one of 'dual-arm' or "
            "'single-arm' in task_tag"
        )
    return has_dual_arm


def _has_data(raw: Any) -> bool:
    """Return whether a raw metadata field contains any payload."""

    if raw is None:
        return False
    try:
        return len(raw) > 0
    except TypeError:
        return True


def _normalize_camera_name(embodiment: str, camera_name: str) -> str:
    """Normalize per-robot camera-name aliases."""

    camera_name = str(camera_name).strip()
    if embodiment == "ur5" and camera_name == "cam_side":
        return "cam_arm"
    return camera_name


def _normalize_camera_mapping(
    mapping: Dict[str, Any],
    embodiment: str,
    source_name: str,
) -> Dict[str, Any]:
    """Normalize camera-name aliases while preserving insertion order."""

    normalized: Dict[str, Any] = {}
    original_names: Dict[str, str] = {}
    for raw_name, value in mapping.items():
        camera_name = _normalize_camera_name(embodiment, raw_name)
        if camera_name in normalized:
            raise ValueError(
                f"{source_name} camera names collide after normalization: "
                f"{original_names[camera_name]!r} and {raw_name!r} -> "
                f"{camera_name!r}"
            )
        normalized[camera_name] = value
        original_names[camera_name] = raw_name
    return normalized


def _load_jsonl_states(
    path: str,
    fields: Tuple[str, ...],
) -> Dict[str, np.ndarray]:
    """Parse a states JSONL into a dict of numpy arrays + timestamps."""
    collected: Dict[str, list] = {name: [] for name in fields}
    collected["timestamp"] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for name in fields:
                if name in row:
                    collected[name].append(row[name])
            collected["timestamp"].append(row.get("timestamp", 0.0))

    num_rows = len(collected["timestamp"])
    out: Dict[str, np.ndarray] = {}
    for name, values in collected.items():
        if len(values) == 0:
            continue
        if name != "timestamp" and len(values) != num_rows:
            raise ValueError(
                f"state field {name!r} in {path} has {len(values)} rows, "
                f"expected {num_rows}"
            )
        arr = np.asarray(values)
        if name == "timestamp":
            out[name] = arr.astype(np.float64)
        else:
            out[name] = arr.astype(np.float64)
    return out


def _parse_intrinsic(raw, camera_name: str, source_name: str) -> np.ndarray:
    """Parse a required 3x3 intrinsic matrix."""

    if not _has_data(raw):
        raise ValueError(
            f"{source_name} camera {camera_name} is missing intrinsics"
        )
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(
            f"{source_name} camera {camera_name} has invalid intrinsic shape "
            f"{arr.shape}"
        )
    return arr


def _parse_transform(
    raw,
    camera_name: str,
    source_name: str,
    field_name: str,
) -> np.ndarray:
    """Parse a required 4x4 transform matrix."""

    if not _has_data(raw):
        raise ValueError(
            f"{source_name} camera {camera_name} is missing {field_name}"
        )
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(
            f"{source_name} camera {camera_name} has invalid {field_name} "
            f"shape {arr.shape}"
        )
    return arr


def _feature_has_any_calibration(feature: Dict[str, Any]) -> bool:
    """Whether a raw feature entry should be treated as env camera."""

    if _has_data(feature.get("intrinsics")):
        return True
    arms = (feature.get("extrinsics") or {}).get("arms") or {}
    return any(_has_data(value) for value in arms.values())


def _parse_environment_camera_feature(
    feature: Dict[str, Any],
    camera_name: str,
    is_dual_arm: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Parse an environment-camera feature entry from ``episode_meta``."""

    intrinsic = _parse_intrinsic(
        feature.get("intrinsics"),
        camera_name,
        "meta",
    )
    raw_arms = (feature.get("extrinsics") or {}).get("arms") or {}
    env_calibrations: Dict[str, np.ndarray] = {}
    if is_dual_arm:
        env_calibrations["left"] = _parse_transform(
            raw_arms.get("left"),
            camera_name,
            "meta",
            "left camera_in_base",
        )
        if _has_data(raw_arms.get("right")):
            env_calibrations["right"] = _parse_transform(
                raw_arms.get("right"),
                camera_name,
                "meta",
                "right camera_in_base",
            )
    else:
        raw = raw_arms.get("arm")
        if not _has_data(raw):
            raw = raw_arms.get("left")
        env_calibrations["arm"] = _parse_transform(
            raw,
            camera_name,
            "meta",
            "camera_in_base",
        )
    return intrinsic, env_calibrations


def _build_camera_matrix_from_intrinsics(
    intrinsics: Dict[str, float],
    camera_name: str,
) -> np.ndarray:
    """Build a 3x3 pinhole camera matrix from bundle intrinsics."""

    try:
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
    except KeyError as exc:
        raise ValueError(
            f"robot calibration camera {camera_name} is missing intrinsic "
            f"field {exc.args[0]!r}"
        ) from exc
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _infer_wrist_camera_side(camera_name: str) -> str:
    """Infer wrist-camera side from naming conventions."""

    lowered = camera_name.lower()
    if "left" in lowered:
        return "left"
    if "right" in lowered:
        return "right"
    return "arm"


def _load_robot_wrist_calibrations(
    calibration_root: Path,
    embodiment: str,
) -> Dict[str, Dict[str, Any]]:
    """Load wrist-camera calibration bundle for one embodiment."""

    bundle_path = calibration_root / embodiment / "final_calibrations.json"
    if not bundle_path.is_file():
        return {}

    with open(bundle_path, "r") as f:
        data = json.load(f)

    bundle_embodiment = _canonical_embodiment(data.get("robotType"))
    if bundle_embodiment and bundle_embodiment != embodiment:
        raise ValueError(
            f"robot calibration bundle {bundle_path} is for "
            f"{data.get('robotType')!r}, expected embodiment {embodiment!r}"
        )

    calibrations: Dict[str, Dict[str, Any]] = {}
    for raw_name, camera_data in (data.get("calibrations") or {}).items():
        if camera_data.get("referenceFrame") != "ee":
            continue
        camera_name = _normalize_camera_name(
            embodiment,
            camera_data.get("cameraId", raw_name),
        )
        if camera_name in calibrations:
            raise ValueError(
                f"duplicate wrist calibration for camera {camera_name!r} in "
                f"{bundle_path}"
            )
        calibrations[camera_name] = {
            "camera_name": camera_name,
            "side": _infer_wrist_camera_side(camera_name),
            "camera_matrix": _build_camera_matrix_from_intrinsics(
                camera_data.get("intrinsics") or {},
                camera_name,
            ),
            "cam_in_ee": _parse_transform(
                (camera_data.get("pose") or {}).get("matrix4"),
                camera_name,
                "robot calibration",
                "camera_in_ee",
            ),
        }
    return calibrations


def _quat_pose_to_matrix(ee_pose: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, qx, qy, qz, qw]`` to a 4x4 pose matrix."""

    ee_pose = np.asarray(ee_pose, dtype=np.float64)
    if ee_pose.shape != (7,):
        raise ValueError(f"expected 7D ee pose, got shape {ee_pose.shape}")
    return _ee_pose_sequence_to_matrix(ee_pose[None, :])[0]


def _ee_pose_sequence_to_matrix(ee_positions: np.ndarray) -> np.ndarray:
    """Convert an ``[N, 7]`` ee-pose sequence to ``[N, 4, 4]`` matrices."""

    ee_positions = np.asarray(ee_positions, dtype=np.float64)
    if ee_positions.ndim != 2 or ee_positions.shape[1] != 7:
        raise ValueError(
            "expected ee_positions with shape [N, 7], "
            f"got {ee_positions.shape}"
        )
    quat = ee_positions[:, 3:7]
    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(quat_norm <= 0):
        raise ValueError("encountered zero-norm ee quaternion")
    normalized_quat = quat / quat_norm
    pose = np.zeros((ee_positions.shape[0], 4, 4), dtype=np.float64)
    pose[:, :3, :3] = Rotation.from_quat(normalized_quat).as_matrix()
    pose[:, :3, 3] = ee_positions[:, :3]
    pose[:, 3, 3] = 1.0
    return pose


def _compute_right_base_in_left_base(
    environment_calibrations: Dict[str, Dict[str, np.ndarray]],
) -> Optional[np.ndarray]:
    """Infer ``right_base_in_left_base`` from environment calibrations."""

    for _camera_name, calibrations in environment_calibrations.items():
        left_calibration = calibrations.get("left")
        right_calibration = calibrations.get("right")
        if left_calibration is None or right_calibration is None:
            continue
        return left_calibration @ _invert_rigid_transform(right_calibration)
    return None


def _invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Invert one rigid 4x4 transform using the SE(3) closed form."""

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(
            f"expected rigid transform with shape (4, 4), got "
            f"{transform.shape}"
        )
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.zeros((4, 4), dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    inverse[3, 3] = 1.0
    return inverse


def _invert_rigid_transform_sequence(
    transforms: np.ndarray,
) -> np.ndarray:
    """Invert a batch of rigid 4x4 transforms using the SE(3) closed form."""

    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(
            "expected rigid transforms with shape [N, 4, 4], "
            f"got {transforms.shape}"
        )
    rotation = transforms[:, :3, :3]
    translation = transforms[:, :3, 3]
    inverse = np.zeros_like(transforms)
    rotation_transposed = np.swapaxes(rotation, 1, 2)
    inverse[:, :3, :3] = rotation_transposed
    inverse[:, :3, 3] = -np.einsum(
        "nij,nj->ni",
        rotation_transposed,
        translation,
    )
    inverse[:, 3, 3] = 1.0
    return inverse


def _encode_rgb_frame_to_jpeg(
    frame_rgb: np.ndarray,
    jpeg_quality: int,
) -> bytes:
    """Encode one RGB frame to JPEG bytes."""

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not ok:
        raise RuntimeError("jpeg encoding failed")
    return buf.tobytes()


def _encode_selected_video_frames(
    video_path: str,
    jpeg_quality: int,
    max_frames: int,
    keep_mask: Optional[np.ndarray] = None,
) -> Tuple[List[bytes], int]:
    """Decode a video and JPEG-encode only the selected frames.

    Args:
        video_path: MP4 path.
        jpeg_quality: JPEG encoder quality.
        max_frames: Stop decoding after this many source frames.
        keep_mask: Optional boolean mask over source-frame indices. When
            provided, only frames where ``keep_mask[i]`` is true are encoded.

    Returns:
        A tuple ``(jpeg_frames, decoded_frame_count)`` where
        ``decoded_frame_count`` counts source frames observed before
        ``max_frames`` or end-of-stream.
    """

    if max_frames < 0:
        raise ValueError(f"max_frames must be non-negative, got {max_frames}")
    if keep_mask is not None and len(keep_mask) < max_frames:
        raise ValueError(
            f"keep_mask length {len(keep_mask)} is smaller than max_frames "
            f"{max_frames}"
        )

    jpeg_frames: List[bytes] = []
    decoded_frame_count = 0
    reader = iio.get_reader(video_path, "ffmpeg")
    try:
        for frame_idx, frame_rgb in enumerate(reader):
            if frame_idx >= max_frames:
                break
            decoded_frame_count += 1
            if keep_mask is not None and not keep_mask[frame_idx]:
                continue
            jpeg_frames.append(
                _encode_rgb_frame_to_jpeg(
                    np.asarray(frame_rgb),
                    jpeg_quality,
                )
            )
    finally:
        reader.close()
    return jpeg_frames, decoded_frame_count


def _build_joint_positions_for_static_filter(
    state_arrays: Dict[str, np.ndarray],
    is_dual_arm: bool,
) -> Optional[np.ndarray]:
    """Build the joint-state array used by static-frame filtering."""

    if is_dual_arm:
        left_jp = state_arrays.get("left/joint_positions")
        right_jp = state_arrays.get("right/joint_positions")
        left_gw = state_arrays.get("left/gripper_width")
        right_gw = state_arrays.get("right/gripper_width")
        if (
            left_jp is None
            or right_jp is None
            or left_gw is None
            or right_gw is None
        ):
            return None
        return np.concatenate(
            [
                left_jp,
                left_gw.reshape(-1, 1),
                right_jp,
                right_gw.reshape(-1, 1),
            ],
            axis=-1,
        )
    return state_arrays.get("joint_positions")


def _stack_joint_positions_with_gripper(
    joint_positions: Optional[np.ndarray],
    gripper_width: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Stack arm joints with gripper width as the final dimension."""

    if joint_positions is None or gripper_width is None:
        return None
    return np.concatenate(
        [joint_positions, gripper_width.reshape(-1, 1)],
        axis=-1,
    )


def _build_dual_arm_joint_positions(
    state_arrays: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Build the packed dual-arm joint-position tensor."""

    left_joint_positions = state_arrays.get("left/joint_positions")
    right_joint_positions = state_arrays.get("right/joint_positions")
    left_gripper_width = state_arrays.get("left/gripper_width")
    right_gripper_width = state_arrays.get("right/gripper_width")
    if (
        left_joint_positions is None
        or right_joint_positions is None
        or left_gripper_width is None
        or right_gripper_width is None
    ):
        return None
    return np.concatenate(
        [
            left_joint_positions,
            left_gripper_width.reshape(-1, 1),
            right_joint_positions,
            right_gripper_width.reshape(-1, 1),
        ],
        axis=-1,
    )


def _build_dual_arm_cartesian_position(
    state_arrays: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Build the packed dual-arm Cartesian-position tensor."""

    left_ee_positions = state_arrays.get("left/ee_positions")
    right_ee_positions = state_arrays.get("right/ee_positions")
    if left_ee_positions is None or right_ee_positions is None:
        return None
    return np.concatenate([left_ee_positions, right_ee_positions], axis=-1)


def _slice_state_arrays(
    state_arrays: Dict[str, np.ndarray],
    stop: int,
) -> None:
    """Slice all state arrays in-place to ``[:stop]``."""

    for key in list(state_arrays.keys()):
        state_arrays[key] = state_arrays[key][:stop]


def _apply_keep_mask_to_state_arrays(
    state_arrays: Dict[str, np.ndarray],
    keep_mask: np.ndarray,
) -> None:
    """Apply a boolean keep mask to every state array in-place."""

    for key in list(state_arrays.keys()):
        state_arrays[key] = state_arrays[key][keep_mask]


def _load_episode_state_arrays(
    states_dir: str,
    is_dual_arm: bool,
    uuid: str,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Load and normalize per-episode robot-state arrays."""

    state_arrays: Dict[str, np.ndarray] = {}
    if is_dual_arm:
        left_state_arrays = _load_jsonl_states(
            os.path.join(states_dir, "left_states.jsonl"),
            DUAL_ARM_STATE_FIELDS,
        )
        right_state_arrays = _load_jsonl_states(
            os.path.join(states_dir, "right_states.jsonl"),
            DUAL_ARM_STATE_FIELDS,
        )
        num_left = len(left_state_arrays["timestamp"])
        num_right = len(right_state_arrays["timestamp"])
        if num_left != num_right:
            logger.warning(
                f"{uuid}: left/right frame count mismatch "
                f"({num_left} vs {num_right}); truncating to min"
            )
        num_state = min(num_left, num_right)
        for key, value in left_state_arrays.items():
            if key == "timestamp":
                state_arrays["timestamp"] = value[:num_state]
            else:
                state_arrays[f"left/{key}"] = value[:num_state]
        for key, value in right_state_arrays.items():
            if key == "timestamp":
                continue
            state_arrays[f"right/{key}"] = value[:num_state]
        return state_arrays, num_state

    single_state_arrays = _load_jsonl_states(
        os.path.join(states_dir, "states.jsonl")
        if os.path.isfile(os.path.join(states_dir, "states.jsonl"))
        else os.path.join(states_dir, "left_states.jsonl"),
        SINGLE_ARM_STATE_FIELDS,
    )
    num_state = len(single_state_arrays["timestamp"])
    state_arrays.update(single_state_arrays)
    packed_joint_positions = _stack_joint_positions_with_gripper(
        state_arrays.get("joint_positions"),
        state_arrays.get("gripper_width"),
    )
    if packed_joint_positions is not None:
        state_arrays["joint_positions"] = packed_joint_positions
    return state_arrays, num_state


def _build_environment_camera_specs(
    features: Dict[str, Any],
    is_dual_arm: bool,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
    """Parse environment-camera calibration specs from raw features."""

    feature_camera_specs: Dict[str, Dict[str, Any]] = {}
    environment_calibrations: Dict[str, Dict[str, np.ndarray]] = {}
    for camera_name, feature in features.items():
        if not _feature_has_any_calibration(feature):
            continue
        intrinsic, env_calibrations = _parse_environment_camera_feature(
            feature,
            camera_name,
            is_dual_arm,
        )
        feature_camera_specs[camera_name] = {
            "kind": "environment",
            "intrinsic": intrinsic,
            "env_calibrations": env_calibrations,
        }
        environment_calibrations[camera_name] = env_calibrations
    return feature_camera_specs, environment_calibrations


def _collect_video_files(
    videos_dir: str,
    embodiment: str,
) -> Dict[str, str]:
    """Collect normalized camera-name -> video-path mappings."""

    video_files_raw: Dict[str, str] = {}
    for filename in sorted(os.listdir(videos_dir)):
        match = VIDEO_FILE_RE.match(filename)
        if match is None:
            continue
        video_files_raw[f"cam_{match.group(1)}"] = os.path.join(
            videos_dir,
            filename,
        )
    return _normalize_camera_mapping(
        video_files_raw,
        embodiment,
        "video",
    )


def _resolve_camera_specs(
    uuid: str,
    is_dual_arm: bool,
    features: Dict[str, Any],
    video_files: Dict[str, str],
    feature_camera_specs: Dict[str, Dict[str, Any]],
    wrist_calibrations: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Resolve the final per-camera calibration spec for one episode."""

    feature_camera_names = list(features.keys())
    missing_in_videos = [
        camera_name
        for camera_name in feature_camera_names
        if camera_name not in video_files
    ]
    if missing_in_videos:
        logger.warning(
            f"{uuid}: cameras in meta but no video found: {missing_in_videos}"
        )

    camera_names = [
        camera_name
        for camera_name in feature_camera_names
        if camera_name in video_files
    ]
    for camera_name in video_files:
        if camera_name not in features:
            camera_names.append(camera_name)

    if not camera_names:
        raise RuntimeError(f"{uuid}: no cameras with both metadata and video")

    camera_specs: Dict[str, Dict[str, Any]] = {}
    for camera_name in camera_names:
        feature = features.get(camera_name)
        if feature is not None and _feature_has_any_calibration(feature):
            env_calibrations = feature_camera_specs[camera_name][
                "env_calibrations"
            ]
            calibration_key = "left" if is_dual_arm else "arm"
            calibration_matrix = env_calibrations.get(calibration_key)
            if calibration_matrix is None:
                raise ValueError(
                    f"{uuid}: environment camera {camera_name} is missing "
                    f"{calibration_key} camera_in_base"
                )
            camera_specs[camera_name] = {
                "kind": "environment",
                "intrinsic": feature_camera_specs[camera_name]["intrinsic"],
                "calibration": calibration_matrix,
            }
            continue

        wrist_calibration = wrist_calibrations.get(camera_name)
        if wrist_calibration is None:
            if feature is None:
                raise ValueError(
                    f"{uuid}: camera {camera_name} has video but no metadata "
                    "and no wrist calibration"
                )
            raise ValueError(
                f"{uuid}: camera {camera_name} is missing calibration in raw "
                f"metadata and wrist calibration bundle"
            )
        camera_specs[camera_name] = {
            "kind": "wrist",
            "side": wrist_calibration["side"],
            "intrinsic": wrist_calibration["camera_matrix"],
            "calibration": wrist_calibration["cam_in_ee"],
        }
    return camera_names, camera_specs


def _build_static_keep_mask(
    uuid: str,
    state_arrays: Dict[str, np.ndarray],
    is_dual_arm: bool,
    static_threshold: Optional[float],
    head_time_to_filter: Optional[float],
    tail_time_to_filter: Optional[float],
) -> Optional[np.ndarray]:
    """Build the static-frame keep mask for one episode."""

    num_state = len(state_arrays.get("timestamp", ()))
    if static_threshold is None or static_threshold <= 0 or num_state <= 1:
        return None

    joint_positions = _build_joint_positions_for_static_filter(
        state_arrays,
        is_dual_arm,
    )
    timestamps = state_arrays.get("timestamp")
    if joint_positions is None or timestamps is None:
        logger.warning(
            f"{uuid}: cannot apply static filter "
            "(missing joint_positions or timestamp); "
            "keeping all frames"
        )
        return None

    keep_mask = np.ones(num_state, dtype=bool)
    keep_mask[1:] = np.any(
        np.abs(np.diff(joint_positions, axis=0)) > static_threshold,
        axis=1,
    )
    time_mask = np.zeros(num_state, dtype=bool)
    if head_time_to_filter is not None and head_time_to_filter > 0:
        head_time = timestamps - timestamps[0]
        time_mask = np.logical_or(time_mask, head_time < head_time_to_filter)
    if tail_time_to_filter is not None and tail_time_to_filter > 0:
        tail_time = timestamps[-1] - timestamps
        time_mask = np.logical_or(time_mask, tail_time < tail_time_to_filter)
    keep_mask = np.logical_or(keep_mask, np.logical_not(time_mask))
    kept = int(keep_mask.sum())
    if kept <= 1:
        raise RuntimeError(f"{uuid}: static filter kept only {kept} frames")
    return keep_mask


def _write_robot_state_metadata(
    meta_pack_file,
    uuid: str,
    state_arrays: Dict[str, np.ndarray],
    is_dual_arm: bool,
) -> None:
    """Write packed robot-state arrays into the episode meta LMDB."""

    obs_root = f"{uuid}/observation/robot_state"
    for key, value in state_arrays.items():
        meta_pack_file.write(f"{obs_root}/{key}", value)

    if is_dual_arm:
        cartesian_position = _build_dual_arm_cartesian_position(state_arrays)
        if cartesian_position is not None:
            meta_pack_file.write(
                f"{obs_root}/cartesian_position",
                cartesian_position,
            )
        joint_positions = _build_dual_arm_joint_positions(state_arrays)
        if joint_positions is not None:
            meta_pack_file.write(
                f"{obs_root}/joint_positions",
                joint_positions,
            )
        return

    if "ee_positions" in state_arrays:
        meta_pack_file.write(
            f"{obs_root}/cartesian_position",
            state_arrays["ee_positions"],
        )


class RCV2LmdbPacker(BaseLmdbManipulationDataPacker):
    """LMDB packer for the RoboChallenge v2 dataset.

    Args:
        input_path: Root containing one subdirectory per task.
        output_path: Target LMDB root; four sub-LMDBs will be created.
        task_names: Optional list of tasks to include. ``None`` means all
            discoverable tasks.
        jpeg_quality: JPEG encoder quality for per-frame image storage
            (0-100). Default 90.
        max_episodes_per_task: Optional cap (useful for quick smoke tests).
        static_threshold: If ``> 0``, frames whose per-joint absolute
            difference from the previous frame is below this threshold on
            every joint are treated as static and dropped, except for frames
            inside the head / tail protection windows. ``None`` disables the
            filter. Mirrors the logic in ``mcap_packer.py``.
        head_time_to_filter: Seconds at the start of the episode where
            static filtering is *active* (i.e. static frames in this window
            may be dropped). Frames *outside* all protection windows are
            always kept regardless of motion. ``None``/``0`` disables the
            head window.
        tail_time_to_filter: Same as ``head_time_to_filter`` but measured
            from the end of the episode.
        robot_calibration_root: Root directory containing wrist-camera
            ``final_calibrations.json`` bundles. Defaults to the shared
            manual-calibration override path.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        task_names: Optional[List[str]] = None,
        jpeg_quality: int = 90,
        max_episodes_per_task: Optional[int] = None,
        static_threshold: Optional[float] = 1e-3,
        head_time_to_filter: Optional[float] = 1e8,
        tail_time_to_filter: Optional[float] = None,
        robot_calibration_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_path, output_path, **kwargs)
        self.task_names = task_names
        self.jpeg_quality = int(jpeg_quality)
        self.max_episodes_per_task = max_episodes_per_task
        self.static_threshold = static_threshold
        self.head_time_to_filter = head_time_to_filter
        self.tail_time_to_filter = tail_time_to_filter
        self.robot_calibration_root = Path(
            robot_calibration_root
            or os.path.join(self.input_path, "wrist_calibration")
        )
        self._wrist_calibration_cache: Dict[
            str,
            Dict[str, Dict[str, Any]],
        ] = {}
        self.episodes = self.input_path_handler(self.input_path)

    # ------------------------------------------------------------------
    # Episode discovery
    # ------------------------------------------------------------------
    def input_path_handler(
        self,
        input_path: str,
    ) -> List[Tuple[str, str, str]]:
        """Scan ``input_path`` and return a sorted list of episode tuples.

        Each tuple is ``(task_name, episode_id_str, episode_dir)``.
        """
        episodes: List[Tuple[str, str, str]] = []
        for task_name in sorted(os.listdir(input_path)):
            if (
                self.task_names is not None
                and task_name not in self.task_names
            ):
                continue
            task_dir = os.path.join(input_path, task_name)
            data_dir = os.path.join(task_dir, "data")
            task_desc_path = os.path.join(task_dir, "task_desc.json")
            if not os.path.isdir(data_dir):
                logger.warning(f"no data dir for task {task_name}; skipping")
                continue
            if not os.path.isfile(task_desc_path):
                logger.warning(
                    f"no task_desc.json for task {task_name}; skipping"
                )
                continue

            task_episodes: List[Tuple[str, str, str]] = []
            for ep_dir_name in sorted(os.listdir(data_dir)):
                m = EPISODE_DIR_RE.match(ep_dir_name)
                if m is None:
                    continue
                ep_id = m.group(1)
                ep_dir = os.path.join(data_dir, ep_dir_name)
                if not self._episode_is_valid(ep_dir):
                    continue
                task_episodes.append((task_name, ep_id, ep_dir))

            if self.max_episodes_per_task is not None:
                task_episodes = task_episodes[: self.max_episodes_per_task]
            episodes.extend(task_episodes)

        episodes.sort(key=lambda x: (x[0], int(x[1])))
        logger.info(f"discovered {len(episodes)} valid episodes")
        return episodes

    def _episode_is_valid(self, ep_dir: str) -> bool:
        meta_path = os.path.join(ep_dir, "meta", "episode_meta.json")
        if not os.path.isfile(meta_path):
            return False
        states_dir = os.path.join(ep_dir, "states")
        if not os.path.isdir(states_dir):
            return False
        has_single = os.path.isfile(os.path.join(states_dir, "states.jsonl"))
        has_dual = os.path.isfile(
            os.path.join(states_dir, "left_states.jsonl")
        ) and os.path.isfile(os.path.join(states_dir, "right_states.jsonl"))
        if not (has_single or has_dual):
            return False
        videos_dir = os.path.join(ep_dir, "videos")
        if not os.path.isdir(videos_dir):
            return False
        if not any(VIDEO_FILE_RE.match(fn) for fn in os.listdir(videos_dir)):
            return False
        return True

    def _get_wrist_calibrations(
        self, embodiment: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load and cache wrist calibrations for one embodiment."""

        if embodiment not in self._wrist_calibration_cache:
            self._wrist_calibration_cache[embodiment] = (
                _load_robot_wrist_calibrations(
                    self.robot_calibration_root,
                    embodiment,
                )
            )
        return self._wrist_calibration_cache[embodiment]

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------
    def _pack(self) -> None:
        num_valid = 0
        total = len(self.episodes)
        for ep_idx, (task_name, ep_id, ep_dir) in enumerate(self.episodes):
            uuid = f"{task_name}_episode{int(ep_id):06d}"
            logger.info(f"[{ep_idx + 1}/{total}] packing {uuid}")
            try:
                num_steps = self._pack_one_episode(
                    ep_idx, uuid, task_name, ep_id, ep_dir
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"failed to pack {uuid}: {exc}; skipping")
                continue
            logger.info(
                f"[{ep_idx + 1}/{total}] finished {uuid} "
                f"(num_steps={num_steps})"
            )
            num_valid += 1

        self.index_pack_file.write("__len__", num_valid, commit=True)
        self.close()
        logger.info(f"done. packed {num_valid}/{total} episodes")

    def _pack_one_episode(
        self,
        ep_idx: int,
        uuid: str,
        task_name: str,
        ep_id: str,
        ep_dir: str,
    ) -> int:
        task_desc_path = os.path.join(
            self.input_path, task_name, "task_desc.json"
        )
        with open(task_desc_path, "r") as f:
            task_desc = json.load(f)
        is_dual_arm = _is_dual_arm_task(task_desc)

        # ---- episode metadata ----
        meta_path = os.path.join(ep_dir, "meta", "episode_meta.json")
        with open(meta_path, "r") as f:
            ep_meta = json.load(f)
        robot_id = ep_meta.get("robot_id")
        meta_frames = int(ep_meta.get("frames", 0))

        states_dir = os.path.join(ep_dir, "states")
        has_left = os.path.isfile(
            os.path.join(states_dir, "left_states.jsonl")
        )
        has_right = os.path.isfile(
            os.path.join(states_dir, "right_states.jsonl")
        )
        embodiment = _detect_embodiment(robot_id, has_left, has_right)

        # ---- robot state ----
        state_arrays, num_state = _load_episode_state_arrays(
            states_dir,
            is_dual_arm,
            uuid,
        )

        # ---- calibration ----
        features = _normalize_camera_mapping(
            ep_meta.get("features") or {},
            embodiment,
            "meta",
        )
        wrist_calibrations = self._get_wrist_calibrations(embodiment)
        feature_camera_specs, environment_calibrations = (
            _build_environment_camera_specs(features, is_dual_arm)
        )

        # ---- video frames ----
        videos_dir = os.path.join(ep_dir, "videos")
        video_files = _collect_video_files(videos_dir, embodiment)
        camera_names, camera_specs = _resolve_camera_specs(
            uuid,
            is_dual_arm,
            features,
            video_files,
            feature_camera_specs,
            wrist_calibrations,
        )
        keep_mask = _build_static_keep_mask(
            uuid,
            state_arrays,
            is_dual_arm,
            self.static_threshold,
            self.head_time_to_filter,
            self.tail_time_to_filter,
        )

        # Decode + re-encode only the frames that survive static filtering.
        per_cam_frames: Dict[str, List[bytes]] = {}
        min_frames = num_state
        for camera_name in camera_names:
            jpeg_bytes, decoded_frame_count = _encode_selected_video_frames(
                video_files[camera_name],
                jpeg_quality=self.jpeg_quality,
                max_frames=num_state,
                keep_mask=keep_mask,
            )
            per_cam_frames[camera_name] = jpeg_bytes
            expected_decoded_frames = (
                min(num_state, meta_frames) if meta_frames > 0 else num_state
            )
            if decoded_frame_count < expected_decoded_frames:
                logger.warning(
                    f"{uuid}/{camera_name}: decoded {decoded_frame_count} "
                    "frames "
                    f"before truncation but expected at least "
                    f"{expected_decoded_frames}"
                )
            min_frames = min(min_frames, decoded_frame_count)

        if min_frames <= 0:
            raise RuntimeError(f"{uuid}: no frames to pack")

        # Truncate state/video to the shared source-frame budget first, then
        # apply static-frame filtering in that common window.
        _slice_state_arrays(state_arrays, min_frames)

        num_steps = min_frames
        if keep_mask is not None:
            final_keep_mask = keep_mask[:min_frames]
            kept = int(final_keep_mask.sum())
            logger.info(
                f"{uuid}: static filter kept {kept}/{min_frames} frames"
            )
            if kept <= 1:
                raise RuntimeError(
                    f"{uuid}: static filter kept only {kept} frames"
                )
            _apply_keep_mask_to_state_arrays(state_arrays, final_keep_mask)
            for camera_name in camera_names:
                per_cam_frames[camera_name] = per_cam_frames[camera_name][
                    :kept
                ]
            num_steps = kept
        else:
            for camera_name in camera_names:
                per_cam_frames[camera_name] = per_cam_frames[camera_name][
                    :min_frames
                ]

        intrinsics: Dict[str, np.ndarray] = {}
        extrinsics: Dict[str, np.ndarray] = {}
        calibration: Dict[str, np.ndarray] = {}

        needs_right_base = is_dual_arm and any(
            spec.get("kind") == "wrist" and spec.get("side") == "right"
            for spec in camera_specs.values()
        )
        right_base_in_left_base = None
        if needs_right_base:
            right_base_in_left_base = _compute_right_base_in_left_base(
                environment_calibrations
            )
            if right_base_in_left_base is None:
                raise ValueError(
                    f"{uuid}: right wrist camera requires at least one "
                    "environment camera with both left/right calibrations"
                )

        for cam_name in camera_names:
            spec = camera_specs[cam_name]
            intrinsics[cam_name] = np.asarray(
                spec["intrinsic"],
                dtype=np.float64,
            )
            calibration_matrix = np.asarray(
                spec["calibration"],
                dtype=np.float64,
            )
            calibration[cam_name] = calibration_matrix

            if spec["kind"] == "environment":
                extrinsics[cam_name] = _invert_rigid_transform(
                    calibration_matrix
                )
                continue

            if is_dual_arm:
                side = spec["side"]
                if side not in ("left", "right"):
                    raise ValueError(
                        f"{uuid}: dual-arm wrist camera {cam_name} has "
                        f"unsupported side {side!r}"
                    )
                ee_positions = state_arrays.get(f"{side}/ee_positions")
            else:
                ee_positions = state_arrays.get("ee_positions")
            if ee_positions is None:
                raise ValueError(
                    f"{uuid}: wrist camera {cam_name} is missing ee_positions"
                )

            cam_in_base = np.matmul(
                _ee_pose_sequence_to_matrix(ee_positions),
                calibration_matrix,
            )
            if is_dual_arm and spec["side"] == "right":
                cam_in_base = np.matmul(right_base_in_left_base, cam_in_base)
            extrinsics[cam_name] = _invert_rigid_transform_sequence(
                cam_in_base
            )

        # ---- write image LMDB ----
        for cam in camera_names:
            frames = per_cam_frames[cam]
            for i, jpeg in enumerate(frames):
                self.image_pack_file.write(f"{uuid}/{cam}/{i}", jpeg)

        # ---- write meta LMDB ----
        self.meta_pack_file.write(f"{uuid}/camera_names", camera_names)
        self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsics)
        self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsics)
        self.meta_pack_file.write(f"{uuid}/calibration", calibration)
        _write_robot_state_metadata(
            self.meta_pack_file,
            uuid,
            state_arrays,
            is_dual_arm,
        )

        # ---- index + meta_data ----------------------------------------------
        index_data = dict(
            uuid=uuid,
            task_name=task_name,
            num_steps=int(num_steps),
            simulation=False,
            embodiment=embodiment,
            robot_id=robot_id,
            episode_id=int(ep_id),
            fps=30,
            start_time=ep_meta.get("start_time"),
            end_time=ep_meta.get("end_time"),
        )
        self.write_index(ep_idx, index_data)

        index_data.update(
            instruction=task_desc.get("prompt", ""),
            description=task_desc.get("description", ""),
        )
        self.meta_pack_file.write(f"{uuid}/meta_data", index_data)
        logger.info(f"{uuid}-embodiment: {embodiment}, robot_id: {robot_id}")
        return num_steps


def main() -> None:
    from robo_orchard_lab.utils import log_basic_config

    log_basic_config(
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--task_names",
        type=str,
        default=None,
        help="Comma-separated list of tasks to include (default: all).",
    )
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--max_episodes_per_task", type=int, default=None)
    args = parser.parse_args()

    task_names = (
        [t.strip() for t in args.task_names.split(",") if t.strip()]
        if args.task_names
        else None
    )

    packer = RCV2LmdbPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        task_names=task_names,
        jpeg_quality=args.jpeg_quality,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    packer()


if __name__ == "__main__":
    main()
