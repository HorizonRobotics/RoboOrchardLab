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

#!/usr/bin/env python3
"""Pack RoboCasa LeRobot datasets into RoboOrchard-style LMDB datasets."""

from __future__ import annotations
import argparse
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

DEFAULT_CAMERAS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
DEFAULT_CAMERA_CONFIGS: dict[str, dict[str, Any]] = {
    "robot0_agentview_center": {
        "pos": [-0.6, 0.0, 1.15],
        "quat": [
            0.636945903301239,
            0.3325185477733612,
            -0.3199238181114197,
            -0.6175596117973328,
        ],
        "parent_body": "mobilebase0_support",
    },
    "robot0_agentview_left": {
        "pos": [-0.5, 0.35, 1.05],
        "quat": [0.55623853, 0.29935253, -0.37678665, -0.6775092],
        "camera_attribs": {"fovy": "60"},
        "parent_body": "mobilebase0_support",
    },
    "robot0_agentview_right": {
        "pos": [-0.5, -0.35, 1.05],
        "quat": [
            0.6775091886520386,
            0.3767866790294647,
            -0.2993525564670563,
            -0.55623859167099,
        ],
        "camera_attribs": {"fovy": "60"},
        "parent_body": "mobilebase0_support",
    },
    "robot0_frontview": {
        "pos": [-0.5, 0.0, 0.95],
        "quat": [
            0.6088936924934387,
            0.3814677894115448,
            -0.3673907518386841,
            -0.5905545353889465,
        ],
        "camera_attribs": {"fovy": "60"},
        "parent_body": "mobilebase0_support",
    },
    "robot0_eye_in_hand": {
        "pos": [0.05, 0.0, 0.0],
        "quat": [0.0, 0.707107, 0.707107, 0.0],
        "camera_attribs": {"fovy": "75"},
        "parent_body": "robot0_right_hand",
    },
}
DEFAULT_EEF_TO_HAND = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.097],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

STATE_SLICES = {
    "base_position": slice(0, 3),
    "base_rotation_xyzw": slice(3, 7),
    "eef_position_relative": slice(7, 10),
    "eef_rotation_relative_xyzw": slice(10, 14),
    "gripper_qpos": slice(14, 16),
}
ACTION_SLICES = {
    "base_motion": slice(0, 4),
    "control_mode": slice(4, 5),
    "eef_position": slice(5, 8),
    "eef_rotation": slice(8, 11),
    "gripper_close": slice(11, 12),
}
BASE_TRANSLATION_THRESHOLD_M = 0.02
BASE_ROTATION_THRESHOLD_RAD = math.radians(2.0)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    frames: int
    width: int
    height: int


@dataclass(frozen=True)
class EpisodeInfo:
    dataset_dir: Path
    split: str
    data_type: str
    task_name: str
    date: str
    if_mimicgen: bool
    mg_variant: str | None
    episode_index: int
    parquet_path: Path
    video_paths: dict[str, Path]
    ep_meta_path: Path | None
    instruction: str
    trajectory_id: str | None


@dataclass(frozen=True)
class EpisodeTimeseries:
    observation: dict[str, np.ndarray]
    action: dict[str, np.ndarray]
    if_mobile: bool

    @property
    def num_steps(self) -> int:
        return len(self.observation["ee_state"])


class FFmpegVideoReader:
    def __init__(self, path: Path, meta: VideoMeta) -> None:
        self.path = path
        self.meta = meta
        self.frame_size = meta.width * meta.height * 3
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def read(self) -> np.ndarray | None:
        if self.proc.stdout is None:
            return None
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return None
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self.meta.height, self.meta.width, 3)
            .copy()
        )

    def close(self) -> None:
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        stderr = b""
        if self.proc.stderr is not None:
            stderr = self.proc.stderr.read()
            self.proc.stderr.close()
        returncode = self.proc.wait()
        if returncode not in (0, 255):
            msg = stderr.decode("utf-8", errors="replace").strip()
            if "Broken pipe" not in msg:
                raise RuntimeError(
                    f"ffmpeg reader failed for {self.path} with code "
                    f"{returncode}: {msg}"
                )


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def camera_to_video_key(camera: str) -> str:
    if camera.startswith("observation.images."):
        return camera
    return f"observation.images.{camera}"


def camera_short_name(camera: str) -> str:
    return camera.removeprefix("observation.images.")


def parse_fraction(value: str, default: float) -> float:
    if not value:
        return default
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if abs(den_f) > 1e-12 else default
    return float(value)


def probe_video(path: Path) -> VideoMeta:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    streams = json.loads(completed.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe did not find a video stream in {path}")
    stream = streams[0]
    frames_raw = stream.get("nb_frames")
    frames = int(frames_raw) if frames_raw not in (None, "N/A") else 0
    return VideoMeta(
        fps=parse_fraction(str(stream.get("avg_frame_rate") or ""), 20.0),
        frames=frames,
        width=int(stream.get("width") or 256),
        height=int(stream.get("height") or 256),
    )


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if np.linalg.norm(q) < 1e-12:
        return np.eye(3, dtype=np.float64)
    return (
        Rotation.from_quat(q, scalar_first=False)
        .as_matrix()
        .astype(np.float64)
    )


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if np.linalg.norm(q) < 1e-12:
        return np.eye(3, dtype=np.float64)
    return (
        Rotation.from_quat(q, scalar_first=True).as_matrix().astype(np.float64)
    )


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat)
    return quat[..., [3, 0, 1, 2]]


def shard_count(num_steps: int, num_steps_per_shard: int | None) -> int:
    if num_steps_per_shard is None:
        return 1
    return int(math.ceil(num_steps / num_steps_per_shard))


def shard_prefix(uuid: str, shard_idx: int | None) -> str:
    return uuid if shard_idx is None else f"{uuid}/{shard_idx}"


def slice_camera_timeseries(
    camera_data: dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
) -> dict[str, np.ndarray]:
    sliced = {}
    shard_len = end_idx - start_idx
    for camera, value in camera_data.items():
        value = np.asarray(value)
        if value.ndim >= 3 and value.shape[0] >= end_idx:
            sliced[camera] = value[start_idx:end_idx]
        elif value.ndim >= 3 and value.shape[0] == 1:
            sliced[camera] = np.broadcast_to(
                value[0], (shard_len, *value.shape[1:])
            ).copy()
        else:
            sliced[camera] = np.broadcast_to(
                value, (shard_len, *value.shape)
            ).copy()
    return sliced


def make_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def pose_inv(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def camera_axis_correction() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def camera_local_pose(cam_cfg: dict[str, Any]) -> np.ndarray:
    local_pos = np.asarray(cam_cfg["pos"], dtype=np.float64)
    local_rot = quat_wxyz_to_mat(np.asarray(cam_cfg["quat"], dtype=np.float64))
    return make_pose(local_pos, local_rot)


def intrinsic_from_fovy(fovy: float, height: int, width: int) -> np.ndarray:
    f = 0.5 * height / math.tan(float(fovy) * math.pi / 360.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def get_camera_fovy(cam_cfg: dict[str, Any] | None, camera_name: str) -> float:
    if cam_cfg is not None:
        attribs = cam_cfg.get("camera_attribs") or {}
        if "fovy" in attribs:
            return float(attribs["fovy"])
    default_cfg = DEFAULT_CAMERA_CONFIGS.get(camera_name, {})
    return float((default_cfg.get("camera_attribs") or {}).get("fovy", 60.0))


def state_base_pose(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = state[STATE_SLICES["base_position"]].astype(np.float64, copy=True)
    rot = quat_xyzw_to_mat(state[STATE_SLICES["base_rotation_xyzw"]])
    return pos, rot


def state_eef_pose_world(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_pos, base_rot = state_base_pose(state)
    eef_rel = state[STATE_SLICES["eef_position_relative"]]
    eef_rel_rot = quat_xyzw_to_mat(
        state[STATE_SLICES["eef_rotation_relative_xyzw"]]
    )
    return base_pos + base_rot @ eef_rel, base_rot @ eef_rel_rot


def infer_if_mobile(
    base_position: np.ndarray, base_rotation_xyzw: np.ndarray
) -> bool:
    if len(base_position) <= 1:
        return False

    base_xy = base_position[:, :2].astype(np.float64, copy=False)
    step_translation = np.linalg.norm(np.diff(base_xy, axis=0), axis=1).sum()
    max_translation = np.linalg.norm(base_xy - base_xy[0], axis=1).max()

    quats = base_rotation_xyzw.astype(np.float64, copy=True)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    quats[valid] /= norms[valid]
    dots = np.abs(quats @ quats[0])
    angles = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    max_rotation = float(angles.max()) if len(angles) else 0.0

    return bool(
        step_translation >= BASE_TRANSLATION_THRESHOLD_M
        or max_translation >= BASE_TRANSLATION_THRESHOLD_M
        or max_rotation >= BASE_ROTATION_THRESHOLD_RAD
    )


def state_camera_world_pose(
    state: np.ndarray,
    camera_name: str,
    cam_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    local_pose = camera_local_pose(cam_cfg)
    parent_body = cam_cfg.get("parent_body")
    if parent_body == "robot0_right_hand":
        parent_pos, parent_rot = state_eef_pose_world(state)
        parent_pose = make_pose(parent_pos, parent_rot) @ DEFAULT_EEF_TO_HAND
    elif parent_body:
        parent_pos, parent_rot = state_base_pose(state)
        parent_pose = make_pose(parent_pos, parent_rot)
    else:
        parent_pose = np.eye(4, dtype=np.float64)
    del camera_name
    camera_pose = parent_pose @ local_pose
    return camera_pose[:3, 3], camera_pose[:3, :3]


def t_base2cam_from_world_pose(
    camera_pos: np.ndarray,
    camera_rot: np.ndarray,
    base_pos: np.ndarray,
    base_rot: np.ndarray,
) -> np.ndarray:
    t_base_to_world = make_pose(base_pos, base_rot)
    t_cam_to_world = (
        make_pose(camera_pos, camera_rot) @ camera_axis_correction()
    )
    t_base_to_cam = pose_inv(t_cam_to_world) @ t_base_to_world
    return t_base_to_cam


def camera_calibration_from_config(
    cam_cfg: dict[str, Any],
) -> np.ndarray:
    local_pose = camera_local_pose(cam_cfg)
    parent_body = cam_cfg.get("parent_body")
    if parent_body == "robot0_right_hand":
        parent_to_camera = DEFAULT_EEF_TO_HAND @ local_pose
    else:
        parent_to_camera = local_pose
    return pose_inv(parent_to_camera @ camera_axis_correction())


def resolve_input_paths(
    input_path: str | Path | list[str | Path],
) -> list[Path]:
    if isinstance(input_path, (str, Path)):
        raw_paths = [Path(input_path)]
    else:
        raw_paths = [Path(p) for p in input_path]
    out: list[Path] = []
    for path in raw_paths:
        path = path.expanduser().resolve()
        if path.name == "lerobot" and (path / "meta" / "info.json").exists():
            out.append(path)
            continue
        if (path / "lerobot" / "meta" / "info.json").exists():
            out.append((path / "lerobot").resolve())
            continue
        if (path / "meta" / "info.json").exists():
            out.append(path)
            continue
        for info_path in sorted(path.glob("**/lerobot/meta/info.json")):
            out.append(info_path.parents[1].resolve())
    unique = []
    seen = set()
    for path in out:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def classify_lerobot_path(
    dataset_dir: Path,
) -> tuple[str, str, str, str, bool, str | None]:
    parts = dataset_dir.parts
    if "v1.0" in parts:
        idx = parts.index("v1.0")
        split = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        data_type = parts[idx + 2] if idx + 2 < len(parts) else "unknown"
        task_name = (
            parts[idx + 3] if idx + 3 < len(parts) else dataset_dir.parent.name
        )
        date = parts[idx + 4] if idx + 4 < len(parts) else "unknown"
        if_mimicgen = "mg" in parts[idx + 5 :]
    else:
        split, data_type, task_name, date = (
            "unknown",
            "unknown",
            dataset_dir.parent.name,
            "unknown",
        )
        if_mimicgen = "mg" in parts
    mg_variant = None
    if if_mimicgen and "mg" in parts:
        mg_idx = parts.index("mg")
        if mg_idx + 2 < len(parts):
            mg_variant = "/".join(parts[mg_idx + 1 : -1])
    return split, data_type, task_name, date, if_mimicgen, mg_variant


def build_default_instruction(task_name: str) -> str:
    words = []
    current = ""
    for ch in task_name:
        if ch.isupper() and current:
            words.append(current)
            current = ch.lower()
        else:
            current += ch.lower()
    if current:
        words.append(current)
    return " ".join(words).strip().capitalize() + "."


def episode_records(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["episode_index"]): row
        for row in read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        if "episode_index" in row
    }


def find_episode_path(
    dataset_dir: Path,
    primary_path: Path,
    fallback_pattern: str,
) -> Path | None:
    if primary_path.exists():
        return primary_path
    matches = sorted(dataset_dir.glob(fallback_pattern))
    return matches[0] if matches else None


def build_episode_timeseries(
    states: np.ndarray,
    actions: np.ndarray,
) -> EpisodeTimeseries:
    base_position = states[:, STATE_SLICES["base_position"]]
    base_rotation_xyzw = states[:, STATE_SLICES["base_rotation_xyzw"]]
    ee_state = np.concatenate(
        [
            states[:, STATE_SLICES["eef_position_relative"]],
            xyzw_to_wxyz(
                states[:, STATE_SLICES["eef_rotation_relative_xyzw"]]
            ),
        ],
        axis=1,
    )
    osc_action = np.concatenate(
        [
            actions[:, ACTION_SLICES["eef_position"]],
            actions[:, ACTION_SLICES["eef_rotation"]],
        ],
        axis=1,
    )
    return EpisodeTimeseries(
        observation={
            "gripper_state": states[:, STATE_SLICES["gripper_qpos"]],
            "ee_state": ee_state,
            "base_position": base_position,
            "base_rotation": xyzw_to_wxyz(base_rotation_xyzw),
        },
        action={
            "gripper": actions[:, ACTION_SLICES["gripper_close"]],
            "osc_action": osc_action,
            "base_motion": actions[:, ACTION_SLICES["base_motion"]],
            "control_mode": actions[:, ACTION_SLICES["control_mode"]],
        },
        if_mobile=infer_if_mobile(base_position, base_rotation_xyzw),
    )


def find_episode_infos(
    input_path: str | Path | list[str | Path],
    cameras: list[str],
    splits: set[str] | None,
    data_types: set[str] | None,
    include_mg: bool,
    max_episodes: int | None,
) -> list[EpisodeInfo]:
    episodes: list[EpisodeInfo] = []
    for dataset_dir in resolve_input_paths(input_path):
        split, data_type, task_name, date, if_mimicgen, mg_variant = (
            classify_lerobot_path(dataset_dir)
        )
        if splits is not None and split not in splits:
            continue
        if data_types is not None and data_type not in data_types:
            continue
        if if_mimicgen and not include_mg:
            continue

        info = read_json(dataset_dir / "meta" / "info.json")
        chunk_size = int(info.get("chunks_size", 1000))
        data_rel = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        video_rel = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        ep_rows = episode_records(dataset_dir)
        total_episodes = int(info.get("total_episodes", len(ep_rows)))
        for episode_index in range(total_episodes):
            episode_chunk = episode_index // chunk_size
            fmt = {
                "episode_chunk": episode_chunk,
                "episode_index": episode_index,
            }
            parquet_path = dataset_dir / data_rel.format(**fmt)
            parquet_path = find_episode_path(
                dataset_dir,
                parquet_path,
                f"data/*/episode_{episode_index:06d}.parquet",
            )
            if parquet_path is None:
                continue
            video_paths: dict[str, Path] = {}
            missing_video = False
            for camera in cameras:
                video_key = camera_to_video_key(camera)
                vfmt = {**fmt, "video_key": video_key}
                video_path = dataset_dir / video_rel.format(**vfmt)
                video_path = find_episode_path(
                    dataset_dir,
                    video_path,
                    f"videos/*/{video_key}/episode_{episode_index:06d}.mp4",
                )
                if video_path is None:
                    missing_video = True
                    break
                video_paths[camera_short_name(camera)] = video_path
            if missing_video:
                continue
            extras_dir = (
                dataset_dir / "extras" / f"episode_{episode_index:06d}"
            )
            ep_meta_path = extras_dir / "ep_meta.json"
            ep_meta_exists = ep_meta_path.exists()
            ep_meta = read_json(ep_meta_path) if ep_meta_exists else {}
            ep_row = ep_rows.get(episode_index, {})
            instruction = (
                ep_meta.get("lang")
                or (ep_row.get("tasks") or [None])[0]
                or build_default_instruction(task_name)
            )
            episodes.append(
                EpisodeInfo(
                    dataset_dir=dataset_dir,
                    split=split,
                    data_type=data_type,
                    task_name=task_name,
                    date=date,
                    if_mimicgen=if_mimicgen,
                    mg_variant=mg_variant,
                    episode_index=episode_index,
                    parquet_path=parquet_path,
                    video_paths=video_paths,
                    ep_meta_path=ep_meta_path if ep_meta_exists else None,
                    instruction=str(instruction),
                    trajectory_id=ep_row.get("trajectory_id"),
                )
            )
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
    return episodes


class RoboCasaLmdbPacker(BaseLmdbManipulationDataPacker):
    def __init__(
        self,
        input_path: str | Path | list[str | Path],
        output_path: str | Path,
        cameras: list[str] | None = None,
        splits: list[str] | None = None,
        data_types: list[str] | None = None,
        include_mg: bool = True,
        max_episodes: int | None = None,
        jpeg_quality: int = 95,
        num_steps_per_shard: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(input_path, str(output_path), **kwargs)
        if num_steps_per_shard is not None and num_steps_per_shard <= 0:
            raise ValueError("num_steps_per_shard must be positive when set.")
        self.cameras = [
            camera_short_name(c) for c in (cameras or DEFAULT_CAMERAS)
        ]
        self.splits = set(splits) if splits else None
        self.data_types = set(data_types) if data_types else None
        self.include_mg = include_mg
        self.max_episodes = max_episodes
        self.jpeg_quality = int(jpeg_quality)
        self.num_steps_per_shard = num_steps_per_shard
        self.episodes = find_episode_infos(
            input_path,
            cameras=self.cameras,
            splits=self.splits,
            data_types=self.data_types,
            include_mg=self.include_mg,
            max_episodes=self.max_episodes,
        )
        LOGGER.info(
            "number of valid RoboCasa episodes: %d", len(self.episodes)
        )

    def _episode_uuid(self, episode: EpisodeInfo) -> str:
        parts = [
            episode.split,
            episode.data_type,
            episode.task_name,
            episode.date,
        ]
        if episode.if_mimicgen:
            parts.append("mg")
            if episode.trajectory_id:
                parts.append(str(episode.trajectory_id))
        parts.append(f"episode_{episode.episode_index:06d}")
        return "_".join(str(p).replace("/", "-") for p in parts if p)

    def _load_episode_arrays(
        self, parquet_path: Path
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        df = pd.read_parquet(parquet_path)
        states = np.stack(df["observation.state"].to_numpy()).astype(
            np.float64
        )
        actions = np.stack(df["action"].to_numpy()).astype(np.float64)
        return df, states, actions

    def _camera_configs(
        self, episode: EpisodeInfo
    ) -> dict[str, dict[str, Any]]:
        if episode.ep_meta_path is not None:
            ep_meta = read_json(episode.ep_meta_path)
            cam_configs = dict(ep_meta.get("cam_configs") or {})
        else:
            cam_configs = {}
        for camera in self.cameras:
            if camera not in cam_configs and camera in DEFAULT_CAMERA_CONFIGS:
                cam_configs[camera] = DEFAULT_CAMERA_CONFIGS[camera]
        return cam_configs

    def _pack_images(
        self,
        uuid: str,
        episode: EpisodeInfo,
        video_meta: dict[str, VideoMeta],
        num_steps: int,
    ) -> None:
        readers = {
            camera: FFmpegVideoReader(
                episode.video_paths[camera], video_meta[camera]
            )
            for camera in self.cameras
        }
        try:
            for step in range(num_steps):
                for camera in self.cameras:
                    frame = readers[camera].read()
                    if frame is None:
                        raise RuntimeError(
                            f"Video {episode.video_paths[camera]} ended "
                            f"before step {step}."
                        )
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if not ok:
                        raise RuntimeError(
                            f"Failed to JPEG encode {uuid}/{camera}/{step}."
                        )
                    self.image_pack_file.write(
                        f"{uuid}/{camera}/{step}", encoded
                    )
        finally:
            for reader in readers.values():
                reader.close()

    def _pack_calibration(
        self,
        episode: EpisodeInfo,
        states: np.ndarray,
        video_meta: dict[str, VideoMeta],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        cam_configs = self._camera_configs(episode)
        missing = [
            camera for camera in self.cameras if camera not in cam_configs
        ]
        if missing:
            raise ValueError(
                f"Missing camera configs for {episode.dataset_dir}: {missing}"
            )

        intrinsics: dict[str, np.ndarray] = {}
        base2cam: dict[str, np.ndarray] = {}
        calibration: dict[str, np.ndarray] = {}
        for camera in self.cameras:
            meta = video_meta[camera]
            cam_cfg = cam_configs[camera]
            intrinsics[camera] = intrinsic_from_fovy(
                get_camera_fovy(cam_cfg, camera), meta.height, meta.width
            )
            calibration[camera] = camera_calibration_from_config(cam_cfg)
            t_base2cam = []
            for state in states:
                base_pos, base_rot = state_base_pose(state)
                cam_pos, cam_rot = state_camera_world_pose(
                    state, camera, cam_cfg
                )
                t_base2cam.append(
                    t_base2cam_from_world_pose(
                        cam_pos, cam_rot, base_pos, base_rot
                    )
                )
            base2cam[camera] = np.stack(t_base2cam).astype(np.float64)
        return intrinsics, base2cam, calibration

    def _write_timeseries_meta(
        self,
        uuid: str,
        base2cam: dict[str, np.ndarray],
        timeseries: EpisodeTimeseries,
        shard_idx: int | None = None,
    ) -> None:
        num_steps = timeseries.num_steps
        if shard_idx is None:
            start_idx = 0
            end_idx = num_steps
        else:
            start_idx = shard_idx * self.num_steps_per_shard
            end_idx = min(start_idx + self.num_steps_per_shard, num_steps)
        prefix = shard_prefix(uuid, shard_idx)
        base2cam_slice = slice_camera_timeseries(base2cam, start_idx, end_idx)

        self.meta_pack_file.write(f"{prefix}/base2cam", base2cam_slice)
        for group_name, arrays in (
            ("observation", timeseries.observation),
            ("action", timeseries.action),
        ):
            for key, value in arrays.items():
                self.meta_pack_file.write(
                    f"{prefix}/{group_name}/{key}", value[start_idx:end_idx]
                )

    def _write_all_timeseries_meta(
        self,
        uuid: str,
        base2cam: dict[str, np.ndarray],
        timeseries: EpisodeTimeseries,
    ) -> None:
        if self.num_steps_per_shard is None:
            self._write_timeseries_meta(uuid, base2cam, timeseries)
            return

        self.meta_pack_file.write(
            f"{uuid}/num_steps_per_shard", self.num_steps_per_shard
        )
        num_shards = shard_count(
            timeseries.num_steps,
            self.num_steps_per_shard,
        )
        for shard_idx in range(num_shards):
            self._write_timeseries_meta(
                uuid,
                base2cam,
                timeseries,
                shard_idx=shard_idx,
            )

    def _pack_episode(self, ep_id: int, episode: EpisodeInfo) -> None:
        uuid = self._episode_uuid(episode)
        LOGGER.info(
            "start process [%d/%d] %s", ep_id + 1, len(self.episodes), uuid
        )
        _, states, actions = self._load_episode_arrays(episode.parquet_path)
        video_meta = {
            camera: probe_video(path)
            for camera, path in episode.video_paths.items()
        }
        video_frames = [meta.frames for meta in video_meta.values()]
        video_frames = [n for n in video_frames if n > 0]
        num_steps = min([len(states), len(actions)] + video_frames)
        states = states[:num_steps]
        actions = actions[:num_steps]

        self._pack_images(uuid, episode, video_meta, num_steps)
        intrinsics, base2cam, calibration = self._pack_calibration(
            episode, states, video_meta
        )

        timeseries = build_episode_timeseries(states, actions)

        index_data = {
            "uuid": uuid,
            "task_name": episode.task_name,
            "num_steps": int(num_steps),
            "split": episode.split,
            "pretrain_or_target": episode.split,
            "data_type": episode.data_type,
            "atomic_or_composite": episode.data_type,
            "if_mimicgen": episode.if_mimicgen,
            "if_mobile": timeseries.if_mobile,
            "date": episode.date,
            "episode_index": episode.episode_index,
            "trajectory_id": episode.trajectory_id,
            "mg_variant": episode.mg_variant,
            "simulation": True,
        }

        self.meta_pack_file.write(f"{uuid}/meta_data", index_data)
        self.meta_pack_file.write(f"{uuid}/camera_names", self.cameras)
        self.meta_pack_file.write(f"{uuid}/instruction", episode.instruction)
        self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsics)
        self.meta_pack_file.write(f"{uuid}/calibration", calibration)
        self._write_all_timeseries_meta(uuid, base2cam, timeseries)
        self.write_index(ep_id, index_data)
        LOGGER.info(
            "finish process [%d/%d] %s, num_steps:%d",
            ep_id + 1,
            len(self.episodes),
            uuid,
            num_steps,
        )

    def _pack(self) -> None:
        num_valid_ep = 0
        try:
            for ep_id, episode in enumerate(self.episodes):
                self._pack_episode(ep_id, episode)
                num_valid_ep += 1
            self.index_pack_file.write("__len__", num_valid_ep, commit=True)
        finally:
            self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack RoboCasa LeRobot data into RoboOrchard-style LMDB."
    )
    parser.add_argument(
        "--input_path",
        nargs="+",
        required=True,
        help="v1.0 root, lerobot dir, or parent dir(s).",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--data_types", nargs="+", default=None)
    parser.add_argument(
        "--include_mg", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--jpeg_quality", type=int, default=99)
    parser.add_argument("--num_steps_per_shard", type=int, default=200)
    parser.add_argument("--commit_step", type=int, default=500)
    parser.add_argument("--map_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
    )
    args = parse_args()
    lmdb_kwargs = {}
    if args.map_size is not None:
        lmdb_kwargs["map_size"] = args.map_size
    packer = RoboCasaLmdbPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        cameras=args.cameras,
        splits=args.splits,
        data_types=args.data_types,
        include_mg=args.include_mg,
        max_episodes=args.max_episodes,
        jpeg_quality=args.jpeg_quality,
        num_steps_per_shard=args.num_steps_per_shard,
        commit_step=args.commit_step,
        **lmdb_kwargs,
    )
    packer()


if __name__ == "__main__":
    main()
