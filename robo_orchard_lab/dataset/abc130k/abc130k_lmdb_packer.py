# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

"""ABC130k LMDB packer — one-shot single pack + task/chunk orchestrator.

Single self-contained entry point. Two modes selected at CLI parse time:

  A) **Single pack** — ``--input_path`` present. Runs
     ``ABC130kMPLmdbPacker`` in-process on that path. The orchestrator
     mode below re-invokes ``python -m ...abc130k_lmdb_packer`` with
     ``--input_path`` for each chunk, so this is also the per-chunk
     subprocess entry point.

  B) **Orchestrator (task/chunk fan-out)** — ``--data_root`` or
     ``--complete_tasks_json`` present. Walks the tasks, splits each
     into N-episode chunks, and launches one ``--input_path``
     subprocess per chunk. Chunk-level subprocess isolation is
     deliberate: the packer's ``Pool(processes=num_workers)`` is spawned
     on construction, so a fresh chunk = fresh pool = clean process
     state — a crashy episode in chunk N cannot poison chunk N+1's
     workers.

Usage — orchestrator (recommended for cloud packing runs)::

    python -m robo_orchard_lab.dataset.abc130k.abc130k_lmdb_packer \\
        --data_root /horizon-bucket/.../ABC-130k/data/train \\
        --output_root /horizon-bucket/.../ABC_130k/lmdb_train \\
        --episodes_per_lmdb 200 \\
        --num_workers 16 \\
        --num_steps_per_shard 64 \\
        --num_shards 50 --shard_idx 0 \\
        --skip_existing

Usage — single pack (a single ``<output_root>/<chunk>/`` slice)::

    python -m robo_orchard_lab.dataset.abc130k.abc130k_lmdb_packer \\
        --input_path /path/to/task_a/episode_0000 \\
        --output_path /out/root/task_a/chunk_000 \\
        --num_workers 16 \\
        --num_steps_per_shard 64 \\
        --stats_path /out/root/task_a/chunk_000/pack_stats.json
"""

from __future__ import annotations
import argparse
import concurrent.futures as cf
import glob
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

logger = logging.getLogger(__name__)


# RealSense D405 native streaming modes.
D405_NATIVE_WIDTHS = (1920, 1280, 848, 640, 480)
D405_NATIVE_HEIGHTS = {1920: 1080, 1280: 720, 848: 480, 640: 480, 480: 270}

TICK_NS = 33333333  # int(1e9 / 30)
TOP_TOPIC_CANDIDATES = ("/top-left-camera", "/top-right-camera", "/top-camera")
CAMERAS = [("left", "/left-wrist-camera"), ("right", "/right-wrist-camera")]
STATE_TOPICS = [
    ("/left-arm-state", 6),
    ("/left-ee-state", 1),
    ("/right-arm-state", 6),
    ("/right-ee-state", 1),
]
CALIB_TOPICS = {
    "/top-camera-info": "top",
    "/left-wrist-camera-info": "left",
    "/right-wrist-camera-info": "right",
    "/top-left-camera-info": "top",
    "/top-right-camera-info": "top",
}
X264 = [
    "-c:v",
    "libx264",
    "-preset",
    "fast",
    "-crf",
    "18",
    "-bf",
    "0",
    "-pix_fmt",
    "yuv420p",
]

# Static reference camera poses from:
# abc-main/assets/put_bottles/assets/i2rt_yam/cameras.yaml
# Values are T_world_camera (world->camera) at zero-joint posture.
ABC_MAIN_T_WORLD_CAMERA = {
    "top": np.array(
        [
            [0.0, 0.8660264716999262, -0.49999815031155626, 0.08600512],
            [-1.0000000000000002, 0.0, 0.0, 1.734723475976807e-18],
            [0.0, 0.49999815031155626, 0.8660264716999262, 1.70432053],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "left": np.array(
        [
            [0.0, 0.766656164198587, -0.642057883602647, 0.42480000000000007],
            [-1.0000000000000004, 0.0, 0.0, 0.3083],
            [0.0, 0.642057883602647, 0.766656164198587, 1.0190000000000001],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "right": np.array(
        [
            [0.0, 0.766656164198587, -0.642057883602647, 0.42480000000000007],
            [-1.0000000000000004, 0.0, 0.0, -0.3117],
            [0.0, 0.642057883602647, 0.766656164198587, 1.0190000000000001],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}

# MuJoCo camera frame -> CV/ROS optical frame:
# x right, y up, z back  ->  x right, y down, z forward
T_MUJOCO_CAM_TO_CV_CAM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# yam.xml bakes a 180 deg X-rotation into the MJCF `<camera>` element that
# sits on top of the URDF `*_camera_frame` link.
T_CAMFRAME_TO_MJCAM = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

# 14-d joint vector layout used by RobotState in MCAP:
#   [L_joint1..6, L_gripper, R_joint1..6, R_gripper]
ABC130K_JOINT_TO_URDF = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    None,  # left gripper
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    None,  # right gripper
]
ABC130K_CAMERA_TO_URDF_LINK = {
    "top": "top_camera_frame",
    "left": "left_camera_frame",
    "right": "right_camera_frame",
}


def _correct_k_to_image_size(K, image_width, image_height):  # noqa: N803
    """Reconcile a possibly-stale 3x3 K with the actual saved image size."""
    K = np.array(K, dtype=np.float64).copy()  # noqa: N806
    cx, cy = float(K[0, 2]), float(K[1, 2])
    candidates = [(w, abs(cx - w / 2.0)) for w in D405_NATIVE_WIDTHS]
    native_w, dist = min(candidates, key=lambda x: x[1])
    if dist > native_w * 0.1:
        logger.warning(
            "_correct_k_to_image_size: cx=%.1f doesn't match any known D405 "
            "native mode for a %dx%d image; K left unchanged.",
            cx,
            image_width,
            image_height,
        )
        return K, False
    if native_w == image_width:
        return K, False
    if native_w > image_width:
        K[0, 2] = cx - (native_w - image_width) / 2.0
        native_h = D405_NATIVE_HEIGHTS.get(native_w, image_height)
        if native_h > image_height:
            K[1, 2] = cy - (native_h - image_height) / 2.0
    else:
        scale_w = image_width / native_w
        native_h = D405_NATIVE_HEIGHTS.get(native_w, image_height)
        scale_h = image_height / native_h if native_h > 0 else scale_w
        K[0, 0] *= scale_w
        K[0, 2] *= scale_w
        K[1, 1] *= scale_h
        K[1, 2] *= scale_h
    return K, True


def _reference_world2cam_cv(cam_name):
    world_t_cam_mj = ABC_MAIN_T_WORLD_CAMERA.get(cam_name)
    if world_t_cam_mj is None:
        return None
    cam_t_world_mj = np.linalg.inv(world_t_cam_mj)
    return T_MUJOCO_CAM_TO_CV_CAM @ cam_t_world_mj


class ABC130kExtrinsicsFK:
    """Reusable URDF-FK helper for ABC130k wrist-camera extrinsics."""

    def __init__(
        self,
        urdf_path,
        joint_to_urdf=None,
        camera_to_urdf_link=None,
    ):
        self.urdf_path = urdf_path
        self.joint_to_urdf = (
            list(joint_to_urdf)
            if joint_to_urdf is not None
            else list(ABC130K_JOINT_TO_URDF)
        )
        self.camera_to_urdf_link = dict(
            camera_to_urdf_link
            if camera_to_urdf_link is not None
            else ABC130K_CAMERA_TO_URDF_LINK
        )
        self._chain = None
        self._qdim = 0
        self._qpos_idx = None
        self._setup()

    def _setup(self):
        try:
            import pytorch_kinematics as pk
        except ImportError as e:
            raise RuntimeError(
                f"ABC130k FK requested with urdf_path={self.urdf_path!r} "
                "but pytorch_kinematics is not installed."
            ) from e
        chain = pk.build_chain_from_urdf(open(self.urdf_path, "rb").read())
        all_joints = [j.name for j in chain.get_joints()]
        qpos_idx = {}
        for data_idx, urdf_name in enumerate(self.joint_to_urdf):
            if urdf_name is None:
                continue
            if urdf_name not in all_joints:
                raise ValueError(
                    f"joint {urdf_name!r} not present in URDF {self.urdf_path}"
                )
            qpos_idx[data_idx] = all_joints.index(urdf_name)
        self._chain = chain
        self._qdim = len(all_joints)
        self._qpos_idx = qpos_idx
        logger.info(
            "ABC130k FK: %s cameras=%s",
            os.path.basename(self.urdf_path),
            list(self.camera_to_urdf_link.keys()),
        )

    def compute(self, camera_names, joint_positions):
        import torch

        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        num_steps = joint_positions.shape[0]
        qs = np.zeros((num_steps, self._qdim), dtype=np.float32)
        for data_idx, q_idx in self._qpos_idx.items():
            qs[:, q_idx] = joint_positions[:, data_idx]
        poses = self._chain.forward_kinematics(torch.from_numpy(qs))
        extrinsics = {}
        for cam_name in camera_names:
            link_name = self.camera_to_urdf_link.get(cam_name)
            zero_ref = _reference_world2cam_cv(cam_name)
            if link_name is None:
                if zero_ref is None:
                    continue
                extrinsics[cam_name] = np.broadcast_to(
                    zero_ref, (num_steps, 4, 4)
                ).copy()
                continue
            link_mats = (
                poses[link_name].get_matrix().numpy().astype(np.float64)
            )
            t_world_cam_mj = link_mats @ T_CAMFRAME_TO_MJCAM
            t_cam_world_mj = np.linalg.inv(t_world_cam_mj)
            t_world2cam = T_MUJOCO_CAM_TO_CV_CAM @ t_cam_world_mj
            extrinsics[cam_name] = t_world2cam
        return extrinsics


def correct_intrinsics_dict(intrinsic, image_shapes):
    """Apply `_correct_k_to_image_size` to each camera K."""
    corrected = {}
    note = {}
    for cam_name, k in intrinsic.items():
        shape = image_shapes.get(cam_name)
        if shape is None:
            corrected[cam_name] = k
            continue
        img_h, img_w = shape
        fixed_k, was_changed = _correct_k_to_image_size(k, img_w, img_h)
        corrected[cam_name] = fixed_k
        note[cam_name] = bool(was_changed)
    return corrected, note


def probe(path, *entries):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *entries,
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return [int(x) for x in out.split(",") if x]


@dataclass
class ABC130kEpisode:
    kind: str
    episode_dir: Path
    mcap_path: Optional[Path] = None
    bin_path: Optional[Path] = None
    video_path: Optional[Path] = None
    meta_path: Optional[Path] = None


ZEDX_WRIST_RESOLUTION = (1920, 1200)


ABC130K_JOINT_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
]


class JointStatsAccumulator:
    """Accumulate per-joint samples across kept episodes for post-pack summary.

    Per-episode ``joint_positions`` is sub-sampled with ``np.linspace`` so a
    handful of very long episodes don't drown out the rest of the dataset.
    Statistics (mean/std/min/max/percentiles) and histograms are computed once
    at ``print_summary`` time from the concatenated buffer.

    ``label`` is used in the section title so we can reuse this class for both
    ``joint_positions`` (state) and ``joint_actions`` (commanded action) —
    gripper columns will differ dramatically between the two (state is finger
    distance after grasp; action is 0/1 open/close intent).
    """

    def __init__(
        self,
        joint_names,
        subsample_per_episode=8,
        label="joint_state",
    ):
        self.joint_names = list(joint_names)
        self.n_joints = len(self.joint_names)
        self.subsample_per_episode = int(subsample_per_episode)
        self.label = label
        self._buf = []
        self.total_raw_rows = 0
        self.num_episodes = 0

    def update(self, joint_positions):
        js = np.asarray(joint_positions, dtype=np.float64)
        if js.ndim != 2 or js.shape[1] != self.n_joints:
            return
        T = js.shape[0]  # noqa: N806
        self.total_raw_rows += T
        self.num_episodes += 1
        if self.subsample_per_episode and T > self.subsample_per_episode:
            idx = np.linspace(
                0,
                T - 1,
                self.subsample_per_episode,
                dtype=int,
            )
            js = js[idx]
        self._buf.append(js.astype(np.float32, copy=False))

    def finalize(self):
        if not self._buf:
            return None
        data = np.concatenate(self._buf, axis=0).astype(np.float64)
        return dict(
            data=data,
            num_samples=int(data.shape[0]),
            num_raw_rows=int(self.total_raw_rows),
            num_episodes=int(self.num_episodes),
            mean=data.mean(axis=0),
            std=data.std(axis=0),
            min=data.min(axis=0),
            max=data.max(axis=0),
            p25=np.percentile(data, 25, axis=0),
            p50=np.percentile(data, 50, axis=0),
            p75=np.percentile(data, 75, axis=0),
        )

    def print_summary(self, log_fn=None, n_bins=20, bar_width=32):
        s = self.finalize()
        log = log_fn or print
        if s is None:
            log(f"[{self.label}] no samples collected.")
            return None
        header = (
            f"[{self.label}] Per-joint stats — {s['num_samples']} subsampled "
            f"rows from {s['num_episodes']} eps ({s['num_raw_rows']} raw rows)"
        )
        log("=" * 100)
        log(header)
        log("=" * 100)
        log(
            f"{'joint':<14} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} "
            f"{'range':>9} {'p25':>9} {'p50':>9} {'p75':>9}"
        )
        log("-" * 100)
        for i, name in enumerate(self.joint_names):
            rng = float(s["max"][i] - s["min"][i])
            log(
                f"{name:<14} "
                f"{s['mean'][i]:>9.4f} {s['std'][i]:>9.4f} "
                f"{s['min'][i]:>9.4f} {s['max'][i]:>9.4f} "
                f"{rng:>9.4f} "
                f"{s['p25'][i]:>9.4f} {s['p50'][i]:>9.4f} {s['p75'][i]:>9.4f}"
            )
        log("=" * 100)
        # Per-joint histograms
        data = s["data"]
        for i, name in enumerate(self.joint_names):
            col = data[:, i]
            lo, hi = float(s["min"][i]), float(s["max"][i])
            if hi - lo < 1e-9:
                log(
                    f"[{self.label}] {name}: constant at {lo:.4f} "
                    f"({col.size} rows)"
                )
                continue
            hist, edges = np.histogram(col, bins=n_bins, range=(lo, hi))
            max_h = int(hist.max()) or 1
            log("")
            log(
                f"[{self.label}] {name}  range=[{lo:.4f}, {hi:.4f}]  "
                f"bins={n_bins}"
            )
            for h, e0, e1 in zip(hist, edges[:-1], edges[1:], strict=False):
                bar_len = int(round(h * bar_width / max_h))
                bar = "█" * bar_len
                log(f"  [{e0:>8.4f}, {e1:>8.4f})  {h:>7d}  {bar}")
        return s


def _is_zedx_calibration(width, height, cam_topic):
    """Wrist or top stream that matches the ZED-X documented resolution.

    A RealSense station ships everything at 640x480; ZED-X ships 1920x1200.
    Some episodes lack the doubled top stream but still carry ZED-X wrist
    cameras (different mechanical mount), so resolution is a more reliable
    signal than topic count alone.
    """
    if (width, height) == ZEDX_WRIST_RESOLUTION:
        return True
    return False


def _camera_name_from_topic(topic):
    if topic in ("/top-camera", "/top-left-camera", "/top-right-camera"):
        return "top"
    if topic == "/left-wrist-camera":
        return "left"
    if topic == "/right-wrist-camera":
        return "right"
    return topic.strip("/")


def validate_intrinsic(K, image_width, image_height):  # noqa: N803
    """Return (K_fixed, ok, reason).

    Mirrors the behavior of correct_k_to_image_size from the dataset module,
    but converts the "can't snap" case into a hard reject instead of silently
    leaving an unusable K.
    """
    K = np.array(K, dtype=np.float64).copy()  # noqa: N806
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if not (fx > 0.0 and fy > 0.0):
        return K, False, f"fx_or_fy_zero(fx={fx:.1f},fy={fy:.1f})"
    if not (cx > 0.0 and cy > 0.0):
        return K, False, f"cx_or_cy_zero(cx={cx:.1f},cy={cy:.1f})"

    candidates = [(w, abs(cx - w / 2.0)) for w in D405_NATIVE_WIDTHS]
    native_w, dist = min(candidates, key=lambda x: x[1])
    if dist > native_w * 0.1:
        return (
            K,
            False,
            (
                f"cx={cx:.1f} doesn't snap to any D405 native mode for "
                f"{image_width}x{image_height} (best native_w={native_w}, "
                f"dist={dist:.1f})"
            ),
        )
    if native_w == image_width:
        return K, True, None
    if native_w > image_width:
        K[0, 2] = cx - (native_w - image_width) / 2.0
        native_h = D405_NATIVE_HEIGHTS.get(native_w, image_height)
        if native_h > image_height:
            K[1, 2] = cy - (native_h - image_height) / 2.0
    else:
        scale_w = image_width / native_w
        native_h = D405_NATIVE_HEIGHTS.get(native_w, image_height)
        scale_h = image_height / native_h if native_h > 0 else scale_w
        K[0, 0] *= scale_w
        K[0, 2] *= scale_w
        K[1, 1] *= scale_h
        K[1, 2] *= scale_h
    return K, True, None


def _get_reference_world2cam(cam_name):
    world_t_cam_mj = ABC_MAIN_T_WORLD_CAMERA.get(cam_name)
    if world_t_cam_mj is None:
        return None
    cam_t_world_mj = np.linalg.inv(world_t_cam_mj)
    return T_MUJOCO_CAM_TO_CV_CAM @ cam_t_world_mj


def _floor_indices(source_ts, target_ts):
    return np.clip(
        np.searchsorted(source_ts, target_ts, side="right") - 1,
        0,
        len(source_ts) - 1,
    )


def _discover_episodes(
    input_path,
    max_episodes_per_task=None,
    scandir_threads=32,
):
    """Fast scandir-based episode discovery.

    ``input_path`` is one or more comma-separated paths. Each may be:
      - a *dataset root* (contains subdir per task, each with ``episode_*/``);
      - a *task directory* (contains ``episode_*/``);
      - an *episode directory* (contains ``episode.mcap``);
      - a shell glob resolving to any of the above.

    Design goals (network FS friendly):

      * **Only two directory listings per task** — one to scandir the task and
        list its ``episode_*`` children, then early-stop once
        ``max_episodes_per_task`` is met.
      * **Zero per-episode ``stat`` calls.** On JFS every ``is_file()`` is a
        network round-trip, so verifying ``episode.mcap`` presence up front
        would be O(#episodes) round-trips. We instead trust the
        ``episode_*/`` naming convention and let ``parse_episode`` in the
        worker return a clean error for any missing mcap.
      * **Parallel task scandir.** Each task-dir listing is a single blocking
        network round-trip (~hundreds of ms on JFS). ``os.scandir`` releases
        the GIL, so a modest thread pool turns 200 sequential round-trips
        into a handful of round-trip batches.
      * **No episode-level globs.** A glob like ``.../train/*/episode_*``
        forces ``glob.glob`` to walk every one of the ~130k episode dirs
        before this function starts. Pass a task-level path
        (``.../train`` or ``.../train/*``) instead.
    """

    def _scandir_children(path):
        """Return ``[(name, DirEntry), ...]`` sorted by name; empty on err."""
        try:
            with os.scandir(path) as it:
                # Materialize now: after ``it`` closes, DirEntries can't be
                # queried on some filesystems.
                entries = [(e.name, e) for e in it]
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.warning("scandir failed on %s: %s", path, exc)
            return []
        entries.sort(key=lambda x: x[0])
        return entries

    def _collect_from_task(task_path):
        """Return up to ``max_episodes_per_task`` ``episode_*/`` dirs."""
        eps = []
        for name, entry in _scandir_children(task_path):
            if not name.startswith("episode_"):
                continue
            # ``entry.is_dir(follow_symlinks=False)`` is served from cached
            # dirent metadata on Linux ext4/xfs/nfs — no extra stat.
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            eps.append(Path(entry.path))
            if (
                max_episodes_per_task is not None
                and len(eps) >= max_episodes_per_task
            ):
                break
        return eps

    # Step 1: expand comma-separated patterns into concrete filesystem roots.
    patterns = [p.strip() for p in input_path.split(",") if p.strip()]
    resolved = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            resolved.extend(sorted(glob.glob(pat)))
        else:
            resolved.append(pat)

    # Step 2: classify each resolved root by its immediate content and
    # collect episodes.
    tasks_seen = set()
    episode_dirs = []
    task_dirs = []  # (task_name, task_path) — filled after classification.

    for root_str in resolved:
        root = Path(root_str)
        # Direct episode.mcap path.
        if root.name == "episode.mcap":
            episode_dirs.append(root.parent)
            continue
        # Direct episode dir (episode_<uuid>).
        if root.name.startswith("episode_"):
            episode_dirs.append(root)
            continue
        # A directory: peek one level down.
        children = _scandir_children(root)
        if not children:
            continue
        has_episode_children = any(
            n.startswith("episode_") for n, _ in children
        )
        if has_episode_children:
            task_dirs.append((root.name, root))
        else:
            # Treat as dataset root: each subdir is a task.
            for name, entry in children:
                if name.startswith("."):
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                task_dirs.append((name, Path(entry.path)))

    # Step 3: walk each task once with per-task cap. Fan out with a thread
    # pool because each task scandir is one blocking network round-trip and
    # scandir releases the GIL.
    unique_tasks = []
    seen_task_names = set()
    for task_name, task_path in task_dirs:
        if task_name in seen_task_names:
            continue
        seen_task_names.add(task_name)
        unique_tasks.append((task_name, task_path))

    total_tasks = len(unique_tasks)
    n_threads = min(max(1, int(scandir_threads)), max(1, total_tasks))
    logger.info(
        "Discovery: %d task dirs to walk (cap %s ep/task, %d threads)",
        total_tasks,
        max_episodes_per_task if max_episodes_per_task is not None else "∞",
        n_threads,
    )

    tasks_seen.update(seen_task_names)  # keep external contract if any

    if n_threads == 1 or total_tasks <= 1:
        for i, (task_name, task_path) in enumerate(unique_tasks, 1):
            eps = _collect_from_task(task_path)
            episode_dirs.extend(eps)
            logger.info(
                "[%d/%d] task=%s eps=%d (total=%d)",
                i,
                total_tasks,
                task_name,
                len(eps),
                len(episode_dirs),
            )
    else:
        done = 0
        with cf.ThreadPoolExecutor(max_workers=n_threads) as pool:
            future_to_meta = {
                pool.submit(_collect_from_task, task_path): (
                    task_name,
                    task_path,
                )
                for task_name, task_path in unique_tasks
            }
            for fut in cf.as_completed(future_to_meta):
                task_name, _ = future_to_meta[fut]
                try:
                    eps = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "scandir task=%s failed: %s", task_name, exc
                    )
                    eps = []
                episode_dirs.extend(eps)
                done += 1
                logger.info(
                    "[%d/%d] task=%s eps=%d (total=%d)",
                    done,
                    total_tasks,
                    task_name,
                    len(eps),
                    len(episode_dirs),
                )

    # De-duplicate while preserving discovery order.
    seen = set()
    unique = []
    for ep in episode_dirs:
        key = str(ep)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ep)

    return [
        ABC130kEpisode(
            kind="mcap", episode_dir=ep, mcap_path=ep / "episode.mcap"
        )
        for ep in unique
    ]


def _compute_static_mask(
    joint_positions,
    ticks_ns,
    static_threshold,
    head_time_to_filter,
    tile_time_to_filter,
):
    """Return a boolean keep-mask matching the horizon_manipulation packer.

    Semantics (mirrors ``PiperMcapPacker._pack`` in mcap_packer.py):

      * A frame is "moving" if the max abs diff against the previous frame
        exceeds ``static_threshold`` on any joint. Frame 0 is always moving.
      * ``head_time_to_filter`` / ``tile_time_to_filter`` (both in **seconds**)
        define a protection window at the start / end of the episode:
        inside that window, only moving frames are kept; outside, every
        frame is kept regardless. Set either to ``None`` or ``<=0`` to
        disable that side. To filter static frames across the whole
        episode, pass a large number (e.g. ``1e8``).

    ``ticks_ns`` is a monotone int64 array of nanoseconds (the 30Hz aligned
    ticks); the function does its own ns->s conversion internally.
    """
    n = joint_positions.shape[0]
    static_mask = np.ones(n, dtype=bool)  # True = keep
    if static_threshold is None or static_threshold <= 0 or n <= 1:
        return static_mask
    moving = np.any(
        np.abs(np.diff(joint_positions, axis=0)) > float(static_threshold),
        axis=1,
    )
    static_mask[1:] = moving

    time_mask = np.zeros(n, dtype=bool)  # True = inside protection window
    t_s = (ticks_ns - ticks_ns[0]).astype(np.float64) / 1e9
    if head_time_to_filter is not None and head_time_to_filter > 0:
        time_mask |= t_s < float(head_time_to_filter)
    if tile_time_to_filter is not None and tile_time_to_filter > 0:
        tail_s = (ticks_ns[-1] - ticks_ns).astype(np.float64) / 1e9
        time_mask |= tail_s < float(tile_time_to_filter)
    # Keep if the frame is moving OR outside the protection window.
    return static_mask | np.logical_not(time_mask)


def parse_episode(
    episode_dir_str,
    state_dim=14,
    action_dim=14,
    static_threshold=None,
    head_time_to_filter=None,
    tile_time_to_filter=None,
):
    """Top-level pickleable worker function.

    Returns one of:
      {"status": "ok", "payload": {...}, "episode_name": str}
      {"status": "skip", "reason": str, "episode_name": str}
      {"status": "error", "error": str, "tb": str, "episode_name": str}

    All results are tagged with the worker PID and wall-clock duration so
    the main process can log parallelism.

    ``static_threshold`` / ``head_time_to_filter`` / ``tile_time_to_filter``
    control static-frame filtering — see ``_apply_static_filter`` for the
    exact semantics. Filtering happens inside the worker so the discarded
    ticks never make it through JPEG encoding + IPC.
    """
    import time

    t0 = time.perf_counter()
    worker_pid = os.getpid()
    episode_dir = Path(episode_dir_str)
    episode_name = episode_dir.name
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as e:
        return {
            "status": "error",
            "error": f"mcap deps missing: {e}",
            "tb": "",
            "episode_name": episode_name,
            "worker_pid": worker_pid,
            "duration": time.perf_counter() - t0,
        }
    try:
        result = _parse_episode_inner(
            episode_dir,
            episode_name,
            state_dim,
            action_dim,
            make_reader,
            DecoderFactory,
            static_threshold=static_threshold,
            head_time_to_filter=head_time_to_filter,
            tile_time_to_filter=tile_time_to_filter,
        )
    except Exception as e:  # noqa: BLE001
        result = {
            "status": "error",
            "error": str(e),
            "tb": traceback.format_exc(),
            "episode_name": episode_name,
        }
    result["worker_pid"] = worker_pid
    result["duration"] = time.perf_counter() - t0
    return result


def _parse_episode_inner(
    episode_dir,
    episode_name,
    state_dim,
    action_dim,
    make_reader,
    DecoderFactory,  # noqa: N803
    static_threshold=None,
    head_time_to_filter=None,
    tile_time_to_filter=None,
):
    row_width = state_dim + action_dim  # noqa: F841 (parity with old packer)

    mcap_path = episode_dir / "episode.mcap"
    task_name = episode_dir.parent.name

    cams = {}
    states = {}
    actions = {}
    calibs = {}
    cam_formats = {}
    instruction = None
    session_uuid = None
    session_meta = {}
    subtask_annotations = []

    scalar_names = {t for t, _ in STATE_TOPICS}
    action_names = {
        t
        for t, _ in [
            ("/left-arm-action", 6),
            ("/left-ee-action", 1),
            ("/right-arm-action", 6),
            ("/right-ee-action", 1),
        ]
    }
    cam_topic_names = {t for _, t in CAMERAS} | set(TOP_TOPIC_CANDIDATES)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for metadata in reader.iter_metadata():
            if metadata.name == "session-metadata":
                session_uuid = metadata.metadata.get("session-uuid")
                session_meta = dict(metadata.metadata)
                break

    if session_uuid:
        uuid = f"{task_name}_{session_uuid}"
    else:
        uuid = f"{task_name}_{episode_name}"

    # Early reject: open the MCAP once and detect ZED-X by scanning channels.
    # We can avoid decoding the full episode body for stations we'll throw
    # away anyway.
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        topic_set = set()
        for ch in reader.get_summary().channels.values():
            topic_set.add(ch.topic)
    is_zedx_by_topics = (
        "/top-left-camera" in topic_set and "/top-right-camera" in topic_set
    )
    if is_zedx_by_topics:
        return {
            "status": "skip",
            "reason": "zedx_station",
            "episode_name": episode_name,
        }
    if "/top-camera" not in topic_set:
        return {
            "status": "skip",
            "reason": "no_top_camera",
            "episode_name": episode_name,
        }

    ann_path = episode_dir / "annotation.mcap"
    if ann_path.is_file():
        with open(ann_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, channel, _, decoded in reader.iter_decoded_messages():
                if channel.topic != "/subtask-annotation":
                    continue
                ts_obj = getattr(decoded, "timestamp", None)
                if ts_obj is None:
                    continue
                ts_ns = int(ts_obj.seconds) * int(1e9) + int(ts_obj.nanos)
                subtask_annotations.append(
                    {
                        "timestamp_ns": ts_ns,
                        "label": getattr(decoded, "data", ""),
                    }
                )
        subtask_annotations.sort(key=lambda x: x["timestamp_ns"])

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for (
            _schema,
            channel,
            message,
            decoded,
        ) in reader.iter_decoded_messages():
            topic = channel.topic
            if topic == "/instruction":
                instruction = getattr(decoded, "data", None)
                continue
            if topic in cam_topic_names:
                cams.setdefault(topic, []).append(
                    (message.log_time, decoded.data)
                )
                cam_formats.setdefault(
                    topic,
                    str(getattr(decoded, "format", "")).lower(),
                )
                continue
            if topic in scalar_names:
                states.setdefault(topic, []).append(
                    (message.log_time, decoded)
                )
                continue
            if topic in action_names:
                actions.setdefault(topic, []).append(
                    (message.log_time, decoded)
                )
                continue
            if topic in CALIB_TOPICS:
                calibs[topic] = decoded
                continue

    for v in (
        list(cams.values()) + list(states.values()) + list(actions.values())
    ):
        v.sort(key=lambda x: x[0])

    top_topic = "/top-camera"
    active_cam_topics = [top_topic] + [t for _, t in CAMERAS if t in cams]

    # ZED-X resolution check (in case a station ships /top-camera but
    # 1920x1200 wrist hardware). Look at wrist camera calibration if present.
    for info_topic, _cam_name in CALIB_TOPICS.items():
        if info_topic not in calibs:
            continue
        c = calibs[info_topic]
        w, h = int(getattr(c, "width", 0)), int(getattr(c, "height", 0))
        if _is_zedx_calibration(w, h, info_topic):
            return {
                "status": "skip",
                "reason": "zedx_resolution",
                "episode_name": episode_name,
            }

    if any(t not in states for t, _ in STATE_TOPICS):
        return {
            "status": "skip",
            "reason": "missing_state_streams",
            "episode_name": episode_name,
        }

    required_action_topics = [
        "/left-arm-action",
        "/left-ee-action",
        "/right-arm-action",
        "/right-ee-action",
    ]
    if any(t not in actions for t in required_action_topics):
        missing = [t for t in required_action_topics if t not in actions]
        return {
            "status": "skip",
            "reason": "missing_action_streams",
            "episode_name": episode_name,
            "detail": ",".join(missing),
        }

    # Aligned 30Hz ticks, same logic as the single-process packer.
    all_streams = [cams[t] for t in active_cam_topics if t in cams] + [
        states[t] for t, _ in STATE_TOPICS if t in states
    ]
    if not all_streams:
        return {
            "status": "skip",
            "reason": "empty_streams",
            "episode_name": episode_name,
        }
    t0 = max(s[0][0] for s in all_streams)
    t_end = min(s[-1][0] for s in all_streams)
    ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
    num_steps = len(ticks)
    if num_steps < 1:
        return {
            "status": "skip",
            "reason": "no_aligned_steps",
            "episode_name": episode_name,
        }
    timestamp = ticks

    def sample_series(topic_dict, topic, attr_name, dim):
        msgs = topic_dict.get(topic)
        if not msgs:
            return np.zeros((num_steps, dim), dtype=np.float64)
        ts = np.array([t for t, _ in msgs], dtype=np.int64)
        raw_vals = []
        for _, msg in msgs:
            arr = np.array(getattr(msg, attr_name, []), dtype=np.float64)
            if arr.size < dim:
                tmp = np.zeros((dim,), dtype=np.float64)
                tmp[: arr.size] = arr
                arr = tmp
            elif arr.size > dim:
                arr = arr[:dim]
            raw_vals.append(arr)
        return np.stack(raw_vals, axis=0)[_floor_indices(ts, ticks)]

    left_joint = sample_series(states, "/left-arm-state", "position", 6)
    right_joint = sample_series(states, "/right-arm-state", "position", 6)
    left_gripper = sample_series(states, "/left-ee-state", "position", 1)
    right_gripper = sample_series(states, "/right-ee-state", "position", 1)
    joint_positions = np.concatenate(
        [left_joint, left_gripper, right_joint, right_gripper], axis=1
    )

    # Action layout mirrors joint_positions:
    #   [L_arm(6), L_gripper(1), R_arm(6), R_gripper(1)]
    # Gripper action is 0/1 (open/close intent), while gripper state is the
    # physical finger distance after contact. Training on state as the BC
    # target lets the model settle for whatever the fingers converged to and
    # never fully close on new objects, so action is packed as a separate
    # stream and downstream can pick it per-dimension.
    left_act = sample_series(actions, "/left-arm-action", "position", 6)
    right_act = sample_series(actions, "/right-arm-action", "position", 6)
    left_act_gripper = sample_series(
        actions,
        "/left-ee-action",
        "position",
        1,
    )
    right_act_gripper = sample_series(
        actions,
        "/right-ee-action",
        "position",
        1,
    )
    joint_actions = np.concatenate(
        [left_act, left_act_gripper, right_act, right_act_gripper],
        axis=1,
    )

    # Static-frame filter — drop ticks where |Δjoint_positions| stays below
    # `static_threshold` inside the head/tail protection windows. Applied
    # here (before video decode) so JPEGs are never encoded for dropped
    # ticks. Skipping this drops peak worker RSS + IPC payload dramatically
    # for long episodes with a static settle at the start/end.
    num_steps_raw = num_steps
    keep_mask = _compute_static_mask(
        joint_positions,
        ticks,
        static_threshold=static_threshold,
        head_time_to_filter=head_time_to_filter,
        tile_time_to_filter=tile_time_to_filter,
    )
    num_kept = int(keep_mask.sum())
    if num_kept < 1:
        return {
            "status": "skip",
            "reason": "all_static",
            "episode_name": episode_name,
            "detail": f"raw_steps={num_steps_raw}",
        }
    if num_kept < num_steps_raw:
        ticks = ticks[keep_mask]
        timestamp = ticks
        joint_positions = joint_positions[keep_mask]
        joint_actions = joint_actions[keep_mask]
        num_steps = int(ticks.shape[0])

    camera_names = [_camera_name_from_topic(t) for t in active_cam_topics]
    intrinsic = {}
    calibration = {}
    camera_info = {}
    extrinsic = {}

    # Decode videos to per-tick JPEG bytes in a temp dir; this is the
    # heavy part. Raw BGR frames are encoded and dropped inside the loop so
    # peak per-worker memory stays around a few hundred MB (JPEG bytes
    # only) instead of scaling with raw video length × 3 cameras.
    image_bytes = {}  # cam_name -> [jpeg bytes per aligned tick]
    image_shapes = {}  # cam_name -> (h, w)
    with tempfile.TemporaryDirectory() as work:
        for topic in active_cam_topics:
            cam_name = _camera_name_from_topic(topic)
            msgs = cams.get(topic)
            if not msgs:
                return {
                    "status": "skip",
                    "reason": f"empty_video_{cam_name}",
                    "episode_name": episode_name,
                }
            cam_ts = np.array([t for t, _ in msgs], dtype=np.int64)

            codec_fmt = cam_formats.get(topic, "h264")
            stream_suffix = "h265" if "265" in codec_fmt else "h264"
            h264_file = Path(work) / f"{cam_name}.{stream_suffix}"
            with open(h264_file, "wb") as f:
                for _, chunk in msgs:
                    f.write(chunk)

            try:
                (n_frames,) = probe(
                    str(h264_file),
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                )
            except Exception as e:  # noqa: BLE001
                return {
                    "status": "skip",
                    "reason": f"probe_failed_{cam_name}",
                    "episode_name": episode_name,
                    "detail": str(e),
                }

            mp4_file = Path(work) / f"{cam_name}.mp4"
            remux_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(h264_file),
                "-c:v",
                "copy",
                str(mp4_file),
            ]
            remux_proc = subprocess.run(
                remux_cmd, capture_output=True, text=True
            )
            if remux_proc.returncode != 0:
                transcode_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(h264_file),
                    *X264,
                    "-threads",
                    "1",
                    str(mp4_file),
                ]
                transcode_proc = subprocess.run(
                    transcode_cmd,
                    capture_output=True,
                    text=True,
                )
                if transcode_proc.returncode != 0:
                    return {
                        "status": "skip",
                        "reason": f"ffmpeg_failed_{cam_name}",
                        "episode_name": episode_name,
                        "detail": transcode_proc.stderr.strip()[:400],
                    }

            if n_frames > 0 and n_frames != len(cam_ts):
                cam_ts = np.linspace(
                    cam_ts[0],
                    cam_ts[-1],
                    n_frames,
                    dtype=np.int64,
                )
            frame_idx = _floor_indices(cam_ts, ticks)
            max_needed_idx = int(frame_idx.max()) if len(frame_idx) > 0 else -1

            # Walk the mp4 exactly once, encoding each unique raw frame to
            # JPEG the moment we hit it and dropping the BGR array. Prior
            # implementation accumulated every decoded BGR frame in
            # ``all_frames`` — at 640x480x3 that's ~900 KB/frame, and
            # long-form episodes at 30-60 fps × 3 cameras easily hit 10+ GB
            # per worker, blowing the 112 GB dev-box open at num_workers=8.
            # ``frame_idx`` is monotone non-decreasing (searchsorted), so
            # ticks that fall on the same raw frame can just point at the
            # last encoded buffer.
            enc_bytes = [None] * len(frame_idx)
            cap = cv2.VideoCapture(str(mp4_file))
            current_idx = -1
            current_frame = None
            last_enc_idx = -1
            last_enc_buf = None
            last_shape = None
            decode_short = False
            jpeg_fail = False
            for i, target in enumerate(frame_idx):
                target = int(target)
                if target == last_enc_idx:
                    enc_bytes[i] = last_enc_buf
                    continue
                while current_idx < target:
                    ok, frm = cap.read()
                    if not ok:
                        decode_short = True
                        break
                    current_idx += 1
                    current_frame = frm
                if decode_short:
                    break
                ok, buf = cv2.imencode(".jpg", current_frame)
                if not ok:
                    jpeg_fail = True
                    break
                last_enc_buf = buf.tobytes()
                last_enc_idx = target
                last_shape = current_frame.shape[:2]
                enc_bytes[i] = last_enc_buf
            cap.release()
            if jpeg_fail:
                return {
                    "status": "skip",
                    "reason": f"jpeg_encode_{cam_name}",
                    "episode_name": episode_name,
                }
            if decode_short or current_idx < max_needed_idx:
                return {
                    "status": "skip",
                    "reason": f"decode_short_{cam_name}",
                    "episode_name": episode_name,
                }
            image_bytes[cam_name] = enc_bytes
            image_shapes[cam_name] = last_shape

    # Validate intrinsics now that we know each saved image's actual size.
    # We *don't* mutate stored K (the dataset's correct_k handles that at
    # load time and only when correct_k=True); we just gate the episode.
    for topic in active_cam_topics:
        cam_name = _camera_name_from_topic(topic)
        info_topic = (
            "/top-camera-info"
            if topic == "/top-camera"
            else topic.replace("-camera", "-camera-info")
        )
        if info_topic not in calibs:
            return {
                "status": "skip",
                "reason": f"no_calib_{cam_name}",
                "episode_name": episode_name,
            }
        c = calibs[info_topic]
        k = np.array(getattr(c, "K", []), dtype=np.float64)
        if k.size != 9:
            return {
                "status": "skip",
                "reason": f"bad_K_size_{cam_name}",
                "episode_name": episode_name,
            }
        # Reject rational_polynomial (wide-angle) calibrations: those episodes
        # come from a different wrist camera hardware than the D405 that the
        # cameras.yaml/URDF extrinsics were captured for. Even after
        # correctly undistorting the image and rescaling K, the FK-derived
        # T_world2cam is wrong because the physical mount is different, so
        # projected joint markers land off-target. Skipping is safer than
        # training on silently-misprojected data.
        dm = str(getattr(c, "distortion_model", "")).lower()
        if dm == "rational_polynomial":
            return {
                "status": "skip",
                "reason": f"wide_angle_{cam_name}",
                "episode_name": episode_name,
                "detail": (
                    "distortion_model=rational_polynomial (non-D405 wrist "
                    "hardware, extrinsics from URDF do not apply)"
                ),
            }
        K_raw = k.reshape(3, 3)  # noqa: N806
        h, w = image_shapes[cam_name]
        _, ok, reason = validate_intrinsic(K_raw, w, h)
        if not ok:
            return {
                "status": "skip",
                "reason": f"bad_intrinsic_{cam_name}",
                "episode_name": episode_name,
                "detail": reason,
            }
        intrinsic[cam_name] = K_raw
        calibration[cam_name] = dict(
            width=int(getattr(c, "width", 0)),
            height=int(getattr(c, "height", 0)),
            K=list(getattr(c, "K", [])),
            P=list(getattr(c, "P", [])),
            R=list(getattr(c, "R", [])),
            D=list(getattr(c, "D", [])),
            distortion_model=getattr(c, "distortion_model", ""),
            frame_id=getattr(c, "frame_id", ""),
        )
        camera_info[cam_name] = {
            "image": {
                "height": int(getattr(c, "height", 0)),
                "width": int(getattr(c, "width", 0)),
                "K": list(getattr(c, "K", [])),
                "P": list(getattr(c, "P", [])),
                "R": list(getattr(c, "R", [])),
                "D": list(getattr(c, "D", [])),
                "distortion_model": getattr(c, "distortion_model", ""),
            }
        }
        extrinsic[cam_name] = _get_reference_world2cam(cam_name)

    # image_bytes already populated inline during the decode loop above.

    meta = dict(
        task_name=task_name,
        instruction=instruction,
        source="mcap",
        has_depth=False,
        mcap_path=str(mcap_path),
        annotation_mcap_path=str(ann_path) if ann_path.is_file() else None,
        session_uuid=session_uuid,
        operator_id=session_meta.get("operator-id"),
        session_instruction=session_meta.get("instruction"),
        start_time_unix_ms=session_meta.get("start-time-unix"),
        end_time_unix_ms=session_meta.get("end-time-unix"),
        video_formats={
            _camera_name_from_topic(t): cam_formats.get(t)
            for t in active_cam_topics
        },
        alignment="tick_floor_33ms",
        extrinsic_source="abc_main_i2rt_yam_zero_joint_reference",
        extrinsic_note=(
            "Reference extrinsics from MJCF (MuJoCo -> CV optical). Top uses "
            "static world pose; left/right wrist cameras must be recomputed "
            "by FK at load time."
        ),
        t0_ns=int(timestamp[0]),
        tick_ns=None,
        num_steps=int(num_steps),
        num_steps_raw=int(num_steps_raw),
        static_filter=dict(
            static_threshold=(
                float(static_threshold)
                if static_threshold is not None
                else None
            ),
            head_time_to_filter=(
                float(head_time_to_filter)
                if head_time_to_filter is not None
                else None
            ),
            tile_time_to_filter=(
                float(tile_time_to_filter)
                if tile_time_to_filter is not None
                else None
            ),
            num_dropped=int(num_steps_raw - num_steps),
        ),
    )

    payload = dict(
        uuid=uuid,
        task_name=task_name,
        num_steps=num_steps,
        camera_names=camera_names,
        timestamp=timestamp,
        joint_positions=joint_positions,
        joint_actions=joint_actions,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        # image_shapes lets the main process apply K correction + FK without
        # re-decoding a frame. Sending only the (h, w) tuples per camera adds
        # a handful of bytes to the IPC payload.
        image_shapes=image_shapes,
        calibration=calibration,
        camera_info=camera_info,
        image_bytes=image_bytes,
        instruction=instruction,
        subtask_annotations=subtask_annotations,
        meta=meta,
    )
    return {"status": "ok", "payload": payload, "episode_name": episode_name}


class ABC130kMPLmdbPacker(BaseLmdbManipulationDataPacker):
    """Multi-worker MCAP -> LMDB packer with filtering and stats."""

    def __init__(
        self,
        input_path,
        output_path,
        state_dim=14,
        action_dim=14,
        num_workers=8,
        num_steps_per_shard=None,
        stats_path=None,
        max_episodes=None,
        max_episodes_per_task=None,
        scandir_threads=32,
        joint_stats_subsample=8,
        joint_stats_bins=20,
        urdf_path=None,
        joint_to_urdf=None,
        camera_to_urdf_link=None,
        static_threshold=None,
        head_time_to_filter=None,
        tile_time_to_filter=None,
        **kwargs,
    ):
        super().__init__(input_path, output_path, **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_workers = max(1, int(num_workers))
        self.num_steps_per_shard = num_steps_per_shard
        self.stats_path = stats_path
        self.joint_stats_bins = int(joint_stats_bins)
        # Static-frame filter params — forwarded to every worker. See
        # `_compute_static_mask` for semantics. All-None disables filtering.
        self.static_threshold = (
            float(static_threshold) if static_threshold is not None else None
        )
        self.head_time_to_filter = (
            float(head_time_to_filter)
            if head_time_to_filter is not None
            else None
        )
        self.tile_time_to_filter = (
            float(tile_time_to_filter)
            if tile_time_to_filter is not None
            else None
        )
        # FK runs on the main process against every payload as it arrives.
        # This keeps the workers pickling-friendly (a `pytorch_kinematics`
        # chain isn't) and adds ~milliseconds per episode next to minutes of
        # video decode. `urdf_path=""` (or falsy) disables FK for the "old
        # zero-joint" layout.
        self.extrinsics_fk = (
            ABC130kExtrinsicsFK(
                urdf_path,
                joint_to_urdf=joint_to_urdf,
                camera_to_urdf_link=camera_to_urdf_link,
            )
            if urdf_path
            else None
        )
        self.joint_stats = JointStatsAccumulator(
            ABC130K_JOINT_NAMES,
            subsample_per_episode=joint_stats_subsample,
            label="joint_state",
        )
        self.action_stats = JointStatsAccumulator(
            ABC130K_JOINT_NAMES,
            subsample_per_episode=joint_stats_subsample,
            label="joint_action",
        )
        # Push per-task cap into discovery so scandir short-circuits after N
        # matches per task. This is what makes "walk 200 tasks × 1 ep" finish
        # in seconds instead of stat-ing all 130k mcap paths on JFS.
        self.episodes = _discover_episodes(
            input_path,
            max_episodes_per_task=max_episodes_per_task,
            scandir_threads=scandir_threads,
        )
        if max_episodes is not None and max_episodes > 0:
            self.episodes = self.episodes[: int(max_episodes)]
            logger.info(
                "Capped to first %d episodes (--max_episodes).",
                len(self.episodes),
            )
        logger.info(
            "Discovered %d episodes under %s",
            len(self.episodes),
            input_path,
        )

    def _write_payload(self, ep_id, payload, episode_name):
        uuid = payload["uuid"]
        num_steps = payload["num_steps"]
        camera_names = payload["camera_names"]

        # Non-time-series meta — written once per episode.
        if payload["instruction"] is not None:
            self.meta_pack_file.write(
                f"{uuid}/instructions",
                payload["instruction"],
            )
        self.meta_pack_file.write(f"{uuid}/meta_data", payload["meta"])
        self.meta_pack_file.write(f"{uuid}/camera_names", camera_names)
        self.meta_pack_file.write(f"{uuid}/has_depth", False)
        self.meta_pack_file.write(f"{uuid}/intrinsic", payload["intrinsic"])
        self.meta_pack_file.write(
            f"{uuid}/intrinsic_corrected", payload["intrinsic_corrected"]
        )
        self.meta_pack_file.write(f"{uuid}/extrinsic", payload["extrinsic"])
        if payload.get("extrinsic_corrected") is not None:
            self.meta_pack_file.write(
                f"{uuid}/extrinsic_corrected", payload["extrinsic_corrected"]
            )
        self.meta_pack_file.write(
            f"{uuid}/calibration", payload["calibration"]
        )
        self.meta_pack_file.write(
            f"{uuid}/camera_info",
            payload.get("camera_info"),
        )
        self.meta_pack_file.write(
            f"{uuid}/subtask_annotations",
            payload.get("subtask_annotations", []),
        )

        timestamp = payload["timestamp"]
        joint_positions = payload["joint_positions"]
        joint_actions = payload["joint_actions"]

        if self.num_steps_per_shard is None:
            self.meta_pack_file.write(f"{uuid}/timestamp", timestamp)
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_positions",
                joint_positions,
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/master_joint_positions",
                joint_actions,
            )
        else:
            sps = int(self.num_steps_per_shard)
            self.meta_pack_file.write(f"{uuid}/num_steps_per_shard", sps)
            num_shards = math.ceil(num_steps / sps)
            for shard_idx in range(num_shards):
                s = shard_idx * sps
                e = min(s + sps, num_steps)
                self.meta_pack_file.write(
                    f"{uuid}/{shard_idx}/timestamp",
                    timestamp[s:e],
                )
                self.meta_pack_file.write(
                    f"{uuid}/{shard_idx}/observation/robot_state/joint_positions",
                    joint_positions[s:e],
                )
                self.meta_pack_file.write(
                    f"{uuid}/{shard_idx}/observation/robot_state/master_joint_positions",
                    joint_actions[s:e],
                )

        # Images stay flat at {uuid}/{cam}/{i} with global step index, so
        # the dataset image reader (which always uses the global step within
        # episode) doesn't need shard-awareness.
        for i in range(num_steps):
            for cam_name in camera_names:
                self.image_pack_file.write(
                    f"{uuid}/{cam_name}/{i}",
                    payload["image_bytes"][cam_name][i],
                )
        self.write_index(
            ep_id,
            dict(
                uuid=uuid,
                task_name=payload["task_name"],
                num_steps=num_steps,
                date=episode_name,
                simulation=False,
                error=False,
            ),
        )

    def _pack(self):
        stats = Counter()
        skip_details = []  # list of (episode_name, reason, detail)
        num_kept = 0
        total = len(self.episodes)
        if total == 0:
            logger.warning("No episodes found.")
            self.index_pack_file.write("__len__", 0)
            self.close()
            return

        episode_dirs = [str(e.episode_dir) for e in self.episodes]
        worker_args = [
            (
                d,
                self.state_dim,
                self.action_dim,
                self.static_threshold,
                self.head_time_to_filter,
                self.tile_time_to_filter,
            )
            for d in episode_dirs
        ]

        def consume(result, ep_id_holder):
            nonlocal num_kept
            status = result.get("status")
            ep_name = result.get("episode_name", "?")
            wpid = result.get("worker_pid", -1)
            dt = result.get("duration", 0.0)
            tag = f"worker={wpid} dt={dt:.2f}s"
            if status == "ok":
                payload = result["payload"]
                # Enrich the payload with K correction + FK extrinsics on the
                # main process. Doing it here keeps the workers picklable
                # (they don't need torch/pytorch_kinematics) and the cost is
                # microseconds next to minutes of video decode per worker.
                payload["intrinsic_corrected"], _ = correct_intrinsics_dict(
                    payload["intrinsic"], payload.get("image_shapes", {})
                )
                payload["extrinsic_corrected"] = (
                    self.extrinsics_fk.compute(
                        payload["camera_names"], payload["joint_positions"]
                    )
                    if self.extrinsics_fk is not None
                    else None
                )
                self._write_payload(num_kept, payload, ep_name)
                self.joint_stats.update(payload["joint_positions"])
                self.action_stats.update(payload["joint_actions"])
                num_kept += 1
                stats["kept"] += 1
                logger.info(
                    "[%d/%d %s kept=%d] %s steps=%d",
                    ep_id_holder[0] + 1,
                    total,
                    tag,
                    num_kept,
                    payload["uuid"],
                    payload["num_steps"],
                )
            elif status == "skip":
                reason = result.get("reason", "unknown")
                stats[f"skip:{reason}"] += 1
                skip_details.append((ep_name, reason, result.get("detail")))
                logger.info(
                    "[%d/%d %s skip=%s] %s",
                    ep_id_holder[0] + 1,
                    total,
                    tag,
                    reason,
                    ep_name,
                )
            else:
                stats["error"] += 1
                skip_details.append(
                    (ep_name, "error", result.get("error", ""))
                )
                logger.warning(
                    "[%d/%d %s ERROR] %s: %s",
                    ep_id_holder[0] + 1,
                    total,
                    tag,
                    ep_name,
                    result.get("error", "")[:300],
                )

        if self.num_workers == 1:
            for i, args in enumerate(worker_args):
                consume(_worker_entry(args), [i])
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=self.num_workers) as pool:
                holder = [0]
                for result in pool.imap_unordered(
                    _worker_entry,
                    worker_args,
                    chunksize=1,
                ):
                    consume(result, holder)
                    holder[0] += 1

        self.index_pack_file.write("__len__", num_kept)
        self.close()

        # Summary
        logger.info("=" * 60)
        logger.info("Pack summary: %d total, %d kept", total, num_kept)
        for k, v in sorted(stats.items()):
            pct = v * 100.0 / max(total, 1)
            logger.info("  %-30s %6d  (%5.2f%%)", k, v, pct)

        joint_stats_final = self.joint_stats.print_summary(
            log_fn=logger.info,
            n_bins=self.joint_stats_bins,
        )
        action_stats_final = self.action_stats.print_summary(
            log_fn=logger.info,
            n_bins=self.joint_stats_bins,
        )

        if self.stats_path:
            import json

            os.makedirs(os.path.dirname(self.stats_path) or ".", exist_ok=True)
            payload = {
                "total": total,
                "kept": num_kept,
                "counts": dict(stats),
                "skipped": [
                    {"episode": e, "reason": r, "detail": d}
                    for e, r, d in skip_details
                ],
            }

            def _serialize(final, names):
                return {
                    "joint_names": names,
                    "num_samples": final["num_samples"],
                    "num_raw_rows": final["num_raw_rows"],
                    "num_episodes": final["num_episodes"],
                    "mean": final["mean"].tolist(),
                    "std": final["std"].tolist(),
                    "min": final["min"].tolist(),
                    "max": final["max"].tolist(),
                    "p25": final["p25"].tolist(),
                    "p50": final["p50"].tolist(),
                    "p75": final["p75"].tolist(),
                }

            if joint_stats_final is not None:
                payload["joint_stats"] = _serialize(
                    joint_stats_final,
                    self.joint_stats.joint_names,
                )
            if action_stats_final is not None:
                payload["action_stats"] = _serialize(
                    action_stats_final,
                    self.action_stats.joint_names,
                )
            with open(self.stats_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info("Wrote stats to %s", self.stats_path)


def _worker_entry(args):
    (
        episode_dir_str,
        state_dim,
        action_dim,
        static_threshold,
        head_time_to_filter,
        tile_time_to_filter,
    ) = args
    return parse_episode(
        episode_dir_str,
        state_dim,
        action_dim,
        static_threshold=static_threshold,
        head_time_to_filter=head_time_to_filter,
        tile_time_to_filter=tile_time_to_filter,
    )


def _build_pack_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=(
            "Comma-separated list of paths. Each entry may be a dataset root "
            "(contains task subdirs), a task directory, an episode directory, "
            "or a shell glob resolving to any of the above. Passing a "
            "task-level path (e.g. /.../train or /.../train/*) is strongly "
            "preferred on network filesystems: an episode-level glob "
            "(/.../train/*/episode_*) forces glob to walk all ~130k episode "
            "dirs before packing starts."
        ),
    )
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--commit_step", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help=(
            "Optional cap: only process the first N episodes after glob "
            "discovery. Useful for smoke tests."
        ),
    )
    parser.add_argument(
        "--max_episodes_per_task",
        type=int,
        default=None,
        help=(
            "Optional per-task cap: keep only the first N episodes of every "
            "distinct task (episode's parent directory). Applied before "
            "--max_episodes. Set N=1 for the fastest end-to-end coverage "
            "sweep."
        ),
    )
    parser.add_argument(
        "--scandir_threads",
        type=int,
        default=32,
        help=(
            "Thread pool size for parallel per-task scandir during discovery. "
            "Only affects wall time on network filesystems where each dir "
            "listing is a blocking round-trip. Set to 1 to disable."
        ),
    )
    parser.add_argument(
        "--num_steps_per_shard",
        type=int,
        default=None,
        help=(
            "If set, time-series metadata (joint_positions, timestamp) is "
            "sliced into shards of this size and keyed as "
            "{uuid}/{shard_idx}/... A {uuid}/num_steps_per_shard marker is "
            "written so the dataset can resolve shards at load time. Images "
            "remain at the flat {uuid}/{cam}/{i} layout. Recommended: 32-128."
        ),
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Optional JSON path for full skip/error breakdown.",
    )
    parser.add_argument(
        "--joint_stats_subsample",
        type=int,
        default=8,
        help=(
            "Per-episode subsample cap for joint stats. Very long episodes "
            "otherwise dominate the histogram. Set 0 to keep every row."
        ),
    )
    parser.add_argument(
        "--joint_stats_bins",
        type=int,
        default=20,
        help="Histogram bin count printed for each joint.",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="/home/users/zhengmao.sun-labs/codes/robo_orchard_lab/abc-main/assets/put_bottles/put_bottle_dual_arm.urdf",
        help=(
            "Path to the YAM dual-arm URDF used to run FK for per-step "
            "wrist-camera extrinsics (`extrinsic_corrected`). Pass empty "
            "string to skip FK and store only zero-joint references."
        ),
    )
    parser.add_argument(
        "--static_threshold",
        type=float,
        default=1e-3,
        help=(
            "Max abs delta on ANY joint below which the tick is treated as "
            "static (units match `joint_positions`, typically radians / "
            "gripper units). Combined with the head/tail protection windows "
            "below: static ticks inside the window are dropped, ticks "
            "outside the window are always kept. Set <=0 to disable "
            "filtering entirely. Default matches horizon_manipulation's "
            "mcap_packer."
        ),
    )
    parser.add_argument(
        "--head_time_to_filter",
        type=float,
        default=1e8,
        help=(
            "Protection window at the START of the episode, in SECONDS. "
            "Static ticks whose (t - t0) < head_time_to_filter are eligible "
            "for filtering. Default 1e8 matches mcap_packer — effectively "
            "'the whole episode', so all static frames are dropped. Set "
            "<=0 to disable head filtering entirely."
        ),
    )
    parser.add_argument(
        "--tile_time_to_filter",
        type=float,
        default=None,
        help=(
            "Protection window at the END of the episode, in SECONDS. "
            "Static ticks whose (t_end - t) < tile_time_to_filter are "
            "eligible for filtering. Default None matches mcap_packer — "
            "no tail-side filter (head window already covers everything "
            "when its default 1e8 is used)."
        ),
    )
    return parser


# --------------------------------------------------------------------------- #
# Orchestrator: task discovery, chunking, per-chunk subprocess dispatch.
# --------------------------------------------------------------------------- #
# This module is also its own per-chunk subprocess entry point. The
# orchestrator invokes ``python -m <this>`` for each chunk, which lands
# in single-pack mode and routes to ABC130kMPLmdbPacker.
_ORCHESTRATOR_WORKER_MODULE = (
    "robo_orchard_lab.dataset.abc130k.abc130k_lmdb_packer"
)


def _chunk(seq, size):
    for i in range(0, len(seq), size):
        yield i // size, seq[i : i + size]


def _chunk_output_dir(output_root, task_name, chunk_idx):
    return Path(output_root) / task_name / f"chunk_{chunk_idx:03d}"


def _scandir_names(path):
    try:
        with os.scandir(path) as it:
            return [e.name for e in it]
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.warning("scandir failed on %s: %s", path, exc)
        return []


def _discover_from_root(data_root, scandir_threads=32):
    """Walk ``data_root/<task>/episode_*/`` and return the task->episodes map.

    No completeness check — for a fully-copied dataset, just enumerate
    everything. Uses ``os.scandir`` + a thread pool because each per-task
    listing on network FS is one blocking round-trip.
    """
    data_root = Path(data_root)
    task_names = sorted(
        n for n in _scandir_names(data_root) if not n.startswith(".")
    )
    logger.info(
        "Discovery: %d task dirs under %s",
        len(task_names),
        data_root,
    )

    def _list_eps(task_name):
        task_dir = data_root / task_name
        eps = sorted(
            n for n in _scandir_names(task_dir) if n.startswith("episode_")
        )
        return task_name, eps

    n_threads = min(max(1, int(scandir_threads)), max(1, len(task_names)))
    tasks = []
    with cf.ThreadPoolExecutor(max_workers=n_threads) as pool:
        futs = [pool.submit(_list_eps, t) for t in task_names]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            task_name, eps = fut.result()
            tasks.append(
                {
                    "task": task_name,
                    "episode_count": len(eps),
                    "episodes": eps,
                }
            )
            if i % 50 == 0 or i == len(task_names):
                logger.info(
                    "  discovered %d/%d tasks (last=%s, %d eps)",
                    i,
                    len(task_names),
                    task_name,
                    len(eps),
                )
    tasks.sort(key=lambda x: x["task"])
    return tasks


def _chunk_is_done(chunk_dir):
    """A chunk is "done" iff ``pack_stats.json`` records ``kept > 0``.

    Presence of ``index/`` + ``meta/`` + ``image/`` alone is NOT enough:
    LMDB pre-allocates ``data.mdb`` so an empty env still shows non-zero
    file sizes. We hit exactly that trap once — every episode inside the
    packer returned ``error="mcap deps missing"`` but the LMDB was
    initialized, so a naive dir-based heuristic falsely marked those
    chunks "done" for the next ``--skip_existing`` pass.
    """
    stats_path = chunk_dir / "pack_stats.json"
    if not stats_path.is_file():
        return False
    try:
        with open(stats_path) as f:
            s = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return int(s.get("kept", 0)) > 0


def _wipe_partial_pack(out_dir):
    """Remove index/meta/image/depth subdirs of a previously crashed pack.

    Called before a chunk is (re)packed. Without this, LMDB entries whose
    keys don't match the new pack pass become orphan bytes (not referenced
    by index, waste disk but harmless). Cleaning up is safe when
    ``_chunk_is_done`` returned False (kept<=0), and LMDB envs re-init
    cleanly on fresh dirs.
    """
    for sub in ("index", "meta", "image", "depth"):
        partial = out_dir / sub
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)


def _run_chunk_subprocess(
    input_path,
    out_dir,
    args,
    stats_path,
    log_path,
):
    """Dispatch one chunk pack as an isolated subprocess.

    Returning the wall-clock and rc so the caller can accumulate stats.
    """
    cmd = [
        sys.executable,
        "-u",
        "-m",
        _ORCHESTRATOR_WORKER_MODULE,
        "--input_path",
        input_path,
        "--output_path",
        str(out_dir),
        "--num_workers",
        str(args.num_workers),
        "--num_steps_per_shard",
        str(args.num_steps_per_shard),
        "--stats_path",
        str(stats_path),
        "--joint_stats_subsample",
        str(args.joint_stats_subsample),
        "--joint_stats_bins",
        str(args.joint_stats_bins),
    ]
    # Forward URDF override if the caller set one — otherwise the child's
    # default (baked into _build_pack_parser above) is used.
    if args.urdf_path is not None:
        cmd += ["--urdf_path", str(args.urdf_path)]
    # Forward static-frame filter params. Only forward flags whose value is
    # not None — argparse `type=float` on the child would choke on "None",
    # and the tile_time_to_filter default is legitimately None.
    if args.static_threshold is not None:
        cmd += ["--static_threshold", str(args.static_threshold)]
    if args.head_time_to_filter is not None:
        cmd += ["--head_time_to_filter", str(args.head_time_to_filter)]
    if args.tile_time_to_filter is not None:
        cmd += ["--tile_time_to_filter", str(args.tile_time_to_filter)]
    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode, time.perf_counter() - t0


def _read_pack_stats(stats_path):
    """Return ``(kept, err)`` for a chunk's pack_stats.json.

    Returns ``(None, None)`` if the file is missing or unparseable.
    """
    if not stats_path.is_file():
        return None, None
    try:
        with open(stats_path) as sf:
            s = json.load(sf)
        return int(s.get("kept", 0)), int(s.get("counts", {}).get("error", 0))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "  could not parse pack_stats.json at %s (%s)", stats_path, e
        )
        return None, None


def _load_task_list(args):
    """Return (candidate_root, tasks) per the CLI source flags."""
    if args.data_root:
        tasks = _discover_from_root(
            args.data_root,
            scandir_threads=args.scandir_threads,
        )
        candidate_root = Path(args.data_root)
        logger.info(
            "Discovered %d tasks directly from %s",
            len(tasks),
            args.data_root,
        )
        return candidate_root, tasks

    if not args.candidate_root:
        raise SystemExit(
            "--candidate_root is required when using --complete_tasks_json"
        )
    with open(args.complete_tasks_json) as f:
        report = json.load(f)
    tasks = report.get("complete_tasks", [])
    candidate_root = Path(args.candidate_root)
    logger.info(
        "Loaded %d complete tasks from %s",
        len(tasks),
        args.complete_tasks_json,
    )
    return candidate_root, tasks


def _slice_task_shard(tasks, num_shards, shard_idx):
    """Return the contiguous ``tasks[start:end)`` slice for this shard.

    Task-level slicing means every chunk of a given task ends up on the
    same job, keeping per-task output cohesive on one machine and logs
    easy to follow. Sizes may vary between shards because episode counts
    per task vary.
    """
    if num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if not (0 <= shard_idx < num_shards):
        raise SystemExit(f"--shard_idx must be in [0, {num_shards})")
    n_all_tasks = len(tasks)
    if num_shards == 1:
        logger.info("Sharding: single job, packing all %d tasks.", n_all_tasks)
        return tasks
    start = n_all_tasks * shard_idx // num_shards
    end = n_all_tasks * (shard_idx + 1) // num_shards
    my = tasks[start:end]
    logger.info(
        "Sharding: shard %d/%d handles tasks[%d:%d) of %d "
        "(%d tasks in this shard)",
        shard_idx,
        num_shards,
        start,
        end,
        n_all_tasks,
        len(my),
    )
    return my


def _plan_summary(my_tasks, per_chunk):
    total_chunks = 0
    total_episodes = 0
    for t in my_tasks:
        eps = t["episodes"]
        total_episodes += len(eps)
        if per_chunk is None:
            total_chunks += 1
        else:
            total_chunks += (len(eps) + per_chunk - 1) // per_chunk
    logger.info(
        "Plan (this shard): %d tasks, %d episodes, %d chunks (%s eps/chunk).",
        len(my_tasks),
        total_episodes,
        total_chunks,
        "task" if per_chunk is None else per_chunk,
    )
    return total_chunks


def _orchestrate(args):
    """Task/chunk fan-out entry point (mode B).

    Reads (or discovers) the task list, slices for this shard, then packs
    each chunk in an isolated subprocess. Handles ``--skip_existing`` and
    silent-failure detection (rc=0 but all episodes errored).
    """
    candidate_root, tasks = _load_task_list(args)

    if args.task_names:
        include = set(args.task_names)
        tasks = [t for t in tasks if t["task"] in include]
        logger.info("Filtered to %d tasks via --task_names.", len(tasks))

    if args.max_tasks is not None and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
        logger.info("Capped to first %d tasks (--max_tasks).", len(tasks))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    per_chunk = args.episodes_per_lmdb if args.episodes_per_lmdb > 0 else None

    my_tasks = _slice_task_shard(tasks, args.num_shards, args.shard_idx)
    total_chunks = _plan_summary(my_tasks, per_chunk)

    if args.dry_run:
        for t in my_tasks[:5]:
            eps = t["episodes"]
            chunks = list(_chunk(eps, per_chunk or len(eps)))
            for chunk_idx, chunk_eps in chunks[:3]:
                out_dir = _chunk_output_dir(output_root, t["task"], chunk_idx)
                logger.info(
                    "  [dry] task=%s chunk=%03d eps=%d -> %s",
                    t["task"],
                    chunk_idx,
                    len(chunk_eps),
                    out_dir,
                )
        return 0

    n_ok = 0
    n_skipped = 0
    n_failed = 0
    t_start = time.perf_counter()

    for task_i, t in enumerate(my_tasks, 1):
        task_name = t["task"]
        eps = t["episodes"]
        task_dir = candidate_root / task_name
        chunks = list(_chunk(eps, per_chunk or len(eps)))
        logger.info(
            "[task %d/%d] %s: %d episodes -> %d chunks",
            task_i,
            len(my_tasks),
            task_name,
            len(eps),
            len(chunks),
        )
        for chunk_idx, chunk_eps in chunks:
            out_dir = _chunk_output_dir(output_root, task_name, chunk_idx)
            if args.skip_existing and _chunk_is_done(out_dir):
                logger.info(
                    "  chunk_%03d: already packed, skipping (%s)",
                    chunk_idx,
                    out_dir,
                )
                n_skipped += 1
                continue
            _wipe_partial_pack(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            input_path = ",".join(str(task_dir / ep) for ep in chunk_eps)
            stats_path = out_dir / "pack_stats.json"
            log_path = out_dir / "pack.log"
            try:
                rc, dt = _run_chunk_subprocess(
                    input_path, out_dir, args, stats_path, log_path
                )
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "  chunk_%03d: subprocess launch failed: %s",
                    chunk_idx,
                    e,
                )
                n_failed += 1
                if args.stop_on_failure:
                    return 1
                continue

            kept, err = _read_pack_stats(stats_path)
            # rc==0 is necessary but not sufficient: we've historically hit
            # runs where 100% of episodes returned error="mcap deps missing"
            # while the packer still exited cleanly. Also verify
            # pack_stats.json shows kept>0 (or every episode was a legit
            # filter skip, not a hard error).
            if rc == 0 and kept is not None and err > 0 and kept == 0:
                n_failed += 1
                logger.error(
                    "  chunk_%03d: SILENT FAILURE (rc=0 but all %d "
                    "episodes returned error). See %s",
                    chunk_idx,
                    err,
                    log_path,
                )
                if args.stop_on_failure:
                    return 1
            elif rc == 0:
                n_ok += 1
                logger.info(
                    "  chunk_%03d: OK (%d eps, %.1fs, kept=%s err=%s) -> %s",
                    chunk_idx,
                    len(chunk_eps),
                    dt,
                    kept,
                    err,
                    out_dir,
                )
            else:
                n_failed += 1
                logger.error(
                    "  chunk_%03d: FAILED rc=%d (%d eps, %.1fs); log=%s",
                    chunk_idx,
                    rc,
                    len(chunk_eps),
                    dt,
                    log_path,
                )
                if args.stop_on_failure:
                    return 1

    dt = time.perf_counter() - t_start
    logger.info(
        "Done in %.1fs. chunks: ok=%d skipped=%d failed=%d "
        "(this shard planned=%d)",
        dt,
        n_ok,
        n_skipped,
        n_failed,
        total_chunks,
    )
    return 0 if n_failed == 0 else 2


# --------------------------------------------------------------------------- #
# Single-pack mode: run the underlying ABC130kMPLmdbPacker in-process.
# This is what each orchestrator-spawned subprocess lands in.
# --------------------------------------------------------------------------- #
def _run_single_pack(args):
    os.makedirs(args.output_path, exist_ok=True)
    packer = ABC130kMPLmdbPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        commit_step=args.commit_step,
        num_workers=args.num_workers,
        num_steps_per_shard=args.num_steps_per_shard,
        stats_path=args.stats_path,
        max_episodes=args.max_episodes,
        max_episodes_per_task=args.max_episodes_per_task,
        scandir_threads=args.scandir_threads,
        joint_stats_subsample=args.joint_stats_subsample,
        joint_stats_bins=args.joint_stats_bins,
        # Empty string keeps the "no FK" behaviour parity with the old
        # CLI. Default (None) falls back to build_parser's baked-in URDF.
        urdf_path=args.urdf_path if args.urdf_path else None,
        static_threshold=args.static_threshold,
        head_time_to_filter=args.head_time_to_filter,
        tile_time_to_filter=args.tile_time_to_filter,
    )
    packer()
    return 0


# --------------------------------------------------------------------------- #
# Unified argparse — reuses _build_pack_parser() (single-pack flags:
# state_dim, commit_step, urdf_path, ...) so per-chunk args stay in one
# place; then layers the orchestrator-only args on top with a mutually
# exclusive source group.
# --------------------------------------------------------------------------- #
def build_parser():
    # Start from the single-pack parser so every flag it exposes is
    # accepted here verbatim. Then override ``--input_path`` to no longer
    # be required — orchestrator mode uses ``--data_root`` instead.
    parser = _build_pack_parser()
    for act in parser._actions:  # noqa: SLF001 — intentional argparse mutate
        if getattr(act, "dest", None) == "input_path":
            act.required = False
            break
    # ``--output_path`` is also only required in single-pack mode; make
    # it optional here and validate below once we know the mode.
    for act in parser._actions:  # noqa: SLF001
        if getattr(act, "dest", None) == "output_path":
            act.required = False
            break

    orch = parser.add_argument_group(
        "orchestrator",
        "Task/chunk fan-out (mutually exclusive with --input_path).",
    )
    src = orch.add_mutually_exclusive_group()
    src.add_argument(
        "--data_root",
        help=(
            "Dataset root that contains ``<task>/episode_*/`` directly. "
            "Walk it and pack every task. Preferred when the copy is "
            "complete — no completeness JSON needed."
        ),
    )
    src.add_argument(
        "--complete_tasks_json",
        help=(
            "Path to select_complete_tasks.py output. Use this when the "
            "candidate root is only partially copied and you want to pack "
            "only the tasks whose episode set matches the reference."
        ),
    )
    orch.add_argument(
        "--candidate_root",
        help=(
            "Only used with --complete_tasks_json: root that holds the "
            "episode dirs (should match candidate_root in the JSON, but "
            "pass explicitly so we don't chase a stale reference path)."
        ),
    )
    orch.add_argument(
        "--output_root",
        help="Chunks land at <output_root>/<task>/chunk_XXX/.",
    )
    orch.add_argument(
        "--episodes_per_lmdb",
        type=int,
        default=200,
        help="How many episodes per chunk. Set to 0 for one LMDB per task.",
    )
    orch.add_argument(
        "--task_names",
        nargs="*",
        default=None,
        help="Optional whitelist of task names.",
    )
    orch.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Optional cap on number of tasks to pack (debug).",
    )
    orch.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help=(
            "Split the sorted task list into N contiguous slices so that "
            "N cluster jobs can pack in parallel across machines. Task-"
            "level slicing means every chunk of a given task ends up on "
            "the same job. Sizes may vary between shards because task "
            "episode counts vary."
        ),
    )
    orch.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Which shard [0, num_shards) this invocation handles.",
    )
    orch.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a chunk if its output dir already looks packed.",
    )
    orch.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the subprocess plan and exit without packing.",
    )
    orch.add_argument(
        "--stop_on_failure",
        action="store_true",
        help=(
            "Abort after the first chunk that returns a non-zero exit "
            "code. Default is to log and continue so a single bad chunk "
            "doesn't kill the whole night."
        ),
    )
    return parser


def _decide_mode(args):
    single = bool(args.input_path)
    orchestrator = bool(args.data_root or args.complete_tasks_json)
    if single and orchestrator:
        raise SystemExit(
            "--input_path (single pack) and --data_root / "
            "--complete_tasks_json (orchestrator) are mutually exclusive."
        )
    if not single and not orchestrator:
        raise SystemExit(
            "Pass either --input_path (single-pack mode) or one of "
            "--data_root / --complete_tasks_json (orchestrator mode)."
        )
    if single and not args.output_path:
        raise SystemExit("--output_path is required in single-pack mode.")
    if orchestrator and not args.output_root:
        raise SystemExit("--output_root is required in orchestrator mode.")
    return "single" if single else "orchestrator"


def main():
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    mode = _decide_mode(args)
    if mode == "single":
        return _run_single_pack(args)
    return _orchestrate(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
