# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

"""Multi-worker ABC130k MCAP -> LMDB packer with filtering and stats.

Workers parse MCAP, validate calibration, JPEG-encode frames, and return a
payload dict. The main process writes payloads to LMDB sequentially (LMDB
writers are not multi-process safe). Filter reasons are accumulated and
reported at the end.

Filters:
  - zedx_station: episode is from a ZED-X bimanual station (4 camera streams,
    no /top-camera). MJCF wrist extrinsics in cameras.yaml are only valid for
    RealSense D405 wrist mounts, so ZED-X is unusable until we get a
    per-station calibration.
  - bad_intrinsic: a camera's K can't be reconciled to its saved image size
    (cx=0, fx=0, or correct_k_to_image_size can't snap to a known D405 native
    mode). The episode is skipped rather than silently producing wrong
    projections.
  - missing_state: state/action streams are absent or empty.
  - no_top_camera: no top camera stream of any kind.
  - decode_error: video decode or other unexpected error.
"""

import argparse
import concurrent.futures as cf
import glob
import logging
import math
import multiprocessing as mp
import os
import subprocess
import tempfile
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from robo_orchard_lab.dataset.abc130k.abc130k_export_lmdb_packer import (
    ABC_MAIN_T_WORLD_CAMERA,
    CALIB_TOPICS,
    CAMERAS,
    D405_NATIVE_HEIGHTS,
    D405_NATIVE_WIDTHS,
    STATE_TOPICS,
    T_MUJOCO_CAM_TO_CV_CAM,
    TICK_NS,
    TOP_TOPIC_CANDIDATES,
    X264,
    ABC130kEpisode,
    ABC130kExtrinsicsFK,
    correct_intrinsics_dict,
    probe,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

logger = logging.getLogger(__name__)


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


def parse_episode(episode_dir_str, state_dim=14, action_dim=14):
    """Top-level pickleable worker function.

    Returns one of:
      {"status": "ok", "payload": {...}, "episode_name": str}
      {"status": "skip", "reason": str, "episode_name": str}
      {"status": "error", "error": str, "tb": str, "episode_name": str}

    All results are tagged with the worker PID and wall-clock duration so
    the main process can log parallelism.
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
        **kwargs,
    ):
        super().__init__(input_path, output_path, **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_workers = max(1, int(num_workers))
        self.num_steps_per_shard = num_steps_per_shard
        self.stats_path = stats_path
        self.joint_stats_bins = int(joint_stats_bins)
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
            (d, self.state_dim, self.action_dim) for d in episode_dirs
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
                consume(parse_episode(*args), [i])
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
    episode_dir_str, state_dim, action_dim = args
    return parse_episode(episode_dir_str, state_dim, action_dim)


def build_parser():
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
    return parser


def main():
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
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
        urdf_path=args.urdf_path or None,
    )
    packer()


if __name__ == "__main__":
    main()
