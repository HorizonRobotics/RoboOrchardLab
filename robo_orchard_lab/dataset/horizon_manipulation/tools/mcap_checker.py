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

import copy
import glob
import json
import logging
import multiprocessing
import os
import subprocess

import cv2
import numpy as np
import pytorch_kinematics as pk
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation

import robo_orchard_lab.dataset.horizon_manipulation.tools.utils as _utils
from robo_orchard_lab.dataset.horizon_manipulation.tools.check_config import (
    build_default_inspect_config,
    load_inspect_config,
)
from robo_orchard_lab.dataset.horizon_manipulation.tools.check_models import (
    EpisodeData,
    EpisodeReport,
)
from robo_orchard_lab.dataset.horizon_manipulation.tools.check_rules import (
    evaluate_episode_rules,
)
from robo_orchard_lab.dataset.horizon_manipulation.tools.utils import (
    FfmpegVideoWriter,
    build_job_summary,
    build_report_row,
    concat_videos,
    count_topics,
    format_numeric_detail_table,
    format_rule_detail_lines,
    format_text_detail_table,
    format_time,
    format_timestamp,
    format_topic_summary_line,
    get_frequency,
    iter_logged_topics,
    iter_rule_results,
    pose_to_mat,
    sample,
    topic_sort_key,
    write_error_logs,
    write_full_log_episode,
    write_manual_review_artifacts,
    write_mcap_lists,
    write_rule_block,
)
from robo_orchard_lab.dataset.horizon_manipulation.utils import (
    decode_depth,
    decode_img,
    depth_visualize,
)

logger = logging.getLogger(__name__)

_default_get_h264_encoder = _utils.get_h264_encoder


def get_h264_encoder() -> str:
    return _default_get_h264_encoder()


def _popen_proxy(*args, **kwargs):
    return subprocess.Popen(*args, **kwargs)


def _run_proxy(*args, **kwargs):
    return subprocess.run(*args, **kwargs)


class _SubprocessProxy:
    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL
    Popen = staticmethod(_popen_proxy)
    run = staticmethod(_run_proxy)


_utils.get_h264_encoder = lambda: get_h264_encoder()
_utils.subprocess = _SubprocessProxy()


class PiperMcapChecker:
    """Parse episodes, evaluate rules, and optionally export videos."""

    def __init__(
        self,
        input_path,
        output_path,
        urdf,
        calibration_dict=None,
        task_names=None,
        user_names=None,
        static_threshold: float = 0,
        head_time_to_filter: float = 1e8,
        tile_time_to_filter: float = None,
        date_prefix: str = None,
        num_workers: int = 4,
        vis_interval: int = 3,
        vis_fps: int = 150,
        vis_resize_scale: float = 4,
        vis_depth: bool = True,
        inspect_config=None,
        inspect_config_path: str | None = None,
        enable_ffmpeg_log: bool = False,
        **kwargs,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.calibration_dict = self.calibration_process(calibration_dict)
        self.task_names = task_names
        self.user_names = user_names
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.static_threshold = static_threshold
        self.head_time_to_filter = head_time_to_filter
        self.tile_time_to_filter = tile_time_to_filter
        if date_prefix is not None:
            self.date_prefix = date_prefix.strip().split(",")
        else:
            self.date_prefix = date_prefix
        self.episodes = self.input_path_handler(input_path)
        self.num_workers = num_workers
        self.vis_interval = vis_interval
        self.vis_fps = vis_fps
        self.vis_resize_scale = vis_resize_scale
        self.vis_depth = vis_depth
        if inspect_config is None:
            inspect_config = load_inspect_config(inspect_config_path)
        self.inspect_config = inspect_config
        self.inspect_config_path = inspect_config_path
        self.enable_ffmpeg_log = enable_ffmpeg_log

    def calibration_process(self, calibration_dict):
        """Normalize camera extrinsics into camera-to-world matrices.

        Args:
            calibration_dict: Raw calibration payload keyed by camera name.

        Returns:
            dict | None: Normalized calibration payload.
        """

        if calibration_dict is None:
            return None
        for camera, calib in calibration_dict.items():
            calibration_dict[camera] = np.linalg.inv(pose_to_mat(calib))
        return calibration_dict

    def get_recorded_joint_limits(
        self,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return URDF joint limits aligned with the 14-D recorded joints.

        Returns:
            tuple[list[str], np.ndarray, np.ndarray]:
                Recorded joint names, lower limits, and upper limits.
        """

        cached = getattr(self, "_recorded_joint_limit_cache", None)
        if cached is not None:
            return cached

        joint_names = list(
            self.chain.get_joint_parameter_names(exclude_fixed=True)
        )
        joint_lower_limits, joint_upper_limits = self.chain.get_joint_limits()
        lower = np.asarray(joint_lower_limits, dtype=np.float64)
        upper = np.asarray(joint_upper_limits, dtype=np.float64)
        if len(joint_names) != 16 or lower.shape != (16,) or upper.shape != (
            16,
        ):
            raise ValueError(
                "Expected 16 non-fixed URDF joints to align with Piper "
                "recorded joint layout"
            )

        def _collapse_gripper_pair(
            left_index: int, right_index: int
        ) -> tuple[str, float, float]:
            pair_lower = max(
                2.0 * float(lower[left_index]),
                -2.0 * float(upper[right_index]),
            )
            pair_upper = min(
                2.0 * float(upper[left_index]),
                -2.0 * float(lower[right_index]),
            )
            if pair_lower > pair_upper:
                raise ValueError(
                    "Failed to derive a valid recorded gripper range from "
                    "URDF joint limits"
                )
            pair_name = f"{joint_names[left_index]}|{joint_names[right_index]}"
            return pair_name, pair_lower, pair_upper

        left_gripper_name, left_gripper_lower, left_gripper_upper = (
            _collapse_gripper_pair(6, 7)
        )
        right_gripper_name, right_gripper_lower, right_gripper_upper = (
            _collapse_gripper_pair(14, 15)
        )

        recorded_joint_names = (
            joint_names[:6]
            + [left_gripper_name]
            + joint_names[8:14]
            + [right_gripper_name]
        )
        recorded_joint_lower_limits = np.array(
            [
                *lower[:6].tolist(),
                left_gripper_lower,
                *lower[8:14].tolist(),
                right_gripper_lower,
            ],
            dtype=np.float64,
        )
        recorded_joint_upper_limits = np.array(
            [
                *upper[:6].tolist(),
                left_gripper_upper,
                *upper[8:14].tolist(),
                right_gripper_upper,
            ],
            dtype=np.float64,
        )

        self._recorded_joint_limit_cache = (
            recorded_joint_names,
            recorded_joint_lower_limits,
            recorded_joint_upper_limits,
        )
        return self._recorded_joint_limit_cache

    def input_path_handler(self, input_paths):
        """Expand configured roots into a sorted list of candidate episodes.

        Args:
            input_paths: Comma-separated episode root directories.

        Returns:
            list: Sorted episode descriptors consumed by the checker.
        """

        episodes = []
        input_paths = input_paths.strip().split(",")
        logger.info(f"input_paths: {input_paths}")
        for input_path in input_paths:
            for user in os.listdir(input_path):
                if self.user_names is not None and user not in self.user_names:
                    continue

                user_path = os.path.join(input_path, user)
                for task in os.listdir(user_path):
                    if (
                        self.task_names is not None
                        and task not in self.task_names
                    ):
                        continue
                    task_path = os.path.join(user_path, task)
                    for ep in os.listdir(task_path):
                        if not ep.startswith("episode_"):
                            logger.warning(f"invalid episode dir name {ep}")
                            continue
                        path = os.path.join(task_path, ep)
                        time = ep.replace("episode_", "")

                        if self.date_prefix is not None and not any(
                            [time.startswith(x) for x in self.date_prefix]
                        ):
                            continue
                        episodes.append([path, user, task, time])
        episodes.sort()
        logger.info(f"number of valid episodes: {len(episodes)}")
        return episodes

    def forward_kinematics(self, left_joint, right_joint):
        """Run FK for the dual-arm robot state used in inspection.

        Args:
            left_joint: Left arm joint trajectory.
            right_joint: Right arm joint trajectory.

        Returns:
            dict: Link poses returned by the kinematics chain.
        """

        joint = np.zeros([left_joint.shape[0], 16])
        if left_joint is not None:
            joint[:, :6] = left_joint[:, :6]
            joint[:, 6:7] = left_joint[:, 6:7] / 2
            joint[:, 7:8] = -left_joint[:, 6:7] / 2
        if right_joint is not None:
            joint[:, 8:14] = right_joint[:, :6]
            joint[:, 14:15] = right_joint[:, 6:7] / 2
            joint[:, 15:16] = -right_joint[:, 6:7] / 2
        link_poses_dict = self.chain.forward_kinematics(joint)
        return link_poses_dict

    def parse(self, ep_id, payload_mode="inspect"):
        """Parse one episode and build the raw payload used by inspection.

        Args:
            ep_id: Episode index in ``self.episodes``.
            payload_mode: ``inspect`` omits raw visual payloads, while
                ``render`` keeps them for video export.

        Returns:
            dict: Raw episode payload used by inspection and rendering.
        """
        keep_visual_payload = payload_mode == "render"

        cameras = ["left", "middle", "right"]
        image_topics = [
            f"/observation/cameras/{x}/color_image/image_raw" for x in cameras
        ]
        image_intrinsic_topics = [
            f"/observation/cameras/{x}/color_image/camera_info"
            for x in cameras
        ]
        depth_topics = [
            f"/observation/cameras/{x}/depth_image/image_raw" for x in cameras
        ]
        depth_intrinsic_topics = [
            f"/observation/cameras/{x}/depth_image/camera_info"
            for x in cameras
        ]
        image_extrinsic_topic = ["/tf_static"]
        left_joint_topic = "/observation/robot_state/left/joint"
        right_joint_topic = "/observation/robot_state/right/joint"
        left_master_joint_topic = "/observation/robot_state/left_master/joint"
        right_master_joint_topic = (
            "/observation/robot_state/right_master/joint"
        )
        left_ee_topic = "/observation/robot_state/left/end_pose"
        right_ee_topic = "/observation/robot_state/right/end_pose"

        frame_id_extrinsic_map = {
            "middle_camera_color_optical_frame/left_base_link": "middle",
            "left_camera_color_optical_frame/left_end_effector": "left",
            "right_camera_color_optical_frame/right_end_effector": "right",
        }

        all_useful_topics = (
            image_topics
            + image_intrinsic_topics
            + depth_topics
            + depth_intrinsic_topics
            + image_extrinsic_topic
            + [
                left_joint_topic,
                right_joint_topic,
                left_ee_topic,
                right_ee_topic,
            ]
            + [left_master_joint_topic, right_master_joint_topic]
        )

        ep = self.episodes[ep_id]
        path, user, task_name, time = ep

        uuid = f"{task_name}/{user}/{time}"
        mcap = glob.glob(f"{path}/episode_*_0.mcap")
        assert len(mcap) == 1
        mcap = mcap[0]
        meta = json.load(open(os.path.join(path, "episode_meta.json"), "r"))
        assert user == meta["user_name"]
        assert task_name == meta["task_name"]

        image_extrinsic = {}
        images = {}
        for t in image_topics:
            images[t] = {
                "time": [],
                "intrinsic": None,
            }
            if keep_visual_payload:
                images[t]["data"] = []
        image_intrinsics = {}
        for t in image_intrinsic_topics:
            image_intrinsics[t] = {
                "time": [],
            }
        depths = {}
        for t in depth_topics:
            depths[t] = {
                "time": [],
                "intrinsic": None,
            }
            if keep_visual_payload:
                depths[t]["data"] = []
        depth_intrinsics = {}
        for t in depth_intrinsic_topics:
            depth_intrinsics[t] = {
                "time": [],
            }
        joints = {}
        for t in [
            left_joint_topic,
            right_joint_topic,
            left_master_joint_topic,
            right_master_joint_topic,
        ]:
            joints[t] = {
                "position": [],
                "velocity": [],
                "effort": [],
                "time": [],
            }
        ee_pose_streams = {}
        for t in [left_ee_topic, right_ee_topic]:
            ee_pose_streams[t] = {
                "pose": [],
                "time": [],
            }
        observed_topics: set[str] = set()
        topic_summaries = {}
        alignment_time_diff_stats = {}

        reader = make_reader(
            open(mcap, "rb"), decoder_factories=[DecoderFactory()]
        )

        for (
            _schema,
            channel,
            _message,
            ros_msg,
        ) in reader.iter_decoded_messages(
            topics=all_useful_topics, log_time_order=True
        ):
            if hasattr(ros_msg, "header"):
                time = (
                    ros_msg.header.stamp.sec,
                    ros_msg.header.stamp.nanosec,
                )
            topic = channel.topic
            if topic != "/tf_static":
                observed_topics.add(topic)
            if topic != "/tf_static":
                if hasattr(ros_msg, "header"):
                    summary_time_ns = (
                        ros_msg.header.stamp.sec * int(1e9)
                        + ros_msg.header.stamp.nanosec
                    )
                else:
                    summary_time_ns = _message.log_time
                summary = topic_summaries.setdefault(
                    topic,
                    {
                        "count": 0,
                        "size_bytes": 0,
                        "start_time_ns": None,
                        "end_time_ns": None,
                        "mean_frequency": 0.0,
                        "min_frequency": 0.0,
                    },
                )
                summary["count"] += 1
                summary["size_bytes"] += len(_message.data)
                if summary["start_time_ns"] is None:
                    summary["start_time_ns"] = summary_time_ns
                    summary["end_time_ns"] = summary_time_ns
                else:
                    summary["start_time_ns"] = min(
                        summary["start_time_ns"], summary_time_ns
                    )
                    summary["end_time_ns"] = max(
                        summary["end_time_ns"], summary_time_ns
                    )
            if topic in images:
                if keep_visual_payload:
                    images[topic]["data"].append(ros_msg.data)
                images[topic]["time"].append(time)
            elif topic in depths:
                if keep_visual_payload:
                    depths[topic]["data"].append(ros_msg.data)
                depths[topic]["time"].append(time)
            elif topic in joints:
                joints[topic]["position"].append(ros_msg.position)
                joints[topic]["velocity"].append(ros_msg.velocity)
                joints[topic]["effort"].append(ros_msg.effort)
                joints[topic]["time"].append(time)
            elif topic in ee_pose_streams:
                pos = ros_msg.pose.position
                rot = ros_msg.pose.orientation
                x, y, z = pos.x, pos.y, pos.z
                qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
                ee_pose_streams[topic]["pose"].append(
                    [x, y, z, qx, qy, qz, qw]
                )
                ee_pose_streams[topic]["time"].append(time)
            elif topic in image_intrinsic_topics:
                image_intrinsics[topic]["time"].append(time)
                img_topic = image_topics[image_intrinsic_topics.index(topic)]
                images[img_topic]["intrinsic"] = np.array(ros_msg.p).reshape(
                    3, 4
                )
            elif topic in depth_intrinsic_topics:
                depth_intrinsics[topic]["time"].append(time)
                dpt_topic = depth_topics[depth_intrinsic_topics.index(topic)]
                depths[dpt_topic]["intrinsic"] = np.array(ros_msg.p).reshape(
                    3, 4
                )
            elif topic in image_extrinsic_topic:
                for tf in ros_msg.transforms:
                    frame_id = f"{tf.child_frame_id}/{tf.header.frame_id}"
                    if frame_id in frame_id_extrinsic_map:
                        camera_name = frame_id_extrinsic_map[frame_id]
                        extrinsic = np.linalg.inv(pose_to_mat(tf.transform))
                        image_extrinsic[camera_name] = extrinsic

        for data in [images, depths, joints, ee_pose_streams]:
            for t in data:
                data[t]["time"] = format_time(data[t]["time"])
        for data in [image_intrinsics, depth_intrinsics]:
            for t in data:
                if data[t]["time"]:
                    data[t]["time"] = format_time(data[t]["time"])
                else:
                    data[t]["time"] = np.array([], dtype="float64")

        def _update_topic_frequency(topic: str, freq: np.ndarray) -> None:
            summary = topic_summaries.setdefault(
                topic,
                {
                    "count": 0,
                    "size_bytes": 0,
                    "start_time_ns": None,
                    "end_time_ns": None,
                    "mean_frequency": 0.0,
                    "min_frequency": 0.0,
                },
            )
            if freq.size > 0:
                summary["mean_frequency"] = float(freq.mean())
                summary["min_frequency"] = float(freq.min())
            else:
                summary["mean_frequency"] = 0.0
                summary["min_frequency"] = 0.0

        raw_topic_counts = {
            topic: len(payload["time"]) for topic, payload in images.items()
        }
        raw_topic_counts.update(
            {topic: len(payload["time"]) for topic, payload in depths.items()}
        )
        raw_topic_counts.update(
            {
                topic: len(payload["position"])
                for topic, payload in joints.items()
            }
        )
        raw_topic_counts.update(
            {
                topic: len(payload["pose"])
                for topic, payload in ee_pose_streams.items()
            }
        )

        base_time = images[image_topics[0]]["time"]
        all_steps = len(base_time)

        def _empty_aligned_time() -> np.ndarray:
            return np.array([], dtype="float64")

        def _aligned_time_or_base(
            src_time: np.ndarray, topic: str
        ) -> np.ndarray:
            if base_time.size == 0:
                return _empty_aligned_time()
            if src_time.size == 0:
                return np.copy(base_time)
            return sample(base_time, src_time, prefix=uuid + " | " + topic)

        def _default_joint_series(step_count: int) -> list[list[float]]:
            return [[0.0] * 7 for _ in range(step_count)]

        def _default_pose_series(step_count: int) -> list[list[float]]:
            return [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                for _ in range(step_count)
            ]
        for t in images:
            images[t]["freq"] = get_frequency(
                images[t]["time"], uuid + " | " + t
            )
            _update_topic_frequency(t, images[t]["freq"])
            if t == image_topics[0]:
                continue
            if "data" in images[t]:
                if base_time.size > 0 and images[t]["time"].size == 0:
                    raise RuntimeError(
                        f"{uuid} | required render stream is empty: {t}"
                    )
                images[t]["time"], (images[t]["data"],) = sample(
                    base_time,
                    images[t]["time"],
                    [images[t]["data"]],
                    prefix=uuid + " | " + t,
                )
            else:
                images[t]["time"] = _aligned_time_or_base(images[t]["time"], t)
            time_diff = np.abs(base_time - images[t]["time"])
            alignment_time_diff_stats[t] = {
                "max_time_diff": float(time_diff.max())
                if time_diff.size
                else 0.0,
                "mean_time_diff": float(time_diff.mean())
                if time_diff.size
                else 0.0,
            }
        for t in depths:
            depths[t]["freq"] = get_frequency(
                depths[t]["time"], uuid + " | " + t
            )
            _update_topic_frequency(t, depths[t]["freq"])
            if "data" in depths[t]:
                if base_time.size > 0 and depths[t]["time"].size == 0:
                    raise RuntimeError(
                        f"{uuid} | required render stream is empty: {t}"
                    )
                depths[t]["time"], (depths[t]["data"],) = sample(
                    base_time,
                    depths[t]["time"],
                    [depths[t]["data"]],
                    prefix=uuid + " | " + t,
                )
            else:
                depths[t]["time"] = _aligned_time_or_base(depths[t]["time"], t)
            time_diff = np.abs(base_time - depths[t]["time"])
            alignment_time_diff_stats[t] = {
                "max_time_diff": float(time_diff.max())
                if time_diff.size
                else 0.0,
                "mean_time_diff": float(time_diff.mean())
                if time_diff.size
                else 0.0,
            }
        for t in joints:
            joints[t]["freq"] = get_frequency(
                joints[t]["time"], uuid + " | " + t
            )
            _update_topic_frequency(t, joints[t]["freq"])
            if base_time.size == 0:
                joints[t]["time"] = _empty_aligned_time()
                joints[t]["position"] = []
                joints[t]["velocity"] = []
                joints[t]["effort"] = []
            elif joints[t]["time"].size == 0:
                joints[t]["time"] = np.copy(base_time)
                joints[t]["position"] = _default_joint_series(len(base_time))
                joints[t]["velocity"] = _default_joint_series(len(base_time))
                joints[t]["effort"] = _default_joint_series(len(base_time))
            else:
                (
                    joints[t]["time"],
                    (
                        joints[t]["position"],
                        joints[t]["velocity"],
                        joints[t]["effort"],
                    ),
                ) = sample(
                    base_time,
                    joints[t]["time"],
                    [
                        joints[t]["position"],
                        joints[t]["velocity"],
                        joints[t]["effort"],
                    ],
                    prefix=uuid + " | " + t,
                )
            time_diff = np.abs(base_time - joints[t]["time"])
            alignment_time_diff_stats[t] = {
                "max_time_diff": float(time_diff.max())
                if time_diff.size
                else 0.0,
                "mean_time_diff": float(time_diff.mean())
                if time_diff.size
                else 0.0,
            }
            joints[t]["position"] = np.asarray(
                joints[t]["position"], dtype="float64"
            ).reshape(-1, 7)
            joints[t]["velocity"] = np.asarray(
                joints[t]["velocity"], dtype="float64"
            ).reshape(-1, 7)
            joints[t]["effort"] = np.asarray(
                joints[t]["effort"], dtype="float64"
            ).reshape(-1, 7)

        if joints[left_joint_topic]["position"].shape[0] == 0:
            left_ee_pose = np.repeat(np.eye(4)[None], 0, axis=0)
            right_ee_pose = np.repeat(np.eye(4)[None], 0, axis=0)
        else:
            poses = self.forward_kinematics(
                joints[left_joint_topic]["position"],
                joints[right_joint_topic]["position"],
            )
            left_ee_pose = poses["left_gripper_base"].get_matrix().numpy()
            right_ee_pose = poses["right_gripper_base"].get_matrix().numpy()

        for t in ee_pose_streams:
            ee_pose_streams[t]["freq"] = get_frequency(
                ee_pose_streams[t]["time"], uuid + " | " + t
            )
            _update_topic_frequency(t, ee_pose_streams[t]["freq"])
            if base_time.size == 0:
                ee_pose_streams[t]["time"] = _empty_aligned_time()
                ee_pose_streams[t]["pose"] = []
            elif ee_pose_streams[t]["time"].size == 0:
                ee_pose_streams[t]["time"] = np.copy(base_time)
                ee_pose_streams[t]["pose"] = _default_pose_series(
                    len(base_time)
                )
            else:
                (
                    ee_pose_streams[t]["time"],
                    (ee_pose_streams[t]["pose"],),
                ) = sample(
                    base_time,
                    ee_pose_streams[t]["time"],
                    [ee_pose_streams[t]["pose"]],
                    prefix=uuid + " | " + t,
                )
            time_diff = np.abs(base_time - ee_pose_streams[t]["time"])
            alignment_time_diff_stats[t] = {
                "max_time_diff": float(time_diff.max())
                if time_diff.size
                else 0.0,
                "mean_time_diff": float(time_diff.mean())
                if time_diff.size
                else 0.0,
            }
        for t in image_intrinsics:
            freq = get_frequency(image_intrinsics[t]["time"], uuid + " | " + t)
            _update_topic_frequency(t, freq)
        for t in depth_intrinsics:
            freq = get_frequency(depth_intrinsics[t]["time"], uuid + " | " + t)
            _update_topic_frequency(t, freq)
        if base_time.size == 0:
            recorded_ee_poses = np.zeros((0, 2, 7), dtype="float64")
        elif (
            raw_topic_counts[left_ee_topic] == 0
            or raw_topic_counts[right_ee_topic] == 0
        ):
            recorded_ee_poses = None
        else:
            recorded_ee_poses = np.concatenate(
                [
                    np.array(ee_pose_streams[left_ee_topic]["pose"]),
                    np.array(ee_pose_streams[right_ee_topic]["pose"]),
                ],
                axis=1,
            ).reshape(-1, 2, 7)

        if self.calibration_dict is not None:
            image_extrinsic = copy.deepcopy(self.calibration_dict)
        extrinsic = {
            "left": image_extrinsic["left"] @ np.linalg.inv(left_ee_pose),
            "right": image_extrinsic["right"] @ np.linalg.inv(right_ee_pose),
            "middle": np.copy(image_extrinsic["middle"]),
        }

        intrinsic = {}
        for cam, t in zip(cameras, image_topics, strict=False):
            intrinsic[cam] = images[t]["intrinsic"]
        joint_positions = np.concatenate(
            [
                np.array(joints[left_joint_topic]["position"]),
                np.array(joints[right_joint_topic]["position"]),
            ],
            axis=-1,
        )
        joint_velocity = np.concatenate(
            [
                np.array(joints[left_joint_topic]["velocity"]),
                np.array(joints[right_joint_topic]["velocity"]),
            ],
            axis=-1,
        )
        joint_effort = np.concatenate(
            [
                np.array(joints[left_joint_topic]["effort"]),
                np.array(joints[right_joint_topic]["effort"]),
            ],
            axis=-1,
        )

        master_joint_positions = np.concatenate(
            [
                np.array(joints[left_master_joint_topic]["position"]),
                np.array(joints[right_master_joint_topic]["position"]),
            ],
            axis=-1,
        )
        master_joint_velocity = np.concatenate(
            [
                np.array(joints[left_master_joint_topic]["velocity"]),
                np.array(joints[right_master_joint_topic]["velocity"]),
            ],
            axis=-1,
        )
        master_joint_effort = np.concatenate(
            [
                np.array(joints[left_master_joint_topic]["effort"]),
                np.array(joints[right_master_joint_topic]["effort"]),
            ],
            axis=-1,
        )

        if left_ee_pose.shape[0] == 0:
            ee_poses = np.zeros((0, 2, 7), dtype="float64")
        else:
            ee_poses = np.concatenate(
                [
                    left_ee_pose[:, :3, 3],
                    Rotation.from_matrix(left_ee_pose[:, :3, :3]).as_quat(
                        scalar_first=False
                    ),
                    right_ee_pose[:, :3, 3],
                    Rotation.from_matrix(right_ee_pose[:, :3, :3]).as_quat(
                        scalar_first=False
                    ),
                ],
                axis=1,
            ).reshape(-1, 2, 7)

        # ==================== filter
        if self.static_threshold is not None and self.static_threshold > 0:
            static_mask = np.ones(joint_positions.shape[0], dtype=bool)
            static_mask[1:] = np.any(
                np.abs(np.diff(joint_positions, axis=0)) > 1e-3, axis=1
            )

            time_mask = np.zeros(static_mask.shape[0], bool)
            if (
                self.head_time_to_filter is not None
                and self.head_time_to_filter > 0
            ):
                head_time = base_time - base_time[0]
                time_mask = np.logical_or(
                    time_mask, head_time < self.head_time_to_filter
                )
            if (
                self.tile_time_to_filter is not None
                and self.tile_time_to_filter > 0
            ):
                tile_time = -base_time + base_time[-1]
                time_mask = np.logical_or(
                    time_mask, tile_time < self.tile_time_to_filter
                )
            static_mask = np.logical_or(static_mask, np.logical_not(time_mask))
            joint_positions = joint_positions[static_mask]
            joint_velocity = joint_velocity[static_mask]
            joint_effort = joint_effort[static_mask]
            master_joint_positions = master_joint_positions[static_mask]
            master_joint_velocity = master_joint_velocity[static_mask]
            master_joint_effort = master_joint_effort[static_mask]
            ee_poses = ee_poses[static_mask]
            if recorded_ee_poses is not None:
                recorded_ee_poses = recorded_ee_poses[static_mask]
            logger.info(
                f"{uuid} | all steps: {all_steps}, "
                f"non static steps: {joint_positions.shape[0]}"
            )

            extrinsic["left"] = extrinsic["left"][static_mask]
            extrinsic["right"] = extrinsic["right"][static_mask]
            base_time = base_time[static_mask]
            for t in image_topics:
                images[t]["time"] = images[t]["time"][static_mask]
                if "data" in images[t]:
                    images[t]["data"] = [
                        x
                        for i, x in enumerate(images[t]["data"])
                        if static_mask[i]
                    ]
            for t in depth_topics:
                depths[t]["time"] = depths[t]["time"][static_mask]
                if "data" in depths[t]:
                    depths[t]["data"] = [
                        x
                        for i, x in enumerate(depths[t]["data"])
                        if static_mask[i]
                    ]
            for t in ee_pose_streams:
                ee_pose_streams[t]["pose"] = [
                    x
                    for i, x in enumerate(ee_pose_streams[t]["pose"])
                    if static_mask[i]
                ]
                ee_pose_streams[t]["time"] = [
                    x
                    for i, x in enumerate(ee_pose_streams[t]["time"])
                    if static_mask[i]
                ]

        non_static_steps = len(base_time)
        # ==================== filter finish

        topic_counts = raw_topic_counts
        topic_frequencies = {
            topic: payload["freq"] for topic, payload in images.items()
        }
        topic_frequencies.update(
            {topic: payload["freq"] for topic, payload in depths.items()}
        )
        topic_frequencies.update(
            {topic: payload["freq"] for topic, payload in joints.items()}
        )
        topic_frequencies.update(
            {
                topic: payload["freq"]
                for topic, payload in ee_pose_streams.items()
            }
        )
        required_topics = (
            image_topics
            + depth_topics
            + [
                left_joint_topic,
                right_joint_topic,
                left_master_joint_topic,
                right_master_joint_topic,
                left_ee_topic,
                right_ee_topic,
            ]
        )
        (
            joint_limit_names,
            joint_lower_limits,
            joint_upper_limits,
        ) = self.get_recorded_joint_limits()

        return dict(
            cam_names=cameras,
            uuid=uuid,
            source_path=path,
            mcap_path=mcap,
            images=images,
            depths=depths,
            joint_positions=joint_positions,
            joint_velocity=joint_velocity,
            joint_effort=joint_effort,
            master_joint_positions=master_joint_positions,
            master_joint_velocity=master_joint_velocity,
            master_joint_effort=master_joint_effort,
            ee_poses=ee_poses,
            fk_ee_poses=ee_poses,
            recorded_ee_poses=recorded_ee_poses,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            meta=meta,
            base_time=base_time,
            all_steps=all_steps,
            non_static_steps=non_static_steps,
            topic_counts=topic_counts,
            topic_frequencies=topic_frequencies,
            required_topics=required_topics,
            observed_topics=observed_topics,
            joint_limit_names=joint_limit_names,
            joint_lower_limits=joint_lower_limits,
            joint_upper_limits=joint_upper_limits,
            topic_summaries=topic_summaries,
            alignment_time_diff_stats=alignment_time_diff_stats,
            static_filter_applied=(
                self.static_threshold is not None and self.static_threshold > 0
            ),
        )

    def build_episode_data(self, data) -> EpisodeData:
        """Convert the parser payload into the typed inspection model.

        Args:
            data: Raw episode payload produced by ``parse``.

        Returns:
            robo_orchard_lab.dataset.horizon_manipulation.tools.check_models.EpisodeData:
                Typed episode model for rule evaluation.
        """

        def _as_array(value):
            if value is None:
                return None
            return np.asarray(value)

        return EpisodeData(
            uuid=data["uuid"],
            meta=data["meta"],
            topic_counts=data.get("topic_counts", {}),
            topic_frequencies=data.get("topic_frequencies", {}),
            base_time=_as_array(data.get("base_time", [])),
            joint_positions=_as_array(data["joint_positions"]),
            master_joint_positions=_as_array(data["master_joint_positions"]),
            fk_ee_poses=_as_array(data.get("fk_ee_poses", data["ee_poses"])),
            recorded_ee_poses=_as_array(data.get("recorded_ee_poses")),
            images=data["images"],
            depths=data["depths"],
            required_topics=data.get("required_topics", []),
            observed_topics=set(data.get("observed_topics", set())),
            joint_limit_names=data.get("joint_limit_names", []),
            joint_lower_limits=_as_array(data.get("joint_lower_limits")),
            joint_upper_limits=_as_array(data.get("joint_upper_limits")),
            parse_warnings=data.get("parse_warnings", []),
            cam_names=data.get("cam_names", []),
            extrinsic=data.get("extrinsic", {}),
            intrinsic=data.get("intrinsic", {}),
            topic_summaries=data.get("topic_summaries", {}),
            alignment_time_diff_stats=data.get(
                "alignment_time_diff_stats", {}
            ),
            static_filter_applied=data.get("static_filter_applied", False),
        )

    def draw_ee_pose(self, img, intrinsic, extrinsic, ee_poses):
        """Project EE axes into the image for visualization output.

        Args:
            img: Target image.
            intrinsic: Camera intrinsic matrix.
            extrinsic: Camera extrinsic matrix.
            ee_poses: End-effector poses to project.

        Returns:
            np.ndarray: Image with EE pose overlays.
        """

        rot = Rotation.from_quat(
            ee_poses[:, 3:], scalar_first=True
        ).as_matrix()
        trans = ee_poses[:, :3]
        axis_length = 0.1
        points = np.float32(
            [
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length],
                [0, 0, 0],
            ]
        )
        # npose, 1, 3, 3 @ [4, 3, 3] + [npose, 1, 3]
        points = (rot[:, None] @ points[..., None]).squeeze(-1)
        points += trans[:, None]  # npose, 4, 3
        projection_mat = intrinsic @ extrinsic

        pts_2d = points @ projection_mat[:3, :3].T
        pts_2d = pts_2d + projection_mat[:3, 3]
        depth = pts_2d[..., 2]
        pts_2d = pts_2d[..., :2] / depth[..., None]
        pts_2d = pts_2d.astype(np.int32)

        for pose_idx in range(len(ee_poses)):
            for i in range(3):
                if depth[pose_idx, i] < 0.02:
                    continue
                cv2.circle(
                    img,
                    (pts_2d[pose_idx, i, 0], pts_2d[pose_idx, i, 1]),
                    6,
                    (0, 0, 255),
                    -1,
                )
                if i == 3:
                    continue
                color = [0, 0, 0]
                color[i] = 255
                cv2.line(
                    img,
                    (pts_2d[pose_idx, i, 0], pts_2d[pose_idx, i, 1]),
                    (pts_2d[pose_idx, 3, 0], pts_2d[pose_idx, 3, 1]),
                    tuple(color),
                    3,
                )
        return img

    def decode_and_visualize(
        self, data, episode_status: str = "pass", render_hints=None
    ):
        """Render one episode into a stitched visualization video.

        Args:
            data: Raw episode payload.
            episode_status: Aggregate inspection status for the episode.
            render_hints: Optional short inspection hints for rendering.

        Returns:
            str | None: Rendered video path when frames are written.
        """

        video_writer = None
        uuid = data["uuid"]
        os.makedirs(self.output_path, exist_ok=True)
        file = os.path.join(
            self.output_path, f"{uuid.replace(os.sep, '-')}.mp4"
        )
        num_steps = len(data["joint_positions"])
        if num_steps == 0:
            return None
        for i in range(0, num_steps, self.vis_interval):
            imgs = []
            depths = []
            ee_poses = data["ee_poses"][i]
            for cam in data["cam_names"]:
                image_topic = (
                    f"/observation/cameras/{cam}/color_image/image_raw"
                )
                depth_topic = (
                    f"/observation/cameras/{cam}/depth_image/image_raw"
                )
                extrinsic = data["extrinsic"][cam]
                if extrinsic.ndim == 3:
                    extrinsic = extrinsic[i]
                intrinsic = data["intrinsic"][cam]

                img = decode_img(data["images"][image_topic]["data"][i])
                img = self.draw_ee_pose(img, intrinsic, extrinsic, ee_poses)
                depth = decode_depth(
                    data["depths"][depth_topic]["data"][i], 1000
                )
                depth = depth_visualize(depth)

                imgs.append(img)
                depths.append(depth)

            shapes = [im.shape[:2] for im in imgs + depths]
            if not all(s == shapes[0] for s in shapes):
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                if i == 0:
                    logger.warning(
                        f"{uuid} | inconsistent image shapes detected, "
                        f"resizing all images to ({max_h},{max_w})"
                    )

                def _resize(im, size):
                    return cv2.resize(im, size, interpolation=cv2.INTER_AREA)

                imgs = [_resize(im, (max_w, max_h)) for im in imgs]
                depths = [_resize(im, (max_w, max_h)) for im in depths]

            if self.vis_depth:
                vis_img = np.concatenate(
                    [
                        np.concatenate(imgs, axis=1),
                        np.concatenate(depths, axis=1),
                    ],
                    axis=0,
                )
            else:
                vis_img = np.concatenate(imgs, axis=1)

            if self.vis_resize_scale is not None:
                size = (
                    vis_img.shape[1] // self.vis_resize_scale,
                    vis_img.shape[0] // self.vis_resize_scale,
                )
                vis_img = cv2.resize(
                    vis_img, size, interpolation=cv2.INTER_AREA
                )

            cv2.putText(
                vis_img,
                uuid,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(1 / self.vis_resize_scale, 0.8),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                vis_img,
                data["meta"]["instruction"],
                (10, vis_img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(1 / self.vis_resize_scale, 0.8),
                (0, 0, 255),
                2,
            )

            if video_writer is None:
                render_fps = max(1, round(self.vis_fps / self.vis_interval))
                video_writer = FfmpegVideoWriter(
                    file,
                    vis_img.shape[:2][::-1],
                    render_fps,
                    enable_ffmpeg_log=getattr(
                        self, "enable_ffmpeg_log", False
                    ),
                )
            video_writer.write(vis_img)
        if video_writer is not None:
            video_writer.release()
            return file
        return None

    def render_episode_video(self, data, inspection):
        """Render the visualization video for one inspected episode.

        Args:
            data: Raw episode payload.
            inspection: Aggregate inspection result.

        Returns:
            str | None: Rendered video path when export succeeds.
        """

        return self.decode_and_visualize(
            data,
            episode_status=inspection.episode_status,
            render_hints=inspection.render_hints,
        )

    def render_handler(self, item):
        """Re-parse one episode and export its visualization video.

        Args:
            item: Episode report payload carrying the episode index.

        Returns:
            dict: Video export update keyed by episode index.
        """

        ep_id = item["_episode_index"]
        uuid = item.get("uuid", self.episodes[ep_id][0])
        try:
            episode_raw = self.parse(ep_id, payload_mode="render")
            video_file = self.decode_and_visualize(
                episode_raw,
                episode_status=item.get("episode_status", "pass"),
                render_hints=item.get("render_hints", []),
            )
            return {
                "_episode_index": ep_id,
                "video_file": video_file,
            }
        except Exception as e:
            logger.exception(
                f"failed to render episode {ep_id} ({uuid}) with error: {e}"
            )
            return {
                "_episode_index": ep_id,
                "video_file": None,
                "runtime_error": str(e),
            }

    def render_videos(self, results):
        """Run the second-stage video export after rule checking completes.

        Args:
            results: Episode report payloads for the whole batch.
        """

        render_jobs = [
            item
            for item in results
            if item.get("_episode_index") is not None
            and not item.get("runtime_error")
        ]
        if not render_jobs:
            return

        if self.num_workers > 1:
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                render_updates = pool.map(self.render_handler, render_jobs)
        else:
            render_updates = [
                self.render_handler(item) for item in render_jobs
            ]

        updates_by_index = {
            item["_episode_index"]: item for item in render_updates
        }
        for item in results:
            episode_index = item.get("_episode_index")
            if episode_index is None:
                continue
            update = updates_by_index.get(episode_index)
            if update is None:
                continue
            item["video_file"] = update.get("video_file")
            if update.get("runtime_error"):
                item["video_runtime_error"] = update["runtime_error"]

    def _iter_rule_results(self, item):
        return iter_rule_results(item)

    def _count_topics(self, metric_value: str | None) -> int:
        return count_topics(metric_value)

    def _build_report_row(self, item):
        return build_report_row(item)

    def _write_mcap_lists(self, rows):
        return write_mcap_lists(self.output_path, rows)

    def _format_numeric_rule_detail(
        self, name: str, actual: float, limit: float
    ) -> str:
        return (
            f"{name} actual={actual:.6f}, "
            f"limit={limit:.6f}, gap={actual - limit:.6f}"
        )

    def _format_numeric_detail_table(
        self, label_header: str, rows: list[tuple[str, float, float, float]]
    ) -> list[str]:
        return format_numeric_detail_table(label_header, rows)

    def _format_text_detail_table(
        self, headers: tuple[str, ...], rows: list[tuple[str, ...]]
    ) -> list[str]:
        return format_text_detail_table(headers, rows)

    def _format_rule_detail_lines(self, rule: dict) -> list[str]:
        return format_rule_detail_lines(rule, self.inspect_config)

    def _write_rule_block(self, fh, rule: dict) -> None:
        return write_rule_block(fh, rule, self.inspect_config)

    def _format_timestamp(self, timestamp_ns: int | None) -> str:
        return format_timestamp(timestamp_ns)

    def _topic_sort_key(self, topic: str) -> tuple[int, str]:
        return topic_sort_key(topic)

    def _iter_logged_topics(
        self, topic_summaries: dict[str, dict]
    ) -> list[tuple[str, dict]]:
        return iter_logged_topics(topic_summaries)

    def _format_topic_summary_line(self, topic: str, summary: dict) -> str:
        return format_topic_summary_line(topic, summary)

    def _write_full_log_episode(self, fh, item: dict) -> None:
        return write_full_log_episode(fh, item, self.inspect_config)

    def _build_runtime_config(self) -> dict[str, object]:
        """Collect runtime options that should be printed in full logs.

        Returns:
            dict[str, object]: Serializable runtime configuration payload.
        """

        return {
            "input_path": getattr(self, "input_path", None),
            "output_path": getattr(self, "output_path", None),
            "task_names": getattr(self, "task_names", None),
            "user_names": getattr(self, "user_names", None),
            "date_prefix": getattr(self, "date_prefix", None),
            "static_threshold": getattr(self, "static_threshold", None),
            "head_time_to_filter": getattr(self, "head_time_to_filter", None),
            "tile_time_to_filter": getattr(self, "tile_time_to_filter", None),
            "num_workers": getattr(self, "num_workers", None),
            "vis_interval": getattr(self, "vis_interval", None),
            "vis_fps": getattr(self, "vis_fps", None),
            "vis_resize_scale": getattr(self, "vis_resize_scale", None),
            "vis_depth": getattr(self, "vis_depth", None),
            "skip_video_export": getattr(self, "skip_video_export", None),
            "enable_ffmpeg_log": getattr(self, "enable_ffmpeg_log", None),
            "inspect_config_path": getattr(self, "inspect_config_path", None),
        }

    def _write_error_logs(self, results):
        """Write the full log and the extracted signal log for one job.

        Args:
            results: Episode report payloads for the whole batch.
        """

        return write_error_logs(
            self.output_path,
            results,
            getattr(self, "inspect_config", None),
            self._build_runtime_config(),
        )

    def write_manual_review_artifacts(self, results):
        """Write review timeline and failure placeholders for one batch."""

        return write_manual_review_artifacts(
            self.output_path,
            results,
            self.vis_interval,
            self.vis_fps,
        )

    def handler(self, ep_id):
        """Parse and inspect one episode without exporting video.

        Args:
            ep_id: Episode index in ``self.episodes``.

        Returns:
            dict: Serializable episode report with logging payloads attached.
        """

        logger.info(f"start process [{ep_id + 1}/{len(self.episodes)}]")
        try:
            episode_raw = self.parse(ep_id, payload_mode="inspect")
            episode_data = self.build_episode_data(episode_raw)
            inspection = evaluate_episode_rules(
                episode_data, self.inspect_config
            )
            uuid = episode_data.uuid
            source_path = episode_raw.get("source_path")
            num_steps = len(episode_data.joint_positions)
            mcap_path = episode_raw.get("mcap_path", source_path)
            topic_summaries = episode_raw.get("topic_summaries", {})
            all_steps = episode_raw.get("all_steps", num_steps)
            non_static_steps = episode_raw.get("non_static_steps", num_steps)
            render_hints = inspection.render_hints
            report = EpisodeReport(
                uuid=uuid,
                episode_status=inspection.episode_status,
                rule_results=inspection.rule_results,
                episode_metrics=inspection.episode_metrics,
                video_file=None,
            )
        except Exception as e:
            uuid = self.episodes[ep_id][0]
            source_path = uuid
            logger.exception(
                f"failed to process episode {ep_id} with error: {e}"
            )
            report = EpisodeReport(
                uuid=uuid,
                episode_status="fail",
                rule_results=[],
                episode_metrics={"rule_hit_count": 0},
                video_file=None,
                runtime_error=str(e),
            )
            return {
                **report.to_dict(),
                "source_path": source_path,
                "mcap_path": source_path,
                "topic_summaries": {},
                "num_steps": 0,
                "all_steps": 0,
                "non_static_steps": 0,
                "render_hints": [],
                "_episode_index": ep_id,
            }
        else:
            logger.info(
                f"finish process [{ep_id + 1}/{len(self.episodes)}] {uuid}, "
                f"num_steps:{num_steps} \n"
            )
        del episode_data
        del episode_raw
        return {
            **report.to_dict(),
            "source_path": source_path,
            "mcap_path": mcap_path,
            "topic_summaries": topic_summaries,
            "render_hints": render_hints,
            "num_steps": num_steps,
            "all_steps": all_steps,
            "non_static_steps": non_static_steps,
            "_episode_index": ep_id,
        }

    def concat_videos(self, video_files):
        """Concatenate rendered episode videos into one mp4.

        Args:
            video_files: Individual episode videos to concatenate.
        """

        return concat_videos(
            video_files,
            self.output_path,
            enable_ffmpeg_log=getattr(self, "enable_ffmpeg_log", False),
        )

    def build_job_summary(self, results):
        """Build aggregate counters for a batch of episode reports.

        Args:
            results: Episode report payloads for the whole batch.

        Returns:
            dict: Batch-level summary counters and artifact paths.
        """

        return build_job_summary(results, self.output_path)

    def write_job_outputs(self, results):
        """Write batch-level log artifacts derived from inspection results.

        Args:
            results: Episode report payloads for the whole batch.
        """

        os.makedirs(self.output_path, exist_ok=True)
        rows = [self._build_report_row(item) for item in results]
        self._write_mcap_lists(rows)
        self._write_error_logs(results)

    def __call__(self):
        """Execute the full job: inspect first, then export videos if enabled.

        Returns:
            list[dict]: Sorted episode report payloads.
        """

        if self.num_workers > 1:
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                results = pool.map(
                    self.handler, list(range(len(self.episodes)))
                )
        else:
            results = []
            for i in range(len(self.episodes)):
                results.append(self.handler(i))

        results.sort(key=lambda x: x["uuid"])
        total_steps = sum([x.get("num_steps", 0) for x in results])

        logger.info(
            f"total steps: {total_steps}, "
            f"total times: {total_steps / 108000:.3f}h"
        )
        self.write_job_outputs(results)
        self.render_videos(results)
        self.concat_videos(
            [x["video_file"] for x in results if x.get("video_file")]
        )
        self.write_manual_review_artifacts(results)


if __name__ == "__main__":
    import argparse

    from robo_orchard_lab.utils import log_basic_config

    log_basic_config(
        format="%(asctime)s %(levelname)s-%(lineno)d: %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--urdf", type=str)
    parser.add_argument("--user_names", type=str, default=None)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--date_prefix", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embodiment_meta_file", type=str, default=None)
    parser.add_argument("--embodiment", type=str, default=None)
    parser.add_argument("--use_extra_calibration", action="store_true")
    parser.add_argument("--inspect_config", type=str, default=None)
    parser.add_argument("--enable_ffmpeg_log", action="store_true")
    args = parser.parse_args()

    calibration_dict = None
    if args.embodiment_meta_file is not None:
        if args.embodiment is None:
            parser.error(
                "--embodiment is required when "
                "--embodiment_meta_file is specified"
            )
        with open(args.embodiment_meta_file, "r") as f:
            embodiment_meta = json.load(f)
        embodiment_meta = embodiment_meta[args.embodiment]
        if args.use_extra_calibration:
            calibration_dict = embodiment_meta["calibration"]
            logger.info(f"use extra calibration: {calibration_dict}")
        if args.urdf is None:
            args.urdf = embodiment_meta["urdf"]

    if args.urdf is None:
        parser.error(
            "Either --urdf or --embodiment_meta_file "
            "(with --embodiment) must be specified"
        )
    logger.info(f"urdf: {args.urdf}")

    if args.task_names is None:
        task_names = None
    else:
        task_names = args.task_names.split(",")

    checker = PiperMcapChecker(
        input_path=args.input_path,
        output_path=args.output_path,
        urdf=args.urdf,
        task_names=task_names,
        user_names=args.user_names,
        date_prefix=args.date_prefix,
        calibration_dict=calibration_dict,
        num_workers=args.num_workers,
        inspect_config=build_default_inspect_config()
        if args.inspect_config is None
        else load_inspect_config(args.inspect_config),
        inspect_config_path=args.inspect_config,
        enable_ffmpeg_log=args.enable_ffmpeg_log,
    )
    checker()
