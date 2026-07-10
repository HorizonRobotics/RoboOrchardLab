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
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytorch_kinematics as pk
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.horizon_manipulation.tools.lmdb_pack_log import (
    write_pack_log,
)
from robo_orchard_lab.dataset.horizon_manipulation.tools.utils import (
    episode_matches_embodiment,
    format_time,
    get_frequency,
    normalize_embodiment_name,
    pose_to_mat,
    sample,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PiperMcapPacker(BaseLmdbManipulationDataPacker):
    """Abstact class of Packing data into target packtype.

    Packer is the recommended base class for being inherited.
    Focus on packer env, read and write.

    Subclass need to override :py:func:`pack_data`.

    Args:
        max_data_num (int): Num of data for packing.
        pack_type (str): The target pack type.
        num_workers (int): Num workers for reading original data.
            while num_workers <= 0 means pack by single process.
            num_workers >= 1 mean pack by num_workers process.
        **kwargs (dict): kwargs for pack type.
    """

    def __init__(
        self,
        input_path,
        output_path,
        urdf,
        calibration_dict=None,
        task_names=None,
        user_names=None,
        static_threshold: float = 1e-3,
        head_time_to_filter: float = 1e8,
        tile_time_to_filter: float = None,
        date_prefix: str = None,
        num_steps_per_shard: int = None,
        embodiment: str | None = None,
        **kwargs,
    ):
        super().__init__(input_path, output_path, **kwargs)
        self.calibration_dict = self.calibration_process(calibration_dict)
        self.task_names = task_names
        self.user_names = user_names
        self.embodiment = normalize_embodiment_name(embodiment)
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.static_threshold = static_threshold
        self.head_time_to_filter = head_time_to_filter
        self.tile_time_to_filter = tile_time_to_filter
        if date_prefix is not None:
            self.date_prefix = date_prefix.strip().split(",")
        else:
            self.date_prefix = date_prefix
        self.episodes = self.input_path_handler(input_path)
        self.num_steps_per_shard = num_steps_per_shard

    def calibration_process(self, calibration_dict):
        if calibration_dict is None:
            return None
        for camera, calib in calibration_dict.items():
            calibration_dict[camera] = np.linalg.inv(pose_to_mat(calib))
        return calibration_dict

    @staticmethod
    def _decode_depth(depth: bytes) -> np.ndarray | None:
        depth_buffer = np.frombuffer(depth, np.uint8)
        return cv2.imdecode(depth_buffer, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def _encode_depth(depth: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(".png", depth)
        if not success:
            raise RuntimeError("Failed to encode replacement zero depth PNG")
        return buffer.tobytes()

    @classmethod
    def _replace_invalid_depths(
        cls,
        depths: list[bytes],
        *,
        uuid: str,
        cam: str,
        expected_shape: tuple[int, int] | None = None,
    ) -> list[bytes]:
        """Replace invalid encoded depth frames with zero-depth PNGs."""

        fallback_depth = None
        invalid_indices = []

        for idx, depth in enumerate(depths):
            decoded = cls._decode_depth(depth)
            if decoded is None:
                invalid_indices.append(idx)
                continue
            if fallback_depth is None:
                fallback_depth = np.zeros_like(decoded)

        if not invalid_indices:
            return depths

        if fallback_depth is None and expected_shape is not None:
            fallback_depth = np.zeros(expected_shape, dtype=np.uint16)

        if fallback_depth is None:
            raise RuntimeError(
                f"{uuid} | {cam} depth has no valid frame to infer "
                "replacement shape"
            )

        zero_depth = cls._encode_depth(fallback_depth)
        sanitized_depths = list(depths)
        for idx in invalid_indices:
            sanitized_depths[idx] = zero_depth

        logger.warning(
            "%s | %s depth contains %d invalid PNG frame(s); "
            "replace with zero depth at indices: %s",
            uuid,
            cam,
            len(invalid_indices),
            invalid_indices[:20],
        )
        return sanitized_depths

    def input_path_handler(self, input_paths):
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
                        if self.embodiment and not episode_matches_embodiment(
                            path, self.embodiment
                        ):
                            continue
                        episodes.append([path, user, task, time])
        episodes.sort()
        logger.info(f"number of valid episodes: {len(episodes)}")
        return episodes

    def forward_kinematics(self, left_joint, right_joint):
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

    def _write_meta_data(
        self,
        uuid,
        cameras,
        intrinsic,
        extrinsic,
        timestamp,
        joint_positions,
        joint_velocity,
        joint_effort,
        cartesian_position,
        master_joint_positions,
        master_joint_velocity,
        master_joint_effort,
        shard_idx=None,
    ):
        if shard_idx is None:
            start_idx = 0
            end_idx = len(timestamp)
            prefix = f"{uuid}"
        else:
            start_idx = shard_idx * self.num_steps_per_shard
            end_idx = start_idx + self.num_steps_per_shard
            prefix = f"{uuid}/{shard_idx}"

        self.meta_pack_file.write(f"{uuid}/camera_names", cameras)
        self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsic)
        self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsic)
        self.meta_pack_file.write(
            f"{prefix}/timestamp", timestamp[start_idx:end_idx]
        )
        self.meta_pack_file.write(
            f"{prefix}/observation/robot_state/joint_positions",
            joint_positions[start_idx:end_idx],
        )
        self.meta_pack_file.write(
            f"{prefix}/observation/robot_state/joint_velocity",
            joint_velocity[start_idx:end_idx],
        )
        self.meta_pack_file.write(
            f"{prefix}/observation/robot_state/joint_effort",
            joint_effort[start_idx:end_idx],
        )
        self.meta_pack_file.write(
            f"{prefix}/observation/robot_state/cartesian_position",
            cartesian_position[start_idx:end_idx],
        )

        self.meta_pack_file.write(
            f"{prefix}/observation/robot_state/master_joint_positions",
            master_joint_positions[start_idx:end_idx],
        )
        self.meta_pack_file.write(
            f"{prefix}/observation/robot_state/master_joint_velocity",
            master_joint_velocity[start_idx:end_idx],
        )
        self.meta_pack_file.write(
            f"{prefix}/observation/robot_state/master_joint_effort",
            master_joint_effort[start_idx:end_idx],
        )

    def _pack(self):
        cameras = ["middle", "left", "right"]
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

        num_valid_ep = 0
        for ep_id, ep in enumerate(self.episodes):
            path, user, task_name, time = ep

            uuid = f"{task_name}/{user}/{time}"
            logger.info(
                f"start process [{ep_id + 1}/{len(self.episodes)}] {uuid}"
            )
            mcap = glob.glob(f"{path}/episode_*_0.mcap")
            assert len(mcap) == 1
            mcap = mcap[0]
            meta = json.load(
                open(os.path.join(path, "episode_meta.json"), "r")
            )
            assert user == meta["user_name"]
            assert task_name == meta["task_name"]

            image_extrinsic = {}
            images = {}
            for t in image_topics:
                images[t] = {
                    "data": [],
                    "time": [],
                    "intrinsic": None,
                }
            depths = {}
            for t in depth_topics:
                depths[t] = {
                    "data": [],
                    "time": [],
                    "intrinsic": None,
                    "shape": None,
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
            ee_poses = {}
            for t in [left_ee_topic, right_ee_topic]:
                ee_poses[t] = {
                    "pose": [],
                    "time": [],
                }

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
                if topic in images:
                    images[topic]["data"].append(ros_msg.data)
                    images[topic]["time"].append(time)
                elif topic in depths:
                    depths[topic]["data"].append(ros_msg.data)
                    depths[topic]["time"].append(time)
                elif topic in joints:
                    joints[topic]["position"].append(ros_msg.position)
                    joints[topic]["velocity"].append(ros_msg.velocity)
                    joints[topic]["effort"].append(ros_msg.effort)
                    joints[topic]["time"].append(time)
                elif topic in ee_poses:
                    pos = ros_msg.pose.position
                    rot = ros_msg.pose.orientation
                    x, y, z = pos.x, pos.y, pos.z
                    qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
                    ee_poses[topic]["pose"].append([x, y, z, qx, qy, qz, qw])
                    ee_poses[topic]["time"].append(time)
                elif topic in image_intrinsic_topics:
                    img_topic = image_topics[
                        image_intrinsic_topics.index(topic)
                    ]
                    images[img_topic]["intrinsic"] = np.array(
                        ros_msg.p
                    ).reshape(3, 4)
                elif topic in depth_intrinsic_topics:
                    dpt_topic = depth_topics[
                        depth_intrinsic_topics.index(topic)
                    ]
                    depths[dpt_topic]["intrinsic"] = np.array(
                        ros_msg.p
                    ).reshape(3, 4)
                    if hasattr(ros_msg, "height") and hasattr(
                        ros_msg, "width"
                    ):
                        depths[dpt_topic]["shape"] = (
                            int(ros_msg.height),
                            int(ros_msg.width),
                        )
                elif topic in image_extrinsic_topic:
                    for tf in ros_msg.transforms:
                        frame_id = f"{tf.child_frame_id}/{tf.header.frame_id}"
                        if frame_id in frame_id_extrinsic_map:
                            camera_name = frame_id_extrinsic_map[frame_id]
                            extrinsic = np.linalg.inv(
                                pose_to_mat(tf.transform)
                            )
                            image_extrinsic[camera_name] = extrinsic

            for data in [images, depths, joints, ee_poses]:
                for t in data:
                    data[t]["time"] = format_time(data[t]["time"])

            base_time = images[image_topics[0]]["time"]
            num_steps = len(base_time)
            for t in images:
                images[t]["freq"] = get_frequency(images[t]["time"], t)
                if t == image_topics[0]:
                    continue
                images[t]["time"], (images[t]["data"],) = sample(
                    base_time, images[t]["time"], [images[t]["data"]], prefix=t
                )
            for t in depths:
                depths[t]["freq"] = get_frequency(depths[t]["time"], t)
                depths[t]["time"], (depths[t]["data"],) = sample(
                    base_time, depths[t]["time"], [depths[t]["data"]], prefix=t
                )
            for t in joints:
                joints[t]["freq"] = get_frequency(joints[t]["time"], t)
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
                    prefix=t,
                )
                joints[t]["position"] = np.array(joints[t]["position"])

            poses = self.forward_kinematics(
                joints[left_joint_topic]["position"],
                joints[right_joint_topic]["position"],
            )
            left_ee_pose = poses["left_gripper_base"].get_matrix().numpy()
            right_ee_pose = poses["right_gripper_base"].get_matrix().numpy()

            for t in ee_poses:
                ee_poses[t]["freq"] = get_frequency(ee_poses[t]["time"], t)
                ee_poses[t]["time"], (ee_poses[t]["pose"],) = sample(
                    base_time,
                    ee_poses[t]["time"],
                    [ee_poses[t]["pose"]],
                    prefix=t,
                )

            if self.calibration_dict is not None:
                image_extrinsic = copy.deepcopy(self.calibration_dict)
            extrinsic = {
                "left": image_extrinsic["left"] @ np.linalg.inv(left_ee_pose),
                "right": image_extrinsic["right"]
                @ np.linalg.inv(right_ee_pose),
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
                static_mask = np.logical_or(
                    static_mask, np.logical_not(time_mask)
                )
                joint_positions = joint_positions[static_mask]
                joint_velocity = joint_velocity[static_mask]
                joint_effort = joint_effort[static_mask]
                master_joint_positions = master_joint_positions[static_mask]
                master_joint_velocity = master_joint_velocity[static_mask]
                master_joint_effort = master_joint_effort[static_mask]
                ee_poses = ee_poses[static_mask]
                logger.info(
                    f"all steps: {num_steps}, "
                    f"non static steps: {joint_positions.shape[0]}"
                )
                num_steps = joint_positions.shape[0]

                extrinsic["left"] = extrinsic["left"][static_mask]
                extrinsic["right"] = extrinsic["right"][static_mask]
                base_time = base_time[static_mask]
                for t in image_topics:
                    images[t]["data"] = [
                        x
                        for i, x in enumerate(images[t]["data"])
                        if static_mask[i]
                    ]
                for t in depth_topics:
                    depths[t]["data"] = [
                        x
                        for i, x in enumerate(depths[t]["data"])
                        if static_mask[i]
                    ]
            # ==================== filter finish

            for cam, t in zip(cameras, image_topics, strict=False):
                for i, img in enumerate(images[t]["data"]):
                    self.image_pack_file.write(f"{uuid}/{cam}/{i}", img)

            for cam, t in zip(cameras, depth_topics, strict=False):
                sanitized_depths = self._replace_invalid_depths(
                    depths[t]["data"],
                    uuid=uuid,
                    cam=cam,
                    expected_shape=depths[t].get("shape"),
                )
                for i, depth in enumerate(sanitized_depths):
                    self.depth_pack_file.write(f"{uuid}/{cam}/{i}", depth)

            self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsic)
            if self.num_steps_per_shard is not None:
                self.meta_pack_file.write(
                    f"{uuid}/num_steps_per_shard", self.num_steps_per_shard
                )
                num_shards = math.ceil(num_steps / self.num_steps_per_shard)
                logger.info(f"num_shards:{num_shards}")
                for shard_idx in range(num_shards):
                    self._write_meta_data(
                        uuid,
                        cameras,
                        intrinsic,
                        extrinsic,
                        base_time,
                        joint_positions,
                        joint_velocity,
                        joint_effort,
                        ee_poses,
                        master_joint_positions,
                        master_joint_velocity,
                        master_joint_effort,
                        shard_idx,
                    )
            else:
                self._write_meta_data(
                    uuid,
                    cameras,
                    intrinsic,
                    extrinsic,
                    base_time,
                    joint_positions,
                    joint_velocity,
                    joint_effort,
                    ee_poses,
                    master_joint_positions,
                    master_joint_velocity,
                    master_joint_effort,
                )

            meta.update(
                uuid=uuid,
                user=user,
                task_name=task_name,
                num_steps=num_steps,
                simulation=False,
            )
            self.meta_pack_file.write(f"{uuid}/meta_data", meta)
            self.write_index(ep_id, meta)

            num_valid_ep += 1
            logger.info(
                f"finish process [{ep_id + 1}/{len(self.episodes)}] {uuid}, "
                f"num_steps:{num_steps} \n"
            )
        self.index_pack_file.write("__len__", num_valid_ep)
        self.close()
        self._write_pack_log()

    def _write_pack_log(self):
        pack_log_path = write_pack_log(Path(self.output_path), overwrite=True)
        logger.info("Wrote LMDB pack log: %s", pack_log_path)


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
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("--user_names", type=str, default=None)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--date_prefix", type=str, default=None)
    parser.add_argument("--num_steps_per_shard", type=int, default=None)
    parser.add_argument("--embodiment_meta_file", type=str, default=None)
    parser.add_argument("--embodiment", type=str, default=None)
    parser.add_argument("--use_extra_calibration", action="store_true")
    args = parser.parse_args()

    if args.task_names is None:
        task_names = None
    else:
        task_names = args.task_names.split(",")

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
    packer = PiperMcapPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        urdf=args.urdf,
        task_names=task_names,
        user_names=args.user_names,
        date_prefix=args.date_prefix,
        num_steps_per_shard=args.num_steps_per_shard,
        calibration_dict=calibration_dict,
        embodiment=args.embodiment,
    )
    packer()
