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

import glob
import json
import logging
import os

import numpy as np
import pytorch_kinematics as pk
import torch
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sample(tgt_time, src_time, src_data=None, prefix=""):
    time_diff = np.abs(tgt_time[:, None] - src_time)
    logger.info(
        f"{prefix:<50} - "
        + f"max time diff: {time_diff.min(axis=-1).max():.4f}, "
        + f"mean time diff: {time_diff.min(axis=-1).mean():.4f}"
    )
    index = np.argmin(time_diff, axis=1)
    output_time = src_time[index]
    if src_data is not None:
        output = []
        for src in src_data:
            _output = []
            for i in index:
                _output.append(src[i])
            output.append(_output)
        return output_time, output
    return output_time


def format_time(timestamp):
    timestamp = np.array(timestamp, dtype="float64")
    timestamp = timestamp[:, 0] + timestamp[:, 1] / 1e9
    return timestamp


def pose_to_mat(pose):
    if isinstance(pose, dict):
        x, y, z = pose["position"]
        qx, qy, qz, w = pose["orientation"]
    else:
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        qx, qy, qz, w = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
    trans = np.array([x, y, z])
    rot = Rotation.from_quat([qx, qy, qz, w], scalar_first=False).as_matrix()
    ret = np.eye(4)
    ret[:3, 3] = trans
    ret[:3, :3] = rot
    return ret


def get_frequency(timestamp, prefix="", window_size=3):
    if not isinstance(timestamp, np.ndarray):
        timestamp = np.array(timestamp)
    time_diff = np.diff(timestamp)
    time_diff = torch.from_numpy(time_diff)[None, None]
    time_diff = torch.nn.functional.avg_pool1d(
        time_diff, window_size, 1
    ).numpy()[0, 0]
    freq = 1 / time_diff
    logger.info(
        f"{prefix:<50} - "
        f"duration: {timestamp[-1] - timestamp[0]:.2f}s, "
        + f"min frequency: {freq.min():.1f}Hz, "
        + f"mean frequency: {freq.mean():.1f}Hz"
    )
    return freq


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
        calibration_dict,
        task_names=None,
        user_names=None,
        static_threshold: float = 1e-3,
        head_time_to_filter: float = 4,
        tile_time_to_filter: float = 4,
        date_prefix: str = None,
        **kwargs,
    ):
        super().__init__(input_path, output_path, **kwargs)
        self.calibration_dict = self.calibration_process(calibration_dict)
        self.task_names = task_names
        self.user_names = user_names
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.episodes = self.input_path_handler(input_path)
        self.static_threshold = static_threshold
        self.head_time_to_filter = head_time_to_filter
        self.tile_time_to_filter = tile_time_to_filter
        self.date_prefix = date_prefix

    def calibration_process(self, calibration_dict):
        for camera, calib in calibration_dict.items():
            calibration_dict[camera] = np.linalg.inv(pose_to_mat(calib))
        return calibration_dict

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

                        if (
                            self.date_prefix is not None
                            and not time.startswith(self.date_prefix)
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

    def _pack(self):
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

            cameras = ["middle", "left", "right"]
            image_topics = [
                f"/observation/cameras/{x}/color_image/image_raw"
                for x in cameras
            ]
            image_intrinsic_topics = [
                f"/observation/cameras/{x}/color_image/camera_info"
                for x in cameras
            ]
            depth_topics = [
                f"/observation/cameras/{x}/depth_image/image_raw"
                for x in cameras
            ]
            depth_intrinsic_topics = [
                f"/observation/cameras/{x}/depth_image/camera_info"
                for x in cameras
            ]
            left_joint_topic = "/observation/robot_state/left/joint"
            right_joint_topic = "/observation/robot_state/right/joint"
            left_master_joint_topic = (
                "/observation/robot_state/left_master/joint"
            )
            right_master_joint_topic = (
                "/observation/robot_state/right_master/joint"
            )
            left_ee_topic = "/observation/robot_state/left/end_pose"
            right_ee_topic = "/observation/robot_state/right/end_pose"

            all_useful_topics = (
                image_topics
                + image_intrinsic_topics
                + depth_topics
                + depth_intrinsic_topics
                + [
                    left_joint_topic,
                    right_joint_topic,
                    left_ee_topic,
                    right_ee_topic,
                ]
                + [left_master_joint_topic, right_master_joint_topic]
            )

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
                time = (ros_msg.header.stamp.sec, ros_msg.header.stamp.nanosec)
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

            extrinsic = {
                "left": self.calibration_dict["left"]
                @ np.linalg.inv(left_ee_pose),
                "right": self.calibration_dict["right"]
                @ np.linalg.inv(right_ee_pose),
                "middle": np.copy(self.calibration_dict["right"]),
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
                for i, depth in enumerate(depths[t]["data"]):
                    self.depth_pack_file.write(f"{uuid}/{cam}/{i}", depth)

            self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsic)
            self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsic)
            self.meta_pack_file.write(f"{uuid}/timestamp", base_time)

            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_positions",
                joint_positions,
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_velocity",
                joint_velocity,
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_effort", joint_effort
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/cartesian_position", ee_poses
            )

            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/master_joint_positions",
                master_joint_positions,
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/master_joint_velocity",
                master_joint_velocity,
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/master_joint_effort",
                master_joint_effort,
            )

            self.meta_pack_file.write(f"{uuid}/camera_names", cameras)
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


if __name__ == "__main__":
    import argparse

    from robo_orchard_lab.utils import log_basic_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--urdf", type=str)
    parser.add_argument("--user_names", type=str, default=None)
    parser.add_argument("--task_names", type=str, default=None)
    args = parser.parse_args()

    if args.task_names is None:
        task_names = None
    else:
        task_names = args.task_names.split(",")

    packer = PiperMcapPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        urdf=args.urdf,
        task_names=task_names,
        user_names=args.user_names,
        calibration_dict={
            "left": {
                "position": [
                    -0.06867924193484086,
                    -0.0005945544447201671,
                    0.03843362824412718,
                ],
                "orientation": [
                    -0.14277810176817451,
                    0.1236499359266293,
                    -0.6680764786273947,
                    0.7197214222917346,
                ],
            },
            "right": {
                "position": [
                    -0.07333788908459828,
                    0.00991803705544634,
                    0.03390080995535155,
                ],
                "orientation": [
                    0.1296176811682453,
                    -0.12171535345636147,
                    0.717362436615576,
                    -0.673628802824318,
                ],
            },
            "middle": {
                "position": [
                    -0.010783568385050412,
                    -0.2559182030838615,
                    0.5173197227547938,
                ],
                "orientation": [
                    -0.6344593881273598,
                    0.6670669773214551,
                    -0.2848079166270871,
                    0.2671467447131103,
                ],
            },
        },
    )
    log_basic_config(
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )
    packer()
