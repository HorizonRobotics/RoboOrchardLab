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

import argparse
import glob
import json
import logging
import os

import numpy as np
import torch
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph
from robo_orchard_core.kinematics.chain import KinematicChain
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.math import Transform3D_M

from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchJointsState,
)
from robo_orchard_lab.dataset.horizon_manipulation.packer.utils import (
    McapParseConfig,
    PackConfig,
    filter_static_frames,
    parse_mcap,
    scale_images_and_update_intrinsics,
    time_sync,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

# Setup Logger
logger = LoggerManager().get_child(__name__)
logger.setLevel(logging.INFO)


class McapLmdbDataPacker(BaseLmdbManipulationDataPacker):
    """Packs robotics episode data from MCAP files into LMDB format.

    This class reads multimodal robotics data (images, joint states, etc.)
    from MCAP files, synchronizes the different data streams to a common
    timeline, filters the data, and then writes the processed information
    into an LMDB (Lightning Memory-Mapped Database) dataset.

    The synchronization is based on a primary camera's timestamps, ensuring
    that all data for a given frame is temporally aligned.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        urdf_path: str,
        pack_config: PackConfig,
        **kwargs,
    ):
        """Initializes the McapLmdbDataPacker.

        Args:
            input_path (str): A glob pattern for the input episode directories.
            output_path (str): The path to the output LMDB database directory.
            urdf (str): The path to the robot's URDF file.
            pack_config: Configuration parameters for packaging.
        """
        super().__init__(input_path, output_path, **kwargs)
        self.episodes = self.input_path_handler(input_path)
        self.pack_config = pack_config
        self.urdf_path = urdf_path

    def input_path_handler(self, input_paths):
        """Scans input paths to find and validate episode data.

        This method takes a comma-separated string of glob patterns, finds all
        matching directories, and verifies that each contains the necessary
        metadata files (`episode_meta.json`).

        Args:
            input_paths (str): A string containing one or more glob patterns,
                separated by commas.

        Returns:
            list: A sorted list where each element is a list containing the
                path, user, task name, and date for a valid episode.
        """
        episodes = []
        input_paths = input_paths.strip().split(",")
        logger.info(f"input_paths: {input_paths}")
        for input_path in input_paths:
            input_files = glob.glob(input_path)
            for input_file in input_files:
                path = "/" + os.path.join(*input_file.split("/")[:-1])
                date = input_file.split("/")[-2]
                meta = json.load(
                    open(os.path.join(path, "episode_meta.json"), "r")
                )
                user = meta["user_name"]
                task = meta["task_name"]
                episodes.append([path, user, task, date])
        episodes.sort()
        logger.info(f"Found {len(episodes)} potential episodes.")
        return episodes

    def forward_kinematics(self, joint_states: BatchJointsState):
        """Computes the forward kinematics for the dual-arm robot.

        This method takes the joint positions for the left and right arms and
        calculates the poses of all links in the kinematic chain, including the
        end-effectors.

        Args:
            left_joint (np.ndarray): An array of joint positions for the left
                arm, shaped (N, 7).
            right_joint (np.ndarray): An array of joint positions for the right
                arm, shaped (N, 7).

        Returns:
            dict: A dictionary mapping link names to their corresponding poses
                as transformation matrices.
        """
        # Load URDF and Initialize Kinematic Chain
        chain = KinematicChain.from_content(
            open(self.urdf_path, "r").read().encode("utf-8"), "urdf"
        )

        position = joint_states.position

        position_urdf = torch.zeros((joint_states.batch_size, 16))
        position_urdf[:, :6] = position[:, :6]
        position_urdf[:, 8:14] = position[:, 7:13]
        position_urdf[:, 6] = position[:, 6] / 2
        position_urdf[:, 7] = -position[:, 6] / 2
        position_urdf[:, 14] = position[:, 13] / 2
        position_urdf[:, 15] = -position[:, 13] / 2

        link_poses = chain.forward_kinematics(position_urdf)
        left_ee_pose: Transform3D_M = link_poses["left_gripper_base"]
        right_ee_pose: Transform3D_M = link_poses["right_gripper_base"]

        batch_left_ee_pose = BatchFrameTransform(
            parent_frame_id="world",
            child_frame_id="left_end_effector",
            xyz=left_ee_pose.get_translation(),
            quat=left_ee_pose.get_rotation_quaternion(),
        )
        batch_right_ee_pose = BatchFrameTransform(
            parent_frame_id="world",
            child_frame_id="right_end_effector",
            xyz=right_ee_pose.get_translation(),
            quat=right_ee_pose.get_rotation_quaternion(),
        )

        return batch_left_ee_pose, batch_right_ee_pose

    def _pack(self):
        """Main data packing loop that processes all episodes.

        This method iterates through each found episode, performs the full
        data loading, synchronization, filtering, and writing pipeline, and
        stores the results in the configured LMDB databases.
        """
        num_valid_ep = 0
        for ep_id, ep in enumerate(self.episodes):
            path, user, task_name, date = ep
            uuid = f"{task_name}/{user}/{date}"
            logger.info(f"Start processing episode: {uuid}")
            mcap_path = glob.glob(os.path.join(path, "*.mcap"))[0]

            pack_config: PackConfig = self.pack_config
            parse_config: McapParseConfig = pack_config.PARSE_CONFIG

            # Parse MCAP
            mcap_data = parse_mcap(
                mcap_path=mcap_path, parse_config=parse_config
            )
            (
                batch_tf_list,
                batch_joint_dict,
                batch_image_dict,
                batch_depth_dict,
            ) = mcap_data

            # Time Synchronization
            base_time = batch_image_dict[pack_config.SYNC_CAMERA].timestamps
            time_sync(
                data=[batch_joint_dict, batch_image_dict, batch_depth_dict],
                base_time=base_time,
            )

            # Filter static frames
            slave_left_pos = batch_joint_dict[
                parse_config.SLAVE_LEFT_JOINT
            ].position
            slave_right_pos = batch_joint_dict[
                parse_config.SLAVE_RIGHT_JOINT
            ].position
            joint_positions = torch.cat(
                [slave_left_pos, slave_right_pos], dim=-1
            )
            filter_static_frames(
                data=[batch_joint_dict, batch_image_dict, batch_depth_dict],
                joint_positions=joint_positions,
                base_time=base_time,
                static_threshold=pack_config.STATIC_THRESHOLD,
                head_time_to_filter=pack_config.HEAD_TIME_TO_FILTER,
                tail_time_to_filter=pack_config.TAIL_TIME_TO_FILTER,
            )

            # Scale images and update intrinsics
            scale_images_and_update_intrinsics(
                image_data=[batch_image_dict, batch_depth_dict],
                image_scale_ratio=pack_config.IMAGE_SCALE,
            )

            tf_graph = BatchFrameTransformGraph(batch_tf_list)
            base_time = batch_image_dict[pack_config.SYNC_CAMERA].timestamps
            num_steps = len(base_time)

            # Load episode metadata
            meta = json.load(
                open(os.path.join(path, "episode_meta.json"), "r")
            )
            meta.update(
                uuid=uuid,
                user=user,
                task_name=task_name,
                date=date,
                mcap_path=mcap_path,
                urdf_path=self.urdf_path,
                num_steps=num_steps,
                simulation=False,
            )
            self.meta_pack_file.write(f"{uuid}/meta_data", meta)
            self.meta_pack_file.write(
                f"{uuid}/camera_names", parse_config.CAMERAS
            )
            self.write_index(ep_id, meta)

            # --- Joints (Slave Arm) ---
            slave_left = batch_joint_dict[parse_config.SLAVE_LEFT_JOINT]
            slave_right = batch_joint_dict[parse_config.SLAVE_RIGHT_JOINT]
            slave_right.timestamps = slave_left.timestamps
            joint_states = BatchJointsState.concat(
                [slave_left, slave_right], dim=1
            )

            # --- Actions (Master Arm) ---
            master_left = batch_joint_dict[parse_config.MASTER_LEFT_JOINT]
            master_right = batch_joint_dict[parse_config.MASTER_RIGHT_JOINT]
            master_right.timestamps = master_left.timestamps
            action_states = BatchJointsState.concat(
                [slave_left, slave_right], dim=1
            )

            # --- Camera Data ---
            batch_left_ee_pose, batch_right_ee_pose = self.forward_kinematics(
                joint_states
            )
            tf_graph.add_tf(batch_left_ee_pose)
            tf_graph.add_tf(batch_right_ee_pose)
            for topic in parse_config.COLOR_IMAGE_TOPICS:
                color_frame = batch_image_dict[topic].frame_id
                pose = tf_graph.get_tf("world", color_frame)
                if pose.batch_size == 1 and num_steps > 1:
                    pose = pose.repeat(num_steps)
                batch_image_dict[topic].pose = pose

            joint_positions = joint_states.position.numpy()
            joint_velocity = joint_states.velocity.numpy()
            joint_effort = joint_states.effort.numpy()

            master_joint_positions = action_states.position.numpy()
            master_joint_velocity = action_states.velocity.numpy()
            master_joint_effort = action_states.effort.numpy()
            ee_poses = np.concatenate(
                [
                    batch_left_ee_pose.xyz,
                    batch_left_ee_pose.quat,
                    batch_right_ee_pose.xyz,
                    batch_right_ee_pose.quat,
                ],
                axis=1,
            ).reshape(-1, 2, 7)

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

            # Camera intrinsics and extrinsics
            extrinsic = {}
            intrinsic = {}
            for cam_idx, cam_name in enumerate(parse_config.CAMERAS):
                color_topic = parse_config.COLOR_IMAGE_TOPICS[cam_idx]
                extrinsic[cam_name] = np.linalg.inv(
                    batch_image_dict[color_topic]
                    .pose.as_Transform3D_M()
                    .get_matrix()
                )
                intrinsic[cam_name] = batch_image_dict[
                    color_topic
                ].intrinsic_matrices[0]
            self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsic)
            self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsic)

            # --- Camera Images ---
            for cam_idx, cam_name in enumerate(parse_config.CAMERAS):
                color_topic = parse_config.COLOR_IMAGE_TOPICS[cam_idx]
                depth_topic = parse_config.DEPTH_IMAGE_TOPICS[cam_idx]
                for i in range(num_steps):
                    img = batch_image_dict[color_topic].sensor_data[i]
                    self.image_pack_file.write(f"{uuid}/{cam_name}/{i}", img)

                for i in range(num_steps):
                    depth = batch_depth_dict[depth_topic].sensor_data[i]
                    self.depth_pack_file.write(f"{uuid}/{cam_name}/{i}", depth)

            num_valid_ep += 1
            logger.info(
                f"finish process [{ep_id + 1}/{len(self.episodes)}] {uuid}, "
                f"num_steps:{num_steps} \n"
            )
        self.index_pack_file.write("__len__", num_valid_ep)
        self.close()
        logger.info(
            f"Packing complete. {num_valid_ep} episodes processed. "
            f"Saved to {self.output_path}"
        )


if __name__ == "__main__":
    import argparse

    from robo_orchard_lab.utils import log_basic_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--urdf_path", type=str)
    parser.add_argument("--image_scale_factor", type=float, default=1.0)
    args = parser.parse_args()

    parse_config = McapParseConfig(
        CAMERAS=["middle", "left", "right"],
        SLAVE_LEFT_JOINT="/observation/robot_state/left/joint",
        SLAVE_RIGHT_JOINT="/observation/robot_state/right/joint",
        MASTER_LEFT_JOINT="/observation/robot_state/left_master/joint",
        MASTER_RIGHT_JOINT="/observation/robot_state/right_master/joint",
        COLOR_IMAGE_TOPICS=[
            "/observation/cameras/middle/color_image/image_raw",
            "/observation/cameras/left/color_image/image_raw",
            "/observation/cameras/right/color_image/image_raw",
        ],
        DEPTH_IMAGE_TOPICS=[
            "/observation/cameras/middle/depth_image/image_raw",
            "/observation/cameras/left/depth_image/image_raw",
            "/observation/cameras/right/depth_image/image_raw",
        ],
        COLOR_INFO_TOPICS=[
            "/observation/cameras/middle/color_image/camera_info",
            "/observation/cameras/left/color_image/camera_info",
            "/observation/cameras/right/color_image/camera_info",
        ],
        DEPTH_INFO_TOPICS=[
            "/observation/cameras/middle/depth_image/camera_info",
            "/observation/cameras/left/depth_image/camera_info",
            "/observation/cameras/right/depth_image/camera_info",
        ],
    )
    pack_config = PackConfig(
        SYNC_CAMERA="/observation/cameras/middle/color_image/image_raw",
        IMAGE_SCALE=args.image_scale_factor,
        STATIC_THRESHOLD=1e-3,
        HEAD_TIME_TO_FILTER=None,
        TAIL_TIME_TO_FILTER=None,
        PARSE_CONFIG=parse_config,
    )

    packer = McapLmdbDataPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        urdf_path=args.urdf_path,
        pack_config=pack_config,
    )
    log_basic_config(
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )
    packer()
