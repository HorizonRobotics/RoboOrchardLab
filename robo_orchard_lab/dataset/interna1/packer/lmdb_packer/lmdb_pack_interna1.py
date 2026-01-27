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

import logging

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robo_orchard_lab.dataset.interna1.packer.arrow_packer.arrow_pack_interna1 import (  # noqa: E501
    InternA1Packaging,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

# Keep file descriptor usage low when DataLoader forks workers.
torch.multiprocessing.set_sharing_strategy("file_system")
logger = logging.getLogger(__name__)
RESERVED_LEROBOT_KEYS = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task",
    "task_index",
}


# Default camera topics and image target size (H, W)
CAMERA_TOPICS = ["head", "hand_left", "hand_right"]
TARGET_SIZE = (480, 640)
JPEG_QUALITY = 95


def forward_kinematics_agilex(chain, batch_joint_state):
    """Computes forward kinematics for the AgileX robot.

    Args:
        chain (KinematicChain): The kinematic chain object.
        batch_joint_state (BatchJointsState): Batch of joint states.

    Returns:
        torch.Tensor: Link poses resulting from forward kinematics.
    """
    joint_num = len(chain._chain.get_joints())
    all_joint_state = torch.zeros((batch_joint_state.batch_size, joint_num))
    all_joint_state[:, 9:15] = batch_joint_state.position[:, :6]
    all_joint_state[:, 15] = batch_joint_state.position[:, 6] / 2
    all_joint_state[:, 16] = -batch_joint_state.position[:, 6] / 2
    all_joint_state[:, 17:23] = batch_joint_state.position[:, 7:13]
    all_joint_state[:, 23] = batch_joint_state.position[:, 13] / 2
    all_joint_state[:, 24] = -batch_joint_state.position[:, 13] / 2

    link_poses = chain.forward_kinematics(all_joint_state)
    return link_poses


def forward_kinematics_arx(chain, batch_joint_state):
    """Computes forward kinematics for the ARX robot.

    Args:
        chain (KinematicChain): The kinematic chain object.
        batch_joint_state (BatchJointsState): Batch of joint states.

    Returns:
        torch.Tensor: Link poses resulting from forward kinematics.
    """
    joint_num = len(chain._chain.get_joints())
    all_joint_state = torch.zeros((batch_joint_state.batch_size, joint_num))
    all_joint_state[:, 6:13] = batch_joint_state.position[:, :7]
    all_joint_state[:, 14:21] = batch_joint_state.position[:, 7:]

    link_poses = chain.forward_kinematics(all_joint_state)
    return link_poses


def forward_kinematics_genie(chain, batch_joint_state):
    """Computes forward kinematics for the Genie robot.

    Args:
        chain (KinematicChain): The kinematic chain object.
        batch_joint_state (BatchJointsState): Batch of joint states.

    Returns:
        torch.Tensor: Link poses resulting from forward kinematics.
    """
    joint_num = len(chain._chain.get_joints())
    all_joint_state = torch.zeros((batch_joint_state.batch_size, joint_num))
    all_joint_state[:, 4:12] = batch_joint_state.position[:, :8]
    all_joint_state[:, 19:27] = batch_joint_state.position[:, 8:]

    link_poses = chain.forward_kinematics(all_joint_state)
    return link_poses


class Lerobot2LmdbDataPacker(BaseLmdbManipulationDataPacker):
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
        episodes_to_package: list,
        output_path: str,
        **kwargs,
    ):
        """Initializes the McapLmdbDataPacker.

        Args:
            input_path (str): A glob pattern for the input episode directories.
            output_path (str): The path to the output LMDB database directory.
            urdf (str): The path to the robot's URDF file.
            pack_config: Configuration parameters for packaging.
        """
        super().__init__(
            input_path=episodes_to_package, output_path=output_path, **kwargs
        )
        self.episodes_to_package = episodes_to_package

    def _pack(self):
        """Main data packing loop that processes all episodes.

        This method iterates through each found episode, performs the full
        data loading, synchronization, filtering, and writing pipeline, and
        stores the results in the configured LMDB databases.
        """
        num_valid_ep = 0
        for episode_idx, episode_packer in enumerate(self.episodes_to_package):
            episode_meta = episode_packer.generate_episode_meta()
            task_name = episode_meta.task.name.replace(" ", "_").replace(
                "-", "_"
            )
            robot_type = episode_packer.dataset.meta.info.get("robot_type")
            dataset_name = args.input_path.replace(
                "/horizon-bucket/robot_lab2/datasets/InternData-A1/sim_updated_lerobotv30/",
                "",
            )
            note = f"{dataset_name}_{args.start_idx + episode_idx}"

            uuid = f"{robot_type}/{task_name}/{note}"
            print(f"Start processing episode: {uuid}")

            # parse episode data
            episode_packer.parse_episode_data()

            dataset_from_index = episode_packer.dataset.meta.episodes[
                args.start_idx + episode_idx
            ]["dataset_from_index"]
            dataset_to_index = episode_packer.dataset.meta.episodes[
                args.start_idx + episode_idx
            ]["dataset_to_index"]

            if episode_packer.max_frames > 0:
                num_steps = min(
                    dataset_to_index - dataset_from_index,
                    episode_packer.max_frames,
                )
            else:
                num_steps = dataset_to_index - dataset_from_index

            # 过滤静止帧
            static_mask = np.ones(num_steps, dtype=bool)
            static_threshold = 1e-4
            if static_threshold > 0 and num_steps > 1:
                static_mask[1:] = np.any(
                    np.abs(
                        np.diff(episode_packer.batch_actions.position, axis=0)
                    )
                    > static_threshold,
                    axis=1,
                )
            if "conveyor" in args.input_path or "water" in args.input_path:
                static_mask = np.ones(num_steps, dtype=bool)

            num_steps = sum(static_mask)

            print(f"num_steps: {num_steps}")
            meta = {
                "uuid": uuid,
                "robot_type": robot_type,
                "task_name": task_name,
                "urdf_path": episode_packer.urdf_path,
                "num_steps": num_steps,
                "simulation": False,
            }
            self.meta_pack_file.write(f"{uuid}/meta_data", meta)
            self.meta_pack_file.write(f"{uuid}/camera_names", CAMERA_TOPICS)
            self.write_index(episode_idx, meta)

            extrinsic = {}
            intrinsic = {}
            for camera_topic in CAMERA_TOPICS:
                extrinsic[camera_topic] = (
                    episode_packer.batch_images_msg[camera_topic]
                    .pose.as_Transform3D_M()
                    .get_matrix()
                ).to(torch.float64)[static_mask]
                intrinsic[camera_topic] = (
                    episode_packer.batch_images_msg[camera_topic]
                    .intrinsic_matrices[0]
                    .to(torch.float64)
                )

            self.meta_pack_file.write(
                f"{uuid}/timestamp",
                episode_packer.batch_timestamps[static_mask],
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/master_joint_positions",
                episode_packer.batch_actions.position[static_mask],
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_positions",
                episode_packer.batch_joints.position[static_mask],
            )

            batch_left_ee_matrix = (
                episode_packer.batch_left_ee.as_Transform3D_M().get_matrix()
            )  # noqa: E501
            batch_right_ee_matrix = (
                episode_packer.batch_right_ee.as_Transform3D_M().get_matrix()
            )  # noqa: E501
            batch_ee_matrix = torch.cat(
                [
                    batch_left_ee_matrix.unsqueeze(1),
                    batch_right_ee_matrix.unsqueeze(1),
                ],
                dim=1,
            )[static_mask]

            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/cartesian_position",
                batch_ee_matrix,
            )
            self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsic)
            self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsic)

            camera_index = 0
            for step_idx in range(
                len(episode_packer.batch_images_msg[camera_topic].sensor_data)
            ):
                if not static_mask[step_idx]:
                    continue
                for camera_topic in CAMERA_TOPICS:
                    self.image_pack_file.write(
                        f"{uuid}/{camera_topic}/{camera_index}",
                        episode_packer.batch_images_msg[
                            camera_topic
                        ].sensor_data[step_idx],
                    )
                camera_index += 1

            num_valid_ep += 1
            print(
                f"finish process [{episode_idx + 1}/{len(self.episodes_to_package)}] "  # noqa: E501
                f"{uuid}, num_steps:{num_steps}, "
                f"static frames count is {np.sum(~static_mask)} \n"
            )

            # Clear cache to avoid OOM
            episode_packer.clear_cache()

        self.index_pack_file.write("__len__", num_valid_ep)
        self.close()
        print(
            f"Packing complete. {num_valid_ep} episodes processed. "  # noqa: E501
            f"Saved to {self.output_path}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index of episodes to process.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=0,
        help="End index of episodes to process.",
    )

    args = parser.parse_args()
    EPISODE_IDX = list(range(args.start_idx, args.end_idx))
    TARGET_LEROBOT_CONVERT_PATH = args.output_path
    print(f"Start Processing Number of Episodes: {EPISODE_IDX}")

    lerobot_dataset = LeRobotDataset(
        args.input_path,
        video_backend="pyav",  # for compatible purpose
        batch_encoding_size=256,
    )

    robot_type = lerobot_dataset.meta.info.get("robot_type")

    if robot_type == "ARX Lift-2":
        link_topics = [
            "left_arm_link1",
            "left_arm_link2",
            "left_arm_link3",
            "left_arm_link4",
            "left_arm_link5",
            "left_arm_link6",
            "right_arm_link1",
            "right_arm_link2",
            "right_arm_link3",
            "right_arm_link4",
            "right_arm_link5",
            "right_arm_link6",
            "left_arm_tcp_link",
            "right_arm_tcp_link",
        ]
        left_arm_key = "left_arm"
        right_arm_key = "right_arm"
        left_ee_key = "left_arm_link6"
        right_ee_key = "right_arm_link6"

    elif robot_type == "AgileX Split Aloha":
        link_topics = [
            "left/link1",
            "left/link2",
            "left/link3",
            "left/link4",
            "left/link5",
            "left/link6",
            "right/link1",
            "right/link2",
            "right/link3",
            "right/link4",
            "right/link5",
            "right/link6",
            "left/link7",
            "right/link7",
        ]
        left_arm_key = "left/"
        right_arm_key = "right/"
        left_ee_key = "left/link6"
        right_ee_key = "right/link6"
    elif robot_type == "Genie-1":
        link_topics = [
            # Left arm joints
            "arm_l_link1",
            "arm_l_link2",
            "arm_l_link3",
            "arm_l_link4",
            "arm_l_link5",
            "arm_l_link6",
            "arm_l_end_link",
            # Left gripper outer joint
            "gripper_l_center_link",
            # Right arm joints
            "arm_r_link1",
            "arm_r_link2",
            "arm_r_link3",
            "arm_r_link4",
            "arm_r_link5",
            "arm_r_link6",
            "arm_r_end_link",
            # Right gripper outer joint
            "gripper_r_center_link",
        ]
        left_arm_key = "arm_l"
        right_arm_key = "arm_r"
        left_ee_key = "gripper_l_center_link"
        right_ee_key = "gripper_r_center_link"

    # for topic in link_topics:
    #     feature_dict[topic] = BatchFrameTransformFeature(dtype="float32")

    episodes_to_package = []
    for ep_idx in EPISODE_IDX:
        # We access the metadata by index
        episode_meta = lerobot_dataset.meta.episodes[ep_idx]
        episodes_to_package.append(
            InternA1Packaging(
                dataset=lerobot_dataset,
                episode_meta=episode_meta,
                robot_type=robot_type,
                # max_frames=100,  # for demo purpose
            )
        )

    packer = Lerobot2LmdbDataPacker(
        episodes_to_package=episodes_to_package,
        output_path=TARGET_LEROBOT_CONVERT_PATH,
    )
    packer()
