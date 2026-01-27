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
# noqa: N806
import logging
import os
import pprint
from typing import Any, Generator

import datasets as hg_datasets
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robo_orchard_core.kinematics.chain import KinematicChain
from robo_orchard_core.utils.math import Transform3D_M

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraDataEncoded,
    BatchCameraDataEncodedFeature,
    BatchFrameTransform,
    BatchFrameTransformFeature,
    BatchJointsState,
    BatchJointsStateFeature,
    Distortion,
)
from robo_orchard_lab.dataset.horizon_manipulation.packer.utils import (
    get_index_camera,
    get_index_tf,
)
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    InstructionData,
    RobotData,
    TaskData,
)

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
episode_count = 0


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


class InternA1Packaging(DatasetPackaging):
    def __init__(
        self,
        dataset: LeRobotDataset,
        episode_meta: dict,
        robot_type: str,
        max_frames: int = 0,
    ):
        self.dataset = dataset
        self.episode_meta = episode_meta
        self.max_frames = max_frames
        urdf_dir = "/horizon-bucket/robot_lab2/datasets/InternData-A1/urdf"
        self.robot_type = robot_type
        self.chain = None
        self.joint_names = None
        if robot_type == "ARX Lift-2":
            self.urdf_path = os.path.join(urdf_dir, "ARX_Lift2_fix/lift.urdf")
            self.chain = KinematicChain.from_content(
                open(self.urdf_path, "rb").read(), "urdf"
            )
            self.joint_names = (
                self.chain.joint_parameter_names[6:12]
                + ["left_arm_gripper"]
                + self.chain.joint_parameter_names[14:20]
                + ["right_arm_gripper"]
            )
            self.forward_kinematics = forward_kinematics_arx
        elif robot_type == "AgileX Split Aloha":
            self.urdf_path = os.path.join(
                urdf_dir,
                "AgileX_Split_Aloha_piper100/split_aloha_mid_360_with_piper.urdf",
            )
            self.chain = KinematicChain.from_content(
                open(self.urdf_path, "r").read(), "urdf"
            )
            self.joint_names = (
                self.chain.joint_parameter_names[9:15]
                + ["left_arm_gripper"]
                + self.chain.joint_parameter_names[17:23]
                + ["right_arm_gripper"]
            )
            self.forward_kinematics = forward_kinematics_agilex
        elif robot_type == "Genie-1":
            self.urdf_path = os.path.join(urdf_dir, "G1_120s/G1_120s.urdf")
            self.chain = KinematicChain.from_content(
                open(self.urdf_path, "r").read(), "urdf"
            )
            self.joint_names = (
                self.chain.joint_parameter_names[4:11]
                + ["left_arm_gripper"]
                + self.chain.joint_parameter_names[19:26]
                + ["right_arm_gripper"]
            )
            self.forward_kinematics = forward_kinematics_genie

        self.distortion = Distortion(
            model="plumb_bob", coefficients=torch.zeros(5, dtype=torch.float32)
        )

    # Efficient RGB CHW -> JPEG bytes
    def _rgb_chw_to_jpeg_bytes(
        self, img_chw: torch.Tensor, quality: int = 95
    ) -> bytes:
        import cv2

        if img_chw.ndim != 3:
            raise ValueError(f"Expected CHW image, got {tuple(img_chw.shape)}")
        if img_chw.dtype != torch.uint8:
            img_hwc = (
                (img_chw.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0)
            )
        else:
            img_hwc = img_chw.permute(1, 2, 0)
        img_np = img_hwc.cpu().numpy()
        if img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np
        ok, buf = cv2.imencode(
            ".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    def generate_episode_meta(self) -> EpisodeMeta:
        tasks = self.episode_meta.get("tasks", [])
        task_data = TaskData(name=tasks[0], description=tasks[0])

        robot_data = RobotData(
            name=self.dataset.meta.robot_type or "UNKNOWN",
            urdf_content=None,
        )
        return EpisodeMeta(
            episode=EpisodeData(), robot=robot_data, task=task_data
        )

    def parse_episode_data(self):
        def column_to_tensor(column) -> torch.Tensor:
            return torch.from_numpy(np.array(column))

        start_idx = self.episode_meta["dataset_from_index"]
        end_idx = self.episode_meta["dataset_to_index"]
        if start_idx > end_idx:
            logger.warning(
                f"Failed to load episode since start_idx({start_idx}) > end_idx({end_idx})"  # noqa: E501
            )

        indices = list(range(start_idx, end_idx))
        if self.max_frames > 0:
            indices = indices[: self.max_frames]

        # 批量按列取出并转换为 PyTorch tensor（加速，避免逐帧转换）
        batch_dataset = self.dataset.hf_dataset.select(indices)
        self.batch_task_indexs = batch_dataset["task_index"]
        self.batch_tasks = [
            self.dataset.meta.tasks.iloc[task_idx.item()].name
            for task_idx in self.batch_task_indexs
        ]
        self.batch_timestamps = (
            column_to_tensor(batch_dataset["timestamp"]) * 1_000_000_000
        )

        # ee 和 tcp
        self.batch_left_ee = column_to_tensor(
            batch_dataset["states.left_ee_to_robot_pose"]
        )
        self.batch_right_ee = column_to_tensor(
            batch_dataset["states.right_ee_to_robot_pose"]
        )
        self.batch_left_tcp = column_to_tensor(
            batch_dataset["states.left_tcp_to_robot_pose"]
        )
        self.batch_right_tcp = column_to_tensor(
            batch_dataset["states.right_tcp_to_robot_pose"]
        )

        def pose_to_tf(pose, frame_id, ts):
            xyz = pose[:, :3]
            wxyz = pose[:, 3:7]
            tf = BatchFrameTransform(
                parent_frame_id="world",
                child_frame_id=frame_id,
                xyz=xyz,
                quat=wxyz,
                timestamps=ts,
            )
            return tf

        self.batch_left_ee = pose_to_tf(
            self.batch_left_ee, "left_ee", self.batch_timestamps
        )
        self.batch_right_ee = pose_to_tf(
            self.batch_right_ee, "right_ee", self.batch_timestamps
        )
        self.batch_left_tcp = pose_to_tf(
            self.batch_left_tcp, "left_tcp", self.batch_timestamps
        )
        self.batch_right_tcp = pose_to_tf(
            self.batch_right_tcp, "right_tcp", self.batch_timestamps
        )

        # 批量取出 states/actions 并转换为 tensor
        self.batch_states_left = column_to_tensor(
            batch_dataset["states.left_joint.position"]
        )
        self.batch_states_left_gripper = column_to_tensor(
            batch_dataset["states.left_gripper.position"]
        ).unsqueeze(1)  # (N,)
        self.batch_states_right = column_to_tensor(
            batch_dataset["states.right_joint.position"]
        )  # (N, D)
        self.batch_states_right_gripper = column_to_tensor(
            batch_dataset["states.right_gripper.position"]
        ).unsqueeze(1)  # (N,)
        self.batch_states = torch.cat(
            [
                self.batch_states_left,
                self.batch_states_left_gripper,
                self.batch_states_right,
                self.batch_states_right_gripper,
            ],
            dim=1,
        )
        self.batch_joints = BatchJointsState(
            position=self.batch_states.to(dtype=torch.float32),
            names=self.joint_names,
            timestamps=self.batch_timestamps.tolist(),
        )

        self.batch_actions_left = column_to_tensor(
            batch_dataset["actions.left_joint.position"]
        )  # (N, D)
        self.batch_actions_left_gripper = column_to_tensor(
            batch_dataset["actions.left_gripper.position"]
        ).unsqueeze(1)  # (N,)
        self.batch_actions_right = column_to_tensor(
            batch_dataset["actions.right_joint.position"]
        )  # (N, D)
        self.batch_actions_right_gripper = column_to_tensor(
            batch_dataset["actions.right_gripper.position"]
        ).unsqueeze(1)  # (N,)
        self.batch_actions = torch.cat(
            [
                self.batch_actions_left,
                self.batch_actions_left_gripper,
                self.batch_actions_right,
                self.batch_actions_right_gripper,
            ],
            dim=1,
        )
        self.batch_actions = BatchJointsState(
            position=self.batch_actions.to(dtype=torch.float32),
            names=self.joint_names,
            timestamps=self.batch_timestamps.tolist(),
        )

        # 图像的内外参
        self.batch_camera_intrinsic_matrices = {}
        self.batch_camera_to_robot_poses = {}
        for topic in CAMERA_TOPICS:
            batch_camera_intrinsics = column_to_tensor(
                batch_dataset[f"{topic}_camera_intrinsics"]
            )
            self.batch_camera_intrinsic_matrices[topic] = torch.stack(
                [
                    torch.tensor(
                        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                        dtype=torch.float32,
                    )
                    for fx, fy, cx, cy in batch_camera_intrinsics
                ]
            )

            batch_camera_to_robot_extrinsic = column_to_tensor(
                batch_dataset[f"{topic}_camera_to_robot_extrinsics"]
            )
            xyz = batch_camera_to_robot_extrinsic[:, :3]
            wxyz = batch_camera_to_robot_extrinsic[:, 3:7]
            frame_id = f"/observation/cameras/{topic}"
            self.batch_camera_to_robot_poses[topic] = BatchFrameTransform(
                parent_frame_id="world",
                child_frame_id=frame_id,
                xyz=xyz,
                quat=wxyz,
                timestamps=self.batch_timestamps.tolist(),
            )

        # fix extrinsic start
        def matrix_to_tf(matrix, frame_id, ts):
            tf_base_link = Transform3D_M(matrix=matrix)
            xyz = tf_base_link.get_translation()
            wxyz = tf_base_link.get_rotation_quaternion()

            tf_msg = BatchFrameTransform(
                parent_frame_id="base_link",
                child_frame_id=frame_id,
                xyz=xyz,
                quat=wxyz,
                timestamps=self.batch_timestamps.tolist(),
            )
            return tf_msg

        # method1: 全部转到 base 坐标系，transform 中就直接 fk 就好
        link_poses = self.forward_kinematics(self.chain, self.batch_joints)
        if self.robot_type == "ARX Lift-2":
            t_base_leftee = link_poses["L_arm_link16"].get_matrix()
            # t_base_rightee = link_poses["R_arm_link26"].get_matrix()
        elif self.robot_type == "AgileX Split Aloha":
            t_base_leftee = link_poses["left/link6"].get_matrix()
            # t_base_rightee = link_poses["right/link6"].get_matrix()
        elif self.robot_type == "Genie-1":
            t_base_leftee = link_poses["gripper_l_center_link"].get_matrix()
            # t_base_rightee = link_poses["gripper_r_center_link"].get_matrix()

        t_robot_leftee = self.batch_left_ee.as_Transform3D_M().get_matrix()
        t_base_robot = t_base_leftee @ torch.linalg.inv(t_robot_leftee)

        self.batch_camera_to_robot_poses["hand_left"] = matrix_to_tf(
            t_base_robot
            @ self.batch_camera_to_robot_poses["hand_left"]
            .as_Transform3D_M()
            .get_matrix(),
            self.batch_camera_to_robot_poses["hand_left"].child_frame_id,
            self.batch_camera_to_robot_poses["hand_left"].timestamps,
        )
        # batch_camera_to_robot_poses["hand_left"] is T_rightrobot_hand_right
        self.batch_camera_to_robot_poses["hand_right"] = matrix_to_tf(
            t_base_robot
            @ self.batch_camera_to_robot_poses["hand_right"]
            .as_Transform3D_M()
            .get_matrix(),
            self.batch_camera_to_robot_poses["hand_right"].child_frame_id,
            self.batch_camera_to_robot_poses["hand_right"].timestamps,
        )

        self.batch_camera_to_robot_poses["head"] = matrix_to_tf(
            t_base_robot
            @ self.batch_camera_to_robot_poses["head"]
            .as_Transform3D_M()
            .get_matrix(),
            self.batch_camera_to_robot_poses["head"].child_frame_id,
            self.batch_camera_to_robot_poses["head"].timestamps,
        )

        self.batch_images = {topic: [] for topic in CAMERA_TOPICS}
        self.batch_images_msg = {topic: [] for topic in CAMERA_TOPICS}

        from torch.utils.data import DataLoader, Subset

        indices = list(range(start_idx, end_idx))
        if self.max_frames > 0:
            indices = indices[: self.max_frames]
        subset = Subset(self.dataset, indices)  # 包装为 PyTorch Subset

        # 使用 DataLoader
        num_workers = 4
        batch_size = 4
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        for item in loader:
            for topic in CAMERA_TOPICS:
                img_t: torch.Tensor = item.get(f"images.rgb.{topic}")
                self.batch_images[topic].append(img_t)

        for topic in CAMERA_TOPICS:
            self.batch_images[topic] = torch.cat(
                self.batch_images[topic], dim=0
            )
            image_hw = (
                self.batch_images[topic].shape[2],
                self.batch_images[topic].shape[3],
            )
            if image_hw != TARGET_SIZE:
                self.batch_images[topic] = torch.nn.functional.interpolate(
                    self.batch_images[topic],
                    size=TARGET_SIZE,
                    mode="bilinear",
                    align_corners=False,
                )
                scale_y = TARGET_SIZE[0] / float(image_hw[0])
                scale_x = TARGET_SIZE[1] / float(image_hw[1])

                # Adjust intrinsic matrices according to the new image scale
                self.batch_camera_intrinsic_matrices[topic][:, 0, 0] *= (
                    scale_x  # f_x
                )
                self.batch_camera_intrinsic_matrices[topic][:, 1, 1] *= (
                    scale_y  # f_y
                )
                self.batch_camera_intrinsic_matrices[topic][:, 0, 2] *= (
                    scale_x  # c_x
                )
                self.batch_camera_intrinsic_matrices[topic][:, 1, 2] *= (
                    scale_y  # c_y
                )

            sensor_data = [
                self._rgb_chw_to_jpeg_bytes(image)
                for image in self.batch_images[topic]
            ]
            frame_id = f"/observation/cameras/{topic}"
            self.batch_images_msg[topic] = BatchCameraDataEncoded(
                topic=frame_id,
                frame_id=frame_id,
                image_shape=image_hw,
                intrinsic_matrices=self.batch_camera_intrinsic_matrices[topic],
                distortion=self.distortion,
                sensor_data=sensor_data,
                format="jpeg",
                timestamps=self.batch_timestamps,
                pose=self.batch_camera_to_robot_poses[topic],
            )

    def clear_cache(self):
        """Clear large data attributes to free memory."""
        attrs_to_clear = [
            "batch_images",
            "batch_images_msg",
            "batch_camera_intrinsic_matrices",
            "batch_camera_to_robot_posesbatch_joints",
            "batch_actions",
            "batch_states",
            "batch_left_ee",
            "batch_right_ee",
            "batch_left_tcp",
            "batch_right_tcp",
            "batch_states_left",
            "batch_states_right",
            "batch_actions_left",
            "batch_actions_right",
        ]
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)

        import gc

        gc.collect()

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        global episode_count
        print(f"start process Episode [{episode_count}/{len(EPISODE_IDX)}] ")  # noqa: E501
        self.parse_episode_data()

        start_idx = self.episode_meta["dataset_from_index"]
        end_idx = self.episode_meta["dataset_to_index"]
        if self.max_frames > 0:
            self.num_steps = min(end_idx - start_idx, self.max_frames)
        else:
            self.num_steps = end_idx - start_idx

        # 过滤静止帧
        static_mask = np.ones(self.num_steps, dtype=bool)
        static_threshold = 5e-3
        if static_threshold > 0:
            static_mask[1:] = np.any(
                np.abs(np.diff(self.batch_joints.position, axis=0))
                > static_threshold,
                axis=1,
            )
        if "conveyor" in args.input_path or "water" in args.input_path:
            static_mask = np.ones(self.num_steps, dtype=bool)

        for idx in range(self.num_steps):
            if not static_mask[idx]:
                continue

            features: dict[str, Any] = {}
            features["joints"] = self.batch_joints[idx]
            features["actions"] = self.batch_actions[idx]
            for topic in CAMERA_TOPICS:
                features[topic] = get_index_camera(
                    self.batch_images_msg[topic], idx
                )
            features["left_ee"] = get_index_tf(self.batch_left_ee, idx)
            features["right_ee"] = get_index_tf(self.batch_right_ee, idx)
            features["left_tcp"] = get_index_tf(self.batch_left_tcp, idx)
            features["right_tcp"] = get_index_tf(self.batch_right_tcp, idx)

            task = self.batch_tasks[idx]
            instruction = InstructionData(
                name=task,
                json_content={"name": task, "description": task},
            )
            yield DataFrame(
                features=features,
                instruction=instruction,
                timestamp_ns_min=self.batch_timestamps[idx],
                timestamp_ns_max=self.batch_timestamps[idx],
            )

        self.clear_cache()

        episode_count += 1
        print(f"finish process Episode [{episode_count}/{len(EPISODE_IDX)}] ")  # noqa: E501


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

    # Build target features to RoboOrchard message format
    feature_dict: dict[str, Any] = {
        "joints": BatchJointsStateFeature(dtype="float32"),
        "actions": BatchJointsStateFeature(dtype="float32"),
        "left_ee": BatchFrameTransformFeature(dtype="float32"),
        "right_ee": BatchFrameTransformFeature(dtype="float32"),
        "left_tcp": BatchFrameTransformFeature(dtype="float32"),
        "right_tcp": BatchFrameTransformFeature(dtype="float32"),
    }

    CAMERA_TOPICS = ["head", "hand_left", "hand_right"]
    for topic in CAMERA_TOPICS:
        feature_dict[topic] = BatchCameraDataEncodedFeature(dtype="float32")

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

    target_features = hg_datasets.Features(feature_dict)
    print(
        "Target Schema (features):\n{}".format(pprint.pformat(target_features))
    )

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

    ro_dataset_packer = DatasetPackaging(
        features=target_features, check_timestamp=True
    )

    ro_dataset_packer.packaging(
        episodes=episodes_to_package,
        dataset_path=TARGET_LEROBOT_CONVERT_PATH,
        max_shard_size="8GB",
        force_overwrite=True,
    )
