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
import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataset,
)
from robo_orchard_lab.utils.build import build

logger = logging.getLogger(__file__)


def cvt_intrinsics_dict_to_mat(intrinsics):
    """Converts camera intrinsics from a dictionary to a 3x3 matrix.

    Handles different camera models by checking the 'distortion_model' key.
    """
    if isinstance(intrinsics, dict):
        if "distortion_model" in intrinsics:
            if intrinsics["distortion_model"] == "equidistant":
                return np.array(
                    [
                        [intrinsics["fu"], 0, intrinsics["pu"], 0],
                        [0, intrinsics["fv"], intrinsics["pv"], 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
            elif intrinsics["distortion_model"] == "plumb bob":
                return np.array(
                    [
                        [intrinsics["fx"], 0, intrinsics["ppx"], 0],
                        [0, intrinsics["fy"], intrinsics["ppy"], 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )
            else:
                logger.warning(
                    f"Unsupported distortion model: {intrinsics['distortion_model']}"
                )
    return intrinsics


class AgiBotLmdbDataset(BaseLmdbManipulationDataset):
    def __init__(
        self,
        paths,
        transforms=None,
        interval=None,
        load_image=True,
        load_depth=True,
        task_names=None,
        lazy_init=False,
        cam_names=None,
        default_space="base",
        lmdb_kwargs=None,
        dataset_name="",
        **kwargs,
    ):
        # Filter out invalid paths before calling the parent constructor.
        valid_paths = []
        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        for path in paths:
            # Check if required LMDB subdirectories exist.
            required_dirs = ["index", "meta"]
            if load_image:
                required_dirs.append("image")
            if load_depth:
                required_dirs.append("depth")

            if all(
                os.path.isdir(os.path.join(path, d)) for d in required_dirs
            ):
                valid_paths.append(path)
            else:
                logger.warning(
                    f"Skipping path due to missing subdirectories: {path}"
                )

        super().__init__(
            paths=valid_paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=load_depth,
            task_names=task_names,
            lazy_init=lazy_init,
            dataset_name=dataset_name,
            **kwargs,
        )
        self.cam_names = cam_names
        assert default_space in ["base", "world", "ego"]
        self.default_space = default_space

        # Define standard camera order: hands first, head last
        self.expected_cam_types = ["hand_left", "hand_right", "head"]
        self.default_head_extrinsic = {
            'rotation_matrix': [
                [-0.012095615235643373, -0.7550696190766631, 0.62648361453548],
                [-0.9983866721384036, 0.0069178206274664745, -0.012049245641575354],
                [0.006845824637032414, -0.6280098898502815, -0.7550385050769443]
            ],
            'translation_vector': [0.4789254482265342, -0.016017166523894975, 1.0054549286501653]
        }

    def _get_task_name(self, index_data):
        return index_data.task_info["task_name"]

    def _load_images_consistent(self, cam_names, uuid, step_index, lmdb_index):
        """Load images with consistent 3-camera format."""
        images = []

        for cam_type in self.expected_cam_types:
            # Find corresponding camera name
            matching_cam = None
            for cam_name in cam_names:
                if (
                    (cam_type == "head" and "head" in cam_name)
                    or (cam_type == "hand_left" and "left" in cam_name)
                    or (cam_type == "hand_right" and "right" in cam_name)
                ):
                    matching_cam = cam_name
                    break

            if matching_cam:
                try:
                    img_key = f"{uuid}/{matching_cam}/{step_index}"
                    img_data = self.img_lmdbs[lmdb_index][img_key]
                    if img_data is not None:
                        # Decode image from byte buffer
                        image = cv2.imdecode(
                            np.frombuffer(img_data, np.uint8),
                            cv2.IMREAD_UNCHANGED,
                        )
                        # Convert BGR to RGB
                        # if len(image.shape) == 3 and image.shape[2] == 3:
                        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
                    else:
                        # Create dummy black image
                        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
                        images.append(dummy_image)
                        # logger.warning(f"No image data for {matching_cam}, using dummy image")
                except KeyError:
                    # Create dummy black image
                    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    images.append(dummy_image)
                    # logger.warning(f"Missing image key for {matching_cam}, using dummy image")
            else:
                # Missing camera - add dummy black image
                dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
                images.append(dummy_image)
                # logger.warning(f"Missing camera {cam_type}, using dummy image")

        return images

    def _load_depths_consistent(self, cam_names, uuid, step_index, lmdb_index):
        """Load depths with consistent 3-camera format."""
        depths = []

        for cam_type in self.expected_cam_types:
            # Find corresponding camera name
            matching_cam = None
            for cam_name in cam_names:
                if (
                    (cam_type == "head" and "head" in cam_name)
                    or (cam_type == "hand_left" and "left" in cam_name)
                    or (cam_type == "hand_right" and "right" in cam_name)
                ):
                    matching_cam = cam_name
                    break

            if matching_cam:
                try:
                    depth_key = f"{uuid}/{matching_cam}/{step_index}"
                    depth_data = self.depth_lmdbs[lmdb_index][depth_key]
                    if depth_data is not None:
                        depth = (
                            cv2.imdecode(
                                np.frombuffer(depth_data, np.uint8),
                                cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
                            )
                            / 1000.0
                        )  # Convert to meters
                        depths.append(depth)
                    else:
                        # Create dummy zero depth
                        dummy_depth = np.zeros((480, 640), dtype=np.float64)
                        depths.append(dummy_depth)
                        # logger.warning(f"No depth data for {matching_cam}, using dummy depth")
                except KeyError:
                    # Create dummy zero depth
                    dummy_depth = np.zeros((480, 640), dtype=np.float64)
                    depths.append(dummy_depth)
                    # logger.warning(f"Missing depth key for {matching_cam}, using dummy depth")
            else:
                # Missing camera - add dummy zero depth
                dummy_depth = np.zeros((480, 640), dtype=np.float64)
                depths.append(dummy_depth)
                logger.warning(f"Missing camera {cam_type}, using dummy depth")

        return depths

    def _convert_cam2world_to_world2cam(self, T_cam2world_raw):
        """Convert AgiBot's cam2world format to world2cam 4x4 matrices.

        Args:
            T_cam2world_raw: List of dict format {'rotation_matrix': [...], 'translation_vector': [...]}

        Returns:
            T_world2cam: List of 4x4 numpy arrays (world2cam transformation matrices)
        """
        T_world2cam = []

        for cam_transform in T_cam2world_raw:
            if isinstance(cam_transform, dict):
                # Extract rotation matrix and translation vector
                R_cam2world = np.array(cam_transform["rotation_matrix"])  # 3x3
                t_cam2world = np.array(
                    cam_transform["translation_vector"]
                )  # 3x1

                # Build 4x4 cam2world matrix
                T_cam2world = np.eye(4)
                T_cam2world[:3, :3] = R_cam2world
                T_cam2world[:3, 3] = t_cam2world

                # Compute world2cam (inverse of cam2world)
                T_world2cam_mat = np.linalg.inv(T_cam2world)
                T_world2cam.append(T_world2cam_mat)

            elif isinstance(
                cam_transform, np.ndarray
            ) and cam_transform.shape == (4, 4):
                # Already in matrix format, just compute inverse
                T_world2cam_mat = np.linalg.inv(cam_transform)
                T_world2cam.append(T_world2cam_mat)
            else:
                logger.error(
                    f"Unsupported camera transform format: {type(cam_transform)}"
                )
                # Fallback to identity matrix
                T_world2cam.append(np.eye(4))

        return T_world2cam

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)
        idx_data = self.idx_lmdbs[lmdb_index][episode_index]
        uuid = idx_data["uuid"]

        # Load camera names - ensure exactly 3 cameras for consistent batch tensor shapes
        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            all_cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]
            # Filter out fisheye cameras, keep 3 views: head + left arm + right arm
            filtered_cams = [
                cam
                for cam in all_cam_names
                if "fisheye" not in cam
                and ("head" in cam or "left" in cam or "right" in cam)
            ]

            # Ensure consistent ordering and exactly 3 cameras: head, left, right
            head_cams = [cam for cam in filtered_cams if "head" in cam]
            left_cams = [cam for cam in filtered_cams if "left" in cam]
            right_cams = [cam for cam in filtered_cams if "right" in cam]

            # Always return exactly 3 cameras in consistent order: hands first, head last
            cam_names = []
            if left_cams:
                cam_names.append(left_cams[0])  # Take first left camera
            if head_cams:
                cam_names.append(head_cams[0])  # Take first head camera
            if right_cams:
                cam_names.append(right_cams[0])  # Take first right camera

            if len(cam_names) != 3:
                logger.warning(
                    f"Sample {uuid} has {len(cam_names)} cameras instead of 3: {cam_names}"
                )

        # Load camera parameters
        _T_cam2world = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]
        if _T_cam2world["head"][0]["translation_vector"][1] > 0.2:
            _T_cam2world["head"] = copy.deepcopy(self.default_head_extrinsic)
        _intrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        # Load images and depths with consistent 3-camera format
        images = (
            self._load_images_consistent(
                cam_names, uuid, step_index, lmdb_index
            )
            if self.load_image
            else None
        )
        depths = (
            self._load_depths_consistent(
                cam_names, uuid, step_index, lmdb_index
            )
            if self.load_depth
            else None
        )

        # Process camera extrinsics and intrinsics with bounds checking
        # Always ensure exactly 3 cameras with consistent tensor shapes
        T_cam2world_raw = []
        intrinsic = []

        for cam_type in self.expected_cam_types:
            # Find corresponding camera name
            matching_cam = None
            for cam_name in cam_names:
                if (
                    (cam_type == "head" and "head" in cam_name)
                    or (cam_type == "hand_left" and "left" in cam_name)
                    or (cam_type == "hand_right" and "right" in cam_name)
                ):
                    matching_cam = cam_name
                    break

            if (
                matching_cam
                and matching_cam in _T_cam2world
                and matching_cam in _intrinsic
            ):
                # Process real camera data
                cam_extrinsic_data = _T_cam2world[matching_cam]

                # Check if cam_extrinsic_data is a dict (single element case - list was compressed)
                # or a list (multiple elements case)
                if (
                    isinstance(cam_extrinsic_data, dict)
                    and "rotation_matrix" in cam_extrinsic_data
                ):
                    # Single element case: the outer list was automatically removed,
                    # leaving just the dict with rotation_matrix and translation_vector
                    T_cam2world_raw.append(cam_extrinsic_data)

                elif isinstance(cam_extrinsic_data, list):
                    # Multiple elements case: normal list of dicts
                    total_steps = len(cam_extrinsic_data)

                    if step_index >= total_steps:
                        logger.warning(
                            f"Step index {step_index} out of bounds for camera {matching_cam} "
                            f"(total steps: {total_steps}). Using last available step {total_steps - 1}."
                        )
                        actual_index = total_steps - 1
                    elif step_index < 0:
                        logger.warning(
                            f"Step index {step_index} is negative for camera {matching_cam}. Using step 0."
                        )
                        actual_index = 0
                    else:
                        actual_index = step_index

                    T_cam2world_raw.append(cam_extrinsic_data[actual_index])

                else:
                    logger.error(
                        f"Unexpected cam_extrinsic_data format for camera {matching_cam}: {type(cam_extrinsic_data)}"
                    )
                    raise ValueError(
                        f"Unsupported cam_extrinsic_data format: {type(cam_extrinsic_data)}"
                    )

                # Add real intrinsic
                intrinsic.append(
                    cvt_intrinsics_dict_to_mat(_intrinsic[matching_cam])
                )
            else:
                # Missing camera - add dummy data with identity matrices
                logger.warning(f"Missing camera {cam_type}, adding dummy data")
                dummy_transform = {
                    "rotation_matrix": np.eye(3).tolist(),
                    "translation_vector": [0.0, 0.0, 0.0],
                }
                T_cam2world_raw.append(dummy_transform)

                dummy_intrinsic = np.eye(4)
                intrinsic.append(dummy_intrinsic)

        T_world2cam = self._convert_cam2world_to_world2cam(T_cam2world_raw)

        # Load robot states
        joint_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/joint_positions"
        ]
        ee_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/cartesian_position"
        ]

        # Get instruction directly
        meta_data = self.meta_lmdbs[lmdb_index][f"{uuid}/meta_data"]
        instruction = ""
        subtask = ""
        if self.instruction_reader is not None:
            task_info = self.instruction_reader.get(uuid, step_index)
            if task_info is None:
                task_info = {"instruction": "", "subtask": ""}
            instruction = task_info["instruction"]
            subtask = task_info.get("subtask")
        if instruction == "":
            instruction = meta_data.get("instruction", "")
        if subtask == "":
            subtasks = meta_data.get("subtask", [])
            # First determine the final step_index by checking subtasks
            if step_index < len(subtasks):
                subtask = subtasks[step_index] if subtasks[step_index] else ""

        # Create data dictionary in RH20T style
        data = dict(
            uuid=uuid,
            step_index=step_index,
            intrinsic=np.stack(intrinsic),  # Stack camera intrinsics
            T_world2cam=np.stack(
                T_world2cam
            ),  # Camera extrinsics (world2cam matrices for projection)
            joint_state=joint_state,  # AgiBot: 20 joints [left_arm(7)+gripper(1), right_arm(7)+gripper(1), head(2), body(2)]
            ee_state=ee_state,  # End-effector cartesian positions for both arms
            text=instruction,  # Main task instruction
            subtask=subtask,  # Current frame's subtask description
        )

        # Add optional data
        if self.load_image and images is not None:
            data["imgs"] = images
        if self.load_depth and depths is not None:
            data["depths"] = depths

        # Apply transforms pipeline
        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data

    def visualize(
        self, episode_index, output_path="./vis_data", vis_iterval=1
    ):
        from tqdm import tqdm

        assert self.load_image

        end_idx = self.cumsum_steps[episode_index]
        if episode_index != 0:
            start_idx = self.cumsum_steps[episode_index - 1]
        else:
            start_idx = 0
        videoWriter = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        uuid = self.__getitem__(start_idx)["uuid"]
        file = os.path.join(output_path, f"{self.dataset_name}-{uuid.replace('/', '-')}.mp4")

        logger.info(f"episode start_idx: {start_idx}, end_idx: {end_idx}")
        logger.info(f"video save path: {file}")
        FPS = 30

        first_ee_pose = None
        for i in tqdm(list(range(start_idx, end_idx, vis_iterval))):
            if self.interval is not None:
                i = i // self.interval
            data = self.__getitem__(i)

            logger.info(f"text: {data.get('text')}, subtask: {data.get('subtask')}")

            link_poses = []
            if "hist_robot_state" in data:
                link_poses.append(data["hist_robot_state"][-1][:, 1:])

            if len(link_poses) != 0:
                link_poses = torch.cat(link_poses)

            vis_imgs = self.draw_robot_state(
                data["imgs"],
                data["projection_mat"],
                link_poses,
            )
            if self.load_depth:
                vis_depths = self.depth_visualize(data["depths"])
                vis_depths = np.reshape(
                    vis_depths.transpose(1, 0, 2, 3), vis_imgs.shape
                )
                vis_imgs = np.concatenate([vis_imgs, vis_depths], axis=0)

            if videoWriter is None:
                videoWriter = cv2.VideoWriter(
                    file,
                    fourcc,
                    FPS,
                    vis_imgs.shape[:2][::-1],
                )
            videoWriter.write(vis_imgs)
        videoWriter.release()

    @staticmethod
    def draw_robot_state(
        imgs,
        projection_mat,
        robot_state,
        ee_indices=(7, 15),
        channel_conversion=False,
    ):
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()
        if isinstance(projection_mat, torch.Tensor):
            projection_mat = projection_mat.cpu().numpy()
        if isinstance(robot_state, torch.Tensor):
            robot_state = robot_state.cpu().numpy()

        vis_imgs = []
        for img_index in range(imgs.shape[0]):
            img = imgs[img_index]
            for joint_index in range(len(robot_state)):
                if robot_state.ndim == 2:
                    rot = Rotation.from_quat(
                        robot_state[joint_index, 3:], scalar_first=True
                    ).as_matrix()
                    trans = robot_state[joint_index, :3]
                else:
                    rot = robot_state[joint_index, :3, :3]
                    trans = robot_state[joint_index, :3, 3]

                if joint_index in ee_indices:
                    axis_length = 0.1
                else:
                    axis_length = 0.03
                points = np.float32(
                    [
                        [axis_length, 0, 0],
                        [0, axis_length, 0],
                        [0, 0, axis_length],
                        [0, 0, 0],
                    ]
                )
                points = points @ rot.T + trans

                pts_2d = points @ projection_mat[img_index, :3, :3].T
                pts_2d = pts_2d + projection_mat[img_index, :3, 3]
                depth = pts_2d[:, 2]
                pts_2d = pts_2d[:, :2] / depth[:, None]

                if depth[3] < 0.02:
                    continue

                pts_2d = pts_2d.astype(np.int32)
                for i in range(3):
                    if depth[i] < 0.02:
                        continue
                    cv2.circle(
                        img, (pts_2d[i, 0], pts_2d[i, 1]), 6, (0, 0, 255), -1
                    )
                    if i == 3:
                        continue
                    color = [0, 0, 0]
                    color[i] = 255
                    cv2.line(
                        img,
                        (pts_2d[i, 0], pts_2d[i, 1]),
                        (pts_2d[3, 0], pts_2d[3, 1]),
                        tuple(color),
                        3,
                    )
            vis_imgs.append(img)

        vis_imgs = np.concatenate(vis_imgs, axis=1)
        vis_imgs = np.uint8(vis_imgs)
        if channel_conversion:
            vis_imgs = vis_imgs[..., ::-1]
        return vis_imgs

    @staticmethod
    def depth_visualize(depth, min_depth=0.01, max_depth=1.2, mode="bwr"):
        import matplotlib.pyplot as plt

        mask = depth > 0
        cmap = plt.cm.get_cmap(mode, 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        cmap = cmap[::-1]

        depth_shape = depth.shape
        if max_depth is None:
            max_depth = depth.max()
        if min_depth is None:
            min_depth = depth.min()

        depth = (depth - min_depth) / (max_depth - min_depth)
        index = np.int32(depth * 255)
        index = np.clip(index, a_min=0, a_max=255)
        depth_color = cmap[index].reshape(*depth_shape, 3)
        depth_color = np.where(mask[..., None], depth_color, 0)
        depth_color = np.uint8(depth_color)
        return depth_color
