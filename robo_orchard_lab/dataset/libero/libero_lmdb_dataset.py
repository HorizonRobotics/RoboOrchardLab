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

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817

from robo_orchard_lab.dataset.libero.utils import (
    pose_inv,
    transform_ee_rotations,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

__all__ = ["LiberoLmdbDataset"]


class LiberoLmdbDataset(BaseLmdbManipulationDataset):
    """Read standard LIBERO RGB-D frames and absolute pose action sequences."""

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
        **kwargs,
    ):
        super().__init__(
            paths=paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=load_depth,
            task_names=task_names,
            lazy_init=lazy_init,
            **kwargs,
        )
        self.cam_names = cam_names

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)
        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        task_name = idx_data.task_name

        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]

        camera_extrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]
        camera_intrinsic = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]

        if self.load_image:
            images = []
        if self.load_depth:
            depths = []

        T_world2cam = []  # noqa: N806
        intrinsic = []

        for cam_name in cam_names:
            if self.load_image:
                image = self.img_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                if isinstance(image, bytes):
                    image = np.frombuffer(image, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                image = image[::-1]
                images.append(image)

            if self.load_depth:
                depth = self.depth_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                if isinstance(depth, bytes):
                    depth = cv2.imdecode(
                        np.frombuffer(depth, np.uint8), cv2.IMREAD_UNCHANGED
                    )
                depth = depth[::-1]
                depths.append(depth[..., 0] if depth.ndim == 3 else depth)

            if camera_intrinsic[cam_name][step_index].shape == (3, 3):
                intrinsic_ = np.eye(4)
                intrinsic_[:3, :3] = camera_intrinsic[cam_name][step_index]

            elif camera_intrinsic[cam_name][step_index].shape == (4, 4):
                intrinsic_ = camera_intrinsic[cam_name][step_index]
            else:
                raise ValueError("Camera intrinsic must be 3x3 or 4x4")

            T_world2cam.append(
                pose_inv(camera_extrinsic[cam_name][step_index])
            )
            intrinsic.append(intrinsic_)

        if self.load_image:
            images = np.stack(images)
        if self.load_depth:
            depths = np.stack(depths)

        T_world2cam = np.stack(T_world2cam)  # noqa: N806
        intrinsic = np.stack(intrinsic)

        ee_states = self.meta_lmdbs[lmdb_index][f"{uuid}/observation/ee_state"]

        r_diff = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        ee_quats = transform_ee_rotations(ee_states, r_diff)
        ee_states = np.concatenate([ee_states[:, :3], ee_quats], axis=1)
        gripper_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/gripper_state"
        ]
        current_width = gripper_state[:, 0] - gripper_state[:, 1]
        MAX_WIDTH = 0.08  # noqa: N806
        gripper_state = current_width / MAX_WIDTH
        robot_states = np.concatenate(
            [gripper_state[:, None], ee_states], axis=1
        )
        robot_states = np.expand_dims(robot_states, axis=1).astype(np.float32)

        actions = self.meta_lmdbs[lmdb_index][f"{uuid}/action"]
        arm_actions = actions[:, :6]
        rot_vecs = arm_actions[:, 3:6]
        rotations = R.from_rotvec(rot_vecs)
        arm_actions_quat = rotations.as_quat()[:, [3, 0, 1, 2]]
        actions = np.concatenate(
            [actions[:, 6:], arm_actions[:, :3], arm_actions_quat], axis=1
        )
        actions = np.expand_dims(actions, axis=1).astype(np.float32)
        actions[..., :1] = (1 - actions[..., :1]) / 2
        data = dict(
            uuid=uuid,
            step_index=step_index,
            joint_state=robot_states,
            action=actions,
            task_name=task_name,
            T_world2cam=T_world2cam,
            intrinsic=intrinsic,
            cam_names=cam_names,
        )
        if self.load_image:
            data["imgs"] = images
        if self.load_depth:
            data["depths"] = depths

        instructions = self.meta_lmdbs[lmdb_index][f"{uuid}/instructions"]
        if isinstance(instructions, str):
            text = instructions
        elif len(instructions) == 0:
            text = ""
        else:
            text = str(instructions[0])

        data["text"] = text
        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data
