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

import random

import cv2
import numpy as np

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)
from robo_orchard_lab.dataset.robocasa.utils import (
    get_gripper_openness,
    osc_action_to_ee_pose,
)


def _decode_image(image_buffer, flags):
    if image_buffer is None:
        return None
    if isinstance(image_buffer, bytes):
        image_buffer = np.frombuffer(image_buffer, np.uint8)
    if isinstance(image_buffer, np.ndarray) and image_buffer.ndim > 1:
        return image_buffer
    image = cv2.imdecode(image_buffer, flags)
    if image is None:
        raise ValueError("Failed to decode RoboCasa image buffer.")
    return image


class RoboCasaLmdbDataset(BaseLmdbManipulationDataset):
    def __init__(
        self,
        paths,
        transforms=None,
        interval=None,
        load_image=True,
        load_depth=False,
        task_names=None,
        lazy_init=False,
        cam_names=None,
        mobile=None,
        mimicgen=None,
        **kwargs,
    ) -> None:
        self.mobile = mobile
        self.mimicgen = mimicgen
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
        assert not self.load_depth
        self.cam_names = cam_names

    def _check_valid(self, index_data):
        if not super()._check_valid(index_data):
            return False
        if self.mobile is not None:
            if self.mobile != index_data.if_mobile:
                return False
        if self.mimicgen is not None:
            if self.mimicgen != index_data.if_mimicgen:
                return False
        return True

    def _get_episode_meta(self, lmdb_index, uuid, key, step_index):
        data = self.meta_lmdbs[lmdb_index][f"{uuid}/{key}"]
        if data is not None:
            return data, step_index

        num_steps_per_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/num_steps_per_shard"
        ]
        if num_steps_per_shard is None:
            raise KeyError(f"RoboCasa LMDB is missing meta key: {uuid}/{key}")
        step_index_in_shard = self._get_step_index_in_shard(
            step_index,
            num_steps_per_shard,
        )
        data = self._get_meta(
            lmdb_index,
            uuid,
            key,
            step_index=step_index,
            num_steps_per_shard=num_steps_per_shard,
        )
        return data, step_index_in_shard

    def _get_instruction(self, lmdb_index, uuid):
        instruction = self.meta_lmdbs[lmdb_index][f"{uuid}/instruction"]
        if instruction is None:
            instruction = self.meta_lmdbs[lmdb_index][f"{uuid}/instructions"]
        if instruction is None:
            meta_data = self.meta_lmdbs[lmdb_index][f"{uuid}/meta_data"]
            if isinstance(meta_data, dict):
                instruction = meta_data.get("instruction")
        if isinstance(instruction, str):
            return instruction
        if isinstance(instruction, (list, tuple)) and len(instruction) > 0:
            return instruction[np.random.randint(len(instruction))]
        return ""

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)
        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        cam_names = (
            self.cam_names
            or self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]
        )
        if isinstance(cam_names[0], list):
            cam_names = random.choice(cam_names)

        intrinsic_meta = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]
        try:
            base2cam_meta, step_index_in_meta = self._get_episode_meta(
                lmdb_index,
                uuid,
                "base2cam",
                step_index,
            )
        except KeyError:
            base2cam_meta, step_index_in_meta = self._get_episode_meta(
                lmdb_index,
                uuid,
                "extrinsic",
                step_index,
            )
        intrinsic = []
        t_base2cam = []
        if self.load_image:
            images = []
        for cam_name in cam_names:
            k4 = np.eye(4, dtype=np.float64)
            cam_intrinsic = intrinsic_meta[cam_name]
            if cam_intrinsic.ndim == 3:
                cam_intrinsic = cam_intrinsic[step_index_in_meta]
            k4[:3, :3] = cam_intrinsic[:3, :3]
            intrinsic.append(k4)
            ext = base2cam_meta[cam_name]
            t_base2cam.append(
                ext[step_index_in_meta] if ext.ndim == 3 else ext
            )
            image = None
            if self.load_image:
                image_buffer = self.img_lmdbs[lmdb_index][
                    f"{uuid}/{cam_name}/{step_index}"
                ]
                image = _decode_image(image_buffer, cv2.IMREAD_UNCHANGED)
                images.append(image)

        gripper_state, _ = self._get_episode_meta(
            lmdb_index,
            uuid,
            "observation/gripper_state",
            step_index,
        )
        gripper_openness = get_gripper_openness(gripper_state)
        ee_state, _ = self._get_episode_meta(
            lmdb_index,
            uuid,
            "observation/ee_state",
            step_index,
        )
        robot_state = np.concatenate([gripper_openness, ee_state], axis=1)
        robot_state = robot_state[:, None].astype(np.float32)

        master_gripper, _ = self._get_episode_meta(
            lmdb_index,
            uuid,
            "action/gripper",
            step_index,
        )
        osc_action, _ = self._get_episode_meta(
            lmdb_index,
            uuid,
            "action/osc_action",
            step_index,
        )
        action_ee_pose = osc_action_to_ee_pose(ee_state, osc_action)
        master_robot_state = np.concatenate(
            [master_gripper, action_ee_pose], axis=1
        )
        master_robot_state = master_robot_state[:, None].astype(np.float32)

        data = {
            "uuid": uuid,
            "step_index": step_index_in_meta,
            "task_name": idx_data.task_name,
            "cam_names": cam_names,
            "intrinsic": np.stack(intrinsic),
            "T_base2cam": np.stack(t_base2cam),
            "text": self._get_instruction(lmdb_index, uuid),
            "robot_state": robot_state,
            "master_robot_state": master_robot_state,
        }
        if self.load_image:
            data["imgs"] = images
        for transform in self.transforms:
            if transform is not None:
                data = transform(data)
        return data
