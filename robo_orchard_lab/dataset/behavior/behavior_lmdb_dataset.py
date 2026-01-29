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

import cv2
import numpy as np

from robo_orchard_lab.dataset.behavior.utils import ROBOT_CAMERA_NAMES
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)


class BehaviorLmdbDataset(BaseLmdbManipulationDataset):
    """Behavior LMDB Dataset.

    Index structure:

    .. code-block:: text

        {episode_idx}:
            ├── uuid: str
            ├── task_name: str
            ├── num_steps: int
            └── simulation: bool

    Meta data structure:

    .. code-block:: text

        {uuid}/meta_data: dict
        {uuid}/camera_names: list(str)
        {uuid}/extrinsic
            └── {cam_name}: np.ndarray[num_steps x 4 x 4]
        {uuid}/intrinsic
            ├── {cam_name}: np.ndarray[3 x 3]
        {uuid}/observation/robot_state/cartesian_position
        {uuid}/observation/robot_state/joint_positions

    Image storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}

    Depth storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}
    """

    def __init__(
        self,
        paths,
        transforms=None,
        interval=None,
        load_image=True,
        load_depth=True,
        task_names=None,
        lazy_init=False,
        num_episode=None,
        cam_names=None,
        T_base2world=None,  # noqa: N803
        T_base2ego=None,  # noqa: N803
        hist_steps=None,
        pred_steps=None,
        reset_step=10000,
        dataset_name="b1k"
    ):
        super().__init__(
            paths=paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=load_depth,
            task_names=task_names,
            lazy_init=lazy_init,
            num_episode=num_episode,
            reset_step=reset_step
        )

        if cam_names is not None:
            self.cam_names = cam_names
        else:
            self.cam_names = ROBOT_CAMERA_NAMES["R1Pro"]

        self.T_base2world = T_base2world
        self.T_base2ego = T_base2ego

        self.hist_steps = hist_steps
        self.pred_steps = pred_steps
        self.dataset_name = dataset_name

    def _concat_shards(self, *shards):
        shards = [x for x in shards if x is not None]
        if len(shards) == 0:
            return None
        elif isinstance(shards[0], np.ndarray):
            return np.concatenate(shards, axis=0)
        elif isinstance(shards[0], list):
            results = []
            for x in shards:
                results.extend(x)
            return results

    def _get_meta_with_shard(
        self,
        lmdb_index,
        uuid,
        key,
        step_index,
        num_steps_per_shard,
    ):
        shard_index = step_index // num_steps_per_shard
        current_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/{shard_index}/{key}"
        ]
        step_index_in_shard = step_index % num_steps_per_shard
        if (
            self.hist_steps is not None
            and step_index_in_shard < self.hist_steps - 1
            and shard_index != 0
        ):
            pre_shard = self.meta_lmdbs[lmdb_index][
                f"{uuid}/{shard_index - 1}/{key}"
            ]
        else:
            pre_shard = None

        if (
            self.pred_steps is not None
            and num_steps_per_shard - step_index_in_shard < self.pred_steps
        ):
            # maby out of bound, return None
            next_shard = self.meta_lmdbs[lmdb_index][
                f"{uuid}/{shard_index + 1}/{key}"
            ]
        else:
            next_shard = None
        if pre_shard is not None:
            step_index_in_shard += len(pre_shard)
        data = self._concat_shards(pre_shard, current_shard, next_shard)
        return data, step_index_in_shard

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        num_steps_per_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/num_steps_per_shard"
        ]
        if num_steps_per_shard is None:
            mobile_traj = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/mobile_traj"
            ]
            joint_state = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/joint_position"
            ]
            action = self.meta_lmdbs[lmdb_index][
                f"{uuid}/robot_action/joint_position"
            ]

            extrinsic = self.meta_lmdbs[lmdb_index][
                f"{uuid}/extrinsic"
            ]
            intrinsic = self.meta_lmdbs[lmdb_index][
                f"{uuid}/intrinsic"
            ]

        else:
            mobile_traj, step_index_in_shard = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "observation/robot_state/mobile_traj",
                step_index,
                num_steps_per_shard,
            )
            joint_state, _ = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "observation/robot_state/joint_position",
                step_index,
                num_steps_per_shard,
            )

            action, _ = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "robot_action/joint_position",
                step_index,
                num_steps_per_shard,
            )

            extrinsic, step_index_in_shard = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "extrinsic",
                step_index,
                num_steps_per_shard,
            )
            extrinsic = extrinsic[step_index_in_shard]

            intrinsic, step_index_in_shard = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "intrinsic",
                step_index,
                num_steps_per_shard,
            )
            intrinsic = intrinsic[step_index_in_shard]

        data = dict(
            uuid=uuid,
            step_index=(
                step_index if step_index_in_shard is None
                else step_index_in_shard
            ),
            mobile_traj=mobile_traj,
            joint_state=joint_state,
            action=action,
            intrinsic=intrinsic,
            T_world2cam=extrinsic,
            T_base2world=copy.deepcopy(self.T_base2world),
            T_base2ego=copy.deepcopy(self.T_base2ego),
        )

        if num_steps_per_shard is not None:
            data["step_index_in_shard"] = step_index_in_shard

        if self.load_image:
            images = []
        if self.load_depth:
            depths = []

        for cam_name in self.cam_names:
            if self.load_image:
                image = self.img_lmdbs[lmdb_index][
                    f"{uuid}/rgb_{cam_name}/{step_index}"
                ]

                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
                #image = image.astype(np.float32) / 255.0
                images.append(image)

            if self.load_depth:
                depth = self.depth_lmdbs[lmdb_index][
                    f"{uuid}/depth_{cam_name}/{step_index}"
                ]

                depth = cv2.imdecode(
                    depth, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
                )
                depth = depth / 1000.0 / 10.0
                depths.append(depth)

        if self.load_image:
            data["imgs"] = images
        if self.load_depth:
            data["depths"] = depths

        instruction = self.meta_lmdbs[lmdb_index][f"{uuid}/instruction"]
        data["text"] = instruction

        subtask_text = self.meta_lmdbs[lmdb_index][f"{uuid}/subtask_text"]
        data["subtask_text"] = subtask_text

        skill_text = self.meta_lmdbs[lmdb_index][f"{uuid}/skill_text"]
        data["skill_text"] = skill_text

        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)

        return data

