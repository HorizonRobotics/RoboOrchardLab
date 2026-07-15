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

from __future__ import annotations
import os
from typing import Any

import cv2
import numpy as np

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)
from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb


def _decode_image(image_buffer: bytes | np.ndarray, key: str) -> np.ndarray:
    if isinstance(image_buffer, bytes):
        image_buffer = np.frombuffer(image_buffer, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode RoboDojo image '{key}'.")
    return image


class RoboDojoLmdbDataset(BaseLmdbManipulationDataset):
    """Read RoboDojo episodes packed in RoboOrchard LMDB format."""

    def __init__(
        self,
        paths: str | list[str],
        transforms: list[Any] | None = None,
        interval: int | None = None,
        load_image: bool = True,
        load_depth: bool = False,
        task_names: str | list[str] | None = None,
        lazy_init: bool = False,
        cam_names: list[str] | None = None,
        hist_steps: int | None = None,
        pred_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        if load_depth:
            raise ValueError(
                "RoboDojo HDF5 data does not contain depth images."
            )
        if isinstance(task_names, str):
            task_names = [task_names]
        self.cam_names = cam_names
        super().__init__(
            paths=paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=False,
            task_names=task_names,
            lazy_init=lazy_init,
            hist_steps=hist_steps,
            pred_steps=pred_steps,
            **kwargs,
        )

    def _init_lmdb(self) -> None:
        if self.initialized:
            return
        for path in self.paths:
            meta_lmdb = Lmdb(
                uri=os.path.join(path, "meta"),
                writable=False,
                encoding_mode=self.encoding_mode,
            )
            try:
                if meta_lmdb["__pack_complete__"] is not True:
                    raise RuntimeError(
                        f"RoboDojo LMDB packing is incomplete: {path}"
                    )
            finally:
                meta_lmdb.close()
        super()._init_lmdb()

    def _get_required_meta(self, lmdb_index: int, uuid: str, key: str) -> Any:
        value = self.meta_lmdbs[lmdb_index][f"{uuid}/{key}"]
        if value is None:
            raise KeyError(f"RoboDojo LMDB is missing meta key: {uuid}/{key}")
        return value

    def _get_timeseries(
        self,
        lmdb_index: int,
        uuid: str,
        key: str,
        step_index: int,
        num_steps_per_shard: int | None,
        num_steps: int,
    ) -> np.ndarray:
        if num_steps_per_shard is None:
            return np.asarray(self._get_required_meta(lmdb_index, uuid, key))
        self._require_shard_steps()
        first_step = max(0, step_index + 1 - self.hist_steps)
        last_step = min(num_steps, step_index + 1 + self.pred_steps)
        first_shard = first_step // num_steps_per_shard
        last_shard = (last_step - 1) // num_steps_per_shard
        shards = []
        for shard_index in range(first_shard, last_shard + 1):
            shard = self.meta_lmdbs[lmdb_index][f"{uuid}/{shard_index}/{key}"]
            if shard is None:
                raise KeyError(
                    "RoboDojo LMDB is missing sharded meta key: "
                    f"{uuid}/{shard_index}/{key}"
                )
            shards.append(np.asarray(shard))
        return np.concatenate(shards, axis=0)

    def _get_local_step_index(
        self,
        step_index: int,
        num_steps_per_shard: int | None,
    ) -> int:
        if num_steps_per_shard is None:
            return step_index
        self._require_shard_steps()
        first_step = max(0, step_index + 1 - self.hist_steps)
        first_shard = first_step // num_steps_per_shard
        return step_index - first_shard * num_steps_per_shard

    def _get_instruction(self, lmdb_index: int, uuid: str) -> str:
        instruction = self.meta_lmdbs[lmdb_index][f"{uuid}/instructions"]
        if instruction is None:
            meta_data = self._get_required_meta(lmdb_index, uuid, "meta_data")
            instruction = meta_data.get("instruction", "")
        if isinstance(instruction, str):
            return instruction
        if isinstance(instruction, (list, tuple)) and instruction:
            return str(instruction[np.random.randint(len(instruction))])
        return ""

    def __getitem__(self, index: int) -> dict[str, Any]:
        lmdb_index, episode_index, step_index = self._get_indices(index)
        index_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = index_data.uuid
        camera_names = self.cam_names or self._get_required_meta(
            lmdb_index, uuid, "camera_names"
        )
        known_camera_names = self._get_required_meta(
            lmdb_index, uuid, "camera_names"
        )
        missing_cameras = sorted(set(camera_names) - set(known_camera_names))
        if missing_cameras:
            raise KeyError(
                f"RoboDojo episode {uuid} does not contain cameras: "
                f"{missing_cameras}."
            )

        num_steps_per_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/num_steps_per_shard"
        ]
        step_index_in_shard = self._get_local_step_index(
            step_index, num_steps_per_shard
        )
        joint_state = self._get_timeseries(
            lmdb_index,
            uuid,
            "observation/robot_state/joint_positions",
            step_index,
            num_steps_per_shard,
            index_data.num_steps,
        )
        master_joint_state = self._get_timeseries(
            lmdb_index,
            uuid,
            "observation/robot_state/master_joint_positions",
            step_index,
            num_steps_per_shard,
            index_data.num_steps,
        )
        ee_state = self._get_timeseries(
            lmdb_index,
            uuid,
            "observation/robot_state/cartesian_position",
            step_index,
            num_steps_per_shard,
            index_data.num_steps,
        ).reshape(len(joint_state), -1)
        delta_ee_state = self._get_timeseries(
            lmdb_index,
            uuid,
            "observation/robot_state/delta_cartesian_position",
            step_index,
            num_steps_per_shard,
            index_data.num_steps,
        ).reshape(len(joint_state), -1)
        intrinsic_meta = self._get_required_meta(lmdb_index, uuid, "intrinsic")
        extrinsic_meta = self._get_required_meta(lmdb_index, uuid, "extrinsic")
        intrinsics = []
        world2cam = []
        images = []
        for camera_name in camera_names:
            intrinsic = np.eye(4, dtype=np.float64)
            intrinsic[:3, :3] = intrinsic_meta[camera_name][:3, :3]
            intrinsics.append(intrinsic)

            extrinsic = np.asarray(extrinsic_meta[camera_name])
            if extrinsic.ndim == 3:
                extrinsic = extrinsic[step_index]
            world2cam.append(extrinsic)

            if self.load_image:
                image_key = f"{uuid}/{camera_name}/{step_index}"
                image_buffer = self.img_lmdbs[lmdb_index][image_key]
                if image_buffer is None:
                    raise KeyError(
                        f"RoboDojo LMDB is missing image key: {image_key}"
                    )
                images.append(_decode_image(image_buffer, image_key))

        data: dict[str, Any] = {
            "uuid": uuid,
            "task_name": index_data.task_name,
            "step_index": step_index,
            "step_index_in_shard": step_index_in_shard,
            "cam_names": list(camera_names),
            "joint_state": joint_state,
            "master_joint_state": master_joint_state,
            "ee_state": ee_state,
            "delta_ee_state": delta_ee_state,
            "text": self._get_instruction(lmdb_index, uuid),
            "intrinsic": np.stack(intrinsics),
            "T_world2cam": np.stack(world2cam),
        }
        if self.load_image:
            data["imgs"] = np.stack(images)

        for transform in self.transforms:
            if transform is not None:
                data = transform(data)
        return data
