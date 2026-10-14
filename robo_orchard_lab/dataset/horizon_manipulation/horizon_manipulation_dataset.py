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

"""Read unsharded Horizon action LMDBs without model dependencies."""

from typing import Callable

import cv2
import numpy as np

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

__all__ = ["HorizonManipulationLmdbDataset"]


class HorizonManipulationLmdbDataset(BaseLmdbManipulationDataset):
    """Read RGB-D frames with observation and controller joint trajectories.

    The root contains index/meta/image/depth LMDBs. Metadata stores full
    ``{uuid}/observation/robot_state/joint_positions`` and
    ``master_joint_positions`` arrays of shape (steps, joints), camera_names,
    intrinsic/extrinsic dictionaries and meta_data.instruction. Frame buffers
    use ``{uuid}/{camera}/{step}``. Sharded metadata is not supported.

    ``__getitem__`` returns the current frame with full episode trajectories,
    then runs transforms in order. ``get_episode_range`` provides the inherited
    half-open global sample range when interval is None. Images remain BGR by
    default; depth is in meters. Packed simulation extrinsics already map the
    robot base to camera: legacy ``T_world2cam`` denotes camera-from-base here,
    and is not inverted. Returned arrays are detached from stored values.

    Database handles are read-only and managed by the existing LMDB base and
    wrapper; each dataset/worker owns its handles. This reader does not perform
    FK and does not require a URDF or a model.

    Args:
        paths (str | list[str]): Action LMDB root directory or directories.
        transforms (list[Callable | dict], optional): Ordered transforms.
            Default is None.
        interval (int, optional): Global frame sampling stride.
            Default is None.
        load_image (bool, optional): Decode images if True; otherwise omit
            imgs. Default is True.
        load_depth (bool, optional): Decode metric depth if True; otherwise
            omit depths without synthesizing zero images. Default is True.
        task_names (str | list[str], optional): Task filter. Default is None.
        lazy_init (bool, optional): Defer opening until first access if True.
            Default is False.
        cam_names (list[str], optional): Camera selection/order, overriding
            stored camera_names. Default is None.
        load_extrinsic (bool, optional): Include current camera-from-base
            matrices if True. Default is True.
        load_calibration (bool, optional): Include calibration when stored
            if True. Default is True.
        load_ee_state (bool, optional): Require cartesian_position and include
            ee_state if True; otherwise leave pose computation to FK.
            Default is False.
        bgr2rgb (bool, optional): Convert decoded BGR to RGB if True.
            Default is False.
        depth_scale (float, optional): Divide packed depth by this positive
            finite scale to obtain meters. Default is 1000.
        **kwargs: Existing base options such as num_episode, dataset_name,
            reset_step and lmdb_kwargs; no sharded-reader options.

    Example::

        dataset = HorizonManipulationLmdbDataset(
            paths="./data/place_a2b/lmdb_balanced_100",
            load_calibration=False,
        )
        sample = dataset[0]
        observations = sample["joint_state"]
        targets = sample["master_joint_state"]
    """

    def __init__(
        self,
        paths: str | list[str],
        transforms: list[Callable | dict] | None = None,
        interval: int | None = None,
        load_image: bool = True,
        load_depth: bool = True,
        task_names: str | list[str] | None = None,
        lazy_init: bool = False,
        cam_names: list[str] | None = None,
        load_extrinsic: bool = True,
        load_calibration: bool = True,
        load_ee_state: bool = False,
        bgr2rgb: bool = False,
        depth_scale: float = 1000,
        **kwargs,
    ):
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale must be positive and finite")
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
        self.load_extrinsic = load_extrinsic
        self.load_calibration = load_calibration
        self.load_ee_state = load_ee_state
        self.bgr2rgb = bgr2rgb
        self.depth_scale = depth_scale

    def get_joint_state(self, lmdb_index: int, data: dict) -> dict:
        """Read aligned full-episode observation and controller targets."""
        arrays = {}
        for key, name in (
            ("joint_positions", "joint_state"),
            ("master_joint_positions", "master_joint_state"),
        ):
            path = f"{data['uuid']}/observation/robot_state/{key}"
            value = self.meta_lmdbs[lmdb_index][path]
            if value is None:
                raise KeyError(f"Missing required LMDB metadata: {path}")
            array = np.array(value)
            if array.ndim != 2 or array.shape[0] != data["num_steps"]:
                raise ValueError(f"Expected (steps, joints) at {path}")
            arrays[name] = array
        if arrays["joint_state"].shape != arrays["master_joint_state"].shape:
            raise ValueError(
                f"Observation/action shapes differ: {data['uuid']}"
            )
        return arrays

    def get_instruction(self, lmdb_index: int, data: dict) -> dict:
        """Read episode text; select one stored variant when provided."""
        meta = self.meta_lmdbs[lmdb_index][f"{data['uuid']}/meta_data"]
        if meta is None:
            raise KeyError(
                f"Missing required LMDB metadata: {data['uuid']}/meta_data"
            )
        instruction = meta.get("instruction")
        if instruction is None or len(instruction) == 0:
            instruction = ""
        elif isinstance(instruction, (list, tuple)):
            instruction = instruction[np.random.randint(len(instruction))]
        return dict(text=instruction, subtask=None)

    def get_intrinsic(self, lmdb_index: int, data: dict) -> dict:
        """Embed stored per-camera intrinsics into homogeneous matrices."""
        path = f"{data['uuid']}/intrinsic"
        intrinsics = self.meta_lmdbs[lmdb_index][path]
        if intrinsics is None:
            raise KeyError(f"Missing required LMDB metadata: {path}")
        matrices = []
        for name in data["cam_names"]:
            matrix = np.eye(4)
            matrix[:3, :3] = np.asarray(intrinsics[name])[:3, :3]
            matrices.append(matrix)
        return dict(intrinsic=np.stack(matrices))

    def get_extrinsic(self, lmdb_index: int, data: dict) -> dict:
        """Select static/current camera-from-base matrices as stored."""
        path = f"{data['uuid']}/extrinsic"
        extrinsics = self.meta_lmdbs[lmdb_index][path]
        if extrinsics is None:
            raise KeyError(f"Missing required LMDB metadata: {path}")
        matrices = []
        for name in data["cam_names"]:
            matrix = np.asarray(extrinsics[name])
            if matrix.ndim == 3:
                matrix = matrix[data["step_index"]]
            matrices.append(matrix)
        # Preserve the consumer key; matrices already map base to camera.
        return dict(T_world2cam=np.stack(matrices))

    def get_images(self, lmdb_index: int, data: dict) -> dict:
        """Decode each selected image in configured order, normally as BGR."""
        images = []
        for name in data["cam_names"]:
            path = f"{data['uuid']}/{name}/{data['step_index']}"
            buffer = self.img_lmdbs[lmdb_index][path]
            if buffer is None:
                raise KeyError(f"Missing LMDB image: {path}")
            image = cv2.imdecode(
                np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError(f"Cannot decode LMDB image: {path}")
            if self.bgr2rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        return dict(imgs=images)

    def get_depths(self, lmdb_index: int, data: dict) -> dict:
        """Decode metric depth; missing/corrupt frames raise errors."""
        depths = []
        for name in data["cam_names"]:
            path = f"{data['uuid']}/{name}/{data['step_index']}"
            buffer = self.depth_lmdbs[lmdb_index][path]
            if buffer is None:
                raise KeyError(f"Missing LMDB depth: {path}")
            depth = cv2.imdecode(
                np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED
            )
            if depth is None:
                raise ValueError(f"Cannot decode LMDB depth: {path}")
            depths.append(depth / self.depth_scale)
        return dict(depths=depths)

    def __getitem__(self, index: int) -> dict:
        """Read one frame plus episode trajectories, then apply transforms.

        Raises:
            NotImplementedError: If metadata is sharded.
            KeyError: If a required field/frame is absent.
            ValueError: If trajectories are misaligned or decoding fails.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")
        lmdb_index, episode_index, step_index = self._get_indices(index)
        row = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        meta_lmdb = self.meta_lmdbs[lmdb_index]
        if meta_lmdb[f"{row.uuid}/num_steps_per_shard"] is not None:
            raise NotImplementedError(
                "Sharded action LMDB metadata is unsupported"
            )
        names = self.cam_names
        if names is None:
            names = meta_lmdb[f"{row.uuid}/camera_names"]
        if not names:
            raise ValueError(f"No cameras selected for episode {row.uuid}")
        data = dict(
            uuid=row.uuid,
            step_index=step_index,
            task_name=row.task_name,
            num_steps=row.num_steps,
            cam_names=list(names),
        )
        data.update(self.get_joint_state(lmdb_index, data))
        if self.load_ee_state:
            path = f"{row.uuid}/observation/robot_state/cartesian_position"
            value = meta_lmdb[path]
            if value is None:
                raise KeyError(f"Missing required LMDB metadata: {path}")
            data["ee_state"] = np.array(value)
        data.update(self.get_intrinsic(lmdb_index, data))
        if self.load_image:
            data.update(self.get_images(lmdb_index, data))
        if self.load_depth:
            data.update(self.get_depths(lmdb_index, data))
        if self.load_extrinsic:
            data.update(self.get_extrinsic(lmdb_index, data))
        if self.load_calibration:
            calibration = meta_lmdb[f"{row.uuid}/calibration"]
            if calibration is not None:
                data["calibration"] = calibration
        data.update(self.get_instruction(lmdb_index, data))
        for transform in self.transforms:
            if transform is not None:
                data = transform(data)
        return data
