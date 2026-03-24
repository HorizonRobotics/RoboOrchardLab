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

from typing import Callable

import cv2
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import Compose

from robo_orchard_lab.dataset.horizon_manipulation.row_sampler import (
    EpisodeChunkSamplerConfig,
)
from robo_orchard_lab.dataset.robot.dataset import (
    ConcatRODataset,
    ROMultiRowDataset,
)
from robo_orchard_lab.dataset.robot.db_orm import Episode

__all__ = ["Table30RODataset"]


class ArrowDataParse:
    def __init__(
        self,
        cam_names: list[str],
        depth_scale: float = 1000,
        hist_steps: int = 1,
        load_image: bool = True,
        load_depth: bool = True,
        load_extrinsic: bool = True,
    ):
        self.cam_names = cam_names
        self.depth_scale = depth_scale
        self.hist_steps = hist_steps
        self.load_image = load_image
        self.load_depth = load_depth
        self.load_extrinsic = load_extrinsic

    def get_instruction(self, data):
        return {"text": data["instruction"].json_content["description"]}

    def get_intrinsic(self, data):
        intrinsic = []
        for cam_name in self.cam_names:
            intrinsic_mat = np.eye(4, dtype=np.float64)
            intrinsic_mat[:3, :3] = data[cam_name].intrinsic_matrices[0]
            intrinsic.append(intrinsic_mat)
        return {"intrinsic": np.stack(intrinsic)}

    def get_joints(self, data):
        joint_state = [item.position for item in data["joints"]]
        joint_state = np.stack(joint_state).squeeze(1).astype(np.float64)
        return {"joint_state": joint_state}

    def get_master_joints(self, data):
        master_joint_state = [item.position for item in data["actions"]]
        master_joint_state = (
            np.stack(master_joint_state).squeeze(1).astype(np.float64)
        )
        return {"master_joint_state": master_joint_state}

    def get_images(self, data):
        images = []
        for cam_name in self.cam_names:
            img_buffer = data[cam_name].sensor_data[0]
            img_buffer = np.ndarray(
                shape=(1, len(img_buffer)), dtype=np.uint8, buffer=img_buffer
            )
            images.append(cv2.imdecode(img_buffer, cv2.IMREAD_ANYCOLOR))
        return {"imgs": np.stack(images)}

    def get_depths(self, data, default_shape):
        depths = []
        for cam_name in self.cam_names:
            feature_name = f"{cam_name}_depth"
            if feature_name in data and data[feature_name]:
                depth_buffer = data[feature_name].sensor_data[0]
                decoded_depth = cv2.imdecode(
                    np.frombuffer(depth_buffer, np.uint8), cv2.IMREAD_UNCHANGED
                )
            else:
                decoded_depth = np.zeros(default_shape, dtype=np.float32)
            assert (
                decoded_depth is not None
            ), f"Failed to decode depth for {feature_name}"
            depths.append(decoded_depth / self.depth_scale)
        return {"depths": depths}

    def get_extrinsic(self, data):
        t_world2cam = []
        for cam_name in self.cam_names:
            assert data[cam_name].pose.parent_frame_id in (
                "world",
                "base_link",
            )
            extrinsic = np.linalg.inv(
                data[cam_name].pose.as_Transform3D_M().get_matrix()[0].numpy()
            )
            t_world2cam.append(extrinsic)
        return {"T_world2cam": np.stack(t_world2cam).astype(np.float64)}

    def __call__(self, data):
        data.update(self.get_instruction(data))
        data.update(self.get_intrinsic(data))
        data.update(self.get_joints(data))
        data.update(self.get_master_joints(data))

        if self.load_image:
            data.update(self.get_images(data))
        if self.load_depth:
            img_shape = data["imgs"][0].shape[:2]
            data.update(self.get_depths(data, default_shape=img_shape))
        if self.load_extrinsic:
            data.update(self.get_extrinsic(data))

        data["step_index"] = data["frame_index"]
        data["step_index_in_chunk"] = self.hist_steps - 1
        data["task_name"] = data["task"].name
        ep_info = data["episode"].info
        data["uuid"] = ep_info.get("uuid") if ep_info else data["task"].name
        return data


class Table30RODataset(TorchDataset):
    def __init__(
        self,
        paths: list[str],
        target_columns: list[str],
        hist_steps: int,
        pred_steps: int,
        cam_names: list[str],
        depth_scale: int = 1000,
        load_image: bool = True,
        load_depth: bool = True,
        load_extrinsic: bool = True,
        transforms: list[Callable] | None = None,
    ):
        resolved_paths = sorted(set(paths))
        assert len(resolved_paths) > 0, "paths must not be empty"
        row_sampler = EpisodeChunkSamplerConfig(
            target_columns=target_columns,
            hist_steps=hist_steps,
            pred_steps=pred_steps,
        )
        datasets = [
            ROMultiRowDataset(
                dataset_path=path,
                row_sampler=row_sampler,
                meta_index2meta=True,
            )
            for path in resolved_paths
        ]
        parser = ArrowDataParse(
            cam_names=cam_names,
            depth_scale=depth_scale,
            hist_steps=hist_steps,
            load_image=load_image,
            load_depth=load_depth,
            load_extrinsic=load_extrinsic,
        )
        composed = Compose([parser] + (transforms or []))
        for dataset in datasets:
            dataset.set_transform(composed)
        self._concat = ConcatRODataset(datasets)

    @property
    def num_episode(self) -> int:
        return sum(dataset.episode_num for dataset in self._concat.datasets)

    def get_episode_range(self, ep_idx: int) -> tuple[int, int]:
        global_ep = 0
        frame_offset = 0
        for dataset in self._concat.datasets:
            for ep in dataset.iterate_meta(Episode):
                if global_ep == ep_idx:
                    start = frame_offset + ep.dataset_begin_index
                    return start, start + ep.frame_num
                global_ep += 1
            frame_offset += len(dataset)
        raise KeyError(f"Episode index {ep_idx} not found in dataset")

    def __len__(self):
        return len(self._concat)

    def __getitem__(self, idx):
        return self._concat[idx]

    def __getitems__(self, indices):
        return self._concat.__getitems__(indices)
