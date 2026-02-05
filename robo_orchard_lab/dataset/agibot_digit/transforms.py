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
from typing import Type

import cv2
import numpy as np
import torch
from datasets import Dataset as HFDataset

from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
    MultiArmKinematics,
)
from robo_orchard_lab.dataset.robot.row_sampler import (
    MultiRowSampler,
    MultiRowSamplerConfig,
)

__all__ = [
    "ArrowDataParse",
    "AgiBotOmniPickerKinematics",
    "SimpleStateSampling",
]


class EpisodeChunkSampler(MultiRowSampler):

    def __init__(self, cfg: "EpisodeChunkSamplerConfig") -> None:
        self.cfg = cfg
        self.hist_steps = cfg.hist_steps
        self.pred_steps = cfg.pred_steps
        self.chunk_size = self.hist_steps + self.pred_steps

    @property
    def column_rows_keys(self) -> dict[str, list[str]]:
        ret = {}
        for column in self.cfg.target_columns:
            ret[column] = [f"chunk_row_{i}" for i in range(self.chunk_size)]
        return ret

    def sample_row_idx(
        self,
        index_dataset: HFDataset,
        index: int,
    ) -> dict[str, list[int | None]]:

        cur_row = index_dataset[index]
        cur_episode_idx = cur_row["episode_index"]

        dataset_len = len(index_dataset)

        start_idx = max(index - self.hist_steps, 0)
        end_idx = min(index + self.pred_steps, dataset_len - 1)

        while start_idx < dataset_len:
            start_row = index_dataset[start_idx]
            if start_row["episode_index"] == cur_episode_idx:
                break
            start_idx += 1

        while end_idx > 0:
            end_row = index_dataset[end_idx - 1]
            if end_row["episode_index"] == cur_episode_idx:
                break
            end_idx -= 1

        raw_indices = np.arange(
            index - self.hist_steps, index + self.pred_steps
        )
        padded_indices = np.clip(raw_indices, start_idx, end_idx)

        chunk_indices: list[int] = padded_indices.tolist()

        ret = {}
        for column in self.cfg.target_columns:
            ret[column] = chunk_indices

        return ret


class EpisodeChunkSamplerConfig(MultiRowSamplerConfig[EpisodeChunkSampler]):
    """Configuration for the EpisodeSampler."""

    class_type: Type[EpisodeChunkSampler] = EpisodeChunkSampler

    target_columns: list[str]

    hist_steps: int = 1
    pred_steps: int = 64


class ArrowDataParse:
    """The dataset class for manipulation tasks in RoboOrchard.

    Args:
        dataset_path (str): Path to the dataset.
        cam_names (list[str]): List of camera names to load data from.
        load_image (bool): Whether to load image data. Default is True.
        load_depth (bool): Whether to load depth data. Default is True.
        load_extrinsic (bool): Whether to load camera extrinsic data.
            Default is True.
        load_ee_state (bool): Whether to load end-effector state data.
            Default is False.
        transforms (list[dict] or dict, optional): List of transformations to
            apply to the data.
        depth_scale (float): Scale factor for depth data. Default is 1000.
        **kwargs: Additional arguments for the base RODataset class.
    """

    def __init__(
        self,
        cam_names: list[str],
        load_image=True,
        load_depth=True,
        load_extrinsic=True,
        load_ee_state=False,
        depth_scale=1000,
        use_detailed_instruction=False,
    ):
        """Initialize the ManipulationRODataset."""
        self.cam_names = cam_names
        self.load_image = load_image
        self.load_depth = load_depth
        self.load_extrinsic = load_extrinsic
        self.load_ee_state = load_ee_state
        self.depth_scale = depth_scale
        self.use_detailed_instruction = use_detailed_instruction

    def get_instruction(self, data):
        """Parse instruction text from the data."""
        content = data["instruction"].json_content
        if self.use_detailed_instruction:
            action_config = content["action_config"]
            action_text = [c["english_action_text"] for c in action_config]
            text = ";".join(action_text)
        else:
            text = content["description"]
        return {"text": text}

    def get_depths(self, data, default_shape):
        """Parse depth images from the data."""
        depths = []
        for cam_name in self.cam_names:
            featue_name = f"cameras/{cam_name}_depth"
            if featue_name in data and data[featue_name]:
                depth_buffer = data[featue_name][0]
                decoded_depth = cv2.imdecode(
                    np.frombuffer(depth_buffer, np.uint8), cv2.IMREAD_UNCHANGED
                )
            else:
                # fill missing depth
                decoded_depth = np.zeros(default_shape)
            assert (
                decoded_depth is not None
            ), f"Failed to decode depth for {featue_name}"
            depth = decoded_depth / self.depth_scale
            depths.append(depth)
        return {"depths": depths}

    def get_images(self, data):
        """Parse rgb images from the data."""
        images = []
        for cam_name in self.cam_names:
            featue_name = f"cameras/{cam_name}_color"
            img_buffer = data[featue_name][0]
            img_buffer = np.ndarray(
                shape=(1, len(img_buffer)), dtype=np.uint8, buffer=img_buffer
            )
            img = cv2.imdecode(img_buffer, cv2.IMREAD_ANYCOLOR)
            images.append(img)
        return {"imgs": images}

    def get_intrinsic(self, data):
        """Parse camera intrinsic matrices from the data."""
        intrinsic = []
        for cam_name in self.cam_names:
            cam_instrinsic = np.eye(4, dtype=np.float64)
            cam_instrinsic[:3, :3] = data[
                f"camera_intrinsics/{cam_name}"
            ].reshape(3, 3)
            intrinsic.append(cam_instrinsic)
        intrinsic = np.stack(intrinsic)
        return {"intrinsic": intrinsic}

    def get_extrinsic(self, data):
        """Parse camera extrinsic matrices from the data."""
        T_world2cam = []
        for cam_name in self.cam_names:
            tf_graph = data["tf_graph"]
            frame_id = f"{cam_name}_camera_color_optical_frame"
            tf_w2c = tf_graph.get_tf(
                parent_frame_id=frame_id, child_frame_id="world"
            )
            t_w2c = tf_w2c.as_Transform3D_M().get_matrix()[0].numpy()
            T_world2cam.append(t_w2c)
        T_world2cam = np.stack(T_world2cam).astype(np.float64)
        return {"T_world2cam": T_world2cam}

    def get_robot_pose(self, data):
        tf_graph = data["tf_graph"]
        if "base_link" not in tf_graph.nodes:
            t_b2w = np.eye(4, dtype=np.float64)
        else:
            tf_b2w = tf_graph.get_tf(
                parent_frame_id="world", child_frame_id="base_link"
            )
            t_b2w = tf_b2w.as_Transform3D_M().get_matrix()[0].numpy()
            t_b2w = t_b2w.astype(np.float64)
        return {"T_base2world": t_b2w}

    def __call__(self, data):

        if self.load_image:
            data.update(self.get_images(data))
        if self.load_depth:
            img_shape = data["imgs"][0].shape[:2]
            data.update(self.get_depths(data, default_shape=img_shape))
        if self.load_extrinsic:
            data.update(self.get_extrinsic(data))

        data.update(self.get_intrinsic(data))
        data.update(self.get_robot_pose(data))
        data.update(
            {
                "step_index": data["frame_index"],
                "step_index_in_chunk": data["frame_index_in_chunk"],
                "raw_step_index": data.get(
                    "raw_frame_index", data["frame_index"]
                ),
                "master_joint_state": data["joint_state"],
            }
        )

        return data


class AgiBotOmniPickerKinematics(MultiArmKinematics):
    def __init__(
        self,
        urdf,
        arm_link_keys,
        arm_joint_id,
        finger_keys=None,
        ee_to_gripper=None,
        head_link_keys=None,
        head_joint_id=None,
        body_link_keys=None,
        body_joint_id=None,
    ):
        self._orig_num_arms = len(arm_link_keys)
        self._has_head = head_link_keys is not None
        self._has_body = body_link_keys is not None

        full_link_keys = list(arm_link_keys)
        full_joint_ids = list(arm_joint_id)
        full_finger_keys = (
            list(finger_keys)
            if finger_keys is not None
            else [[]] * self._orig_num_arms
        )
        full_ee_to_gripper = (
            list(ee_to_gripper)
            if ee_to_gripper is not None
            else [None] * self._orig_num_arms
        )

        if self._has_head:
            full_link_keys.append(head_link_keys)
            full_joint_ids.append(head_joint_id or [])
            full_finger_keys.append([])
            full_ee_to_gripper.append(None)

        if self._has_body:
            full_link_keys.append(body_link_keys)
            full_joint_ids.append(body_joint_id or [])
            full_finger_keys.append([])
            full_ee_to_gripper.append(None)

        super().__init__(
            urdf=urdf,
            arm_link_keys=full_link_keys,
            arm_joint_id=full_joint_ids,
            finger_keys=full_finger_keys,
            ee_to_gripper=full_ee_to_gripper,
        )

    def get_joint_relative_pos(self):
        part_sizes = []
        for i in range(len(self.arm_link_keys)):
            n = len(self.arm_link_keys[i]) + (
                len(self.finger_keys[i]) > 0
                or self.ee_to_gripper[i] is not None
            )
            part_sizes.append(n)

        total_joints = sum(part_sizes)
        num_parts = len(part_sizes)
        matrix = torch.zeros((total_joints, total_joints), dtype=torch.float32)

        hub_idx = num_parts - 1 if self._has_body else 0

        base_dist = torch.full((num_parts, num_parts), 2.0)
        for i in range(num_parts):
            base_dist[i, i] = 0
            base_dist[i, hub_idx] = base_dist[hub_idx, i] = 1.0

        offsets = torch.cat(
            [torch.tensor([0]), torch.cumsum(torch.tensor(part_sizes), dim=0)]
        )

        for i in range(num_parts):
            idx_i = torch.arange(part_sizes[i])
            for j in range(num_parts):
                idx_j = torch.arange(part_sizes[j])
                if i == j:
                    val = torch.abs(idx_i[:, None] - idx_j)
                else:
                    val = idx_i[:, None] + base_dist[i, j] + idx_j
                matrix[
                    offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]
                ] = val

        self._joint_relative_pos = matrix


class SimpleStateSampling:
    def __init__(
        self,
        hist_steps,
        pred_steps,
        pred_steps_sampling_rate=1,
        use_master_gripper=True,
        use_master_joint=False,
        gripper_indices=None,
        limitation=3.14,
        static_threshold=1e-3,
        only_hist=False,
        check_adc_frames=False,
        adc_skip_frames_prev=20,
        adc_skip_frames_next=10,
        adc_anno_results_data=None,
    ):
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps
        assert (
            isinstance(pred_steps_sampling_rate, int)
            and pred_steps_sampling_rate >= 1
        )
        self.pred_steps_sampling_rate = pred_steps_sampling_rate
        self.use_master_gripper = use_master_gripper
        self.use_master_joint = use_master_joint
        if use_master_joint ^ use_master_gripper:
            assert gripper_indices is not None
        self.gripper_indices = gripper_indices
        self.limitation = limitation
        self.static_threshold = static_threshold
        self.only_hist = only_hist
        self.check_adc_frames = check_adc_frames
        self.adc_skip_frames_prev = adc_skip_frames_prev
        self.adc_skip_frames_next = adc_skip_frames_next
        self.adc_anno_results_data = adc_anno_results_data

    def __call__(self, data):
        if "joint_state" not in data and "hist_joint_state" in data:
            data["hist_joint_state"] = np.clip(
                data["hist_joint_state"], -self.limitation, self.limitation
            )
            return data

        joint_state = copy.deepcopy(data["joint_state"])  # N x num_joint
        mask = np.all(
            (joint_state > -self.limitation) & (joint_state < self.limitation),
            axis=-1,
        )
        joint_state = np.clip(joint_state, -self.limitation, self.limitation)

        # check ADC and set mask
        adc_mask = np.ones_like(mask, dtype=bool)
        if self.check_adc_frames and self.adc_anno_results_data:
            uuid = data["uuid"]
            joint_raw_frame_index = data["joint_raw_frame_index"]
            if uuid in self.adc_anno_results_data:
                adc_step_indexes = self.adc_anno_results_data[uuid]
                for adc_step_index in adc_step_indexes:
                    lower_bound = adc_step_index - self.adc_skip_frames_prev
                    upper_bound = adc_step_index + self.adc_skip_frames_next
                    is_in_adc_zone = (joint_raw_frame_index >= lower_bound) & (
                        joint_raw_frame_index <= upper_bound
                    )
                    adc_mask[is_in_adc_zone] = False
            mask = mask & adc_mask

        if "step_index_in_shard" in data:
            step_index = data["step_index_in_shard"]
        elif "step_index_in_chunk" in data:
            step_index = data["step_index_in_chunk"]
        else:
            step_index = data["step_index"]

        hist_steps = self.hist_steps
        pred_steps = self.pred_steps * self.pred_steps_sampling_rate

        if "ee_state" in data:
            ee_state = data["ee_state"]  # N x [num_gripper*[xyzqxqyqzqw]]
            state = np.concatenate(
                [joint_state, ee_state.reshape(joint_state.shape[0], -1)],
                axis=1,
            )
        else:
            state = joint_state
        num_joint = joint_state.shape[1]

        if mask[step_index]:
            hist_state = state[
                max(0, step_index + 1 - hist_steps) : step_index + 1
            ]
        else:
            idx = step_index
            while idx > 0 and not mask[idx]:
                idx -= 1
            if idx < 0:
                idx = step_index + 1
                while idx < len(mask) and not mask[idx]:
                    idx += 1
            hist_state = state[max(0, idx - hist_steps) : idx]
        if hist_state.shape[0] != hist_steps:
            padding = np.tile(state[:1], (hist_steps - hist_state.shape[0], 1))
            hist_state = np.concatenate([padding, hist_state], axis=0)
        hist_state = np.copy(hist_state)
        hist_joint_state = hist_state[:, :num_joint]

        data["hist_joint_state"] = hist_joint_state
        if "ee_state" in data:
            hist_ee_state = hist_state[:, num_joint:]
            data["hist_ee_state"] = hist_ee_state
        if self.only_hist:
            return data

        if "master_joint_state" in data:
            if self.use_master_gripper and self.use_master_joint:
                joint_state = data["master_joint_state"]
            elif self.use_master_gripper:
                joint_state[:, self.gripper_indices] = data[
                    "master_joint_state"
                ][:, self.gripper_indices]
            elif self.use_master_joint:
                master_joint_state = copy.deepcopy(data["master_joint_state"])
                master_joint_state[:, self.gripper_indices] = joint_state[
                    :, self.gripper_indices
                ]
                joint_state = master_joint_state
            state[:, : joint_state.shape[1]] = joint_state

        idx = step_index + 1
        if idx < len(joint_state) - 1 and self.static_threshold > 0:
            static_mask = np.any(
                np.abs(joint_state[idx:] - hist_joint_state[-1])
                > self.static_threshold,
                axis=-1,
            )
            idx += np.argmax(static_mask)

        pred_state = state[idx : idx + pred_steps].copy()
        pred_mask = mask[idx : idx + pred_steps].copy()
        pred_adc_mask = adc_mask[idx : idx + pred_steps].copy()

        if self.check_adc_frames and not np.all(pred_adc_mask):
            first_invalid_idx = np.argmin(pred_adc_mask)
            pred_adc_mask[first_invalid_idx:] = False

            if first_invalid_idx > 0:
                last_valid_state = pred_state[first_invalid_idx - 1]
                pred_state[first_invalid_idx:] = last_valid_state
            else:
                pred_state[:] = state[
                    step_index
                ]  # all invalid, just repeat current state

            pred_mask = pred_mask & pred_adc_mask

        if pred_state.shape[0] != pred_steps:
            padding = np.tile(
                state[-1:], (pred_steps - pred_state.shape[0], 1)
            )
            pred_mask = np.concatenate(
                [pred_mask, np.zeros(padding.shape[0], dtype=bool)], axis=0
            )
            pred_state = np.concatenate([pred_state, padding], axis=0)
        pred_joint_state = pred_state[:, :num_joint]
        if "ee_state" in data:
            pred_ee_state = pred_state[:, num_joint:]

        # downsample pred steps
        r = self.pred_steps_sampling_rate
        pred_joint_state = pred_joint_state[(r - 1) :: r]
        pred_mask = pred_mask[(r - 1) :: r]
        if "ee_state" in data:
            pred_ee_state = pred_ee_state[(r - 1) :: r]

        data.update(
            pred_joint_state=pred_joint_state,
            pred_mask=pred_mask,
        )
        if "ee_state" in data:
            data.update(
                pred_ee_state=pred_ee_state,
            )

        return data
