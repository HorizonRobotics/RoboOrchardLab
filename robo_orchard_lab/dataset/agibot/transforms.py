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

import cv2
import numpy as np
import pytorch_kinematics as pk
import torch
from pytorch3d.transforms import matrix_to_quaternion
from scipy.spatial.transform import Rotation

__all__ = [
    "AddItems",
    "AddScaleShift",
    "ConvertDataType",
    "IdentityTransform",
    "ImageChannelFlip",
    "ItemSelection",
    "SimpleStateSampling",
    "Resize",
    "ToTensor",
    "GetProjectionMat",
    "UnsqueezeBatch",
    "AgiBotDualArmKinematics",
    "JointSelection",
]


logger = logging.getLogger(__name__)


class IdentityTransform:
    def __call__(self, data):
        return data


class ImageChannelFlip:
    def __init__(self, output_channel=None):
        if output_channel is None:
            output_channel = [2, 1, 0]
        self.output_channel = output_channel

    def __call__(self, data):
        if isinstance(data["imgs"], (list, tuple)):
            data["imgs"] = [x[..., self.output_channel] for x in data["imgs"]]
        else:
            data["imgs"] = data["imgs"][..., self.output_channel]
        return data


class AddItems:
    def __init__(self, **kwargs):
        self.items = copy.deepcopy(kwargs)

    def __call__(self, data):
        for k, v in self.items.items():
            data[k] = copy.deepcopy(v)
        return data


class AddScaleShift:
    def __init__(self, scale_shift):
        if isinstance(scale_shift, (list, tuple)):
            scale_shift = torch.Tensor(scale_shift)
        elif isinstance(scale_shift, np.ndarray):
            scale_shift = torch.from_numpy(scale_shift)
        self.scale_shift = scale_shift

    def __call__(self, data):
        data["joint_scale_shift"] = copy.deepcopy(self.scale_shift)
        return data


class JointStateNoise:
    def __init__(self, noise_range, add_to_pred=False):
        self.range = np.array(noise_range)
        self.add_to_pred = add_to_pred

    def __call__(self, data):
        assert "hist_robot_state" not in data
        num_steps, num_joints = data["hist_joint_state"].shape
        if self.add_to_pred:
            num_steps = 1
        noise = np.random.uniform(
            self.range[..., 0],
            self.range[..., 1],
            size=[num_steps, num_joints],
        )
        noise = torch.from_numpy(noise).to(data["hist_joint_state"])
        data["hist_joint_state"] = data["hist_joint_state"] + noise
        if self.add_to_pred:
            data["pred_joint_state"] = data["pred_joint_state"] + noise
        return data


class SimpleStateSampling:
    def __init__(
        self,
        hist_steps,
        pred_steps,
        gripper_indices,
        limitation=3.14,
        static_threshold=1e-3,
        only_hist=False,
    ):
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps
        self.gripper_indices = gripper_indices
        self.limitation = limitation
        self.static_threshold = static_threshold
        self.only_hist = only_hist

    def __call__(self, data):
        if "joint_state" not in data and "hist_joint_state" in data:
            data["hist_joint_state"] = np.clip(
                data["hist_joint_state"], -self.limitation, self.limitation
            )
            return data

        joint_state = copy.deepcopy(data["joint_state"])  # N x num_joint
        non_gripper_indices = [
            i
            for i in range(joint_state.shape[1])
            if i not in self.gripper_indices
        ]
        mask = np.all(
            (joint_state[..., non_gripper_indices] > -self.limitation)
            & (joint_state[..., non_gripper_indices] < self.limitation),
            axis=-1,
        )
        joint_state = np.clip(joint_state, -self.limitation, self.limitation)

        step_index = data["step_index"]
        hist_steps = self.hist_steps
        pred_steps = self.pred_steps

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
            while not mask[idx]:
                idx -= 1
            if idx < 0:
                idx = step_index + 1
                while not mask[idx]:
                    idx += 1
            hist_state = state[max(0, idx - hist_steps) : idx]
        if hist_state.shape[0] != hist_steps:
            padding = np.tile(state[:1], (hist_steps - hist_state.shape[0], 1))
            hist_state = np.concatenate([padding, hist_state], axis=0)
        hist_joint_state = hist_state[:, :num_joint]
        data["hist_joint_state"] = hist_joint_state
        if "ee_state" in data:
            hist_ee_state = hist_state[:, num_joint:]
            data["hist_ee_state"] = hist_ee_state
        if self.only_hist:
            return data

        idx = step_index + 1
        if idx < len(joint_state) - 1 and self.static_threshold > 0:
            static_mask = np.any(
                np.abs(joint_state[idx:] - hist_joint_state[-1])
                > self.static_threshold,
                axis=-1,
            )
            idx += np.argmax(static_mask)

        pred_state = state[idx : idx + pred_steps]
        pred_mask = mask[idx : idx + pred_steps]
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

        data.update(
            pred_joint_state=pred_joint_state,
            pred_mask=pred_mask,
        )
        if "ee_state" in data:
            data.update(
                pred_ee_state=pred_ee_state,
            )
        return data


class UpSampleJointState:
    def __init__(self, pred_steps, hist_steps=None):
        self.pred_steps = pred_steps
        self.hist_steps = hist_steps

    def __call__(self, data):
        joint_state = torch.cat(
            [data["hist_joint_state"][-1:], data["pred_joint_state"]]
        )  # steps x num_joint
        pred_mask = torch.cat([data["pred_mask"][:1], data["pred_mask"]])[
            :, None
        ]
        joint_state = torch.cat(
            [joint_state, pred_mask.to(joint_state)], dim=-1
        )
        joint_state = joint_state.T[None]  # 1 x num_joint x steps

        joint_state = torch.nn.functional.interpolate(
            joint_state, self.pred_steps + 1, mode="linear", align_corners=True
        )
        data["pred_joint_state"] = joint_state[0].T[1:, :-1]
        data["pred_mask"] = joint_state[0].T[1:, -1].to(dtype=torch.bool)
        if (
            self.hist_steps is not None
            and data["hist_joint_state"].shape[0] != self.hist_steps
        ):
            data["hist_joint_state"] = torch.nn.functional.interpolate(
                data["hist_joint_state"].T[None],
                self.hist_steps,
                mode="linear",
                align_corners=True,
            )[0].T
        return data


class Resize:
    def __init__(self, dst_wh, dst_intrinsic=None):
        self.dst_wh = dst_wh
        if isinstance(dst_intrinsic, (list, tuple)):
            dst_intrinsic = np.array(dst_intrinsic)

        if dst_intrinsic is not None:
            _tmp = np.eye(4)
            _tmp[:3, :3] = dst_intrinsic[:3, :3]
            self.dst_intrinsic = _tmp
            u, v = np.arange(dst_wh[0]), np.arange(dst_wh[1])
            u = np.repeat(u[None], dst_wh[1], 0)
            v = np.repeat(v[:, None], dst_wh[0], 1)
            uv = np.stack([u, v, np.ones_like(u)], axis=-1)
            self.dst_pts = uv @ np.linalg.inv(self.dst_intrinsic[:3, :3]).T
        else:
            self.dst_intrinsic = None

    def __call__(self, data):
        if "imgs" in data:
            imgs = data["imgs"]
            resized_imgs = []
        else:
            imgs = None
        if "depths" in data:
            depths = data["depths"]
            resized_depths = []
        else:
            depths = None

        for i in range(data["intrinsic"].shape[0]):
            intrinsic = data["intrinsic"][i]
            inputs = []
            if imgs is not None:
                inputs.append(imgs[i])
            if depths is not None:
                inputs.append(depths[i])
            results, intrinsic = self.resize(inputs, intrinsic)
            data["intrinsic"][i] = intrinsic
            if imgs is not None:
                resized_imgs.append(results[0])
            if depths is not None:
                resized_depths.append(results[-1])
        if imgs is not None:
            data["imgs"] = np.stack(resized_imgs)
        if depths is not None:
            data["depths"] = np.stack(resized_depths)
        data["image_wh"] = np.array(data["imgs"].shape[1:3][::-1])
        return data

    def resize(self, inputs, intrinsic=None):
        if self.dst_intrinsic is not None:
            src_intrinsic = intrinsic[:3, :3]
            src_uv = self.dst_pts @ src_intrinsic.T
            src_uv = src_uv.astype(np.float32)
            for i, x in enumerate(inputs):
                inputs[i] = cv2.remap(
                    x,
                    src_uv[..., 0],
                    src_uv[..., 1],
                    cv2.INTER_LINEAR,
                )
            intrinsic = self.dst_intrinsic
        elif self.dst_wh is not None:
            origin_wh = inputs[0].shape[:2][::-1]
            trans_mat = np.eye(4)
            trans_mat[0, 0] = self.dst_wh[0] / origin_wh[0]
            trans_mat[1, 1] = self.dst_wh[1] / origin_wh[1]
            intrinsic = trans_mat @ intrinsic
            for i, x in enumerate(inputs):
                inputs[i] = cv2.resize(x, self.dst_wh)
        return inputs, intrinsic


class ToTensor:
    def __call__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = self.__call__(v)
            elif isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v)
            elif isinstance(v, (list, tuple)) and all(
                [isinstance(x, np.ndarray) for x in v]
            ):
                data[k] = type(v)([torch.from_numpy(x) for x in v])
        return data


class ConvertDataType:
    def __init__(self, convert_map, strict=True):
        self.convert_map = convert_map
        self.strict = strict

    def __call__(self, data):
        for data_name, dtype in self.convert_map.items():
            if data_name not in data and not self.strict:
                continue
            if isinstance(data[data_name], np.ndarray):
                data[data_name] = data[data_name].astype(dtype)
            elif isinstance(data[data_name], torch.Tensor):
                data[data_name] = data[data_name].to(dtype)
            else:
                raise TypeError(
                    f"Unsupport convert {data_name}'s "
                    f"type {type(data[data_name])} to {dtype}"
                )
        return data


class ItemSelection:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for k in list(data.keys()):
            if k not in self.keys:
                data.pop(k)
        return data


class GetProjectionMat:
    def __init__(self, target_coordinate="ego"):
        assert target_coordinate in ["base", "world", "ego"]
        self.target_coordinate = target_coordinate

    def __call__(self, data):
        intrinsic = data["intrinsic"]
        if self.target_coordinate == "world":
            projection_mat = intrinsic @ data["T_world2cam"]
            embodiedment_mat = data["T_base2world"]
        elif self.target_coordinate == "base":
            projection_mat = (
                intrinsic @ data["T_world2cam"] @ data["T_base2world"]
            )
            embodiedment_mat = torch.eye(4).to(projection_mat)
        elif self.target_coordinate == "ego":
            projection_mat = (
                intrinsic
                @ data["T_world2cam"]
                @ data["T_base2world"]
                @ torch.linalg.inv(data["T_base2ego"])
            )
            embodiedment_mat = data["T_base2ego"]
        data["projection_mat"] = projection_mat
        data["embodiedment_mat"] = embodiedment_mat
        return data


class UnsqueezeBatch:
    def __call__(self, data):
        for k, v in data.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                data[k] = v[None]
            else:
                data[k] = [v]
        return data


class AgiBotDualArmKinematics:
    """GPU-optimized dual-arm kinematics for AgiBot robot using pytorch_kinematics.

    This implementation replaces the CPU-based numpy forward kinematics with
    pytorch_kinematics for GPU acceleration and batch processing.

    AgiBot has 20 joints: left_arm(7)+gripper(1) + right_arm(7)+gripper(1) + head(2) + body(2)
    """

    def __init__(
        self,
        urdf,
        left_arm_joint_id=None,
        right_arm_joint_id=None,
        left_arm_link_keys=None,
        right_arm_link_keys=None,
        left_finger_keys=None,
        right_finger_keys=None,
        use_ee_state=True,
    ):
        """Initialize AgiBot dual-arm kinematics using pytorch_kinematics for GPU acceleration."""
        self.urdf = urdf

        # Build pytorch_kinematics chains
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.chain_gpu = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.chain.to(dtype=torch.float32)
        self.chain_gpu.to(dtype=torch.float32)

        # Set default configurations matching verified implementation
        if left_arm_joint_id is None:
            left_arm_joint_id = list(range(7))  # [0,1,2,3,4,5,6]
        if right_arm_joint_id is None:
            right_arm_joint_id = list(range(8, 15))  # [8,9,10,11,12,13,14]

        if left_arm_link_keys is None:
            left_arm_link_keys = [
                "Link1_l",
                "Link2_l",
                "Link3_l",
                "Link4_l",
                "Link5_l",
                "Link6_l",
                "Link7_l",
            ]
        if right_arm_link_keys is None:
            right_arm_link_keys = [
                "Link1_r",
                "Link2_r",
                "Link3_r",
                "Link4_r",
                "Link5_r",
                "Link6_r",
                "Link7_r",
            ]
        if left_finger_keys is None:
            left_finger_keys = [
                "gripper_center"
            ]  # From URDF: gripper_center for left gripper
        if right_finger_keys is None:
            right_finger_keys = [
                "right_gripper_center"
            ]  # From URDF: right_gripper_center

        # Store configurations
        self.left_arm_joint_id = left_arm_joint_id
        self.right_arm_joint_id = right_arm_joint_id
        self.left_arm_link_keys = left_arm_link_keys
        self.right_arm_link_keys = right_arm_link_keys
        self.left_finger_keys = left_finger_keys
        self.right_finger_keys = right_finger_keys

        # AgiBot specific joint mappings for 20-joint configuration
        self.joint_name_to_data_index = {
            "Joint1_l": 0,
            "Joint2_l": 1,
            "Joint3_l": 2,
            "Joint4_l": 3,
            "Joint5_l": 4,
            "Joint6_l": 5,
            "Joint7_l": 6,
            "left_gripper": 7,
            "Joint1_r": 8,
            "Joint2_r": 9,
            "Joint3_r": 10,
            "Joint4_r": 11,
            "Joint5_r": 12,
            "Joint6_r": 13,
            "Joint7_r": 14,
            "right_gripper": 15,
            "joint_head_pitch": 16,
            "joint_head_yaw": 17,
            "joint_body_pitch": 18,
            "joint_lift_body": 19,
        }

        # Create combined keys list for pytorch_kinematics
        self.keys = (
            self.left_arm_link_keys
            + self.left_finger_keys
            + self.right_arm_link_keys
            + self.right_finger_keys
        )

        # logger.info(f"Initialized AgiBotDualArmKinematics with pytorch_kinematics")
        # logger.info(f"Available URDF joints: {[j.name for j in self.chain.get_joints()]}")
        # logger.info(f"Target link keys: {self.keys}")

    @property
    def joint_relative_pos(self):
        """Joint relative positions for attention mechanisms."""
        # Create relative position matrix for all 20 joints
        joint_idx = torch.cat(
            [
                torch.arange(8),  # Left arm: 7 joints + 1 gripper
                torch.arange(8, 16),  # Right arm: 7 joints + 1 gripper
                torch.arange(16, 20),  # Head + Body: 2 + 2 joints
            ]
        )
        return torch.abs(joint_idx[:, None] - joint_idx)

    def __call__(self, data):
        """Process data dictionary with dual-arm joint states."""
        # Process all joint state keys
        for key in ["pred_joint_state", "joint_state", "hist_joint_state"]:
            if key in data:
                # Convert to robot state using pytorch_kinematics
                robot_state = self.joint_state_to_robot_state(
                    data[key], data.get("embodiedment_mat")
                )
                # Update data with corresponding robot state
                robot_key = key.replace("joint_state", "robot_state")
                data[robot_key] = robot_state

        # Add required metadata
        data["joint_relative_pos"] = self.joint_relative_pos
        data["kinematics"] = self
        return data

    def joint_state_to_robot_state(self, joint_states, embodiedment_mat=None):
        """Convert joint states to robot states using optimized GPU batch computation.

        Args:
            joint_states: Joint positions [batch, seq, N] or [seq, N] where N can be 16, 18, or 20
            embodiedment_mat: Optional transformation matrix

        Returns:
            robot_states: Robot states [batch, seq, N, 8] for input joints
        """
        original_shape = joint_states.shape
        input_device = joint_states.device

        if len(original_shape) == 2:
            joint_states = joint_states.unsqueeze(0)  # Add batch dimension

        batch_size, seq_len, num_joints = joint_states.shape

        # Handle different joint configurations
        if num_joints == 20:
            joint_mapping = self.joint_name_to_data_index
        elif num_joints == 18:
            joint_mapping = {
                **{
                    name: idx
                    for name, idx in self.joint_name_to_data_index.items()
                    if idx <= 15
                },
                "joint_body_pitch": 16,
                "joint_lift_body": 17,
            }
        elif num_joints == 16:
            joint_mapping = {
                name: idx
                for name, idx in self.joint_name_to_data_index.items()
                if idx <= 15
            }
        else:
            logger.error(
                f"Unsupported joint configuration: {num_joints} joints"
            )
            joint_mapping = self.joint_name_to_data_index

        return self._compute_robot_state_batch_optimized(
            joint_states, joint_mapping, embodiedment_mat, original_shape
        )

    def _compute_robot_state_gpu_accelerated(
        self, joint_positions, joint_mapping, embodiedment_mat, device
    ):
        """Compute robot state using GPU-accelerated forward kinematics
        but following the exact same logic as the original CPU implementation.
        """

        joint_states = joint_states.to(torch.float32)

        # Select appropriate chain based on device
        if device.type == "cpu":
            chain = self.chain
        else:
            if self.chain_gpu.device != device:
                self.chain_gpu.to(device=device, dtype=torch.float32)
            chain = self.chain_gpu

        # Prepare joint states for pytorch_kinematics
        all_joints = chain.get_joints()
        all_joint_states = torch.zeros(len(all_joints)).to(device)

        # Map control joints to URDF joints
        urdf_joint_names = [j.name for j in all_joints]
        for joint_name, data_idx in joint_mapping.items():
            if joint_name in urdf_joint_names and data_idx < len(
                joint_positions
            ):
                urdf_idx = urdf_joint_names.index(joint_name)
                all_joint_states[urdf_idx] = float(
                    joint_positions[data_idx]
                )  # Convert numpy to float for torch tensor

        # Perform forward kinematics using pytorch_kinematics (GPU accelerated)
        link_poses_dict = chain.forward_kinematics(
            all_joint_states.unsqueeze(0)
        )

        # Follow the exact same joint processing order as original CPU implementation
        robot_states = []
        joint_order = [
            # Left arm joints
            ("Joint1_l", "Link1_l"),
            ("Joint2_l", "Link2_l"),
            ("Joint3_l", "Link3_l"),
            ("Joint4_l", "Link4_l"),
            ("Joint5_l", "Link5_l"),
            ("Joint6_l", "Link6_l"),
            ("Joint7_l", "Link7_l"),
            # Left gripper
            ("left_gripper", "gripper_center"),
            # Right arm joints
            ("Joint1_r", "Link1_r"),
            ("Joint2_r", "Link2_r"),
            ("Joint3_r", "Link3_r"),
            ("Joint4_r", "Link4_r"),
            ("Joint5_r", "Link5_r"),
            ("Joint6_r", "Link6_r"),
            ("Joint7_r", "Link7_r"),
            # Right gripper
            ("right_gripper", "right_gripper_center"),
            # Head joints (may be filtered out)
            ("joint_head_pitch", None),
            ("joint_head_yaw", None),
            # Body joints (may be filtered out)
            ("joint_body_pitch", None),
            ("joint_lift_body", None),
        ]

        for joint_name, link_key in joint_order:
            # Skip joints not in the current mapping (filtered out)
            if joint_name not in joint_mapping:
                continue

            joint_idx = joint_mapping[joint_name]
            joint_val = joint_positions[joint_idx]

            # Get pose from FK results or use identity for head/body joints
            if link_key and link_key in link_poses_dict:
                pose_transform = link_poses_dict[
                    link_key
                ]  # This is already a single transform for batch of 1
                pose_matrix = (
                    pose_transform.get_matrix()
                )  # This should be [1, 4, 4]
                pose = (
                    pose_matrix[0].cpu().numpy()
                )  # Extract first batch item and convert to [4, 4] numpy
            else:
                # Use identity for head/body joints or missing poses
                pose = np.eye(4)

            # Apply embodiment transform if provided (same as original)
            if embodiedment_mat is not None:
                if isinstance(embodiedment_mat, torch.Tensor):
                    embodiedment_mat_np = embodiedment_mat.cpu().numpy()
                else:
                    embodiedment_mat_np = embodiedment_mat
                pose = embodiedment_mat_np @ pose

            # Extract position and quaternion (exact same logic as original)
            position = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            quaternion = Rotation.from_matrix(rotation_matrix).as_quat()[
                [3, 0, 1, 2]
            ]  # xyzw -> wxyz

            # Combine: [joint_val(1), position(3), quaternion(4)] = 8 elements
            robot_state = np.concatenate([[joint_val], position, quaternion])
            robot_states.append(robot_state)

        return np.stack(robot_states)  # [N, 8] - filtered joints

    def _compute_robot_state_batch_optimized(
        self, joint_states, joint_mapping, embodiedment_mat, original_shape
    ):
        """Optimized batch GPU computation for large batches.

        This method implements true batch processing for maximum GPU utilization,
        replacing the serial for loops with vectorized operations.
        """
        batch_size, seq_len, num_joints = joint_states.shape
        device = joint_states.device
        joint_states = joint_states.to(dtype=torch.float32)

        # Select appropriate chain based on device
        if device.type == "cpu":
            chain = self.chain
        else:
            if self.chain_gpu.device != device:
                self.chain_gpu.to(device=device, dtype=torch.float32)
            chain = self.chain_gpu

        # Flatten batch and sequence dimensions for batch processing
        total_samples = batch_size * seq_len
        joint_states_flat = joint_states.reshape(total_samples, num_joints)

        # Prepare joint states for pytorch_kinematics
        all_joints = chain.get_joints()
        all_joint_states = torch.zeros(total_samples, len(all_joints)).to(
            device
        )

        # Map control joints to URDF joints (vectorized)
        urdf_joint_names = [j.name for j in all_joints]
        for joint_name, data_idx in joint_mapping.items():
            if joint_name in urdf_joint_names and data_idx < num_joints:
                urdf_idx = urdf_joint_names.index(joint_name)
                all_joint_states[:, urdf_idx] = joint_states_flat[:, data_idx]

        # Batch forward kinematics (GPU accelerated)
        link_poses_dict = chain.forward_kinematics(all_joint_states)

        # Process poses in batch
        robot_states_list = []
        joint_order = [
            # Left arm joints
            ("Joint1_l", "Link1_l"),
            ("Joint2_l", "Link2_l"),
            ("Joint3_l", "Link3_l"),
            ("Joint4_l", "Link4_l"),
            ("Joint5_l", "Link5_l"),
            ("Joint6_l", "Link6_l"),
            ("Joint7_l", "Link7_l"),
            # Left gripper
            ("left_gripper", "gripper_center"),
            # Right arm joints
            ("Joint1_r", "Link1_r"),
            ("Joint2_r", "Link2_r"),
            ("Joint3_r", "Link3_r"),
            ("Joint4_r", "Link4_r"),
            ("Joint5_r", "Link5_r"),
            ("Joint6_r", "Link6_r"),
            ("Joint7_r", "Link7_r"),
            # Right gripper
            ("right_gripper", "right_gripper_center"),
            # Head joints (may be filtered out)
            ("joint_head_pitch", None),
            ("joint_head_yaw", None),
            # Body joints (may be filtered out)
            ("joint_body_pitch", None),
            ("joint_lift_body", None),
        ]

        for joint_name, link_key in joint_order:
            # Skip joints not in the current mapping (filtered out)
            if joint_name not in joint_mapping:
                continue

            joint_idx = joint_mapping[joint_name]
            joint_vals = joint_states_flat[:, joint_idx]  # [total_samples]

            # Get poses from FK results or use identity
            if link_key and link_key in link_poses_dict:
                pose_transforms = link_poses_dict[
                    link_key
                ]  # [total_samples, 4, 4]
                pose_matrices = (
                    pose_transforms.get_matrix()
                )  # [total_samples, 4, 4]

                # Apply embodiment transform if provided
                if embodiedment_mat is not None:
                    if isinstance(embodiedment_mat, torch.Tensor):
                        embodiedment_mat = embodiedment_mat.to(device)
                    else:
                        embodiedment_mat = torch.tensor(
                            embodiedment_mat,
                            device=device,
                            dtype=torch.float32,
                        )

                    # Batch matrix multiplication: [total_samples, 4, 4] @ [4, 4] -> [total_samples, 4, 4]
                    if embodiedment_mat.ndim > 2:
                        em_mat = embodiedment_mat.flatten(0, -3)  # [x, 4, 4]
                    else:
                        em_mat = embodiedment_mat.unsqueeze(0)
                    pose_matrices = torch.matmul(em_mat, pose_matrices)

                # Extract positions and orientations (vectorized)
                positions = pose_matrices[:, :3, 3]  # [total_samples, 3]
                rotation_matrices = pose_matrices[
                    :, :3, :3
                ]  # [total_samples, 3, 3]

                # Convert to quaternions using pytorch3d (GPU accelerated)
                quaternions = matrix_to_quaternion(
                    rotation_matrices
                )  # [total_samples, 4] in wxyz format

            else:
                # Use identity for head/body joints or missing poses
                positions = torch.zeros(total_samples, 3, device=device)
                quaternions = (
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                    .unsqueeze(0)
                    .repeat(total_samples, 1)
                )

            # Combine: [joint_val(1), position(3), quaternion(4)] = 8 elements
            robot_state = torch.cat(
                [
                    joint_vals.unsqueeze(1),  # [total_samples, 1]
                    positions,  # [total_samples, 3]
                    quaternions,  # [total_samples, 4]
                ],
                dim=1,
            )  # [total_samples, 8]

            robot_states_list.append(robot_state)

        # Stack all joints and reshape back to original dimensions
        robot_states = torch.stack(
            robot_states_list, dim=1
        )  # [total_samples, num_joints, 8]
        robot_states = robot_states.reshape(batch_size, seq_len, -1, 8)

        # Remove batch dimension if it was added
        if len(original_shape) == 2:
            robot_states = robot_states.squeeze(0)

        return robot_states


class JointSelection:
    """Transform to select specific joints from the full robot state.
    Useful for filtering out head and body joints when only dual-arm control is needed.
    """

    def __init__(self, selected_joints=None, selection_mode="dual_arm"):
        """Args:
        selected_joints: List of joint indices to keep. If None, uses selection_mode.
        selection_mode: Pre-defined selection modes:
            - "dual_arm": Select only dual-arm joints (16 joints: 2x(7+1))
            - "all": Keep all joints (20 joints)
            - "arms_only": Select only arm joints, no grippers (14 joints)
            - "no_head": Keep all joints except head (18 joints: arms+grippers+body)
        """
        if selected_joints is None:
            if selection_mode == "dual_arm":
                # Left arm (7) + left gripper (1) + right arm (7) + right gripper (1) = 16 joints
                selected_joints = list(range(16))
            elif selection_mode == "all":
                # All joints including head and body
                selected_joints = list(range(20))
            elif selection_mode == "arms_only":
                # Only arm joints, no grippers, no head/body
                selected_joints = list(range(7)) + list(
                    range(8, 15)
                )  # Skip grippers at 7,15
            elif selection_mode == "no_head":
                # All joints except head: arms + grippers + body (exclude head joints 16,17)
                # Indices: 0-15 (arms+grippers) + 18-19 (body) = 18 joints total
                selected_joints = list(range(16)) + list(range(18, 20))
            else:
                raise ValueError(f"Unknown selection_mode: {selection_mode}")

        self.selected_joints = selected_joints
        self.selection_mode = selection_mode

        logger.info(
            f"JointSelection initialized with {len(selected_joints)} joints ({selection_mode})"
        )

    def __call__(self, data):
        """Apply joint selection to robot states, joint states, and scale_shift parameters."""
        # Filter robot state data
        for key in ["pred_robot_state", "robot_state", "hist_robot_state"]:
            if key in data:
                original_shape = data[key].shape
                if (
                    len(original_shape) >= 2
                ):  # [batch, joints, features] or [joints, features]
                    joint_dim = -2  # Second to last dimension is joints
                    data[key] = torch.index_select(
                        data[key],
                        joint_dim,
                        torch.tensor(self.selected_joints, dtype=torch.long),
                    ).contiguous()

        # Filter joint state data
        for key in ["pred_joint_state", "joint_state", "hist_joint_state"]:
            if key in data:
                original_shape = data[key].shape
                if len(original_shape) >= 1:  # [batch, joints] or [joints]
                    joint_dim = -1  # Last dimension is joints for joint states
                    data[key] = torch.index_select(
                        data[key],
                        joint_dim,
                        torch.tensor(self.selected_joints, dtype=torch.long),
                    ).contiguous()
        # Filter joint scale_shift parameters to match selected joints
        if "joint_scale_shift" in data:
            original_shape = data["joint_scale_shift"].shape
            if len(original_shape) >= 2:  # [batch, joints, 2] or [joints, 2]
                joint_dim = -2
                data["joint_scale_shift"] = torch.index_select(
                    data["joint_scale_shift"],
                    joint_dim,
                    torch.tensor(self.selected_joints, dtype=torch.long),
                ).contiguous()

        # Filter joint_relative_pos to match selected joints (critical for attention mechanisms)
        if "joint_relative_pos" in data:
            joint_relative_pos = data[
                "joint_relative_pos"
            ]  # [num_joints, num_joints]
            selected_tensor = torch.tensor(
                self.selected_joints, dtype=torch.long
            )

            # Filter both dimensions: [selected_joints, selected_joints]
            filtered_pos = torch.index_select(
                joint_relative_pos, 0, selected_tensor
            )  # Filter rows
            filtered_pos = torch.index_select(
                filtered_pos, 1, selected_tensor
            )  # Filter columns
            data["joint_relative_pos"] = filtered_pos.contiguous()

            logger.debug(
                f"Filtered joint_relative_pos from {joint_relative_pos.shape} to {filtered_pos.shape}"
            )

        return data
