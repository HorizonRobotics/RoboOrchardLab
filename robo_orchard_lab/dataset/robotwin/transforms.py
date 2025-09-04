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
import pytorch_kinematics as pk
import torch
from pytorch3d.transforms import matrix_to_quaternion

__all__ = [
    "AddItems",
    "AddScaleShift",
    "ConvertDataType",
    "ImageChannelFlip",
    "ItemSelection",
    "JointStateNoise",
    "SimpleStateSampling",
    "Resize",
    "ToTensor",
    "DualArmKinematics",
    "GetProjectionMat",
    "UnsqueezeBatch",
]


class ImageChannelFlip:
    def __init__(self, output_channel=None):
        if output_channel is None:
            output_channel = [2, 1, 0]
        self.output_channel = output_channel

    def __call__(self, data):
        if isinstance(data["imgs"], (list, tuple)):
            data["imgs"] = [
                np.ascontiguousarray(x[..., self.output_channel])
                for x in data["imgs"]
            ]
        else:
            data["imgs"] = np.ascontiguousarray(
                data["imgs"][..., self.output_channel]
            )
        return data


class AddItems:
    def __init__(self, to_numpy=True, **kwargs):
        self.items = copy.deepcopy(kwargs)
        for k, v in self.items.items():
            if to_numpy and not isinstance(v, np.ndarray):
                self.items[k] = np.array(v)

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
    def __init__(self, hist_steps, pred_steps):
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps

    def __call__(self, data):
        if "joint_state" not in data and "hist_joint_state" in data:
            return data
        joint_state = data["joint_state"]  # N x num_joint
        step_index = data["step_index"]
        hist_steps = self.hist_steps
        pred_steps = self.pred_steps

        if "ee_state" in data:
            ee_state = data["ee_state"]  # N x [num_gripper*[xyzqxqyqzqw]]
            state = np.concatenate(
                [joint_state, ee_state],
                axis=1,
            )
        else:
            state = joint_state
        num_joint = joint_state.shape[1]

        pred_state = state[step_index + 1 : step_index + 1 + pred_steps]
        if pred_state.shape[0] != pred_steps:
            padding = np.tile(
                state[-1:], (pred_steps - pred_state.shape[0], 1)
            )
            pred_state = np.concatenate([pred_state, padding], axis=0)
        pred_joint_state = pred_state[:, :num_joint]
        if "ee_state" in data:
            pred_ee_state = pred_state[:, num_joint:]

        hist_state = state[
            max(0, step_index + 1 - hist_steps) : step_index + 1
        ]
        if hist_state.shape[0] != hist_steps:
            padding = np.tile(state[:1], (hist_steps - hist_state.shape[0], 1))
            hist_state = np.concatenate([padding, hist_state], axis=0)
        hist_joint_state = hist_state[:, :num_joint]
        if "ee_state" in data:
            hist_ee_state = hist_state[:, num_joint:]

        data.update(
            pred_joint_state=pred_joint_state,
            hist_joint_state=hist_joint_state,
        )
        data.pop("joint_state")
        if "ee_state" in data:
            data.update(
                pred_ee_state=pred_ee_state,
                hist_ee_state=hist_ee_state,
            )
            data.pop("ee_state")
        return data


class UpSampleJointState:
    def __init__(self, pred_steps, hist_steps=None):
        self.pred_steps = pred_steps
        self.hist_steps = hist_steps

    def __call__(self, data):
        joint_state = torch.cat(
            [data["hist_joint_state"][-1:], data["pred_joint_state"]]
        )  # steps x num_joint
        joint_state = joint_state.T[None]  # 1 x num_joint x steps

        joint_state = torch.nn.functional.interpolate(
            joint_state, self.pred_steps + 1, mode="linear", align_corners=True
        )
        data["pred_joint_state"] = joint_state[0].T[1:]
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
            if isinstance(data[data_name], list):
                data[data_name] = torch.tensor(data[data_name])
            if isinstance(data[data_name], np.ndarray):
                data[data_name] = data[data_name].astype(dtype)
            elif isinstance(data[data_name], torch.Tensor):
                if isinstance(dtype, str):
                    dtype = getattr(torch, dtype)
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


class DualArmKinematics:
    def __init__(
        self,
        urdf,
        left_arm_link_keys=None,
        right_arm_link_keys=None,
        left_arm_joint_id=None,
        right_arm_joint_id=None,
        left_finger_keys=None,
        right_finger_keys=None,
    ):
        super().__init__()
        self.urdf = urdf
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        if left_arm_joint_id is None:
            left_arm_joint_id = [10, 11, 12, 13, 14, 15]
        if right_arm_joint_id is None:
            right_arm_joint_id = [18, 19, 20, 21, 22, 23]
        if left_arm_link_keys is None:
            left_arm_link_keys = [
                "fl_link1",
                "fl_link2",
                "fl_link3",
                "fl_link4",
                "fl_link5",
                "fl_link6",
            ]
        if right_arm_link_keys is None:
            right_arm_link_keys = [
                "fr_link1",
                "fr_link2",
                "fr_link3",
                "fr_link4",
                "fr_link5",
                "fr_link6",
            ]
        if left_finger_keys is None:
            left_finger_keys = [
                "fl_link7",
                "fl_link8",
            ]
        if right_finger_keys is None:
            right_finger_keys = [
                "fr_link7",
                "fr_link8",
            ]

        self.left_arm_joint_id = left_arm_joint_id
        self.right_arm_joint_id = right_arm_joint_id
        self.left_arm_link_keys = left_arm_link_keys
        self.right_arm_link_keys = right_arm_link_keys
        self.left_finger_keys = left_finger_keys
        self.right_finger_keys = right_finger_keys

        self.keys = (
            self.left_arm_link_keys
            + self.left_finger_keys
            + self.right_arm_link_keys
            + self.right_finger_keys
        )

    def __eq__(self, other):
        if isinstance(other, DualArmKinematics):
            return self.urdf == other.urdf
        return NotImplemented

    @property
    def joint_relative_pos(self):
        joint_idx = torch.cat(
            [
                torch.arange(len(self.left_arm_link_keys) + 1),
                torch.arange(-1, -(len(self.right_arm_link_keys) + 2), -1),
            ]
        )
        return torch.abs(joint_idx[:, None] - joint_idx)

    def __call__(self, data):
        if "pred_joint_state" in data:
            data["pred_robot_state"] = self.joint_state_to_robot_state(
                data["pred_joint_state"], data.get("embodiedment_mat")
            )
        elif "joint_state" in data:
            data["robot_state"] = self.joint_state_to_robot_state(
                data["joint_state"], data.get("embodiedment_mat")
            )

        if "hist_joint_state" in data:
            data["hist_robot_state"] = self.joint_state_to_robot_state(
                data["hist_joint_state"], data.get("embodiedment_mat")
            )
        data["joint_relative_pos"] = self.joint_relative_pos
        data["kinematics"] = self
        return data

    def joint_state_to_robot_state(self, joint_state, embodiedment_mat=None):
        input_shape = joint_state.shape
        joint_state = joint_state.to(torch.float32)

        if (
            self.chain.device != joint_state.device
            or self.chain.dtype != joint_state.dtype
        ):
            self.chain = self.chain.to(
                device=joint_state.device,
                dtype=joint_state.dtype,
            )

        all_joint_state = torch.zeros(
            [*input_shape[:-1], len(self.chain.get_joints())]
        ).to(joint_state)
        n_left_joint = len(self.left_arm_joint_id)
        n_right_joint = len(self.right_arm_joint_id)
        all_joint_state[..., self.left_arm_joint_id] = joint_state[
            ..., :n_left_joint
        ]
        all_joint_state[..., self.right_arm_joint_id] = joint_state[
            ..., n_left_joint + 1 : n_left_joint + n_right_joint + 1
        ]
        all_joint_state = all_joint_state.flatten(end_dim=-2)
        link_poses_dict = self.chain.forward_kinematics(all_joint_state)

        link_poses = []
        for key in self.keys:
            link_poses.append(link_poses_dict[key])
        link_poses = link_poses[0].stack(*link_poses[1:])
        link_poses = link_poses.get_matrix()  # [N * xxx, 4, 4]

        if embodiedment_mat is not None:
            link_poses = embodiedment_mat @ link_poses

        robot_states = torch.cat(
            [
                link_poses[..., :3, 3],
                matrix_to_quaternion(link_poses[..., :3, :3]),
            ],
            dim=-1,
        )
        robot_states = robot_states.reshape(len(self.keys), -1, 7)

        start = 0
        results = []
        for keys in [
            self.left_arm_link_keys,
            self.left_finger_keys,
            self.right_arm_link_keys,
            self.right_finger_keys,
        ]:
            end = start + len(keys)
            results.append(robot_states[start:end])
            start = end

        results[1] = results[1].mean(dim=0, keepdim=True)
        results[3] = results[3].mean(dim=0, keepdim=True)

        robot_states = torch.cat(results, dim=0)
        robot_states = robot_states.permute(1, 0, 2)
        robot_states = robot_states.reshape(*input_shape[:-1], -1, 7)
        robot_states = torch.cat(
            [joint_state[..., None], robot_states], dim=-1
        )
        return robot_states


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
