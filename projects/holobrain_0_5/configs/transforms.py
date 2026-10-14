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


import copy

import cv2
import numpy as np
import pytorch_kinematics as pk
import torch

__all__ = [
    "MultiArmKinematics",
    "AddItems",
    "ConvertDataType",
    "Resize",
    "ImageChannelFlip",
]


class MultiArmKinematics:
    """Computes multi-chain robot forward kinematics and joint distances.

    This transform converts joint-space trajectories into per-joint robot
    states. Each "arm" in this class is a serial chain segment described by
    its actuated URDF joint indices and output link keys. Despite the name,
    the segments do not have to be physical arms: callers may also model
    head, torso, body, or other serial chains as additional arms when they
    should participate in the same robot-state tensor.

    Output ordering follows the input configuration exactly. For each arm,
    robot states are emitted for all `arm_link_keys`, followed by one gripper
    state when either `finger_keys[i]` is non-empty or `ee_to_gripper[i]` is
    provided. If multiple finger links are configured for one arm, their
    poses are averaged into a single gripper state to keep one gripper slot
    per arm.

    The `joint_relative_pos` matrix is computed as shortest-path distances on
    a joint graph. Consecutive joints within each arm are connected by unit
    edges. Connections between arms are controlled by
    `arm_connection_joint_indices`. By default, all arms are connected at
    their first joint, preserving the historical dual-arm behavior.
    """

    def __init__(
        self,
        urdf,
        arm_link_keys,
        arm_joint_id=None,
        finger_keys=None,
        ee_to_gripper=None,
        arm_connection_joint_indices=None,
        arm_joint_names=None,
    ):
        """Initialize the multi-arm kinematics transform.

        Args:
            urdf (str): Path to the URDF file used to build the kinematic
                chain.
            arm_link_keys (list[list[str]]): Output link names for each arm.
                The order defines the robot-state joint order for the arm.
            arm_joint_id (list[list[int]], optional): URDF joint indices for
                each arm. The joint values from the input state are scattered
                into these indices before running forward kinematics.
            finger_keys (list[list[str]], optional): Finger or gripper link
                names for each arm. All finger poses for one arm are averaged
                into one gripper state. Default is one empty list per arm.
            ee_to_gripper (list[list[list[float]] | None], optional): Optional
                homogeneous transforms from the last configured end-effector
                link to a synthetic gripper frame. It is mutually exclusive
                with `finger_keys[i]` for the same arm. Default is `None` for
                every arm.
            arm_connection_joint_indices (list[int] | list[list[int | None]],
                optional): Joint graph connections between arms used only for
                `joint_relative_pos`.

                When `None`, all arms are connected at joint index 0. A 1D
                list gives one shared connection joint per arm and connects
                every arm pair through those joints. A 2D matrix gives the
                per-pair connection point: `matrix[i][j]` is the joint index
                on arm `i` that connects to arm `j`, and `None` means no edge
                for that direction. For each connected pair, both directions
                must be defined.
            arm_joint_names (list[list[str]], optional): URDF joint names for
                each arm, resolved using the loaded chain. Mutually exclusive
                with arm_joint_id. Default is None.
        """
        super().__init__()
        self.urdf = urdf
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.chain_gpu = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.chain.to(dtype=torch.float32)
        self.chain_gpu.to(dtype=torch.float32)

        self.arm_link_keys = arm_link_keys
        if arm_joint_names is not None:
            if arm_joint_id is not None:
                raise ValueError(
                    "Only one of arm_joint_id and arm_joint_names may be set."
                )
            parameter_names = self.chain.get_joint_parameter_names()
            try:
                arm_joint_id = [
                    [parameter_names.index(name) for name in names]
                    for names in arm_joint_names
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Kinematics URDF does not contain joint {exc.args[0]}"
                ) from exc
        if arm_joint_id is None:
            raise ValueError(
                "One of arm_joint_id and arm_joint_names must be provided."
            )
        self.arm_joint_id = arm_joint_id
        for keys, ids in zip(
            self.arm_link_keys, self.arm_joint_id, strict=False
        ):
            assert len(keys) == len(ids)
        self.num_arms = len(self.arm_link_keys)
        if finger_keys is None:
            finger_keys = [[]] * len(arm_link_keys)
        else:
            assert len(finger_keys) == self.num_arms, (
                "Number of gripper should equal to number of arms"
            )
        self.finger_keys = finger_keys
        if ee_to_gripper is not None:
            assert len(ee_to_gripper) == self.num_arms
            for i in range(len(ee_to_gripper)):
                if ee_to_gripper[i] is not None:
                    assert len(self.finger_keys[i]) == 0
            self.ee_to_gripper = ee_to_gripper
        else:
            self.ee_to_gripper = [None] * self.num_arms
        self.arm_connection_joint_indices = arm_connection_joint_indices

        self.num_joints = 0
        self.num_keys = 0
        self.arm_num_joints = []
        for i, single_arm_link_keys in enumerate(self.arm_link_keys):
            num_joints = len(single_arm_link_keys) + (
                len(self.finger_keys[i]) > 0
                or self.ee_to_gripper[i] is not None
            )  # One extra robot-state slot is reserved for the gripper.
            self.arm_num_joints.append(num_joints)
            self.num_joints += num_joints
            self.num_keys += len(single_arm_link_keys)
            self.num_keys += len(self.finger_keys[i])
            self.num_keys += self.ee_to_gripper[i] is not None
        self.get_joint_relative_pos()

    def get_joint_relative_pos(self):
        """Build the pairwise shortest-path distance matrix between joints.

        The matrix indexes the flattened robot-state joint order. For each
        arm, adjacent joints are connected with distance 1. Inter-arm edges
        are added from `arm_connection_joint_indices`, then Floyd-Warshall is
        used to compute shortest-path distances between all joints.

        The result is stored in `self._joint_relative_pos` and returned via
        the `joint_relative_pos` property as a clone.
        """
        connection_matrix = self._get_arm_connection_matrix()
        graph = torch.full(
            (self.num_joints, self.num_joints),
            fill_value=self.num_joints + 1,
            dtype=torch.long,
        )
        graph.fill_diagonal_(0)

        arm_start_indices = []
        start_idx = 0
        for num_joints in self.arm_num_joints:
            arm_start_indices.append(start_idx)
            if num_joints > 1:
                # Serial joints in one arm are adjacent in robot-state order.
                joint_ids = torch.arange(start_idx, start_idx + num_joints)
                graph[joint_ids[:-1], joint_ids[1:]] = 1
                graph[joint_ids[1:], joint_ids[:-1]] = 1
            start_idx += num_joints

        for i in range(self.num_arms):
            for j in range(i + 1, self.num_arms):
                idx_i = connection_matrix[i][j]
                idx_j = connection_matrix[j][i]
                if idx_i is None and idx_j is None:
                    continue
                if idx_i is None or idx_j is None:
                    raise ValueError(
                        "arm_connection_joint_indices must define both "
                        f"sides for arm pair ({i}, {j}) or neither side."
                    )
                joint_i = arm_start_indices[i] + idx_i
                joint_j = arm_start_indices[j] + idx_j
                graph[joint_i, joint_j] = 1
                graph[joint_j, joint_i] = 1

        # Floyd-Warshall on a tiny dense graph keeps the logic simple and
        # supports arbitrary arm connection matrices.
        for k in range(self.num_joints):
            graph = torch.minimum(
                graph,
                graph[:, k : k + 1] + graph[k : k + 1],
            )

        self._joint_relative_pos = graph
        assert self._joint_relative_pos.shape[0] == self.num_joints
        assert self._joint_relative_pos.shape[1] == self.num_joints

    def _get_arm_connection_matrix(self):
        """Normalize arm connection input into a square matrix.

        Returns:
            list[list[int | None]]: A `num_arms x num_arms` matrix. Diagonal
            entries are always `None`; off-diagonal entries are local joint
            indices or `None` when the arm pair is not connected.
        """
        indices = self.arm_connection_joint_indices
        if indices is None:
            indices = [0] * self.num_arms

        if not isinstance(indices, (list, tuple)):
            raise TypeError(
                "arm_connection_joint_indices must be None, a 1D list, "
                "or a 2D list."
            )
        if len(indices) != self.num_arms:
            raise ValueError(
                "arm_connection_joint_indices length must equal "
                f"num_arms={self.num_arms}, got {len(indices)}."
            )

        if all(not isinstance(x, (list, tuple)) for x in indices):
            self._validate_connection_indices(indices)
            return [
                [None if i == j else indices[i] for j in range(self.num_arms)]
                for i in range(self.num_arms)
            ]

        if not all(isinstance(x, (list, tuple)) for x in indices):
            raise TypeError(
                "arm_connection_joint_indices must be either a 1D list "
                "or a 2D list, not a mixed structure."
            )
        if any(len(row) != self.num_arms for row in indices):
            raise ValueError(
                "2D arm_connection_joint_indices must have shape "
                f"({self.num_arms}, {self.num_arms})."
            )

        matrix = []
        for i, row in enumerate(indices):
            matrix_row = []
            for j, idx in enumerate(row):
                if j == i:
                    matrix_row.append(None)
                    continue
                if idx is not None:
                    self._validate_connection_index(i, idx)
                matrix_row.append(idx)
            matrix.append(matrix_row)
        return matrix

    def _validate_connection_indices(self, indices):
        """Validate a 1D connection-index list."""
        for arm_idx, joint_idx in enumerate(indices):
            self._validate_connection_index(arm_idx, joint_idx)

    def _validate_connection_index(self, arm_idx, joint_idx):
        """Validate one local connection joint index for an arm.

        Args:
            arm_idx (int): Arm index in `self.arm_link_keys`.
            joint_idx (int): Local robot-state joint index within that arm.
        """
        if not isinstance(joint_idx, int):
            raise TypeError(
                "arm_connection_joint_indices values must be int or None, "
                f"got {type(joint_idx)} for arm {arm_idx}."
            )
        if not 0 <= joint_idx < self.arm_num_joints[arm_idx]:
            raise ValueError(
                "arm_connection_joint_indices contains out-of-range "
                f"index {joint_idx} for arm {arm_idx}; valid range is "
                f"[0, {self.arm_num_joints[arm_idx] - 1}]."
            )

    def __eq__(self, other):
        """Compare kinematics instances by URDF path."""
        if isinstance(other, MultiArmKinematics):
            return self.urdf == other.urdf
        return False

    @property
    def joint_relative_pos(self):
        """Return a clone of the cached joint-relative-position matrix."""
        return torch.clone(self._joint_relative_pos)

    def __call__(self, data):
        """Apply the transform to a sample dictionary.

        Args:
            data (dict): Sample containing `pred_joint_state` and/or
                `hist_joint_state`. If neither exists, `joint_state` is used.
                Each joint-state tensor is expected to have shape
                `(..., num_joints)`.

        Returns:
            dict: The input dictionary updated with `pred_robot_state`,
            `hist_robot_state`, or `robot_state`, plus `joint_relative_pos`
            and `kinematics`.
        """
        joint_states = []
        valid_keys = []
        for key in ["pred_joint_state", "hist_joint_state"]:
            if key in data:
                joint_states.append(data[key])
                valid_keys.append(key)
        if len(joint_states) == 0 and "joint_state" in data:
            joint_states.append(data["joint_state"])
            valid_keys.append("joint_state")
        joint_states = torch.cat(joint_states, dim=0)
        robot_states = self.joint_state_to_robot_state(
            joint_states, data.get("embodiedment_mat")
        )
        start_idx = 0
        for key in valid_keys:
            steps = data[key].shape[0]
            data[key.replace("joint", "robot")] = robot_states[
                start_idx : start_idx + steps
            ]
            start_idx += steps

        data["joint_relative_pos"] = self.joint_relative_pos
        data["kinematics"] = self
        return data

    def joint_state_to_robot_state(
        self, joint_state, embodiedment_mat=None, return_matrix=False
    ):
        """Convert joint states to robot states through forward kinematics.

        Args:
            joint_state (torch.Tensor): Joint-state tensor with shape
                `(..., num_joints)`. Values are ordered by arm, with one
                gripper value slot after each arm when that arm has a gripper.
                Gripper value slots are carried into the output state but are
                not scattered into the URDF chain.
            embodiedment_mat (torch.Tensor, optional): Optional homogeneous
                transform(s) applied to every output link pose. It may be a
                single `[4, 4]` matrix or batched matrices broadcastable after
                flattening the leading dimensions.
            return_matrix (bool, optional): If True, return pose matrices with
                shape `(..., num_keys, 4, 4)` instead of robot-state vectors.
                Default is False.

        Returns:
            torch.Tensor: If `return_matrix` is False, returns a tensor of
            shape `(..., num_joints, 8)`. The last dimension is
            `[joint_value, x, y, z, qw, qx, qy, qz]`. If `return_matrix` is
            True, returns homogeneous matrices with shape
            `(..., num_keys, 4, 4)`.
        """
        from pytorch3d.transforms import matrix_to_quaternion

        input_shape = joint_state.shape
        link_poses_dict = self._forward_kinematics_from_packed_joint_state(
            joint_state
        )

        link_poses = []
        split_size = []
        for i, (single_arm_link_keys, single_finger_keys) in enumerate(
            zip(self.arm_link_keys, self.finger_keys, strict=False)
        ):
            for key in single_arm_link_keys + single_finger_keys:
                link_poses.append(link_poses_dict[key].get_matrix())
            num_finger_keys = len(single_finger_keys)
            if self.ee_to_gripper[i] is not None:
                ee_pose = link_poses[-1]
                gripper_pose = ee_pose @ ee_pose.new_tensor(
                    self.ee_to_gripper[i]
                )
                link_poses.append(gripper_pose)
                num_finger_keys += 1

            split_size.extend([len(single_arm_link_keys), num_finger_keys])
        link_poses = torch.cat(link_poses)

        if embodiedment_mat is not None:
            if embodiedment_mat.ndim != 2:
                embodiedment_mat = embodiedment_mat.flatten(0, -3)  # [x, 4, 4]
                embodiedment_mat = embodiedment_mat.repeat(self.num_keys, 1, 1)
            link_poses = embodiedment_mat @ link_poses

        if return_matrix:
            link_poses = link_poses.unflatten(0, (self.num_keys, -1))
            link_poses = link_poses.transpose(0, 1)
            link_poses = link_poses.unflatten(0, input_shape[:-1])
            return link_poses

        robot_states = torch.cat(
            [
                link_poses[..., :3, 3],
                matrix_to_quaternion(link_poses[..., :3, :3]),
            ],
            dim=-1,
        )
        robot_states = robot_states.reshape(self.num_keys, -1, 7)

        results = list(robot_states.split(split_size))
        for i in range(self.num_arms):
            if results[i * 2 + 1].shape[0] > 1:
                # Multiple finger links represent one gripper state in the
                # policy/state tensor, so average their FK poses.
                results[i * 2 + 1] = results[i * 2 + 1].mean(
                    dim=0, keepdim=True
                )

        robot_states = torch.cat(results, dim=0)
        robot_states = robot_states.permute(1, 0, 2)
        robot_states = robot_states.reshape(*input_shape[:-1], -1, 7)
        robot_states = torch.cat(
            [joint_state[..., None].to(torch.float32), robot_states], dim=-1
        )
        return robot_states

    def _forward_kinematics_from_packed_joint_state(self, joint_state):
        """Run FK from policy-ordered joint state."""

        joint_state = joint_state.to(torch.float32)

        if joint_state.device.type == "cpu":
            chain = self.chain
        else:
            if self.chain_gpu.device != joint_state.device:
                self.chain_gpu.to(device=joint_state.device)
            chain = self.chain_gpu

        all_joint_state = torch.zeros(
            [*joint_state.shape[:-1], len(chain.get_joints())]
        ).to(joint_state)

        start = 0
        for i, single_arm_joint_id in enumerate(self.arm_joint_id):
            num_joint = len(single_arm_joint_id)
            # Input state contains one optional gripper slot after each arm,
            # but only actuated arm joints are written into the URDF chain.
            all_joint_state[..., single_arm_joint_id] = joint_state[
                ..., start : start + num_joint
            ]
            start += num_joint + (
                len(self.finger_keys[i]) > 0
                or self.ee_to_gripper[i] is not None
            )
        return chain.forward_kinematics(all_joint_state.flatten(end_dim=-2))


class AddItems:
    """Add configured constant fields to a mutable action sample."""

    def __init__(self, to_numpy=True, **kwargs):
        self.items = copy.deepcopy(kwargs)
        for k, v in self.items.items():
            if to_numpy and not isinstance(v, np.ndarray):
                self.items[k] = self._to_numpy(v)

    def _to_numpy(self, x):
        if isinstance(x, dict):
            return {k: self._to_numpy(v) for k, v in x.items()}
        if isinstance(x, str):
            return x
        return np.array(x)

    def __call__(self, data):
        for k, v in self.items.items():
            data[k] = copy.deepcopy(v)
        return data


class ConvertDataType:
    """Convert selected sample arrays or tensors to configured dtypes."""

    def __init__(self, convert_map, strict=False):
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


class Resize:
    """Resize RGB-D views and update camera calibration consistently."""

    def __init__(
        self, dst_wh, dst_intrinsic=None, strict_image=True, strict_depth=True
    ):
        self.dst_wh = dst_wh
        self.strict_image = strict_image
        self.strict_depth = strict_depth
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
        elif self.strict_image:
            raise KeyError(
                "'imgs' is missing; pass strict_image=False when the "
                "dataset is built with load_image=False."
            )
        else:
            imgs = None
        if "depths" in data:
            depths = data["depths"]
            resized_depths = []
        elif self.strict_depth:
            raise KeyError(
                "'depths' is missing; pass strict_depth=False when the "
                "dataset is built with load_depth=False."
            )
        else:
            depths = None

        if imgs is None and depths is None:
            return data

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
            data["image_wh"] = np.array(data["imgs"].shape[1:3][::-1])
        if depths is not None:
            data["depths"] = np.stack(resized_depths)
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


class ImageChannelFlip:
    """Reorder the last channel dimension of sample images."""

    def __init__(self, output_channel=None, strict_image=True):
        if output_channel is None:
            output_channel = [2, 1, 0]
        self.output_channel = output_channel
        self.strict_image = strict_image

    def __call__(self, data):
        if "imgs" not in data:
            if self.strict_image:
                raise KeyError(
                    "'imgs' is missing; pass strict_image=False when the "
                    "dataset is built with load_image=False."
                )
            return data
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
