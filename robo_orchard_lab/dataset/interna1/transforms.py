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


import pytorch_kinematics as pk
import torch
from pytorch3d.transforms import matrix_to_quaternion

__all__ = [
    "InternA1MultiArmKinematics",
]


class InternA1MultiArmKinematics:
    def __init__(
        self,
        urdf,
        arm_link_keys,
        arm_joint_id=None,
        finger_keys=None,
        ee_to_gripper=None,
    ):
        super().__init__()
        self.urdf = urdf
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.chain_gpu = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.chain.to(dtype=torch.float32)
        self.chain_gpu.to(dtype=torch.float32)

        self.arm_link_keys = arm_link_keys
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

        self.num_joints = 0
        self.num_keys = 0
        for i, single_arm_link_keys in enumerate(self.arm_link_keys):
            self.num_joints += len(single_arm_link_keys)
            self.num_joints += (
                len(self.finger_keys[i]) > 0
                or self.ee_to_gripper[i] is not None
            )  # Gripper
            self.num_keys += len(single_arm_link_keys)
            self.num_keys += len(self.finger_keys[i])
            self.num_keys += self.ee_to_gripper[i] is not None
        self.get_joint_relative_pos()

    def get_joint_relative_pos(self):
        joint_relative_pos = []
        for i, single_arm_joint_id_a in enumerate(self.arm_joint_id):
            joint_ids_a = torch.arange(
                len(single_arm_joint_id_a)
                + (
                    len(self.finger_keys[i]) > 0
                    or self.ee_to_gripper[i] is not None
                )
            )
            joint_relative_pos_per_arm = []
            for j, single_arm_joint_id_b in enumerate(self.arm_joint_id):
                if j == i:
                    joint_ids_b = joint_ids_a
                else:
                    joint_ids_b = torch.arange(
                        -1,
                        -(
                            len(single_arm_joint_id_b)
                            + 1
                            + (len(self.finger_keys[j]) > 0)
                        ),
                        -1,
                    )
                joint_relative_pos_per_arm.append(
                    torch.abs(joint_ids_a[:, None] - joint_ids_b)
                )
            joint_relative_pos_per_arm = torch.cat(
                joint_relative_pos_per_arm, dim=1
            )
            joint_relative_pos.append(joint_relative_pos_per_arm)
        self._joint_relative_pos = torch.cat(joint_relative_pos, dim=0)
        assert self._joint_relative_pos.shape[0] == self.num_joints
        assert self._joint_relative_pos.shape[1] == self.num_joints

    def __eq__(self, other):
        if isinstance(other, InternA1MultiArmKinematics):
            return self.urdf == other.urdf
        return False

    @property
    def joint_relative_pos(self):
        return torch.clone(self._joint_relative_pos)

    def __call__(self, data):
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
        input_shape = joint_state.shape
        joint_state = joint_state.to(torch.float32)

        if joint_state.device.type == "cpu":
            chain = self.chain
        else:
            if self.chain_gpu.device != joint_state.device:
                self.chain_gpu.to(device=joint_state.device)
            chain = self.chain_gpu

        all_joint_state = torch.zeros(
            [*input_shape[:-1], len(chain.get_joints())]
        ).to(joint_state)

        start = 0
        for i, single_arm_joint_id in enumerate(self.arm_joint_id):
            num_joint = len(single_arm_joint_id)
            all_joint_state[..., single_arm_joint_id] = joint_state[
                ..., start : start + num_joint
            ]
            start += num_joint + (
                len(self.finger_keys[i]) > 0
                or self.ee_to_gripper[i] is not None
            )
        all_joint_state = all_joint_state.flatten(end_dim=-2)
        link_poses_dict = chain.forward_kinematics(all_joint_state)

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
        # link_poses = link_poses[0].stack(*link_poses[1:])
        # link_poses = link_poses.get_matrix()  # [N * xxx, 4, 4]
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
                results[i * 2 + 1] = results[i * 2 + 1].mean(
                    dim=0, keepdim=True
                )

        robot_states = torch.cat(results, dim=0)
        robot_states = robot_states.permute(1, 0, 2)
        robot_states = robot_states.reshape(*input_shape[:-1], -1, 7)
        robot_states = torch.cat(
            [joint_state[..., None], robot_states], dim=-1
        )
        return robot_states
