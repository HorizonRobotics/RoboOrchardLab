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

import pytest
import torch

from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
    MultiArmKinematics,
)


class _FakeChain:
    def to(self, *args, **kwargs):
        return self


@pytest.fixture()
def fake_urdf(tmp_path, monkeypatch):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot />")

    def build_chain_from_urdf(_):
        return _FakeChain()

    monkeypatch.setattr(
        "robo_orchard_lab.dataset.horizon_manipulation.transforms.pk"
        ".build_chain_from_urdf",
        build_chain_from_urdf,
    )
    return str(urdf)


def _make_kinematics(fake_urdf, connection_indices=None):
    return MultiArmKinematics(
        urdf=fake_urdf,
        arm_joint_id=[
            list(range(4)),
            list(range(7)),
            list(range(7)),
        ],
        arm_link_keys=[
            [f"torso_{i}" for i in range(4)],
            [f"left_{i}" for i in range(7)],
            [f"right_{i}" for i in range(7)],
        ],
        finger_keys=[[], ["left_finger"], ["right_finger"]],
        arm_connection_joint_indices=connection_indices,
    )


def _old_multi_arm_relative_pos(arm_num_joints):
    joint_relative_pos = []
    for i, num_joints_a in enumerate(arm_num_joints):
        joint_ids_a = torch.arange(num_joints_a)
        joint_relative_pos_per_arm = []
        for j, num_joints_b in enumerate(arm_num_joints):
            if j == i:
                joint_ids_b = joint_ids_a
            else:
                joint_ids_b = torch.arange(-1, -(num_joints_b + 1), -1)
            joint_relative_pos_per_arm.append(
                torch.abs(joint_ids_a[:, None] - joint_ids_b)
            )
        joint_relative_pos.append(torch.cat(joint_relative_pos_per_arm, dim=1))
    return torch.cat(joint_relative_pos, dim=0)


def _old_r1pro_relative_pos():
    arm_num_joints = [4, 8, 8]
    joint_relative_pos = []
    for i, num_joints_a in enumerate(arm_num_joints):
        joint_ids_a = torch.arange(num_joints_a)
        if i == 0:
            joint_ids_a = joint_ids_a.flip(0)
        joint_relative_pos_per_arm = []
        for j, num_joints_b in enumerate(arm_num_joints):
            if j == i:
                joint_ids_b = joint_ids_a
            else:
                joint_ids_b = torch.arange(-1, -(num_joints_b + 1), -1)
                if j == 0:
                    joint_ids_b = joint_ids_b.flip(0)
            joint_relative_pos_per_arm.append(
                torch.abs(joint_ids_a[:, None] - joint_ids_b)
            )
        joint_relative_pos.append(torch.cat(joint_relative_pos_per_arm, dim=1))
    return torch.cat(joint_relative_pos, dim=0)


def test_multi_arm_default_joint_relative_pos_is_unchanged(fake_urdf):
    kinematics = _make_kinematics(fake_urdf)

    assert torch.equal(
        kinematics.joint_relative_pos,
        _old_multi_arm_relative_pos([4, 8, 8]),
    )


def test_multi_arm_1d_connection_indices_match_old_r1pro(fake_urdf):
    kinematics = _make_kinematics(fake_urdf, [3, 0, 0])

    assert torch.equal(
        kinematics.joint_relative_pos,
        _old_r1pro_relative_pos(),
    )


def test_multi_arm_2d_connection_indices_support_torso_hub(fake_urdf):
    kinematics = _make_kinematics(
        fake_urdf,
        [
            [None, 3, 3],
            [0, None, None],
            [0, None, None],
        ],
    )
    relative_pos = kinematics.joint_relative_pos

    assert relative_pos[3, 4] == 1
    assert relative_pos[3, 12] == 1
    assert relative_pos[4, 12] == 2


def test_multi_arm_rejects_invalid_connection_indices(fake_urdf):
    with pytest.raises(ValueError, match="length must equal"):
        _make_kinematics(fake_urdf, [[None, 0], [0, None]])

    with pytest.raises(ValueError, match="both sides"):
        _make_kinematics(
            fake_urdf,
            [
                [None, 3, 3],
                [None, None, 0],
                [0, 0, None],
            ],
        )

    with pytest.raises(ValueError, match="out-of-range"):
        _make_kinematics(fake_urdf, [4, 0, 0])
