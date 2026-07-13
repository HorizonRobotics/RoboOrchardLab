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

from pathlib import Path

import pytest
import torch

from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
    CalibrationToExtrinsic,
)


def _write_one_joint_urdf(path: Path) -> None:
    path.write_text(
        """
<robot name="camera_ref_probe">
  <link name="base" />
  <link name="link1" />
  <joint name="joint1" type="revolute">
    <parent link="base" />
    <child link="link1" />
    <origin xyz="1 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
  </joint>
</robot>
""".strip(),
        encoding="utf-8",
    )


def test_calibration_to_extrinsic_uses_total_camera_ref_links(tmp_path: Path):
    urdf = tmp_path / "camera_ref_probe.urdf"
    _write_one_joint_urdf(urdf)
    transform = CalibrationToExtrinsic(
        urdf=str(urdf),
        calibration={
            "wrist": torch.eye(4),
            "static": torch.eye(4),
        },
        cam_ref_links={
            "wrist": "link1",
            "static": None,
        },
        cam_names=["wrist", "static"],
        arm_joint_id=[list(range(1))],
        arm_link_keys=[["link1"]],
        finger_keys=[[]],
    )

    data = transform({"hist_joint_state": torch.zeros(1, 1)})

    assert data["T_world2cam"].shape == (2, 4, 4)


def test_calibration_to_extrinsic_rejects_missing_camera_refs(tmp_path: Path):
    urdf = tmp_path / "camera_ref_probe.urdf"
    _write_one_joint_urdf(urdf)
    transform = CalibrationToExtrinsic(
        urdf=str(urdf),
        calibration={
            "wrist": torch.eye(4),
            "missing": torch.eye(4),
        },
        cam_ref_links={"wrist": "link1"},
        cam_names=["wrist", "missing"],
        arm_joint_id=[list(range(1))],
        arm_link_keys=[["link1"]],
        finger_keys=[[]],
    )

    with pytest.raises(KeyError, match="missing"):
        transform({"hist_joint_state": torch.zeros(1, 1)})
