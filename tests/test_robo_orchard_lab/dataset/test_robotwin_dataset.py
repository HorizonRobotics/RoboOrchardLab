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

import os
import subprocess
import tempfile

import pytest


def test_robotwin_lmdb_data_packer(
    PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str
):
    """Test robotwin lmdb data packer."""
    test_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/",
        "robotwin/cvpr_round2_branch",
    )
    test_data_path_v2 = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/",
        "robotwin/v2.0/origin_data",
    )
    os.chdir(PROJECT_ROOT)
    with tempfile.TemporaryDirectory() as workspace_root:
        cmd = (
            " ".join(
                [
                    "python3",
                    "robo_orchard_lab/dataset/robotwin/robotwin_packer.py",
                    f"--input_path {test_data_path}",
                    f"--output_path {workspace_root}",
                    "--task_names blocks_stack_three",
                    "--embodiment aloha-agilex-1",
                    "--robotwin_aug m1_b1_l1_h0.03_c0",
                    "--camera_name D435",
                ]
            ),
        )
        ret_code = subprocess.check_call(cmd, shell=True)

        # Check if the script ran successfully
        assert ret_code == 0, f"Script failed with return code: {ret_code}"

        cmd = (
            " ".join(
                [
                    "python3",
                    "robo_orchard_lab/dataset/robotwin/robotwin_packer.py",
                    f"--input_path {test_data_path_v2}",
                    f"--output_path {workspace_root}",
                    "--task_names place_empty_cup",
                    "--config_name base_setting",
                ]
            ),
        )
        ret_code = subprocess.check_call(cmd, shell=True)

        # Check if the script ran successfully
        assert ret_code == 0, f"Script failed with return code: {ret_code}"


def test_robotwin_lmdb_data(
    PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str
):
    """Test robotwin lmdb data packer."""
    test_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/",
        "robotwin/main_branch",
    )
    os.chdir(PROJECT_ROOT)
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    dataset = RoboTwinLmdbDataset(
        paths=os.path.join(test_data_path, "lmdb"),
    )
    assert dataset.num_episode == 1
    assert len(dataset) == 268

    data = dataset[0]
    for key in [
        "uuid",
        "step_index",
        "intrinsic",
        "T_world2cam",
        "T_base2world",
        "joint_state",
        "ee_state",
        "imgs",
        "depths",
        "text",
    ]:
        assert key in data


def test_calib_to_ext_transform(
    PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str
):
    urdf = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/robo_orchard_lab_projects_ut/"
        "sem_robotwin/urdf/arx5_description_isaac.urdf",
    )
    os.chdir(PROJECT_ROOT)

    import torch

    from robo_orchard_lab.dataset.robotwin.transforms import (
        CalibrationToExtrinsic,
    )

    calib_to_ext = CalibrationToExtrinsic(
        urdf=urdf,
        calibration=dict(
            middle={
                "position": [
                    -0.010783568385050412,
                    -0.2559182030838615,
                    0.5173197227547938,
                ],
                "orientation": [
                    -0.6344593881273598,
                    0.6670669773214551,
                    -0.2848079166270871,
                    0.2671467447131103,
                ],
            },
            left={
                "position": [-0.0693628, 0.04614798, 0.02938585],
                "orientation": [
                    -0.13265687,
                    0.13223542,
                    -0.6930087,
                    0.69615791,
                ],
            },
            right={
                "position": [-0.0693628, 0.04614798, 0.02938585],
                "orientation": [
                    -0.13265687,
                    0.13223542,
                    -0.6930087,
                    0.69615791,
                ],
            },
        ),
        cam_ee_joint_indices=dict(left=5, right=12),
        cam_names=["left", "middle", "right"],
    )
    data = dict(hist_joint_state=torch.zeros([1, 14]))
    data = calib_to_ext(data)
    assert "T_world2cam" in data
    assert data["T_world2cam"].shape == (3, 4, 4)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
