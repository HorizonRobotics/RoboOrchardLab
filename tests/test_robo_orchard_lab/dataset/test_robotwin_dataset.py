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


def test_mcap_to_lmdb_packer(
    PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str
):
    """Test the MCAP to LMDB data packer script.

    This test runs the mcap_to_lmdb_packer.py script on a sample
    MCAP episode and verifies that the process completes successfully.
    """
    # Define paths to test data and URDF file
    test_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/mcap/episode_2025_09_09-17_55_56/episode_2025_09_09-17_55_56_0.mcap",
    )
    urdf_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/urdf/piper_description_dualarm.urdf",
    )

    # Change to the project root directory to ensure correct script paths
    os.chdir(PROJECT_ROOT)

    with tempfile.TemporaryDirectory() as workspace_root:
        # Construct the command to run the packer script
        lmdb_dataset_path = os.path.join(workspace_root, "lmdb_dataset1")

        from robo_orchard_lab.dataset.horizon_manipulation.packer.mcap_lmdb_packer import (  # noqa: E501
            McapLmdbDataPacker,
        )
        from robo_orchard_lab.dataset.horizon_manipulation.packer.utils import (  # noqa: E501
            PackConfig,
        )

        pack_config = PackConfig()

        packer = McapLmdbDataPacker(
            input_path=test_data_path,
            output_path=lmdb_dataset_path,
            urdf_path=urdf_path,
            pack_config=pack_config,
        )
        packer()

        # Verify that output directories have been created
        assert os.path.isdir(os.path.join(lmdb_dataset_path, "meta"))
        assert os.path.isdir(os.path.join(lmdb_dataset_path, "image"))
        assert os.path.isdir(os.path.join(lmdb_dataset_path, "depth"))
        assert os.path.isdir(os.path.join(lmdb_dataset_path, "index"))


def test_mcap_to_arrow_packer(
    PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str
):
    """Test the MCAP to Arrow data packer script.

    This test runs the mcap_to_arrow_packer.py script on a sample
    MCAP episode and verifies that the process completes successfully.
    """
    # Define paths to test data and URDF file
    test_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/mcap/episode_2025_09_09-17_55_56/episode_2025_09_09-17_55_56_0.mcap",
    )
    urdf_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/urdf/piper_description_dualarm.urdf",
    )

    os.chdir(PROJECT_ROOT)

    with tempfile.TemporaryDirectory() as workspace_root:
        from robo_orchard_lab.dataset.horizon_manipulation.packer.mcap_arrow_packer import (  # noqa E501
            make_dataset_from_mcap,
        )
        from robo_orchard_lab.dataset.horizon_manipulation.packer.utils import (  # noqa E501
            PackConfig,
        )

        pack_config = PackConfig()
        arrow_dataset_path = os.path.join(workspace_root, "arrow_dataset1")
        make_dataset_from_mcap(
            input_path=test_data_path,
            output_path=arrow_dataset_path,
            urdf_path=urdf_path,
            pack_config=pack_config,
            force_overwrite=True,
        )

        # Verify that Arrow dataset files have been created
        import glob

        assert os.path.isfile(
            os.path.join(arrow_dataset_path, "dataset_info.json")
        )
        assert len(glob.glob(os.path.join(arrow_dataset_path, "*.arrow"))) > 0
        assert len(glob.glob(os.path.join(arrow_dataset_path, "*.duckdb"))) > 0


def test_dataset_consistency(
    PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str
):
    test_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/mcap/episode_2025_09_09-17_55_56/episode_2025_09_09-17_55_56_0.mcap",
    )
    urdf_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/urdf/piper_description_dualarm.urdf",
    )

    os.chdir(PROJECT_ROOT)

    import numpy as np

    from robo_orchard_lab.dataset.robot.dataset import ROMultiRowDataset
    from robo_orchard_lab.dataset.robotwin.transforms import (
        ArrowDataParse,
        EpisodeSamplerConfig,
    )

    with tempfile.TemporaryDirectory() as workspace_root:
        # Construct the command to run the Arrow packer script
        arrow_dataset_path = os.path.join(workspace_root, "arrow_dataset2")
        lmdb_dataset_path = os.path.join(workspace_root, "lmdb_dataset2")

        from robo_orchard_lab.dataset.horizon_manipulation.packer.mcap_lmdb_packer import (  # noqa E501
            McapLmdbDataPacker,
        )
        from robo_orchard_lab.dataset.horizon_manipulation.packer.utils import (  # noqa E501
            PackConfig,
        )

        pack_config = PackConfig()
        packer = McapLmdbDataPacker(
            input_path=test_data_path,
            output_path=lmdb_dataset_path,
            urdf_path=urdf_path,
            pack_config=pack_config,
        )
        packer()

        from robo_orchard_lab.dataset.horizon_manipulation.packer.mcap_arrow_packer import (  # noqa E501
            make_dataset_from_mcap,
        )

        make_dataset_from_mcap(
            input_path=test_data_path,
            output_path=arrow_dataset_path,
            urdf_path=urdf_path,
            pack_config=pack_config,
            force_overwrite=True,
        )

        print(f"lmdb_dataset_path is {lmdb_dataset_path}")
        print(f"arrow_dataset_path is {arrow_dataset_path}")

        # build arrow_dataset
        data_parser = ArrowDataParse(
            cam_names=["left", "middle", "right"],
            load_image=True,
            load_depth=True,
            load_extrinsic=True,
            depth_scale=1000,
        )
        joint_sampler = EpisodeSamplerConfig(
            target_columns=["joints", "actions"]
        )
        arrow_dataset = ROMultiRowDataset(
            dataset_path=arrow_dataset_path, row_sampler=joint_sampler
        )
        arrow_dataset.set_transform(data_parser)

        # build lmdb_dataset
        from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
            RoboTwinLmdbDataset,
        )

        lmdb_dataset = RoboTwinLmdbDataset(
            paths=[lmdb_dataset_path],
            cam_names=["left", "middle", "right"],
            task_names=["empty_cup_place"],
            load_image=True,
            load_depth=True,
        )

        # Assert data consistency between lmdb_dataset and arrow_dataset
        for idx in range(0, len(lmdb_dataset), 100):
            lmdb_dataitem = lmdb_dataset[idx]
            mcap_dataitem = arrow_dataset[idx]

            assert np.array_equal(lmdb_dataitem["imgs"], mcap_dataitem["imgs"])
            assert np.array_equal(
                lmdb_dataitem["depths"], mcap_dataitem["depths"]
            )
            assert np.array_equal(
                lmdb_dataitem["intrinsic"], mcap_dataitem["intrinsic"]
            )
            assert lmdb_dataitem["step_index"] == mcap_dataitem["step_index"]
            assert lmdb_dataitem["text"] == mcap_dataitem["text"]
            assert np.array_equal(
                lmdb_dataitem["joint_state"].astype(np.float32),
                mcap_dataitem["joint_state"].astype(np.float32),
            )

            assert (
                abs(
                    lmdb_dataitem["T_world2cam"].astype(np.float32)
                    - mcap_dataitem["T_world2cam"].astype(np.float32)
                ).max()
                < 1e-6
            )
            print(f"idx is {idx}, assert pass")


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
