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
import tempfile

import pytest


def _project_root() -> str:
    project_root = os.path.abspath(__file__)
    for _ in range(3):
        project_root = os.path.dirname(project_root)
    return project_root


@pytest.fixture(scope="module")
def packer_input_paths(ROBO_ORCHARD_TEST_WORKSPACE: str) -> tuple[str, str]:
    test_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/mcap/episode_2025_09_09-17_55_56/episode_2025_09_09-17_55_56_0.mcap",
    )
    urdf_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/urdf/piper_description_dualarm.urdf",
    )
    return test_data_path, urdf_path


@pytest.fixture(scope="module")
def pack_config():
    import torch

    from robo_orchard_lab.dataset.datatypes import BatchFrameTransform
    from robo_orchard_lab.dataset.horizon_manipulation.packer.utils import (
        PackConfig,
    )

    # Use one shared packed dataset for both consistency checks and the
    # explicit extrinsic override assertions.
    return PackConfig(
        EXTRINSIC_OVERRIDES={
            "/observation/cameras/middle/color_image/image_raw": BatchFrameTransform(  # noqa: E501
                parent_frame_id="world",
                child_frame_id="middle_camera_color_optical_frame",
                xyz=torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32),
                quat=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            )
        }
    )


@pytest.fixture(scope="module")
def prepared_datasets(packer_input_paths, pack_config):
    os.chdir(_project_root())

    test_data_path, urdf_path = packer_input_paths

    from robo_orchard_lab.dataset.horizon_manipulation.packer.mcap_arrow_packer import (  # noqa: E501
        make_dataset_from_mcap,
    )
    from robo_orchard_lab.dataset.horizon_manipulation.packer.mcap_lmdb_packer import (  # noqa: E501
        McapLmdbDataPacker,
    )

    with tempfile.TemporaryDirectory() as workspace_root:
        lmdb_dataset_path = os.path.join(workspace_root, "lmdb_dataset")
        arrow_dataset_path = os.path.join(workspace_root, "arrow_dataset")

        packer = McapLmdbDataPacker(
            input_path=test_data_path,
            output_path=lmdb_dataset_path,
            urdf_path=urdf_path,
            pack_config=pack_config,
        )
        packer()

        make_dataset_from_mcap(
            input_path=test_data_path,
            output_path=arrow_dataset_path,
            urdf_path=urdf_path,
            pack_config=pack_config,
            force_overwrite=True,
        )

        yield {
            "lmdb_dataset_path": lmdb_dataset_path,
            "arrow_dataset_path": arrow_dataset_path,
        }


def test_mcap_to_lmdb_packer(prepared_datasets):
    """Test the MCAP to LMDB data packer script."""
    lmdb_dataset_path = prepared_datasets["lmdb_dataset_path"]

    assert os.path.isdir(os.path.join(lmdb_dataset_path, "meta"))
    assert os.path.isdir(os.path.join(lmdb_dataset_path, "image"))
    assert os.path.isdir(os.path.join(lmdb_dataset_path, "depth"))
    assert os.path.isdir(os.path.join(lmdb_dataset_path, "index"))


def test_mcap_to_arrow_packer(prepared_datasets):
    """Test the MCAP to Arrow data packer script."""
    import glob

    arrow_dataset_path = prepared_datasets["arrow_dataset_path"]

    assert os.path.isfile(
        os.path.join(arrow_dataset_path, "dataset_info.json")
    )
    assert len(glob.glob(os.path.join(arrow_dataset_path, "*.arrow"))) > 0
    assert len(glob.glob(os.path.join(arrow_dataset_path, "*.duckdb"))) > 0


def test_dataset_consistency(
    prepared_datasets,
):
    import numpy as np

    from robo_orchard_lab.dataset.robot.dataset import ROMultiRowDataset
    from robo_orchard_lab.dataset.robotwin.transforms import (
        ArrowDataParse,
        EpisodeSamplerConfig,
    )

    arrow_dataset_path = prepared_datasets["arrow_dataset_path"]
    lmdb_dataset_path = prepared_datasets["lmdb_dataset_path"]

    data_parser = ArrowDataParse(
        cam_names=["left", "middle", "right"],
        load_image=True,
        load_depth=True,
        load_extrinsic=True,
        depth_scale=1000,
    )
    joint_sampler = EpisodeSamplerConfig(target_columns=["joints", "actions"])
    arrow_dataset = ROMultiRowDataset(
        dataset_path=arrow_dataset_path, row_sampler=joint_sampler
    )
    arrow_dataset.set_transform(data_parser)

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

    for idx in range(0, len(lmdb_dataset), 100):
        lmdb_dataitem = lmdb_dataset[idx]
        mcap_dataitem = arrow_dataset[idx]

        assert np.array_equal(lmdb_dataitem["imgs"], mcap_dataitem["imgs"])
        assert np.array_equal(lmdb_dataitem["depths"], mcap_dataitem["depths"])
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


def test_dataset_extrinsic_override(prepared_datasets):
    import numpy as np

    from robo_orchard_lab.dataset.robot.dataset import ROMultiRowDataset
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )
    from robo_orchard_lab.dataset.robotwin.transforms import (
        ArrowDataParse,
        EpisodeSamplerConfig,
    )

    expected_middle_extrinsic = np.eye(4, dtype=np.float32)
    expected_middle_extrinsic[:3, 3] = np.array(
        [-0.25, 0.5, -0.75], dtype=np.float32
    )

    arrow_dataset_path = prepared_datasets["arrow_dataset_path"]
    lmdb_dataset_path = prepared_datasets["lmdb_dataset_path"]

    data_parser = ArrowDataParse(
        cam_names=["left", "middle", "right"],
        load_image=False,
        load_depth=False,
        load_extrinsic=True,
        depth_scale=1000,
    )
    joint_sampler = EpisodeSamplerConfig(target_columns=["joints", "actions"])
    arrow_dataset = ROMultiRowDataset(
        dataset_path=arrow_dataset_path, row_sampler=joint_sampler
    )
    arrow_dataset.set_transform(data_parser)

    lmdb_dataset = RoboTwinLmdbDataset(
        paths=[lmdb_dataset_path],
        cam_names=["left", "middle", "right"],
        task_names=["empty_cup_place"],
        load_image=False,
        load_depth=False,
    )

    arrow_dataitem = arrow_dataset[0]
    lmdb_dataitem = lmdb_dataset[0]

    assert np.array_equal(
        arrow_dataitem["T_world2cam"][1], expected_middle_extrinsic
    )
    assert np.array_equal(
        lmdb_dataitem["T_world2cam"][1], expected_middle_extrinsic
    )
    assert np.array_equal(
        lmdb_dataitem["T_world2cam"], arrow_dataitem["T_world2cam"]
    )

    # Only the middle camera topic is overridden in pack_config.
    assert not np.allclose(
        arrow_dataitem["T_world2cam"][0], expected_middle_extrinsic, atol=1e-6
    )
    assert not np.allclose(
        arrow_dataitem["T_world2cam"][2], expected_middle_extrinsic, atol=1e-6
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
