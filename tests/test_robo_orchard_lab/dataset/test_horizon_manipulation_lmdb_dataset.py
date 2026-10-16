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

"""Real LMDB fixtures for Horizon action reader and Panda config assembly."""

import copy
from contextlib import contextmanager

import cv2
import numpy as np
import pytest

from robo_orchard_lab.dataset.horizon_manipulation.horizon_manipulation_dataset import (  # noqa: E501
    HorizonManipulationLmdbDataset,
)
from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb

__all__ = []


@pytest.fixture()
def action_lmdb(tmp_path):
    """Write two episodes with distinct views and controller values."""
    root = tmp_path / "action_lmdb"
    root.mkdir()
    databases = {
        name: Lmdb(str(root / name), map_size=1024**2 * 4)
        for name in ("index", "meta", "image", "depth")
    }
    cameras = ["ext1_camera", "ext2_camera", "wrist_camera"]
    try:
        for episode, steps in enumerate((3, 2)):
            uuid = f"episode_{episode}"
            databases["index"].write(
                episode,
                dict(uuid=uuid, num_steps=steps, task_name=f"task_{episode}"),
            )
            joints = np.arange(steps * 8, dtype=np.float32).reshape(steps, 8)
            databases["meta"].write(
                f"{uuid}/observation/robot_state/joint_positions", joints
            )
            databases["meta"].write(
                f"{uuid}/observation/robot_state/master_joint_positions",
                joints + 0.5,
            )
            databases["meta"].write(f"{uuid}/camera_names", cameras)
            databases["meta"].write(
                f"{uuid}/meta_data", dict(instruction=f"Instruction {episode}")
            )
            databases["meta"].write(f"{uuid}/calibration", dict(source="unit"))
            intrinsics, extrinsics = {}, {}
            for camera_index, name in enumerate(cameras):
                matrix = np.eye(3)
                matrix[0, 0] = 100 + camera_index
                intrinsics[name] = matrix
                matrix = np.eye(4)
                matrix[0, 3] = camera_index + 0.1
                if name == "wrist_camera":
                    matrix = np.repeat(matrix[None], steps, axis=0)
                    matrix[:, 1, 3] = np.arange(steps)
                extrinsics[name] = matrix
                for step in range(steps):
                    image = np.full(
                        (4, 6, 3), [camera_index, 20 + step, 100], np.uint8
                    )
                    depth = np.full(
                        (4, 6), 1000 * (camera_index + 1), np.uint16
                    )
                    for database, array in (
                        ("image", image),
                        ("depth", depth),
                    ):
                        valid, encoded = cv2.imencode(".png", array)
                        assert valid
                        databases[database].write(
                            f"{uuid}/{name}/{step}", encoded.tobytes()
                        )
            databases["meta"].write(f"{uuid}/intrinsic", intrinsics)
            databases["meta"].write(f"{uuid}/extrinsic", extrinsics)
        databases["index"].write("__len__", 2)
    finally:
        for database in databases.values():
            database.close()
    return root


@contextmanager
def read_dataset(path, **kwargs):
    """Release the real fixture handles after each read, including failures."""
    dataset = HorizonManipulationLmdbDataset(str(path), **kwargs)
    try:
        yield dataset
    finally:
        if dataset.initialized:
            for group in (
                dataset.idx_lmdbs,
                dataset.meta_lmdbs,
                dataset.img_lmdbs,
                dataset.depth_lmdbs,
            ):
                for database in group:
                    if database is not None:
                        database.close()


def write_metadata(root, key, value, database="meta"):
    """Alter an explicit temporary fixture field before opening its reader."""
    writer = Lmdb(str(root / database))
    try:
        writer.write(key, value)
    finally:
        writer.close()


class TestHorizonActionReader:
    def test_observations_targets_and_frame_order(self, action_lmdb):
        cameras = ["wrist_camera", "ext1_camera", "ext2_camera"]
        with read_dataset(action_lmdb, cam_names=cameras) as dataset:
            assert len(dataset) == 5
            assert dataset.num_episode == 2
            assert dataset.get_episode_range(0) == (0, 3)
            assert dataset.get_episode_range(1) == (3, 5)
            for index, uuid, step in (
                (0, "episode_0", 0),
                (2, "episode_0", 2),
                (3, "episode_1", 0),
                (4, "episode_1", 1),
            ):
                sample = dataset[index]
                assert sample["uuid"] == uuid
                assert sample["step_index"] == step
                assert sample["cam_names"] == cameras
                np.testing.assert_allclose(
                    sample["master_joint_state"] - sample["joint_state"], 0.5
                )
                np.testing.assert_array_equal(
                    sample["imgs"][0][0, 0], [2, 20 + step, 100]
                )
                assert sample["depths"][0][0, 0] == 3.0
                assert sample["intrinsic"].shape == (3, 4, 4)
                assert sample["intrinsic"][0, 0, 0] == 102
                assert sample["T_world2cam"][0, 0, 3] == 2.1
                assert sample["T_world2cam"][0, 1, 3] == step
                assert sample["T_world2cam"][1, 0, 3] == 0.1
                assert sample["subtask"] is None
                assert sample["calibration"] == dict(source="unit")
                assert "ee_state" not in sample
            sample["joint_state"][:] = -100
            assert dataset[4]["joint_state"][0, 0] == 0

    def test_loading_flags_lazy_init_task_filter_and_stride(self, action_lmdb):
        with read_dataset(
            action_lmdb,
            lazy_init=True,
            task_names=["task_1"],
            load_image=False,
            load_depth=False,
            load_extrinsic=False,
            load_calibration=False,
        ) as dataset:
            assert not dataset.initialized
            assert len(dataset) == 2
            sample = dataset[0]
            assert sample["uuid"] == "episode_1"
            assert sample["cam_names"] == [
                "ext1_camera",
                "ext2_camera",
                "wrist_camera",
            ]
            assert (
                not {
                    "imgs",
                    "depths",
                    "T_world2cam",
                    "calibration",
                    "ee_state",
                }
                & sample.keys()
            )
        with read_dataset(action_lmdb, interval=2) as dataset:
            assert len(dataset) == 2
            assert dataset[1]["step_index"] == 2
        with read_dataset(action_lmdb, task_names=["missing"]) as dataset:
            assert len(dataset) == 0
            with pytest.raises(IndexError):
                dataset[0]

    def test_rgb_conversion_and_depth_scale(self, action_lmdb):
        with read_dataset(
            action_lmdb, bgr2rgb=True, depth_scale=2000
        ) as dataset:
            np.testing.assert_array_equal(
                dataset[0]["imgs"][0][0, 0], [100, 20, 0]
            )
            assert dataset[0]["depths"][0][0, 0] == 0.5

    @pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
    def test_invalid_depth_scale(self, scale):
        with pytest.raises(ValueError, match="depth_scale"):
            HorizonManipulationLmdbDataset("unused", depth_scale=scale)

    @pytest.mark.parametrize("instruction", [None, [], ["pick", "place"]])
    def test_instruction_variants(self, action_lmdb, instruction):
        write_metadata(
            action_lmdb, "episode_0/meta_data", dict(instruction=instruction)
        )
        with read_dataset(action_lmdb) as dataset:
            text = dataset[0]["text"]
            assert text in instruction if instruction else text == ""

    @pytest.mark.parametrize(
        "key,value,error",
        [
            ("num_steps_per_shard", 200, NotImplementedError),
            ("observation/robot_state/master_joint_positions", None, KeyError),
            (
                "observation/robot_state/master_joint_positions",
                np.zeros((2, 8)),
                ValueError,
            ),
            (
                "observation/robot_state/master_joint_positions",
                np.zeros((3, 7)),
                ValueError,
            ),
        ],
    )
    def test_metadata_errors(self, action_lmdb, key, value, error):
        write_metadata(action_lmdb, f"episode_0/{key}", value)
        with read_dataset(action_lmdb) as dataset, pytest.raises(error):
            dataset[0]

    @pytest.mark.parametrize("database", ["image", "depth"])
    @pytest.mark.parametrize(
        "buffer,error", [(None, KeyError), (b"invalid", ValueError)]
    )
    def test_frame_errors(self, action_lmdb, database, buffer, error):
        write_metadata(
            action_lmdb, "episode_0/ext1_camera/0", buffer, database
        )
        with (
            read_dataset(action_lmdb) as dataset,
            pytest.raises(error, match=f"LMDB {database}"),
        ):
            dataset[0]

    def test_ee_state_is_only_required_when_enabled(self, action_lmdb):
        with read_dataset(action_lmdb, load_ee_state=True) as dataset:
            with pytest.raises(KeyError, match="cartesian_position"):
                dataset[0]
        write_metadata(
            action_lmdb,
            "episode_0/observation/robot_state/cartesian_position",
            np.zeros((3, 7)),
        )
        with read_dataset(action_lmdb, load_ee_state=True) as dataset:
            assert dataset[0]["ee_state"].shape == (3, 7)

    def test_transform_runs_on_raw_sample(self, action_lmdb):
        def select_frame(data):
            data["current_joints"] = data["joint_state"][data["step_index"]]
            return data

        with read_dataset(action_lmdb, transforms=[select_frame]) as dataset:
            np.testing.assert_array_equal(
                dataset[1]["current_joints"], np.arange(8, 16)
            )

    def test_config_factory_connects_real_reader(
        self, action_lmdb, monkeypatch
    ):
        from projects.holobrain_0_5.configs import (
            config_holobrain_0_5,
            dataset_factory,
        )
        from projects.holobrain_0_5.configs.data_configs import (
            config_issac_dataset,
        )

        # Isolate config assembly; real LMDB decoding is not mocked.
        monkeypatch.setattr(
            config_issac_dataset, "build_transforms", lambda *args: []
        )
        config = dict(
            copy.deepcopy(config_holobrain_0_5.config),
            training_datasets=["panda_droid_rigid"],
            data_paths={"panda_droid_rigid": [str(action_lmdb)]},
            task_names=["task_1"],
        )
        combined = dataset_factory.build_training_dataset(
            config, lazy_init=True
        )
        reader = combined.datasets[0]
        try:
            assert isinstance(reader, HorizonManipulationLmdbDataset)
            assert len(combined) == 2
            assert reader.load_extrinsic and not reader.load_calibration
            assert reader.depth_scale == 1000
            assert combined[0]["cam_names"] == [
                "wrist_camera",
                "ext1_camera",
                "ext2_camera",
            ]
            assert combined[0]["depths"][0][0, 0] == 3.0
        finally:
            for group in (
                reader.idx_lmdbs,
                reader.meta_lmdbs,
                reader.img_lmdbs,
                reader.depth_lmdbs,
            ):
                for database in group:
                    if database is not None:
                        database.close()
