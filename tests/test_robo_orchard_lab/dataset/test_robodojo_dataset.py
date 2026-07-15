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

import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb
from robo_orchard_lab.dataset.robodojo.robodojo_lmdb_dataset import (
    RoboDojoLmdbDataset,
)
from robo_orchard_lab.dataset.robodojo.robodojo_lmdb_packer import (
    USD_TO_OPENCV,
    RoboDojoLmdbPacker,
    cam2world_usd_to_world2cam_cv,
    discover_episodes,
)

TEST_CAMERAS = (
    "cam_head",
    "cam_left_wrist",
    "cam_right_wrist",
)


def _encoded_image(value: int) -> bytes:
    image = np.full((4, 6, 3), value, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def _write_episode(
    path: Path,
    num_steps: int = 5,
    camera_names: tuple[str, ...] = TEST_CAMERAS,
) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    step = np.arange(num_steps, dtype=np.float64)[:, None]
    left_arm = step + np.arange(6, dtype=np.float64)[None]
    right_arm = left_arm + 100
    left_gripper = step + 200
    right_gripper = step + 300
    action_left_arm = left_arm + 1000
    action_right_arm = right_arm + 1000
    action_left_gripper = left_gripper + 1000
    action_right_gripper = right_gripper + 1000
    left_pose = np.concatenate(
        [step, step + 1, step + 2, np.ones((num_steps, 1)), step.repeat(3, 1)],
        axis=1,
    )
    right_pose = left_pose + 10
    left_delta_pose = left_pose + 20
    right_delta_pose = right_pose + 20
    image_buffers = [_encoded_image(20 + index) for index in range(num_steps)]
    image_dtype = f"S{max(len(image) for image in image_buffers)}"

    with h5py.File(path, "w") as episode:
        episode.create_dataset("data_format_version", data="v1.0")
        episode.create_dataset("instruction", data="Test the RoboDojo packer.")
        episode.create_dataset("additional_info/frequency", data=25)
        for group, arrays in {
            "state": {
                "left_arm_joint_states": left_arm,
                "left_ee_joint_states": left_gripper,
                "right_arm_joint_states": right_arm,
                "right_ee_joint_states": right_gripper,
                "left_ee_poses": left_pose,
                "right_ee_poses": right_pose,
                "left_delta_ee_poses": left_delta_pose,
                "right_delta_ee_poses": right_delta_pose,
            },
            "action": {
                "left_arm_joint_states": action_left_arm,
                "left_ee_joint_states": action_left_gripper,
                "right_arm_joint_states": action_right_arm,
                "right_ee_joint_states": action_right_gripper,
            },
        }.items():
            for name, value in arrays.items():
                episode.create_dataset(f"{group}/{name}", data=value)

        for camera_index, camera_name in enumerate(camera_names):
            prefix = f"vision/{camera_name}"
            episode.create_dataset(
                f"{prefix}/colors",
                data=np.asarray(image_buffers, dtype=image_dtype),
            )
            episode.create_dataset(
                f"{prefix}/intrinsic_matrix",
                data=np.diag([camera_index + 1.0, camera_index + 2.0, 1.0]),
            )
            cam2world = np.tile(np.eye(4), (num_steps, 1, 1))
            cam2world[:, 0, 3] = step[:, 0] + camera_index
            episode.create_dataset(
                f"{prefix}/extrinsic_matrix", data=cam2world
            )
            episode.create_dataset(f"{prefix}/shape", data=[4, 6, 3])

    return {
        "joint_positions": np.concatenate(
            [left_arm, left_gripper, right_arm, right_gripper], axis=1
        ),
        "master_joint_positions": np.concatenate(
            [
                action_left_arm,
                action_left_gripper,
                action_right_arm,
                action_right_gripper,
            ],
            axis=1,
        ),
        "cartesian_position": np.stack([left_pose, right_pose], axis=1),
        "delta_cartesian_position": np.stack(
            [left_delta_pose, right_delta_pose], axis=1
        ),
        "image_buffers": image_buffers,
    }


def _open_lmdb(root: Path, name: str) -> Lmdb:
    return Lmdb(uri=str(root / name), writable=False, encoding_mode="utf-8")


def test_discover_episodes_filters_sorts_and_limits(tmp_path: Path):
    for task_name in ("task_b", "task_a"):
        data_dir = tmp_path / task_name / "arx_x5" / "data"
        data_dir.mkdir(parents=True)
        for episode_index in (10, 2, 1):
            (data_dir / f"episode_{episode_index:07d}.hdf5").touch()

    episodes = discover_episodes(
        tmp_path,
        task_names=["task_b", "task_a"],
        max_episodes_per_task=2,
    )

    assert [
        (episode.task_name, episode.episode_index) for episode in episodes
    ] == [("task_a", 1), ("task_a", 2), ("task_b", 1), ("task_b", 2)]


def test_cam2world_usd_to_world2cam_cv_has_positive_forward_depth():
    cam2world = np.eye(4)
    cam2world[:3, 3] = [1.0, 2.0, 3.0]

    world2cam = cam2world_usd_to_world2cam_cv(cam2world)
    camera_center = world2cam @ np.array([1.0, 2.0, 3.0, 1.0])
    point_in_front = world2cam @ np.array([1.0, 2.0, 2.0, 1.0])

    np.testing.assert_allclose(camera_center, [0.0, 0.0, 0.0, 1.0])
    assert point_in_front[2] > 0


def test_packer_writes_shards_and_robotwin_compatible_metadata(tmp_path: Path):
    input_root = tmp_path / "input"
    episode_path = (
        input_root / "test_task" / "arx_x5" / "data" / "episode_0000000.hdf5"
    )
    camera_names = (*TEST_CAMERAS, "cam_top")
    expected = _write_episode(episode_path, camera_names=camera_names)
    output_root = tmp_path / "lmdb"

    RoboDojoLmdbPacker(
        input_path=input_root,
        output_path=output_root,
        num_steps_per_shard=2,
    )()

    index_lmdb = _open_lmdb(output_root, "index")
    meta_lmdb = _open_lmdb(output_root, "meta")
    image_lmdb = _open_lmdb(output_root, "image")
    try:
        uuid = "test_task_arx_x5_episode_0000000"
        assert index_lmdb["__len__"] == 1
        assert index_lmdb[0]["uuid"] == uuid
        assert index_lmdb[0]["num_steps"] == 5
        assert meta_lmdb["__pack_complete__"] is True
        assert meta_lmdb[f"{uuid}/camera_names"] == list(camera_names)
        assert meta_lmdb[f"{uuid}/num_steps_per_shard"] == 2
        assert (
            meta_lmdb[f"{uuid}/observation/robot_state/joint_positions"]
            is None
        )

        for shard_index, expected_slice in enumerate(
            (slice(0, 2), slice(2, 4), slice(4, 5))
        ):
            prefix = f"{uuid}/{shard_index}/observation/robot_state"
            np.testing.assert_array_equal(
                meta_lmdb[f"{prefix}/joint_positions"],
                expected["joint_positions"][expected_slice],
            )
            np.testing.assert_array_equal(
                meta_lmdb[f"{prefix}/master_joint_positions"],
                expected["master_joint_positions"][expected_slice],
            )
            np.testing.assert_array_equal(
                meta_lmdb[f"{prefix}/cartesian_position"],
                expected["cartesian_position"][expected_slice],
            )

        assert image_lmdb[f"{uuid}/cam_head/4"] == expected["image_buffers"][4]
        assert image_lmdb[f"{uuid}/cam_top/4"] == expected["image_buffers"][4]
        assert meta_lmdb[f"{uuid}/instructions"] == "Test the RoboDojo packer."
        extrinsic = meta_lmdb[f"{uuid}/extrinsic"]["cam_head"][2]
        cam2world = np.eye(4)
        cam2world[0, 3] = 2
        np.testing.assert_allclose(
            extrinsic, USD_TO_OPENCV @ np.linalg.inv(cam2world)
        )
    finally:
        index_lmdb.close()
        meta_lmdb.close()
        image_lmdb.close()


def test_dataset_reads_across_shards(tmp_path: Path):
    input_root = tmp_path / "input"
    episode_path = (
        input_root / "test_task" / "arx_x5" / "data" / "episode_0000000.hdf5"
    )
    expected = _write_episode(episode_path)
    output_root = tmp_path / "lmdb"
    RoboDojoLmdbPacker(
        input_path=input_root,
        output_path=output_root,
        num_steps_per_shard=2,
    )()

    dataset = RoboDojoLmdbDataset(
        paths=str(output_root),
        hist_steps=2,
        pred_steps=3,
    )
    data = dataset[2]

    assert len(dataset) == 5
    assert data["step_index"] == 2
    assert data["step_index_in_shard"] == 2
    assert data["cam_names"] == list(TEST_CAMERAS)
    assert data["imgs"].shape == (3, 4, 6, 3)
    assert data["intrinsic"].shape == (3, 4, 4)
    assert data["T_world2cam"].shape == (3, 4, 4)
    assert data["ee_state"].shape == (5, 14)
    assert data["delta_ee_state"].shape == (5, 14)
    assert data["text"] == "Test the RoboDojo packer."
    np.testing.assert_array_equal(
        data["joint_state"], expected["joint_positions"]
    )
    np.testing.assert_array_equal(
        data["master_joint_state"], expected["master_joint_positions"]
    )


def test_dataset_reads_unsharded_data_and_rejects_depth(tmp_path: Path):
    input_root = tmp_path / "input"
    episode_path = (
        input_root / "test_task" / "arx_x5" / "data" / "episode_0000000.hdf5"
    )
    expected = _write_episode(episode_path, num_steps=2)
    output_root = tmp_path / "lmdb"
    RoboDojoLmdbPacker(input_path=input_root, output_path=output_root)()

    dataset = RoboDojoLmdbDataset(paths=str(output_root), load_image=False)
    data = dataset[1]

    assert data["step_index_in_shard"] == 1
    assert "imgs" not in data
    np.testing.assert_array_equal(
        data["joint_state"], expected["joint_positions"]
    )
    with pytest.raises(ValueError, match="does not contain depth"):
        RoboDojoLmdbDataset(paths=str(output_root), load_depth=True)


def test_sharded_dataset_requires_history_and_prediction_lengths(
    tmp_path: Path,
):
    input_root = tmp_path / "input"
    episode_path = (
        input_root / "test_task" / "arx_x5" / "data" / "episode_0000000.hdf5"
    )
    _write_episode(episode_path)
    output_root = tmp_path / "lmdb"
    RoboDojoLmdbPacker(
        input_path=input_root,
        output_path=output_root,
        num_steps_per_shard=2,
    )()

    dataset = RoboDojoLmdbDataset(paths=str(output_root), load_image=False)
    with pytest.raises(ValueError, match="hist_steps and pred_steps"):
        dataset[0]


def test_dataset_reads_windows_larger_than_one_neighbor_shard(tmp_path: Path):
    input_root = tmp_path / "input"
    episode_path = (
        input_root / "test_task" / "arx_x5" / "data" / "episode_0000000.hdf5"
    )
    expected = _write_episode(episode_path, num_steps=8)
    output_root = tmp_path / "lmdb"
    RoboDojoLmdbPacker(
        input_path=input_root,
        output_path=output_root,
        num_steps_per_shard=2,
    )()

    dataset = RoboDojoLmdbDataset(
        paths=str(output_root),
        load_image=False,
        hist_steps=5,
        pred_steps=5,
    )
    data = dataset[4]

    assert data["step_index_in_shard"] == 4
    np.testing.assert_array_equal(
        data["joint_state"], expected["joint_positions"]
    )
    np.testing.assert_array_equal(
        data["master_joint_state"], expected["master_joint_positions"]
    )


def test_failed_pack_is_not_readable_as_partial_dataset(tmp_path: Path):
    input_root = tmp_path / "input"
    data_dir = input_root / "test_task" / "arx_x5" / "data"
    _write_episode(data_dir / "episode_0000000.hdf5", num_steps=2)
    broken_path = data_dir / "episode_0000001.hdf5"
    _write_episode(broken_path, num_steps=2)
    with h5py.File(broken_path, "r+") as episode:
        episode["vision/cam_head/colors"][0] = b"not-a-jpeg"
    output_root = tmp_path / "lmdb"

    with pytest.raises(ValueError, match="invalid JPEG"):
        RoboDojoLmdbPacker(
            input_path=input_root,
            output_path=output_root,
        )()

    index_lmdb = _open_lmdb(output_root, "index")
    try:
        assert list(index_lmdb.keys()) == []
    finally:
        index_lmdb.close()
    with pytest.raises(RuntimeError, match="packing is incomplete"):
        RoboDojoLmdbDataset(paths=str(output_root), load_image=False)


def test_close_failure_does_not_publish_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_root = tmp_path / "input"
    episode_path = (
        input_root / "test_task" / "arx_x5" / "data" / "episode_0000000.hdf5"
    )
    _write_episode(episode_path, num_steps=2)
    output_root = tmp_path / "lmdb"
    packer = RoboDojoLmdbPacker(
        input_path=input_root,
        output_path=output_root,
    )
    packer._init_lmdbs()
    real_close = packer.image_pack_file.close
    close_calls = 0

    def fail_first_close():
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise OSError("simulated image sync failure")
        real_close()

    monkeypatch.setattr(packer.image_pack_file, "close", fail_first_close)
    with pytest.raises(OSError, match="simulated image sync failure"):
        packer._pack()

    meta_lmdb = _open_lmdb(output_root, "meta")
    try:
        assert meta_lmdb["__pack_complete__"] is None
    finally:
        meta_lmdb.close()


def test_module_cli_help_has_no_duplicate_import_warning():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robo_orchard_lab.dataset.robodojo.robodojo_lmdb_packer",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage:" in result.stdout
    assert "RuntimeWarning" not in result.stderr


def test_packer_rejects_non_positive_shard_size(tmp_path: Path):
    with pytest.raises(ValueError, match="num_steps_per_shard"):
        RoboDojoLmdbPacker(
            input_path=tmp_path,
            output_path=tmp_path / "output",
            num_steps_per_shard=0,
        )
