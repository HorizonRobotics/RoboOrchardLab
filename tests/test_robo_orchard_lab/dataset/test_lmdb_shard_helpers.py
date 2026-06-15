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

import numpy as np
import pytest

from robo_orchard_lab.dataset.behavior.behavior_lmdb_dataset import (
    BehaviorLmdbDataset,
)
from robo_orchard_lab.dataset.horizon_manipulation import (
    HorizonManipulationLmdbDataset,
)
from robo_orchard_lab.dataset.interna1 import InternA1LmdbDataset
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataset,
)


class DictLmdb(dict):
    """Dict test double that matches missing-key LMDB reads."""

    def __getitem__(self, key):
        return self.get(key)


def test_base_get_meta_reads_neighbor_shards_for_chunk_sampling():
    dataset = BaseLmdbManipulationDataset(
        paths=[],
        lazy_init=True,
        hist_steps=2,
        pred_steps=2,
    )
    dataset.meta_lmdbs = [
        DictLmdb(
            {
                "episode/state": np.arange(100, 104)[:, None],
                "episode/0/state": np.arange(0, 3)[:, None],
                "episode/1/state": np.arange(3, 6)[:, None],
                "episode/2/state": np.arange(6, 9)[:, None],
                "episode/1/events": ["s1-0", "s1-1", "s1-2"],
                "episode/2/events": ["s2-0", "s2-1", "s2-2"],
            }
        )
    ]

    assert np.array_equal(
        dataset._get_meta(0, "episode", "state"),
        np.arange(100, 104)[:, None],
    )
    assert dataset._get_step_index_in_shard(7) == 7

    first_step_in_shard = dataset._get_meta(
        0,
        "episode",
        "state",
        step_index=3,
        num_steps_per_shard=3,
    )
    assert np.array_equal(first_step_in_shard[:, 0], np.arange(0, 6))
    assert dataset._get_step_index_in_shard(3, 3) == 3

    last_step_in_shard = dataset._get_meta(
        0,
        "episode",
        "state",
        step_index=5,
        num_steps_per_shard=3,
    )
    assert np.array_equal(last_step_in_shard[:, 0], np.arange(3, 9))
    assert dataset._get_step_index_in_shard(5, 3) == 2

    events = dataset._get_meta(
        0,
        "episode",
        "events",
        step_index=5,
        num_steps_per_shard=3,
    )
    assert events == ["s1-0", "s1-1", "s1-2", "s2-0", "s2-1", "s2-2"]


def test_base_get_meta_concats_dict_shards_recursively():
    dataset = BaseLmdbManipulationDataset(
        paths=[],
        lazy_init=True,
        hist_steps=2,
        pred_steps=2,
    )
    dataset.meta_lmdbs = [
        DictLmdb(
            {
                "episode/1/camera_info": {
                    "left": {
                        "pose": np.array([[10], [11]]),
                        "tags": ["left-1", "left-2"],
                    },
                    "right": {
                        "pose": np.array([[20], [21]]),
                        "tags": ["right-1", "right-2"],
                    },
                },
                "episode/2/camera_info": {
                    "left": {
                        "pose": np.array([[12], [13]]),
                        "tags": ["left-3", "left-4"],
                    },
                    "right": {
                        "pose": np.array([[22], [23]]),
                        "tags": ["right-3", "right-4"],
                    },
                },
            }
        )
    ]

    camera_info = dataset._get_meta(
        0,
        "episode",
        "camera_info",
        step_index=3,
        num_steps_per_shard=2,
    )

    assert np.array_equal(
        camera_info["left"]["pose"][:, 0],
        np.array([10, 11, 12, 13]),
    )
    assert np.array_equal(
        camera_info["right"]["pose"][:, 0],
        np.array([20, 21, 22, 23]),
    )
    assert camera_info["left"]["tags"] == [
        "left-1",
        "left-2",
        "left-3",
        "left-4",
    ]
    assert camera_info["right"]["tags"] == [
        "right-1",
        "right-2",
        "right-3",
        "right-4",
    ]


def test_base_concat_dict_shards_requires_matching_keys():
    dataset = BaseLmdbManipulationDataset(
        paths=[],
        lazy_init=True,
        hist_steps=1,
        pred_steps=1,
    )

    with pytest.raises(ValueError, match="same keys"):
        dataset._concat_shards(
            {"left": np.array([[0]])},
            {"right": np.array([[1]])},
        )


def test_base_get_meta_requires_sampling_steps_for_sharded_meta():
    dataset = BaseLmdbManipulationDataset(paths=[], lazy_init=True)
    dataset.meta_lmdbs = [
        DictLmdb(
            {
                "episode/0/state": np.array([[0]]),
            }
        )
    ]

    with pytest.raises(ValueError, match="hist_steps and pred_steps"):
        dataset._get_meta(
            0,
            "episode",
            "state",
            step_index=0,
            num_steps_per_shard=1,
        )

    with pytest.raises(ValueError, match="hist_steps and pred_steps"):
        dataset._get_step_index_in_shard(0, num_steps_per_shard=1)


@pytest.mark.parametrize(
    "dataset_cls",
    [HorizonManipulationLmdbDataset, InternA1LmdbDataset],
)
def test_joint_state_readers_use_base_shard_meta_and_master_key(dataset_cls):
    dataset = dataset_cls(
        paths=[],
        lazy_init=True,
        hist_steps=2,
        pred_steps=2,
        load_image=False,
        load_depth=False,
        load_extrinsic=False,
        load_calibration=False,
    )
    dataset.meta_lmdbs = [
        DictLmdb(
            {
                "episode/0/observation/robot_state/joint_positions": (
                    np.array([[0], [1]])
                ),
                "episode/1/observation/robot_state/joint_positions": (
                    np.array([[2], [3]])
                ),
                "episode/0/observation/robot_state/master_joint_positions": (
                    np.array([[10], [11]])
                ),
                "episode/1/observation/robot_state/master_joint_positions": (
                    np.array([[12], [13]])
                ),
            }
        )
    ]

    result = dataset.get_joint_state(
        0,
        {
            "uuid": "episode",
            "step_index": 2,
            "num_steps_per_shard": 2,
        },
    )

    assert np.array_equal(result["joint_state"][:, 0], np.array([0, 1, 2, 3]))
    assert np.array_equal(
        result["master_joint_state"][:, 0],
        np.array([10, 11, 12, 13]),
    )
    assert result["step_index_in_shard"] == 2


def test_behavior_dataset_uses_base_shard_meta_for_local_step_index():
    dataset = BehaviorLmdbDataset(
        paths=[],
        lazy_init=True,
        hist_steps=2,
        pred_steps=2,
        load_image=False,
        load_depth=False,
        cam_names=[],
    )
    dataset.initialized = True
    dataset.cumsum_steps = np.array([6])
    dataset.lmdb_indices = [0]
    dataset.episode_indices = [0]
    dataset.read_times = [0]
    dataset.idx_lmdbs = [
        DictLmdb(
            {
                0: {
                    "uuid": "episode",
                    "num_steps": 6,
                    "task_name": "task",
                }
            }
        )
    ]

    def sequence(start):
        return np.array([[start], [start + 1]])

    def matrices(start):
        return np.stack(
            [
                np.full((4, 4), start, dtype=np.float32),
                np.full((4, 4), start + 1, dtype=np.float32),
            ]
        )

    dataset.meta_lmdbs = [
        DictLmdb(
            {
                "episode/num_steps_per_shard": 2,
                "episode/1/observation/robot_state/mobile_traj": sequence(2),
                "episode/2/observation/robot_state/mobile_traj": sequence(4),
                "episode/1/observation/robot_state/joint_position": (
                    sequence(12)
                ),
                "episode/2/observation/robot_state/joint_position": (
                    sequence(14)
                ),
                "episode/1/robot_action/joint_position": sequence(22),
                "episode/2/robot_action/joint_position": sequence(24),
                "episode/1/extrinsic": matrices(32),
                "episode/2/extrinsic": matrices(34),
                "episode/1/intrinsic": matrices(42),
                "episode/2/intrinsic": matrices(44),
                "episode/instruction": "instruction",
                "episode/subtask_text": "subtask",
                "episode/skill_text": "skill",
            }
        )
    ]
    dataset.img_lmdbs = [None]
    dataset.depth_lmdbs = [None]

    data = dataset[4]

    assert data["step_index"] == 2
    assert data["step_index_in_shard"] == 2
    assert np.array_equal(data["mobile_traj"][:, 0], np.array([2, 3, 4, 5]))
    assert np.array_equal(
        data["joint_state"][:, 0],
        np.array([12, 13, 14, 15]),
    )
    assert np.array_equal(data["action"][:, 0], np.array([22, 23, 24, 25]))
    assert data["T_world2cam"][0, 0] == 34
    assert data["intrinsic"][0, 0] == 44
