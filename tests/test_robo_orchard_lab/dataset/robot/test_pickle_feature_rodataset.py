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
from typing import Literal

import torch
from datasets import (
    Dataset as HFDataset,
    Features,
    Value,
)
from sqlalchemy import create_engine
from test_robo_orchard_lab.dataset.datatypes._hf_datasets_compat import (
    get_generator_example,
)

from robo_orchard_lab.dataset.datatypes.hg_features import PickleFeature
from robo_orchard_lab.dataset.datatypes.joint_state import (
    BatchJointsState,
    BatchJointsStateFeature,
)
from robo_orchard_lab.dataset.robot.dataset import RODataset


def _make_state(offset: int) -> BatchJointsState:
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3) + offset
    return BatchJointsState(
        position=values,
        velocity=values + 0.5,
        effort=values + 1.0,
        names=["joint_0", "joint_1", "joint_2"],
        timestamps=[offset, offset + 1],
    )


def _build_rodataset(
    tmp_path: Path,
    representation: Literal["structured", "torch_legacy", "numpy_v1"],
) -> tuple[RODataset, list[BatchJointsState]]:
    if representation == "structured":
        state_feature = BatchJointsState.dataset_feature()
    else:
        state_feature = PickleFeature(
            class_type=BatchJointsState,
            tensor_encoding=representation,
        )
    features = Features(
        {
            "index": Value("int64"),
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "task_index": Value("int64"),
            "robot_index": Value("int64"),
            "instruction_index": Value("int64"),
            "timestamp_min": Value("int64"),
            "timestamp_max": Value("int64"),
            "state": state_feature,
        }
    )
    states = [_make_state(0), _make_state(10)]

    def generate_data():
        for index, state in enumerate(states):
            yield get_generator_example(
                features,
                {
                    "index": index,
                    "episode_index": 0,
                    "frame_index": index,
                    "task_index": 0,
                    "robot_index": 0,
                    "instruction_index": 0,
                    "timestamp_min": index,
                    "timestamp_max": index + 1,
                    "state": state,
                },
            )

    frame_dataset = HFDataset.from_generator(
        generate_data,
        features=features,
        cache_dir=tmp_path / f"{representation}_cache",
    )
    dataset_path = tmp_path / representation
    frame_dataset.save_to_disk(dataset_path)
    reloaded = HFDataset.load_from_disk(dataset_path)
    dataset = RODataset.from_dataset(
        frame_dataset=reloaded,
        meta_db_engine=create_engine("sqlite:///:memory:"),
    )
    return dataset, states


def _assert_state_equal(
    actual: BatchJointsState,
    expected: BatchJointsState,
) -> None:
    assert type(actual) is type(expected)
    assert torch.equal(actual.position, expected.position)
    assert torch.equal(actual.velocity, expected.velocity)
    assert torch.equal(actual.effort, expected.effort)
    assert actual.names == expected.names
    assert actual.timestamps == expected.timestamps


def test_batch_joints_state_factory_keeps_pickle_opt_in() -> None:
    structured = BatchJointsState.dataset_feature()
    pickled = BatchJointsState.dataset_feature(use_pickle=True)

    assert isinstance(structured, BatchJointsStateFeature)
    assert isinstance(pickled, PickleFeature)
    assert pickled.tensor_encoding == "numpy_v1"


def test_rodataset_structured_and_pickle_representations_are_equivalent(
    tmp_path: Path,
) -> None:
    datasets_and_states = {
        representation: _build_rodataset(tmp_path, representation)
        for representation in (
            "structured",
            "torch_legacy",
            "numpy_v1",
        )
    }

    for representation, (dataset, states) in datasets_and_states.items():
        _assert_state_equal(dataset[0]["state"], states[0])
        batch = dataset.__getitems__([0, 1])
        _assert_state_equal(batch[0]["state"], states[0])
        _assert_state_equal(batch[1]["state"], states[1])

        feature = dataset.frame_dataset.features["state"]
        if representation == "structured":
            assert isinstance(feature, BatchJointsStateFeature)
        else:
            assert isinstance(feature, PickleFeature)
            assert feature.tensor_encoding == representation
