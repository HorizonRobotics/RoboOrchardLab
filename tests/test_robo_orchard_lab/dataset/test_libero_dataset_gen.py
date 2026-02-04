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

import os

import pytest

try:
    import libero  # noqa: F401
except ImportError:
    pytest.skip("libero is not installed", allow_module_level=True)
from robo_orchard_lab.dataset.datatypes import (
    BatchCameraDataEncodedFeature,
    BatchFrameTransformFeature,
    BatchFrameTransformGraph,
    BatchFrameTransformGraphFeature,
    BatchJointsStateFeature,
)
from robo_orchard_lab.dataset.libero.generate_dataset import (
    make_libero_dataset,
)
from robo_orchard_lab.dataset.robot import RODataset


@pytest.fixture(scope="session")
def libero_example_dataset_path(tmp_local_folder: str):
    return os.path.join(tmp_local_folder, "test_libero_dataset_gen")


@pytest.fixture(scope="session")
def libero_example_dataset(libero_example_dataset_path: str) -> RODataset:
    target_path = libero_example_dataset_path

    make_libero_dataset(dataset_path=target_path, max_episode=1)
    dataset = RODataset(target_path)
    return dataset


def test_libero_dataset_loading(libero_example_dataset: RODataset):
    dataset = libero_example_dataset
    print(dataset.features)


def test_dataset_datatypes_has_timestamps(
    libero_example_dataset: RODataset,
):
    dataset = libero_example_dataset

    frame = dataset[2]

    for key, feature in dataset.features.items():
        if isinstance(
            feature,
            (
                BatchJointsStateFeature,
                BatchFrameTransformFeature,
                BatchCameraDataEncodedFeature,
            ),
        ):
            assert frame[key].timestamps is not None
            assert frame[key].timestamps[0] > 0
        elif isinstance(feature, BatchFrameTransformGraphFeature):
            tf_graph: BatchFrameTransformGraph = frame[key]
            tf_list = tf_graph.as_state().tf_list
            assert len(tf_list) > 0
            for tf in tf_list:
                assert tf.timestamps is not None
                assert tf.timestamps[0] > 0
