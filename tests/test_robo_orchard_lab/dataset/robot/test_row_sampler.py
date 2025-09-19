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

import pytest

from robo_orchard_lab.dataset.robot.dataset import ROMultiRowDataset
from robo_orchard_lab.dataset.robot.row_sampler import (
    DeltaTimestampSamplerConfig,
    IndexFrameCache,
    time_range_match_frame,
)
from robo_orchard_lab.dataset.robotwin.transforms import EpisodeSamplerConfig


class TestIndexFrameCache:
    @pytest.fixture()
    def sample_cache_fixture(self):
        """Fixture to create a sample IndexFrameCache."""
        cache = IndexFrameCache()
        cache.add_frame(0, {"timestamp_min": 1, "timestamp_max": 2})
        cache.add_frame(1, {"timestamp_min": 3, "timestamp_max": 3})
        cache.add_frame(2, {"timestamp_min": 3, "timestamp_max": 4})
        return cache

    def test_find_frame(self, sample_cache_fixture: IndexFrameCache):
        """Test finding a frame in the cache."""
        cache = sample_cache_fixture
        assert cache.get_frame_range(5, 5) is None
        assert cache.get_frame_range(0, 0) is None
        assert cache.get_frame_range(1, 1) == (0, 0)
        assert cache.get_frame_range(3, 3) == (1, 2)
        assert cache.get_frame_range(1, 5) == (0, 2)
        assert cache.get_frame_range(1, 3) == (0, 2)

    def test_match_frame(self, sample_cache_fixture: IndexFrameCache):
        """Test matching frames with timestamp ranges."""
        cache = sample_cache_fixture

        for begin in range(0, 5):
            for end in range(begin, 6):
                r = cache.get_frame_range(begin, end)
                if r is None:
                    continue
                for i in range(r[0], r[1] + 1):
                    frame = cache.get_frame(i)
                    assert frame is not None
                    assert time_range_match_frame(frame, begin, end), (
                        f"Frame {i} does not match range {begin}-{end}"
                    )


class TestDeltaTimestampSampler:
    @pytest.mark.parametrize(
        "cfg, is_none_expected",
        [
            (  # RoboTwin dataset is 25FPS
                DeltaTimestampSamplerConfig(
                    column_delta_ts={
                        "joints": [0, 0.0 + 1.0 / 25 * 1],
                    },
                    tolerance=1e-5,
                ),
                [False, False],
            ),
            (  # RoboTwin dataset is 25FPS
                DeltaTimestampSamplerConfig(
                    column_delta_ts={
                        "joints": [0, 0.0 + 1.0 / 25 * 1 - 0.01 - 0.00001],
                    },
                    tolerance=0.01,
                ),
                [False, True],
            ),
            (  # RoboTwin dataset is 25FPS
                DeltaTimestampSamplerConfig(
                    column_delta_ts={
                        "joints": [0, 0.0 + 1.0 / 25 * 1 - 0.01],
                    },
                    tolerance=0.01,
                ),
                [False, False],
            ),
            (  # RoboTwin dataset is 25FPS
                DeltaTimestampSamplerConfig(
                    column_delta_ts={
                        "joints": [0, 0.0 + 1.0 / 25 * 1 + 0.01],
                    },
                    tolerance=0.01,
                ),
                [False, False],
            ),
            (  # RoboTwin dataset is 25FPS
                DeltaTimestampSamplerConfig(
                    column_delta_ts={
                        "joints": [0, 0.0 + 1.0 / 25 * 1 + 0.01 + 0.00001],
                    },
                    tolerance=0.01,
                ),
                [False, True],
            ),
        ],
    )
    def test_with_delta_ts(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
        cfg: DeltaTimestampSamplerConfig,
        is_none_expected: list[bool],
    ):
        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/robotwin/ro_dataset",
        )
        # RoboTwin dataset is 25FPS

        dataset = ROMultiRowDataset(
            dataset_path=path,
            row_sampler=cfg,
        )
        print(len(dataset))
        joints = dataset[0]["joints"]
        for data, should_be_none in zip(joints, is_none_expected, strict=True):
            assert (data is None) is should_be_none
        print(joints)


class TestEpisodeSampler:
    @pytest.mark.parametrize(
        "cfg, is_true_expected",
        [
            (
                EpisodeSamplerConfig(
                    target_columns=["joints"],
                ),
                [True, False],
            ),
            (
                EpisodeSamplerConfig(
                    target_columns=["actions"],
                ),
                [False, True],
            ),
            (
                EpisodeSamplerConfig(
                    target_columns=["joints", "actions"],
                ),
                [True, True],
            ),
        ],
    )
    def test_sample_episode_indices(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
        cfg: EpisodeSamplerConfig,
        is_true_expected: list[bool],
    ):
        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/horizon_aloha/arrow_dataset_4episode",
        )
        dataset = ROMultiRowDataset(
            dataset_path=path,
            row_sampler=cfg,
        )

        print(len(dataset))
        data_idx = 0
        target_columns = cfg.target_columns
        for col in target_columns:
            assert (
                len(dataset[data_idx][col])
                == dataset[data_idx]["episode"].frame_num
            )

        for col, should_be_true in zip(
            ["joints", "actions"], is_true_expected, strict=True
        ):
            assert isinstance(dataset[data_idx][col], list) is should_be_true
            if isinstance(dataset[data_idx][col], list):
                assert (
                    len(dataset[data_idx][col])
                    == dataset[data_idx]["episode"].frame_num
                ) is should_be_true
