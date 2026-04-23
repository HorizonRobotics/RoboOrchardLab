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

import pytest
from datasets import Dataset

from robo_orchard_lab.dataset.horizon_manipulation.row_sampler import (
    EpisodeChunkSampler,
    EpisodeChunkSamplerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(episode_lengths: list[int]) -> Dataset:
    """Build a minimal HF Dataset with an `episode_index` column.

    Each episode has ``episode_lengths[i]`` consecutive rows.
    """
    episode_indices = []
    for ep_idx, length in enumerate(episode_lengths):
        episode_indices.extend([ep_idx] * length)
    return Dataset.from_dict({"episode_index": episode_indices})


def _sampler(
    hist_steps: int = 1,
    pred_steps: int = 4,
    pred_interval: int = 1,
    target_columns: list[str] | None = None,
) -> EpisodeChunkSampler:
    if target_columns is None:
        target_columns = ["obs"]
    cfg = EpisodeChunkSamplerConfig(
        target_columns=target_columns,
        hist_steps=hist_steps,
        pred_steps=pred_steps,
        pred_interval=pred_interval,
    )
    return EpisodeChunkSampler(cfg)


# Two episodes of 10 rows each: global indices 0-9 (ep 0), 10-19 (ep 1).
EP_LEN = 10
NUM_EPS = 2


@pytest.fixture(scope="module")
def two_episode_dataset() -> Dataset:
    return _make_dataset([EP_LEN, EP_LEN])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEpisodeChunkSamplerPredInterval1:
    """pred_interval=1 preserves contiguous future indices."""

    def test_contiguous_future_indices(self, two_episode_dataset):
        """Future indices must use step 1."""
        sampler = _sampler(hist_steps=1, pred_steps=4, pred_interval=1)
        ds = two_episode_dataset
        # index=5 is mid-episode-0, plenty of room in both directions
        result = sampler.sample_row_idx(ds, index=5)
        expected = [5, 6, 7, 8, 9]  # hist=[5], future=[6,7,8,9]
        assert result["obs"] == expected

    def test_chunk_length(self, two_episode_dataset):
        """Returned list length must equal hist_steps + pred_steps."""
        hist, pred = 3, 5
        sampler = _sampler(hist_steps=hist, pred_steps=pred, pred_interval=1)
        result = sampler.sample_row_idx(two_episode_dataset, index=5)
        assert len(result["obs"]) == hist + pred


class TestEpisodeChunkSamplerPredInterval2:
    """pred_interval=2 spaces future kept-row indices by 2."""

    def test_sparse_future_indices(self, two_episode_dataset):
        """Future indices must use kept-row step pred_interval."""
        sampler = _sampler(hist_steps=1, pred_steps=4, pred_interval=2)
        ds = two_episode_dataset
        # index=1 in episode 0; future with interval=2: 3, 5, 7, 9
        result = sampler.sample_row_idx(ds, index=1)
        expected = [1, 3, 5, 7, 9]  # hist=[1], future=[3,5,7,9]
        assert result["obs"] == expected

    def test_pred_interval_3(self, two_episode_dataset):
        """Verify step-3 spacing."""
        sampler = _sampler(hist_steps=1, pred_steps=3, pred_interval=3)
        ds = two_episode_dataset
        # index=0 in episode 0; future: 3, 6, 9
        result = sampler.sample_row_idx(ds, index=0)
        expected = [0, 3, 6, 9]
        assert result["obs"] == expected

    def test_raw_frame_index_gaps_do_not_change_kept_row_stride(self):
        """raw_frame_index is trace metadata, not the sampling coordinate."""
        ds = Dataset.from_dict(
            {
                "episode_index": [0, 0, 0, 0, 0],
                "raw_frame_index": [0, 10, 20, 30, 40],
            }
        )
        sampler = _sampler(hist_steps=1, pred_steps=2, pred_interval=2)

        result = sampler.sample_row_idx(ds, index=0)

        assert result["obs"] == [0, 2, 4]


class TestEpisodeChunkSamplerFutureBoundaryClipping:
    """Future indices beyond the episode end must be clipped to last row."""

    def test_future_clipped_at_episode_end(self, two_episode_dataset):
        """Indices past episode boundary clip to episode's last row index."""
        sampler = _sampler(hist_steps=1, pred_steps=4, pred_interval=1)
        ds = two_episode_dataset
        # index=8 in episode 0; future 9,10,11,12 clips to 9.
        result = sampler.sample_row_idx(ds, index=8)
        expected = [8, 9, 9, 9, 9]
        assert result["obs"] == expected

    def test_future_clipped_interval2(self, two_episode_dataset):
        """Clipping also works with pred_interval=2."""
        sampler = _sampler(hist_steps=1, pred_steps=4, pred_interval=2)
        ds = two_episode_dataset
        # index=7 in episode 0; future 9,11,13,15 clips to 9.
        result = sampler.sample_row_idx(ds, index=7)
        expected = [7, 9, 9, 9, 9]
        assert result["obs"] == expected

    def test_no_crossover_between_episodes(self, two_episode_dataset):
        """Future indices must never reference rows from the next episode."""
        sampler = _sampler(hist_steps=1, pred_steps=4, pred_interval=1)
        ds = two_episode_dataset
        # index=9 is last row of episode 0
        result = sampler.sample_row_idx(ds, index=9)
        clipped = result["obs"]
        # All indices should belong to episode 0 (0-9)
        assert all(0 <= idx <= 9 for idx in clipped)


class TestEpisodeChunkSamplerHistoryBoundaryClipping:
    """History indices before the episode start clip to the first row."""

    def test_history_clipped_at_episode_start(self, two_episode_dataset):
        """Indices before episode start clip to the episode's first row."""
        sampler = _sampler(hist_steps=4, pred_steps=1, pred_interval=1)
        ds = two_episode_dataset
        # index=1 in episode 0; hist raw: [-2,-1,0,1], clipped to 0
        result = sampler.sample_row_idx(ds, index=1)
        expected = [0, 0, 0, 1, 2]  # hist=[0,0,0,1], future=[2]
        assert result["obs"] == expected

    def test_history_clipped_at_second_episode_start(
        self,
        two_episode_dataset,
    ):
        """History clipping works correctly for the second episode too."""
        sampler = _sampler(hist_steps=3, pred_steps=1, pred_interval=1)
        ds = two_episode_dataset
        # index=10 is first row of episode 1; 8 and 9 belong to ep0.
        result = sampler.sample_row_idx(ds, index=10)
        expected = [10, 10, 10, 11]  # hist=[10,10,10], future=[11]
        assert result["obs"] == expected

    def test_no_crossover_from_previous_episode(self, two_episode_dataset):
        """History indices must never reference rows from the prior episode."""
        sampler = _sampler(hist_steps=4, pred_steps=0, pred_interval=1)
        ds = two_episode_dataset
        # index=11 in episode 1; 8 and 9 belong to ep0.
        result = sampler.sample_row_idx(ds, index=11)
        clipped = result["obs"]
        assert all(10 <= idx <= 19 for idx in clipped)


class TestEpisodeChunkSamplerMultipleColumns:
    """target_columns: each column receives the same index list."""

    def test_multiple_columns_same_indices(self, two_episode_dataset):
        sampler = _sampler(
            hist_steps=1,
            pred_steps=2,
            pred_interval=1,
            target_columns=["obs", "action"],
        )
        result = sampler.sample_row_idx(two_episode_dataset, index=5)
        assert result["obs"] == result["action"]
        assert len(result["obs"]) == 3  # 1 hist + 2 pred


class TestEpisodeChunkSamplerConfig:
    """Config validation."""

    def test_invalid_pred_interval_raises(self):
        with pytest.raises(AssertionError):
            EpisodeChunkSampler(
                EpisodeChunkSamplerConfig(
                    target_columns=["obs"],
                    pred_interval=0,
                )
            )

    def test_pred_steps_zero(self):
        """pred_steps=0 returns only history."""
        ds = _make_dataset([10])
        sampler = _sampler(hist_steps=3, pred_steps=0, pred_interval=1)
        result = sampler.sample_row_idx(ds, index=5)
        assert result["obs"] == [3, 4, 5]
