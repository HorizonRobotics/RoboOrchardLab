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
import pickle
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
from datasets import Dataset as HFDataset
from sqlalchemy import create_engine

from robo_orchard_lab.dataset.datatypes import BatchCameraData
from robo_orchard_lab.dataset.robot.dataset import (
    RODataset,
    RODatasetImageDecodeOptions,
    ROMultiRowDataset,
)
from robo_orchard_lab.dataset.robot.dataset_db_engine import create_tables
from robo_orchard_lab.dataset.robot.row_sampler import (
    CachedIndexDataset,
    ColumnIndexOffsetSamplerConfig,
    CustomizedColumnIndexSampler,
    DeltaTimestampSamplerConfig,
    IndexFrameCache,
    MultiRowSampler,
    time_range_match_frame,
)
from robo_orchard_lab.dataset.robotwin.transforms import EpisodeSamplerConfig


class _RecordingRowDataSampler(MultiRowSampler):
    """Record current-row context before sampling one existing value."""

    def __init__(self) -> None:
        self.scalar_row_data_calls: list[dict[str, Any] | None] = []
        self.batch_row_data_calls: list[dict[str, list[Any]] | None] = []

    @property
    def column_rows_keys(self) -> dict[str, None]:
        return {"value": None}

    def sample_row_idx(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        *,
        row_data: Mapping[str, Any] | None = None,
    ) -> dict[str, list[int | None]]:
        self.scalar_row_data_calls.append(
            None if row_data is None else dict(row_data)
        )
        return {"value": [index]}

    def sample_row_idx_batch(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        *,
        row_data: Mapping[str, Sequence[Any]] | None = None,
    ) -> dict[str, list[list[int | None]]]:
        self.batch_row_data_calls.append(
            None
            if row_data is None
            else {column: list(values) for column, values in row_data.items()}
        )
        return super().sample_row_idx_batch(
            index_dataset,
            index_batch,
            row_data=row_data,
        )


class _LegacyRowDataSampler(MultiRowSampler):
    """Preserve the scalar sampler signature published before row data."""

    @property
    def column_rows_keys(self) -> dict[str, None]:
        return {"value": None}

    def sample_row_idx(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
    ) -> dict[str, list[int | None]]:
        return {"value": [index]}


class _LegacyCustomizedOffsetsSampler(CustomizedColumnIndexSampler):
    """Preserve custom scalar and batch hook signatures without row data."""

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(columns=["value"]))

    def _sample_column_offsets(
        self,
        index_dataset: HFDataset,
        index: int,
    ) -> dict[str, list[int | None]]:
        return {"value": [0]}

    def _sample_column_offsets_batch(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
    ) -> list[dict[str, list[int | None]]]:
        return [{"value": [0]} for _ in index_batch]


class TestMultiRowSamplerRowData:
    def test_multi_row_dataset_passes_materialized_current_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scalar and batch sampling receive the complete current-row view."""

        frame_dataset = HFDataset.from_dict(
            {
                "index": [0, 1, 2],
                "episode_index": [0, 0, 0],
                "frame_index": [0, 1, 2],
                "task_index": [0, 0, 0],
                "robot_index": [0, 0, 0],
                "instruction_index": [0, 0, 0],
                "timestamp_min": [0, 1, 2],
                "timestamp_max": [0, 1, 2],
                "value": [10, 20, 30],
                "sampling_context": ["first", "second", "third"],
            }
        )
        base_dataset = RODataset.from_dataset(
            frame_dataset=frame_dataset,
            meta_db_engine=create_engine("sqlite:///:memory:"),
        )
        dataset = ROMultiRowDataset.from_dataset(
            base_dataset,
            row_sampler=ColumnIndexOffsetSamplerConfig(
                column_offsets={"value": [0]}
            ),
        )
        sampler = _RecordingRowDataSampler()
        dataset.row_sampler = sampler

        def convert_meta_index2meta(data: dict, column_name=None) -> dict:
            assert column_name is None
            converted = data.copy()
            converted["episode"] = converted.pop("episode_index")
            return converted

        monkeypatch.setattr(
            dataset,
            "convert_meta_index2meta",
            convert_meta_index2meta,
        )
        dataset.meta_index2meta = True

        try:
            batch = dataset.__getitem_no_transform__([2, 0, 2])

            assert batch["value"] == [[30], [10], [30]]
            assert batch["episode"] == [0, 0, 0]
            assert "episode_index" not in batch
            assert len(sampler.batch_row_data_calls) == 1
            batch_row_data = sampler.batch_row_data_calls[0]
            assert batch_row_data is not None
            assert set(batch_row_data) == set(frame_dataset.column_names)
            assert batch_row_data["episode_index"] == [0, 0, 0]
            assert batch_row_data["sampling_context"] == [
                "third",
                "first",
                "third",
            ]
            assert [
                row["sampling_context"]
                for row in sampler.scalar_row_data_calls
                if row is not None
            ] == ["third", "first", "third"]
            assert [
                row["value"]
                for row in sampler.scalar_row_data_calls
                if row is not None
            ] == [30, 10, 30]

            sampler.scalar_row_data_calls.clear()
            row = dataset.__getitem_no_transform__(1)

            assert row["value"] == [20]
            assert row["episode"] == 0
            assert "episode_index" not in row
            assert sampler.scalar_row_data_calls[0] is not None
            assert sampler.scalar_row_data_calls[0]["sampling_context"] == (
                "second"
            )
            assert sampler.scalar_row_data_calls[0]["value"] == 20
        finally:
            dataset.close()

    def test_batch_row_data_must_align_with_indices(self) -> None:
        """Malformed column-major context fails before scalar dispatch."""

        sampler = _RecordingRowDataSampler()
        index_dataset = HFDataset.from_dict({"episode_index": [0, 0]})

        with pytest.raises(
            ValueError,
            match="row_data columns must have the same length as index_batch",
        ):
            sampler.sample_row_idx_batch(
                index_dataset,
                [0, 1],
                row_data={"sampling_context": ["only-one-row"]},
            )

        assert sampler.scalar_row_data_calls == []

    def test_legacy_sampler_signature_remains_compatible(self) -> None:
        """Reader scalar and batch access omit row data for old overrides."""

        frame_dataset = HFDataset.from_dict(
            {
                "index": [0, 1],
                "episode_index": [0, 0],
                "frame_index": [0, 1],
                "task_index": [0, 0],
                "robot_index": [0, 0],
                "instruction_index": [0, 0],
                "timestamp_min": [0, 1],
                "timestamp_max": [0, 1],
                "value": [10, 20],
            }
        )
        base_dataset = RODataset.from_dataset(
            frame_dataset=frame_dataset,
            meta_db_engine=create_engine("sqlite:///:memory:"),
        )
        dataset = ROMultiRowDataset.from_dataset(
            base_dataset,
            row_sampler=ColumnIndexOffsetSamplerConfig(
                column_offsets={"value": [0]}
            ),
        )
        dataset.row_sampler = _LegacyRowDataSampler()

        try:
            assert dataset.__getitem_no_transform__(1)["value"] == [20]
            assert dataset.__getitem_no_transform__([1, 0])["value"] == [
                [20],
                [10],
            ]
        finally:
            dataset.close()

    def test_legacy_custom_offset_hooks_remain_compatible(self) -> None:
        """Custom scalar and batch planning hooks omit unsupported row data."""

        sampler = _LegacyCustomizedOffsetsSampler()
        index_dataset = HFDataset.from_dict({"episode_index": [0, 0]})
        row_data = {"episode_index": [0, 0]}

        assert sampler.sample_row_idx(
            index_dataset,
            1,
            row_data={"episode_index": 0},
        ) == {"value": [1]}
        assert sampler.sample_row_idx_batch(
            index_dataset,
            [1, 0],
            row_data=row_data,
        ) == {"value": [[1], [0]]}

    def test_pickle_preserves_related_reader_terminal_lifecycle(
        self, tmp_path
    ) -> None:
        """Related reader views retain one process-local resource after pickle."""  # noqa: E501

        frame_dataset = HFDataset.from_dict(
            {
                "index": [0],
                "episode_index": [0],
                "frame_index": [0],
                "task_index": [0],
                "robot_index": [0],
                "instruction_index": [0],
                "timestamp_min": [0],
                "timestamp_max": [0],
                "value": [10],
            }
        )
        db_engine = create_engine(f"sqlite:///{tmp_path / 'meta.sqlite'}")
        create_tables(db_engine)
        base_dataset = RODataset.from_dataset(
            frame_dataset=frame_dataset,
            meta_db_engine=db_engine,
        )
        view_dataset = ROMultiRowDataset.from_dataset(
            base_dataset,
            row_sampler=ColumnIndexOffsetSamplerConfig(
                column_offsets={"value": [0]}
            ),
        )
        restored_base, restored_view = pickle.loads(
            pickle.dumps((base_dataset, view_dataset))
        )

        try:
            assert restored_base._db_resource is restored_view._db_resource
            restored_base.close()
            with pytest.raises(RuntimeError, match="RODataset is closed"):
                _ = restored_view.db_engine
        finally:
            base_dataset.close()
            view_dataset.close()
            restored_view.close()


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
    @pytest.fixture()
    def timestamp_index_dataset(self) -> HFDataset:
        """Return two timestamp-aligned episodes with overlapping clocks."""
        timestamps = [
            0,
            1_000_000_000,
            2_000_000_000,
            3_000_000_000,
        ] * 2
        return HFDataset.from_dict(
            {
                "index": list(range(len(timestamps))),
                "episode_index": [0] * 4 + [1] * 4,
                "timestamp_min": timestamps,
                "timestamp_max": timestamps,
            }
        )

    def test_sample_row_idx_batch_matches_scalar(
        self,
        timestamp_index_dataset: HFDataset,
    ) -> None:
        """Batch timestamp sampling preserves scalar order and boundaries."""
        sampler = DeltaTimestampSamplerConfig(
            column_delta_ts={"joints": [-1.0, 0.0, 1.0]},
            tolerance=0.0,
        )()
        index_batch = [1, 2, 5, 6, 6, -1]
        scalar_index_dataset = CachedIndexDataset(timestamp_index_dataset)
        scalar_samples = [
            sampler.sample_row_idx(scalar_index_dataset, index)
            for index in index_batch
        ]

        actual = sampler.sample_row_idx_batch(
            CachedIndexDataset(timestamp_index_dataset),
            index_batch,
        )

        assert actual == {
            column: [sample[column] for sample in scalar_samples]
            for column in sampler.column_rows_keys
        }

    def test_sample_row_idx_batch_coalesces_current_cache_chunks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Timestamp batch sampling loads disjoint current chunks together."""
        dataset_length = 1_000
        timestamp_interval_ns = 40_000_000
        index_dataset = HFDataset.from_dict(
            {
                "index": list(range(dataset_length)),
                "episode_index": [0] * dataset_length,
                "timestamp_min": [
                    index * timestamp_interval_ns
                    for index in range(dataset_length)
                ],
                "timestamp_max": [
                    index * timestamp_interval_ns
                    for index in range(dataset_length)
                ],
            }
        )
        sampler = DeltaTimestampSamplerConfig(
            column_delta_ts={"joints": [0.04]},
            tolerance=0.0,
        )()
        index_batch = [50, 250, 450, 650, 850]
        index_read_calls: list[list[int]] = []
        original_getitems = index_dataset.__getitems__

        def record_getitems(indices: list[int]) -> list[dict]:
            index_read_calls.append(indices)
            return original_getitems(indices)

        monkeypatch.setattr(index_dataset, "__getitems__", record_getitems)

        scalar_actual = MultiRowSampler.sample_row_idx_batch(
            sampler,
            CachedIndexDataset(index_dataset),
            index_batch,
        )
        scalar_read_count = len(index_read_calls)
        index_read_calls.clear()

        batch_actual = sampler.sample_row_idx_batch(
            CachedIndexDataset(index_dataset),
            index_batch,
        )

        assert batch_actual == scalar_actual
        assert scalar_read_count == len(index_batch)
        assert len(index_read_calls) == 1

    def test_row_data_keeps_current_index_out_of_neighbor_reads(
        self,
        timestamp_index_dataset: HFDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Timestamp scans reuse current index fields supplied by row_data."""

        sampler = DeltaTimestampSamplerConfig(
            column_delta_ts={"joints": [-1.0, 0.0, 1.0]},
            tolerance=0.0,
        )()
        current_index = 1
        row_data = timestamp_index_dataset[current_index]
        index_read_calls: list[list[int]] = []
        original_getitems = timestamp_index_dataset.__getitems__

        def record_getitems(indices: list[int]) -> list[dict]:
            index_read_calls.append(indices)
            return original_getitems(indices)

        monkeypatch.setattr(
            timestamp_index_dataset,
            "__getitems__",
            record_getitems,
        )

        assert sampler.sample_row_idx(
            timestamp_index_dataset,
            current_index,
            row_data=row_data,
        ) == {"joints": [0, 1, 2]}
        assert index_read_calls
        assert all(
            current_index not in indices for indices in index_read_calls
        )

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
            meta_index2meta=True,
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


class TestColumnIndexOffsetSampler:
    @pytest.fixture()
    def index_dataset(self) -> HFDataset:
        """Return index rows with episode boundaries for sampler tests."""
        return HFDataset.from_dict(
            {
                "episode_index": [0, 0, 0, 1, 1, 1, 2, 2],
            }
        )

    @pytest.mark.parametrize(
        "cfg, expected_indices",
        [
            (
                ColumnIndexOffsetSamplerConfig(
                    column_offsets={
                        "joints": [0, 1, 2],
                        "left_camera": [0, -1],
                    },
                ),
                {
                    "joints": [0, 1, 2],
                    "left_camera": [0, None],
                },
            ),
            (
                ColumnIndexOffsetSamplerConfig(
                    column_offsets={
                        "joints": [0, 5, 10],
                        "left_camera": [0, -5],
                    },
                ),
                {
                    "joints": [0, 5, 10],
                    "left_camera": [0, None],
                },
            ),
            (
                ColumnIndexOffsetSamplerConfig(
                    column_offsets={
                        "joints": [0, 5, 420, 421],
                        "left_camera": [0, -5],
                    },
                ),
                {
                    "joints": [0, 5, 420, None],  # episode 0 has 421 frames
                    "left_camera": [0, None],
                },
            ),
        ],
    )
    def test_sample_column_index_offsets(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
        cfg: ColumnIndexOffsetSamplerConfig,
        expected_indices: dict[str, list[int | None]],
    ):
        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/robotwin/ro_dataset",
        )

        dataset = ROMultiRowDataset(
            dataset_path=path,
            row_sampler=cfg,
        )

        print(len(dataset))
        sampled_indices = dataset.row_sampler.sample_row_idx(
            dataset.index_dataset,
            0,
        )
        assert sampled_indices == expected_indices

    @pytest.mark.parametrize("force_in_episode", [True, False])
    @pytest.mark.parametrize(
        "use_cached_index_dataset",
        [False, True],
    )
    def test_sample_row_idx_batch_matches_scalar_with_one_index_read(
        self,
        index_dataset: HFDataset,
        monkeypatch: pytest.MonkeyPatch,
        force_in_episode: bool,
        use_cached_index_dataset: bool,
    ) -> None:
        """Batch sampling preserves scalar output with one source read."""
        sampler = ColumnIndexOffsetSamplerConfig(
            column_offsets={
                "camera": [-1, 0, 1, None, 1],
                "joints": [-2, 2],
            },
            force_in_episode=force_in_episode,
        )()
        index_batch = [0, 1, 3, 4, 6, 6, -1]
        scalar_samples = [
            sampler.sample_row_idx(index_dataset, index)
            for index in index_batch
        ]
        expected = {
            column: [sample[column] for sample in scalar_samples]
            for column in sampler.column_rows_keys
        }
        index_read_calls: list[list[int]] = []
        original_getitems = index_dataset.__getitems__

        def record_getitems(indices: list[int]) -> list[dict]:
            index_read_calls.append(indices)
            return original_getitems(indices)

        monkeypatch.setattr(index_dataset, "__getitems__", record_getitems)
        batch_index_dataset: HFDataset | CachedIndexDataset = (
            CachedIndexDataset(index_dataset)
            if use_cached_index_dataset
            else index_dataset
        )

        actual = sampler.sample_row_idx_batch(batch_index_dataset, index_batch)

        assert actual == expected
        expected_read_indices = list(range(len(index_dataset)))
        assert index_read_calls == [expected_read_indices]

    def test_sample_row_idx_batch_empty_does_not_read_index_dataset(
        self,
        index_dataset: HFDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty batch preserves the base sampler's no-read behavior."""
        sampler = ColumnIndexOffsetSamplerConfig(
            column_offsets={"camera": [0, None]},
        )()
        index_read_calls: list[list[int]] = []
        original_getitems = index_dataset.__getitems__

        def record_getitems(indices: list[int]) -> list[dict]:
            index_read_calls.append(indices)
            return original_getitems(indices)

        monkeypatch.setattr(index_dataset, "__getitems__", record_getitems)

        assert sampler.sample_row_idx_batch(index_dataset, []) == {
            "camera": []
        }
        assert index_read_calls == []

    def test_row_data_avoids_reloading_current_index_rows(
        self,
        index_dataset: HFDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Current index fields seed scalar and batch offset resolution."""

        sampler = ColumnIndexOffsetSamplerConfig(
            column_offsets={"camera": [0]},
        )()
        index_read_calls: list[list[int]] = []
        original_getitems = index_dataset.__getitems__

        def record_getitems(indices: list[int]) -> list[dict]:
            index_read_calls.append(indices)
            return original_getitems(indices)

        monkeypatch.setattr(index_dataset, "__getitems__", record_getitems)

        assert sampler.sample_row_idx(
            index_dataset,
            1,
            row_data={"episode_index": 0},
        ) == {"camera": [1]}
        assert sampler.sample_row_idx_batch(
            index_dataset,
            [1, 3, 1],
            row_data={"episode_index": [0, 1, 0]},
        ) == {"camera": [[1], [3], [1]]}
        assert index_read_calls == []

    def test_negative_index_shares_cache_without_changing_sample_output(
        self,
        index_dataset: HFDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cache negative aliases while sampler output preserves them."""

        last_index = len(index_dataset) - 1
        last_row = index_dataset[last_index]
        index_read_calls: list[list[int]] = []
        original_getitems = index_dataset.__getitems__

        def record_getitems(indices: list[int]) -> list[dict]:
            index_read_calls.append(indices)
            return original_getitems(indices)

        monkeypatch.setattr(index_dataset, "__getitems__", record_getitems)
        cached_index_dataset = CachedIndexDataset(index_dataset)
        cached_index_dataset.cache_row(-1, last_row)

        assert cached_index_dataset[last_index] == last_row

        sampler = ColumnIndexOffsetSamplerConfig(
            column_offsets={"camera": [0]},
        )()
        assert sampler.sample_row_idx(cached_index_dataset, -1) == {
            "camera": [-1]
        }
        assert index_read_calls == []

    @pytest.mark.parametrize("invalid_index", [-9, 8])
    def test_sample_row_idx_batch_preserves_invalid_current_index_error(
        self,
        index_dataset: HFDataset,
        invalid_index: int,
    ) -> None:
        """Invalid current indices surface the same HFDataset error."""
        sampler = ColumnIndexOffsetSamplerConfig(
            column_offsets={"camera": [0]},
        )()

        with pytest.raises(IndexError) as scalar_error:
            sampler.sample_row_idx(index_dataset, invalid_index)
        with pytest.raises(IndexError) as batch_error:
            sampler.sample_row_idx_batch(index_dataset, [0, invalid_index])

        assert str(batch_error.value) == str(scalar_error.value)

    def test_multi_row_batch_reuses_current_and_deduplicates_references(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
    ) -> None:
        """Overlapping windows read each non-current row only once."""

        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/robotwin/ro_dataset",
        )
        dataset = ROMultiRowDataset(
            dataset_path=path,
            row_sampler=ColumnIndexOffsetSamplerConfig(
                column_offsets={"joints": [0, 1, 2]}
            ),
        )
        with dataset, RODataset(dataset_path=path) as reference_dataset:
            expected_joints = reference_dataset.frame_dataset[[0, 1, 2, 3]][
                "joints"
            ]
            projected_joints = dataset._column_datasets["joints"]
            projection_calls: list[list[int]] = []

            class _RecordingProjection:
                def __getitem__(self, indices: list[int]):
                    projection_calls.append(indices)
                    return projected_joints[indices]

            dataset._column_datasets["joints"] = _RecordingProjection()

            rows = dataset.__getitem_no_transform__([0, 1])

            assert rows["joints"] == [
                expected_joints[0:3],
                expected_joints[1:4],
            ]
            assert projection_calls == [[2, 3]]

            projection_calls.clear()
            row = dataset.__getitem_no_transform__(0)

            assert row["joints"] == expected_joints[0:3]
            assert projection_calls == [[1, 2]]

    def test_multi_row_materializes_sampled_encoded_image_column(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
    ) -> None:
        """Sampled ImageEncoded rows use the same reader decode seam."""

        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/robotwin/ro_dataset",
        )
        with ROMultiRowDataset(
            dataset_path=path,
            row_sampler=ColumnIndexOffsetSamplerConfig(
                column_offsets={"head_camera": [0, 1]}
            ),
            image_decode_options=RODatasetImageDecodeOptions(
                backend="cv2",
                columns=("head_camera",),
            ),
        ) as dataset:
            batch = dataset.__getitem_no_transform__([0, 1])
            rows = dataset.__getitems__([0, 1])

        assert all(
            isinstance(value, BatchCameraData)
            for sampled_values in batch["head_camera"]
            for value in sampled_values
        )
        assert all(
            isinstance(value, BatchCameraData)
            for row in rows
            for value in row["head_camera"]
        )

    def test_multi_row_image_decode_skips_sampled_signal_seam(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Enabling one camera decoder does not touch sampled signals."""

        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/robotwin/ro_dataset",
        )
        with ROMultiRowDataset(
            dataset_path=path,
            row_sampler=ColumnIndexOffsetSamplerConfig(
                column_offsets={"joints": [0, 1, 2]}
            ),
            image_decode_options=RODatasetImageDecodeOptions(
                columns=("head_camera",),
            ),
        ) as dataset:
            original_materialize = dataset._materialize_storage_features
            sampled_signal_calls: list[dict] = []

            def record_materialize(raw_row):
                if set(raw_row) == {"joints"}:
                    sampled_signal_calls.append(raw_row)
                return original_materialize(raw_row)

            monkeypatch.setattr(
                dataset,
                "_materialize_storage_features",
                record_materialize,
            )
            rows = dataset.__getitem_no_transform__([0, 1])

        assert len(rows["joints"]) == 2
        assert sampled_signal_calls == []


class _DefaultCustomizedOffsetsSampler(CustomizedColumnIndexSampler):
    """Custom sampler fixture with deterministic offset planning."""

    def __init__(self) -> None:
        super().__init__(SimpleNamespace(columns=["camera", "joints"]))
        self.scalar_offset_calls: list[int] = []
        self.scalar_row_data_calls: list[dict[str, Any] | None] = []

    @staticmethod
    def _offsets() -> dict[str, list[int | None]]:
        return {
            "camera": [0, 1, None, 1],
            "joints": [-1, 0],
        }

    def _sample_column_offsets(
        self,
        index_dataset: HFDataset,
        index: int,
        *,
        row_data: Mapping[str, Any] | None = None,
    ) -> dict[str, list[int | None]]:
        self.scalar_offset_calls.append(index)
        self.scalar_row_data_calls.append(
            None if row_data is None else dict(row_data)
        )
        return self._offsets()


class _BatchCustomizedOffsetsSampler(_DefaultCustomizedOffsetsSampler):
    """Custom sampler fixture that coalesces dynamic offset planning."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_offset_calls: list[list[int]] = []
        self.batch_row_data_calls: list[dict[str, list[Any]] | None] = []

    def _sample_column_offsets_batch(
        self,
        index_dataset: HFDataset,
        index_batch: Sequence[int],
        *,
        row_data: Mapping[str, Sequence[Any]] | None = None,
    ) -> list[dict[str, list[int | None]]]:
        self.batch_offset_calls.append(list(index_batch))
        self.batch_row_data_calls.append(
            None
            if row_data is None
            else {column: list(values) for column, values in row_data.items()}
        )
        return [self._offsets() for _ in index_batch]


class TestCustomizedColumnIndexSampler:
    @pytest.fixture()
    def index_dataset(self) -> HFDataset:
        """Return index rows with episode boundaries for custom samplers."""
        return HFDataset.from_dict(
            {
                "episode_index": [0, 0, 0, 1, 1, 1, 2, 2],
            }
        )

    def test_default_batch_offsets_match_scalar_and_share_one_read(
        self,
        index_dataset: HFDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The compatibility hook uses one shared offset-resolution read."""
        sampler = _DefaultCustomizedOffsetsSampler()
        index_batch = [0, 1, 3, 4, 6, 6, -1]
        scalar_samples = [
            sampler.sample_row_idx(index_dataset, index)
            for index in index_batch
        ]
        expected = {
            column: [sample[column] for sample in scalar_samples]
            for column in sampler.column_rows_keys
        }
        sampler.scalar_offset_calls.clear()
        sampler.scalar_row_data_calls.clear()
        index_read_calls: list[list[int]] = []
        original_getitems = index_dataset.__getitems__

        def record_getitems(indices: list[int]) -> list[dict]:
            index_read_calls.append(indices)
            return original_getitems(indices)

        monkeypatch.setattr(index_dataset, "__getitems__", record_getitems)

        row_data = {
            "sampling_context": [
                f"row-{offset}" for offset in range(len(index_batch))
            ]
        }
        actual = sampler.sample_row_idx_batch(
            index_dataset,
            index_batch,
            row_data=row_data,
        )

        assert actual == expected
        assert sampler.scalar_offset_calls == index_batch
        assert sampler.scalar_row_data_calls == [
            {"sampling_context": f"row-{offset}"}
            for offset in range(len(index_batch))
        ]
        assert index_read_calls == [list(range(len(index_dataset)))]

    def test_custom_batch_offsets_override_bypasses_scalar_planning(
        self,
        index_dataset: HFDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Custom samplers may replace compatibility planning with one hook."""
        sampler = _BatchCustomizedOffsetsSampler()
        index_read_calls: list[list[int]] = []
        original_getitems = index_dataset.__getitems__

        def record_getitems(indices: list[int]) -> list[dict]:
            index_read_calls.append(indices)
            return original_getitems(indices)

        monkeypatch.setattr(index_dataset, "__getitems__", record_getitems)

        row_data = {"sampling_context": ["first", "second"]}
        actual = sampler.sample_row_idx_batch(
            index_dataset,
            [0, 3],
            row_data=row_data,
        )

        assert actual == {
            "camera": [[0, 1, None, 1], [3, 4, None, 4]],
            "joints": [[None, 0], [2, 3]],
        }
        assert sampler.batch_offset_calls == [[0, 3]]
        assert sampler.batch_row_data_calls == [row_data]
        assert sampler.scalar_offset_calls == []
        assert index_read_calls == [[0, 1, 2, 3, 4]]
