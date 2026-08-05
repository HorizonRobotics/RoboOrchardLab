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

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from inspect import Parameter, signature
from typing import Any, Callable, Mapping, Sequence, Type, TypeVar

from datasets import Dataset as HFDataset
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType,
)
from sortedcontainers import SortedList

__all__ = [
    "DeltaTimestampSampler",
    "DeltaTimestampSamplerConfig",
    "MultiRowSampler",
    "MultiRowSamplerConfig",
    "ColumnIndexOffsetSampler",
    "ColumnIndexOffsetSamplerConfig",
    "CustomizedColumnIndexSampler",
    "CustomizedColumnIndexSamplerConfig",
]


@lru_cache
def _accepts_row_data_keyword(callback: Callable[..., Any]) -> bool:
    """Return whether a sampler hook accepts ``row_data`` as a keyword.

    Public sampler hooks predate read-time materialization context. Inspecting
    the class-level override lets the reader pass that opt-in context to new
    hooks without converting a legacy signature mismatch into a runtime
    ``TypeError``.
    """

    return any(
        parameter.kind is Parameter.VAR_KEYWORD
        or (
            parameter.name == "row_data"
            and parameter.kind
            in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        )
        for parameter in signature(callback).parameters.values()
    )


class CachedIndexDataset:
    def __init__(self, dataset: HFDataset):
        self._dataset = dataset
        self._cache = {}

    @classmethod
    def ensure(
        cls, index_dataset: HFDataset | CachedIndexDataset
    ) -> CachedIndexDataset:
        """Return an existing cache wrapper or create one for a source."""
        if isinstance(index_dataset, cls):
            return index_dataset
        return cls(index_dataset)

    @property
    def source_dataset(self) -> HFDataset:
        """Expose the source only for legacy scalar custom-planning hooks."""
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def cache_row(
        self,
        index: int,
        row_data: Mapping[str, Any] | None,
    ) -> None:
        """Cache current index fields already available in a materialized row.

        A dataset view may omit preserved index columns. In that case this
        method leaves the cache unchanged so a later lookup can fall back to
        the source index dataset.
        """

        if row_data is None:
            return
        index_columns = self._dataset.column_names
        if any(column not in row_data for column in index_columns):
            return
        normalized_index = self._normalize_index(index)
        self._cache[normalized_index] = {
            column: row_data[column] for column in index_columns
        }

    def cache_rows(
        self,
        indices: Sequence[int],
        row_data: Mapping[str, Sequence[Any]] | None,
    ) -> None:
        """Cache current index fields from aligned column-major row data."""

        if row_data is None:
            return
        index_columns = self._dataset.column_names
        if any(column not in row_data for column in index_columns):
            return
        for offset, index in enumerate(indices):
            normalized_index = self._normalize_index(index)
            self._cache[normalized_index] = {
                column: row_data[column][offset] for column in index_columns
            }

    def _normalize_index(self, index: int) -> int:
        """Return a non-negative cache key with HFDataset index semantics."""
        normalized_index = index if index >= 0 else len(self) + index
        if 0 <= normalized_index < len(self):
            return normalized_index

        # Delegate the exception type and message to the wrapped dataset.
        self._dataset.__getitems__([index])
        raise IndexError(f"Index {index} is out of range.")

    def _cache_chunk(self, index: int) -> None:
        """Cache a chunk of the dataset at the given index."""
        normalized_index = self._normalize_index(index)
        min_idx = max(0, normalized_index - 100)
        max_idx = min(len(self._dataset), normalized_index + 100)
        missing_indices = [
            cached_index
            for cached_index in range(min_idx, max_idx)
            if cached_index not in self._cache
        ]
        for cached_index, row in zip(
            missing_indices,
            self._dataset.__getitems__(missing_indices),
            strict=True,
        ):
            self._cache[cached_index] = row

    def _cache_chunks(self, indices: Sequence[int]) -> None:
        """Populate all index-centered cache chunks through one source read."""
        missing_indices = set()
        for index in indices:
            normalized_index = self._normalize_index(index)
            min_idx = max(0, normalized_index - 100)
            max_idx = min(len(self._dataset), normalized_index + 100)
            missing_indices.update(
                cached_index
                for cached_index in range(min_idx, max_idx)
                if cached_index not in self._cache
            )

        if not missing_indices:
            return

        sorted_indices = sorted(missing_indices)
        for index, row in zip(
            sorted_indices,
            self._dataset.__getitems__(sorted_indices),
            strict=True,
        ):
            self._cache[index] = row

    def __getitem__(self, index: int) -> dict:
        """Get the item at the given index, caching if necessary."""
        normalized_index = self._normalize_index(index)
        if normalized_index not in self._cache:
            self._cache_chunk(normalized_index)
        return self._cache[normalized_index]

    def __getitems__(self, indices: Sequence[int]) -> list[dict]:
        """Return rows in input order and batch-fetch only cache misses."""
        normalized_indices = [
            self._normalize_index(index) for index in indices
        ]
        missing_indices = sorted(
            {index for index in normalized_indices if index not in self._cache}
        )
        if missing_indices:
            for index, row in zip(
                missing_indices,
                self._dataset.__getitems__(missing_indices),
                strict=True,
            ):
                self._cache[index] = row
        return [self._cache[index] for index in normalized_indices]


def sec2nanosec(sec: float) -> int:
    """Convert seconds to nanoseconds."""
    return int(sec) * 1000000000 + int((sec - int(sec)) * 1000000000)


def nanosec2sec(nanosec: int) -> float:
    """Convert nanoseconds to seconds."""
    return nanosec / 1000000000.0


def int_iou_1d(min_1: int, max_1: int, min_2: int, max_2: int) -> float:
    """Calculate the intersection over union (IoU) of two 1D intervals.

    Args:
        min_1 (int): The minimum of the first interval.
        max_1 (int): The maximum of the first interval (inclusive).
        min_2 (int): The minimum of the second interval.
        max_2 (int): The maximum of the second interval (inclusive).

    """
    if min_1 > max_1 or min_2 > max_2:
        return 0.0
    intersection = max(0, min(max_1, max_2) - max(min_1, min_2) + 1)
    union = max(max_1, max_2) - min(min_1, min_2) + 1
    return float(intersection) / union


def time_range_match_frame(frame: dict, ts_min: int, ts_max: int) -> bool:
    """Check if the frame matches the given timestamp range.

    Args:
        frame (dict): The frame dictionary containing 'timestamp_min' and
            'timestamp_max'.
        ts_min (int): The minimum timestamp in nanoseconds.
        ts_max (int): The maximum timestamp in nanoseconds (included).

    """
    if frame["timestamp_min"] is None or frame["timestamp_max"] is None:
        raise ValueError(
            "Frame must have both timestamp_min and timestamp_max defined."
        )
    # calculate the iou
    iou = int_iou_1d(
        ts_min, ts_max, frame["timestamp_min"], frame["timestamp_max"]
    )
    return iou > 0


class MultiRowSampler(ClassInitFromConfigMixin, metaclass=ABCMeta):
    """Class for sampling multiple rows of specific columns from a dataset."""

    @abstractmethod
    def sample_row_idx(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        *,
        row_data: Mapping[str, Any] | None = None,
    ) -> dict[str, list[int | None]]:
        """Sample a list of row indices from the index dataset.

        Note:
            This method should be implemented by subclasses to define
            the specific sampling strategy, based on the provided index.

        Args:
            index_dataset (HFDataset): The dataset from which to sample rows.
            index (int): The index or indices to sample.
            row_data (Mapping[str, Any] | None, optional): Materialized data
                for the current dataset-view row, including preserved index
                fields retained by that view, before metadata expansion,
                multi-row columns, and user transforms are applied.
                Implementations must treat it as read-only and must not retain
                it after the call. Defaults to None.

                Legacy overrides that omit this keyword remain supported by
                RODataset access paths but do not receive row data.

        Returns:
            dict[str, list[int | None]]: A dictionary where keys are column
            names and values are lists of row indices.

        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    def _sample_row_idx_with_row_data(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        row_data: Mapping[str, Any] | None,
    ) -> dict[str, list[int | None]]:
        """Call scalar hooks with row data when their signature accepts it."""

        if _accepts_row_data_keyword(type(self).sample_row_idx):
            return self.sample_row_idx(
                index_dataset,
                index,
                row_data=row_data,
            )
        return self.sample_row_idx(index_dataset, index)

    def sample_row_idx_batch(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        *,
        row_data: Mapping[str, Sequence[Any]] | None = None,
    ) -> dict[str, list[list[int | None]]]:
        """Sample a batch of row indices from the index dataset.

        This method is a batch version of `sample_row_idx`, which
        processes multiple indices at once.

        Note:
            The implementation provided here is a simple loop over
            `sample_row_idx`. Subclasses may override this method
            for more efficient batch processing.

        Args:
            index_dataset (HFDataset): The dataset from which to sample rows.
            index_batch (Sequence[int]): A sequence of indices to sample.
            row_data (Mapping[str, Sequence[Any]] | None, optional):
                Materialized current-row data, including preserved index
                fields retained by the view, in column-major form before
                metadata expansion. Every column must align one-to-one with
                ``index_batch``.
                Implementations must treat it as read-only and must not retain
                it after the call. Defaults to None.

        Returns:
            dict[str, list[list[int | None]]]: A dictionary where keys are
            column names and values are lists of lists of row indices.

        """
        self._validate_batch_row_data(row_data, len(index_batch))
        ret: dict[str, list[list[int | None]]] = {
            k: [] for k in self.column_rows_keys
        }
        for offset, idx in enumerate(index_batch):
            current_row_data = self._row_data_at(row_data, offset)
            for column, indices in self._sample_row_idx_with_row_data(
                index_dataset,
                idx,
                current_row_data,
            ).items():
                ret[column].append(indices)
        return ret

    def _sample_row_idx_batch_with_row_data(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        row_data: Mapping[str, Sequence[Any]] | None,
    ) -> dict[str, list[list[int | None]]]:
        """Call batch hooks with row data when their signature accepts it."""

        if _accepts_row_data_keyword(type(self).sample_row_idx_batch):
            return self.sample_row_idx_batch(
                index_dataset,
                index_batch,
                row_data=row_data,
            )
        return self.sample_row_idx_batch(index_dataset, index_batch)

    @staticmethod
    def _validate_batch_row_data(
        row_data: Mapping[str, Sequence[Any]] | None,
        expected_size: int,
    ) -> None:
        """Validate column-major row data against its index batch."""

        if row_data is None:
            return
        invalid_lengths = {
            column: len(values)
            for column, values in row_data.items()
            if len(values) != expected_size
        }
        if invalid_lengths:
            details = ", ".join(
                f"{column}={length}"
                for column, length in invalid_lengths.items()
            )
            raise ValueError(
                "row_data columns must have the same length as index_batch "
                f"({expected_size}); got {details}."
            )

    @staticmethod
    def _row_data_at(
        row_data: Mapping[str, Sequence[Any]] | None,
        offset: int,
    ) -> dict[str, Any] | None:
        """Return one row from validated column-major batch data."""

        if row_data is None:
            return None
        return {column: values[offset] for column, values in row_data.items()}

    @property
    @abstractmethod
    def column_rows_keys(self) -> dict[str, Any]:
        """Get the keys of the rows that are sampled.

        This property is expected to return a dictionary where keys are
        column names and values are the corresponding configuration or
        parameters used for sampling rows from that column.
        It is useful for understanding which columns are sampled and what
        are the sampling strategies or parameters associated with each column.
        """
        raise NotImplementedError(
            "This property should be implemented by subclasses."
        )


MultiRowSamplerType = TypeVar("MultiRowSamplerType", bound=MultiRowSampler)


class MultiRowSamplerConfig(ClassConfig[MultiRowSamplerType]):
    """Configuration class for MultiRowSampler."""

    class_type: Type[MultiRowSamplerType]


class IndexFrameCache:
    """Cache for frames indexed by their timestamps.

    Note that the cached frame should be in the same episode,
    and the timestamp_min and timestamp_max should be defined
    in the frame.
    """

    def __init__(self):
        """Initialize the IndexFrameCache."""
        self._frame_ts_min_list = SortedList(key=lambda x: x[0])
        self._frame_ts_max_list = SortedList(key=lambda x: x[0])
        self._cached_frames = {}

    def get_frame(self, index: int) -> dict | None:
        """Get the frame with the given index from the cache.

        Args:
            index (int): The index of the frame to retrieve.

        Returns:
            dict | None: The frame dictionary if found, otherwise None.

        """
        return self._cached_frames.get(index, None)

    def contain_frame(self, index: int) -> bool:
        """Check if the frame with the given index is in the cache."""
        return index in self._cached_frames

    def add_frame(self, index: int, frame: dict) -> bool:
        """Add a frame to the cache."""
        if index in self._cached_frames:
            return False

        if frame["timestamp_min"] is None or frame["timestamp_max"] is None:
            raise ValueError(
                "Frame must have both timestamp_min and timestamp_max defined."
            )
        self._cached_frames[index] = frame
        self._frame_ts_max_list.add((frame["timestamp_max"], index))
        self._frame_ts_min_list.add((frame["timestamp_min"], index))
        return True

    def get_frame_range(
        self, ts_min: int, ts_max: int
    ) -> None | tuple[int, int]:
        """Get the frames that overlap the given timestamp range.

        Args:
            ts_min (int): The minimum timestamp in nanoseconds.
            ts_max (int): The maximum timestamp in nanoseconds (included).
        """
        if len(self._frame_ts_min_list) == 0:
            return None
        # makesure that ts_max is always greater than candidate_ts_min.
        # any idx before max_idx will have candidate_ts_min <= ts_max
        max_idx = self._frame_ts_min_list.bisect_right((ts_max, None))
        # makesure that ts_min is always less than candidate_ts_max.
        # any idx after min_idx will have ts_min <= candidate_ts_max
        min_idx = self._frame_ts_max_list.bisect_left((ts_min, None))
        if min_idx >= max_idx:
            return None
        max_idx -= 1
        return (
            self._frame_ts_min_list[min_idx][1],
            self._frame_ts_max_list[max_idx][1],
        )  # type: ignore


class _DeltaTimestampOffsetPlanner:
    """Timestamp-to-offset planner shared by DeltaTimestampSampler.

    The planner resolves timestamp matches before the shared offset resolver
    restores the final batch output shape.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

        self._ts_delta_min: int = (
            sec2nanosec(
                min(
                    min(self.cfg.column_delta_ts[k])
                    for k in self.cfg.column_delta_ts
                )
                - self.cfg.tolerance
            )
            if self.cfg.column_delta_ts
            else 0
        )
        self._ts_delta_max: int = (
            sec2nanosec(
                max(
                    max(self.cfg.column_delta_ts[k])
                    for k in self.cfg.column_delta_ts
                )
                + self.cfg.tolerance
            )
            if self.cfg.column_delta_ts
            else 0
        )

    def _sample_column_offsets(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        *,
        row_data: Mapping[str, Any] | None = None,
    ) -> dict[str, list[int | None]]:
        cur_row = index_dataset[index]
        cache = self._prepare_cache(
            index_dataset,
            index,
            current_row=cur_row,
        )
        sampled_indices_by_column = self._sample_row_indices_from_cache(
            index,
            cur_row,
            cache,
        )
        return {
            column: [
                None if sampled_index is None else sampled_index - index
                for sampled_index in sampled_indices
            ]
            for column, sampled_indices in sampled_indices_by_column.items()
        }

    def _sample_column_offsets_batch(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        *,
        row_data: Mapping[str, Sequence[Any]] | None = None,
    ) -> list[dict[str, list[int | None]]]:
        """Plan timestamp-derived offsets with one shared index-row cache.

        Timestamp planning owns the per-episode scan, while the inherited
        custom sampler delegates final candidate resolution to the common
        offset batch helper using this same cache instance.

        Args:
            index_dataset (HFDataset | CachedIndexDataset): Source index rows.
            index_batch (Sequence[int]): Current row indices to plan.
            row_data (Mapping[str, Sequence[Any]] | None, optional):
                Materialized current-row data aligned with ``index_batch``.
                Timestamp planning continues to use ``index_dataset`` as its
                canonical index source. Defaults to None.

        Returns:
            list[dict[str, list[int | None]]]: Offset mappings aligned with
            input current-index order.
        """
        cached_index_dataset = CachedIndexDataset.ensure(index_dataset)
        cached_index_dataset._cache_chunks(index_batch)
        offsets_by_index: dict[int, dict[str, list[int | None]]] = {}
        caches_by_episode: dict[tuple[Any, int | None], IndexFrameCache] = {}
        for index in dict.fromkeys(index_batch):
            cur_row = cached_index_dataset[index]
            # A negative HFDataset index has scalar traversal semantics that
            # differ from its normalized positive cache key, so do not share
            # its timestamp cache with a positive current index.
            cache_key = (
                cur_row["episode_index"],
                index if index < 0 else None,
            )
            cache = caches_by_episode.setdefault(cache_key, IndexFrameCache())
            self._prepare_cache(
                cached_index_dataset,
                index,
                cache=cache,
                current_row=cur_row,
            )
            sampled_indices_by_column = self._sample_row_indices_from_cache(
                index,
                cur_row,
                cache,
            )
            offsets_by_index[index] = {
                column: [
                    None if sampled_index is None else sampled_index - index
                    for sampled_index in sampled_indices
                ]
                for column, sampled_indices in (
                    sampled_indices_by_column.items()
                )
            }
        return [offsets_by_index[index] for index in index_batch]

    def _sample_row_indices_from_cache(
        self,
        index: int,
        cur_row: dict,
        cache: IndexFrameCache,
    ) -> dict[str, list[int | None]]:
        """Resolve configured timestamp deltas from an episode cache."""
        ret: dict[str, list[int | None]] = {}
        for column, delta_ts_list in self.cfg.column_delta_ts.items():
            sampled_rows = []
            for delta_ts in delta_ts_list:
                if delta_ts == 0:
                    # if delta_ts is 0, we just return the current row
                    sampled_rows.append(index)
                    continue

                ts_min = cur_row["timestamp_min"] + sec2nanosec(
                    delta_ts - self.cfg.tolerance
                )
                ts_max = cur_row["timestamp_max"] + sec2nanosec(
                    delta_ts + self.cfg.tolerance
                )
                frame_range = cache.get_frame_range(ts_min, ts_max)
                if frame_range is None:
                    sampled_rows.append(None)
                else:
                    # Return the nearest row. If look ahead, return the first
                    # row (the smallest timestamp) that matches; if behind,
                    # return the last (the largest timestamp) that matches.
                    sampled_rows.append(
                        frame_range[0] if delta_ts > 0 else frame_range[1]
                    )
            ret[column] = sampled_rows
        return ret

    def _prepare_cache(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        cache: IndexFrameCache | None = None,
        current_row: dict | None = None,
    ) -> IndexFrameCache:
        """Prepare the cache for the given index.

        This function relies on the assumption that the index_dataset
        is ordered by episode_index and timestamp.

        """

        if cache is None:
            cache = IndexFrameCache()
        cur_row = index_dataset[index] if current_row is None else current_row
        cache.add_frame(index, cur_row)
        cur_episode = cur_row["episode_index"]
        cur_ts_delta_max = cur_row["timestamp_max"] + self._ts_delta_max
        cur_ts_delta_min = cur_row["timestamp_min"] + self._ts_delta_min

        # generate index cache
        prev_idx = index - 1
        while prev_idx >= 0:
            prev_row = index_dataset[prev_idx]
            prev_row_ts_min = prev_row["timestamp_min"]
            prev_row_ts_max = prev_row["timestamp_max"]
            if prev_row_ts_min is None or prev_row_ts_max is None:
                raise ValueError(
                    "Previous row must have both timestamp_min and "
                    "timestamp_max defined."
                )
            if (
                prev_row_ts_max < cur_ts_delta_min
                or prev_row_ts_min > cur_ts_delta_max
                or prev_row["episode_index"] != cur_episode
            ):
                break
            cache.add_frame(prev_idx, prev_row)
            prev_idx -= 1
        next_idx = index + 1
        while next_idx < len(index_dataset):
            next_row = index_dataset[next_idx]
            next_row_ts_min = next_row["timestamp_min"]
            next_row_ts_max = next_row["timestamp_max"]
            if next_row_ts_min is None or next_row_ts_max is None:
                raise ValueError(
                    "Next row must have both timestamp_min and "
                    "timestamp_max defined."
                )
            if (
                next_row_ts_max < cur_ts_delta_min
                or next_row_ts_min > cur_ts_delta_max
                or next_row["episode_index"] != cur_episode
            ):
                break
            cache.add_frame(next_idx, next_row)
            next_idx += 1
        return cache


class ColumnIndexOffsetSampler(MultiRowSampler):
    """Sampler that samples rows based on column index offsets.

    This sampler selects rows from the dataset based on specified
    index offsets for each column in the same episode.

    Example:
        For example, if the current index is 10, and the column_offsets
        is {"camera": [-1, 0, 1]}, then the sampler will return the indices
        [9, 10, 11] for the "camera" column, provided that these indices
        belong to the same episode as index 10. If any of these indices
        do not belong to the same episode, None will be returned for that
        position.

    """

    cfg: ColumnIndexOffsetSamplerConfig

    def __init__(self, cfg: ColumnIndexOffsetSamplerConfig) -> None:
        self.cfg = cfg

    @property
    def column_rows_keys(self) -> dict[str, list[int | None]]:
        """Get the keys of the rows that are sampled."""
        return self.cfg.column_offsets

    def sample_row_idx(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        *,
        row_data: Mapping[str, Any] | None = None,
    ) -> dict[str, list[int | None]]:
        """Resolve fixed offsets while reusing current-row index fields."""

        index_dataset = CachedIndexDataset.ensure(index_dataset)
        index_dataset.cache_row(index, row_data)
        sampled_indices_by_column = self.sample_row_idx_by_offsets_batch(
            index_dataset,
            [index],
            [self.cfg.column_offsets],
            force_in_episode=self.cfg.force_in_episode,
        )
        return {
            column: sampled_indices[0]
            for column, sampled_indices in sampled_indices_by_column.items()
        }

    def sample_row_idx_batch(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        *,
        row_data: Mapping[str, Sequence[Any]] | None = None,
    ) -> dict[str, list[list[int | None]]]:
        """Sample offset rows for a batch with one index-dataset read.

        The returned batch has the same column, sample, and offset ordering as
        calling :meth:`sample_row_idx` for every current index. Candidate
        indices remain bounded to the non-negative dataset range, matching the
        scalar sampler's offset behavior. A valid negative current index is
        retained for its zero offset because the scalar sampler reads it as
        the current row before bounding non-current candidates.

        Args:
            index_dataset (HFDataset | CachedIndexDataset): Source index rows.
            index_batch (Sequence[int]): Current row indices to sample.
            row_data (Mapping[str, Sequence[Any]] | None, optional):
                Materialized current rows aligned with ``index_batch``. Their
                preserved index fields seed the index cache. Defaults to None.

        Returns:
            dict[str, list[list[int | None]]]: Sampled indices grouped by
            configured column and input current-index order.
        """
        self._validate_batch_row_data(row_data, len(index_batch))
        if not index_batch:
            return {column: [] for column in self.cfg.column_offsets}

        index_dataset = CachedIndexDataset.ensure(index_dataset)
        index_dataset.cache_rows(index_batch, row_data)
        return self.sample_row_idx_by_offsets_batch(
            index_dataset,
            index_batch,
            [self.cfg.column_offsets] * len(index_batch),
            force_in_episode=self.cfg.force_in_episode,
        )

    @staticmethod
    def sample_row_idx_by_offsets(
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        column_offsets: dict[str, list[int | None]],
        force_in_episode: bool,
    ) -> dict[str, list[int | None]]:
        """Sample row indices based on column index offsets.

        This scalar compatibility entrypoint delegates offset resolution to
        :meth:`sample_row_idx_by_offsets_batch` so scalar and batch callers
        preserve one source of truth for bounds and episode semantics.

        Args:
            index_dataset (HFDataset): The dataset from which to sample rows.
            index (int): The index to sample from.
            column_offsets (dict[str, list[int|None]]): A dictionary where
                keys are column names and values are lists of index offsets.
            force_in_episode (bool): Whether to force the sampled rows to be
                in the same episode as the current index.

        Returns:
            dict[str, list[int | None]]: A dictionary where keys are column
                names and values are lists of row indices.
        """

        sampled_indices_by_column = (
            ColumnIndexOffsetSampler.sample_row_idx_by_offsets_batch(
                index_dataset,
                [index],
                [column_offsets],
                force_in_episode=force_in_episode,
            )
        )
        return {
            column: sampled_indices[0]
            for column, sampled_indices in sampled_indices_by_column.items()
        }

    @staticmethod
    def sample_row_idx_by_offsets_batch(
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        column_offsets_batch: Sequence[dict[str, list[int | None]]],
        force_in_episode: bool,
    ) -> dict[str, list[list[int | None]]]:
        """Resolve per-row column offsets through one index-dataset read.

        ``column_offsets_batch`` must align one-to-one with ``index_batch``;
        every entry must contain the same column keys. Callers that use fixed
        offsets may repeat one mapping, while custom samplers can provide
        per-row mappings created by their batch planning hook.

        Args:
            index_dataset (HFDataset): Source index rows.
            index_batch (Sequence[int]): Current row indices to sample.
            column_offsets_batch (Sequence[dict[str, list[int | None]]]):
                Offset mappings in input current-index order.
            force_in_episode (bool): Whether candidates must share an episode
                with their current row.

        Returns:
            dict[str, list[list[int | None]]]: Sampled indices grouped by
            column, current-index order, and offset order.

        Raises:
            ValueError: If offset mappings do not align with the input batch
                or do not expose a stable set of columns.
        """
        if len(index_batch) != len(column_offsets_batch):
            raise ValueError(
                "column_offsets_batch must contain one mapping per index."
            )
        if not index_batch:
            return {}

        columns = tuple(column_offsets_batch[0])
        expected_columns = set(columns)
        if any(
            set(column_offsets) != expected_columns
            for column_offsets in column_offsets_batch[1:]
        ):
            raise ValueError(
                "column_offsets_batch must use the same columns for every "
                "index."
            )

        dataset_length = len(index_dataset)
        for index in index_batch:
            if index < -dataset_length or index >= dataset_length:
                # Preserve the underlying HFDataset exception for invalid
                # current indices without adding a second read on valid input.
                index_dataset.__getitems__([index])

        index_ids = set(index_batch)
        for index, column_offsets in zip(
            index_batch, column_offsets_batch, strict=True
        ):
            for offsets in column_offsets.values():
                for offset in offsets:
                    if offset is None:
                        continue
                    candidate_index = index + offset
                    if 0 <= candidate_index < dataset_length:
                        index_ids.add(candidate_index)

        sorted_index_ids = sorted(index_ids)
        index_rows_by_id = dict(
            zip(
                sorted_index_ids,
                index_dataset.__getitems__(sorted_index_ids),
                strict=True,
            )
        )

        def sample_offsets(
            index: int, offsets: list[int | None]
        ) -> list[int | None]:
            current_episode = index_rows_by_id[index]["episode_index"]
            sampled_indices: list[int | None] = []
            for offset in offsets:
                if offset is None:
                    sampled_indices.append(None)
                    continue

                candidate_index = index + offset
                is_valid_index = (
                    candidate_index == index
                    or 0 <= candidate_index < dataset_length
                )
                if not is_valid_index:
                    sampled_indices.append(None)
                elif (
                    force_in_episode
                    and index_rows_by_id[candidate_index]["episode_index"]
                    != current_episode
                ):
                    sampled_indices.append(None)
                else:
                    sampled_indices.append(candidate_index)
            return sampled_indices

        return {
            column: [
                sample_offsets(index, column_offsets[column])
                for index, column_offsets in zip(
                    index_batch,
                    column_offsets_batch,
                    strict=True,
                )
            ]
            for column in columns
        }


class ColumnIndexOffsetSamplerConfig(
    MultiRowSamplerConfig[ColumnIndexOffsetSampler]
):
    """Configuration class for ColumnIndexOffsetSampler."""

    class_type: ClassType[ColumnIndexOffsetSampler] = ColumnIndexOffsetSampler

    column_offsets: dict[str, list[int | None]]
    """A dictionary where keys are column names and values are lists of
    index offsets. This is used to sample rows based on index offsets
    for each column."""

    force_in_episode: bool = True
    """Whether to force the sampled rows to be in the same episode
    as the current index."""


class CustomizedColumnIndexSampler(MultiRowSampler):
    """Sampler that samples rows based on customized column index list.

    This sampler selects rows from the dataset based on specified
    index lists for each column in the same episode.


    User should inherit this class and implement the method
    `_sample_column_offsets` to define how to sample the column offsets
    for each index.

    """

    cfg: CustomizedColumnIndexSamplerConfig

    def __init__(self, cfg: CustomizedColumnIndexSamplerConfig) -> None:
        self.cfg = cfg

    @property
    def column_rows_keys(self) -> dict[str, None]:
        """Get the keys of the rows that are sampled."""
        return {k: None for k in self.cfg.columns}

    @abstractmethod
    def _sample_column_offsets(
        self,
        index_dataset: HFDataset,
        index: int,
        *,
        row_data: Mapping[str, Any] | None = None,
    ) -> dict[str, list[int | None]]:
        """Sample column offsets for the given index.

        Args:
            index_dataset (HFDataset): The dataset from which to sample rows.
            index (int): The index to sample from.
            row_data (Mapping[str, Any] | None, optional): Materialized data
                for the current dataset-view row. Defaults to None.

                Legacy overrides that omit this keyword remain supported by
                RODataset access paths but do not receive row data.

        Returns:
            dict[str, list[int | None]]: A dictionary where keys are column
                names and values are lists of index offsets.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    def _sample_column_offsets_with_row_data(
        self,
        index_dataset: HFDataset,
        index: int,
        row_data: Mapping[str, Any] | None,
    ) -> dict[str, list[int | None]]:
        """Call custom scalar planning hooks without breaking old overrides."""

        if _accepts_row_data_keyword(type(self)._sample_column_offsets):
            return self._sample_column_offsets(
                index_dataset,
                index,
                row_data=row_data,
            )
        return self._sample_column_offsets(index_dataset, index)

    def _sample_column_offsets_batch(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        *,
        row_data: Mapping[str, Sequence[Any]] | None = None,
    ) -> list[dict[str, list[int | None]]]:
        """Plan column offsets for a batch of current indices.

        The default preserves existing custom samplers by invoking the scalar
        planning hook in input order. A custom sampler that can coalesce its
        own index reads should override this method while returning one offset
        mapping per input index in the same order.

        Args:
            index_dataset (HFDataset | CachedIndexDataset): Source index rows
                for offset planning.
            index_batch (Sequence[int]): Current row indices to plan.
            row_data (Mapping[str, Sequence[Any]] | None, optional):
                Materialized current rows aligned with ``index_batch``.
                Defaults to None.

        Returns:
            list[dict[str, list[int | None]]]: Offset mappings aligned with
            ``index_batch``.
        """
        cached_index_dataset = CachedIndexDataset.ensure(index_dataset)
        return [
            self._sample_column_offsets_with_row_data(
                cached_index_dataset.source_dataset,
                index,
                self._row_data_at(row_data, offset),
            )
            for offset, index in enumerate(index_batch)
        ]

    def _sample_column_offsets_batch_with_row_data(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        row_data: Mapping[str, Sequence[Any]] | None,
    ) -> list[dict[str, list[int | None]]]:
        """Call custom batch planning hooks without breaking old overrides."""

        if _accepts_row_data_keyword(type(self)._sample_column_offsets_batch):
            return self._sample_column_offsets_batch(
                index_dataset,
                index_batch,
                row_data=row_data,
            )
        return self._sample_column_offsets_batch(index_dataset, index_batch)

    def sample_row_idx(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        *,
        row_data: Mapping[str, Any] | None = None,
    ) -> dict[str, list[int | None]]:
        cached_index_dataset = CachedIndexDataset.ensure(index_dataset)
        cached_index_dataset.cache_row(index, row_data)
        column_offsets = self._sample_column_offsets_with_row_data(
            cached_index_dataset.source_dataset,
            index,
            row_data,
        )
        return ColumnIndexOffsetSampler.sample_row_idx_by_offsets(
            cached_index_dataset,
            index,
            column_offsets,
            force_in_episode=False,
        )

    def sample_row_idx_batch(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index_batch: Sequence[int],
        *,
        row_data: Mapping[str, Sequence[Any]] | None = None,
    ) -> dict[str, list[list[int | None]]]:
        """Sample custom offset rows through the shared batch resolver.

        Dynamic offset planning remains owned by
        :meth:`_sample_column_offsets_batch`; once planned, all valid current
        and candidate index rows are resolved by one shared batch read.

        Args:
            index_dataset (HFDataset | CachedIndexDataset): Source index rows.
            index_batch (Sequence[int]): Current row indices to sample.
            row_data (Mapping[str, Sequence[Any]] | None, optional):
                Materialized current rows aligned with ``index_batch``.
                Defaults to None.

        Returns:
            dict[str, list[list[int | None]]]: Sampled indices grouped by
            configured column and input current-index order.
        """
        self._validate_batch_row_data(row_data, len(index_batch))
        if not index_batch:
            return {column: [] for column in self.column_rows_keys}

        cached_index_dataset = CachedIndexDataset.ensure(index_dataset)
        cached_index_dataset.cache_rows(index_batch, row_data)
        column_offsets_batch = self._sample_column_offsets_batch_with_row_data(
            cached_index_dataset,
            index_batch,
            row_data,
        )
        return ColumnIndexOffsetSampler.sample_row_idx_by_offsets_batch(
            cached_index_dataset,
            index_batch,
            column_offsets_batch,
            force_in_episode=False,
        )


class CustomizedColumnIndexSamplerConfig(
    MultiRowSamplerConfig[CustomizedColumnIndexSampler]
):
    class_type: ClassType[CustomizedColumnIndexSampler]

    columns: list[str]
    """The list of columns to sample from. """


class DeltaTimestampSampler(
    _DeltaTimestampOffsetPlanner,
    CustomizedColumnIndexSampler,
):
    """Timestamp sampler expressed as dynamically planned column offsets.

    Timestamp matching remains this class's responsibility. Once matching rows
    are converted to offsets, inherited custom-sampler orchestration resolves
    them through the shared batch helper and the same index-row cache.
    """

    cfg: DeltaTimestampSamplerConfig

    def __init__(self, cfg: DeltaTimestampSamplerConfig) -> None:
        _DeltaTimestampOffsetPlanner.__init__(self, cfg)

    @property
    def column_rows_keys(self) -> dict[str, list[float]]:
        """Get the configured timestamp-sampled columns."""
        return self.cfg.column_delta_ts

    def sample_row_idx(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
        *,
        row_data: Mapping[str, Any] | None = None,
    ) -> dict[str, list[int | None]]:
        """Plan and resolve one timestamp sample through one row cache."""
        cached_index_dataset = CachedIndexDataset.ensure(index_dataset)
        cached_index_dataset.cache_row(index, row_data)
        column_offsets = self._sample_column_offsets(
            cached_index_dataset,
            index,
            row_data=row_data,
        )
        return ColumnIndexOffsetSampler.sample_row_idx_by_offsets(
            cached_index_dataset,
            index,
            column_offsets,
            force_in_episode=False,
        )


class DeltaTimestampSamplerConfig(
    MultiRowSamplerConfig[DeltaTimestampSampler]
):
    """Configuration class for DeltaTimestampSampler.

    This configuration defines the sampling strategy based on delta timestamps
    for each column. It allows specifying the delta timestamps and tolerance
    for matching timestamps.
    """

    class_type: ClassType[DeltaTimestampSampler] = DeltaTimestampSampler

    column_delta_ts: dict[str, list[float]]
    """A dictionary where keys are column names and values are lists of
    delta timestamps in seconds. This is used to sample rows based on
    the delta timestamps for each column."""

    tolerance: float = 0.01
    """The tolerance in seconds for matching timestamps.

    The first row that matches the delta timestamp +/- tolerance is selected.
    This allows modest variations in timestamp alignment.
    """
