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
from typing import Type, TypeVar

from datasets import Dataset as HFDataset
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
)
from sortedcontainers import SortedList

__all__ = [
    "DeltaTimestampSampler",
    "DeltaTimestampSamplerConfig",
    "MultiRowSampler",
    "MultiRowSamplerConfig",
]


class CachedIndexDataset:
    def __init__(self, dataset: HFDataset):
        self._dataset = dataset
        self._cache = {}

    def __len__(self) -> int:
        return len(self._dataset)

    def _cache_chunk(self, index: int) -> None:
        """Cache a chunk of the dataset at the given index."""
        min_idx = max(0, index - 100)
        max_idx = min(len(self._dataset), index + 100)
        sliced_dataset = self._dataset.__getitems__(
            [i for i in range(min_idx, max_idx)]
        )
        for i, row in enumerate(sliced_dataset):
            self._cache[min_idx + i] = row

    def __getitem__(self, index: int) -> dict:
        """Get the item at the given index, caching if necessary."""
        if index not in self._cache:
            self._cache_chunk(index)
        return self._cache[index]


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
    ) -> dict[str, list[int | None]]:
        """Sample a list of row indices from the index dataset.

        Args:
            index_dataset (HFDataset): The dataset from which to sample rows.
            index (int): The index or indices to sample.

        Returns:
            dict[str, list[int | None]]: A dictionary where keys are column
            names and values are lists of row indices.

        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    @property
    @abstractmethod
    def column_rows_keys(self) -> dict[str, list]:
        """Get the keys of the rows that are sampled.

        This property is expected to return a dictionary where keys are
        column names and values are lists of row keys. It is useful
        for understanding which columns are sampled and what are the
        corresponding row keys.
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


class DeltaTimestampSampler(MultiRowSampler):
    def __init__(self, cfg: DeltaTimestampSamplerConfig) -> None:
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

    @property
    def column_rows_keys(self) -> dict[str, list]:
        """Get the keys of the rows that are sampled."""
        return self.cfg.column_delta_ts

    def sample_row_idx(
        self, index_dataset: HFDataset | CachedIndexDataset, index: int
    ) -> dict[str, list[int | None]]:
        cur_row = index_dataset[index]
        cache = self._prepare_cache(index_dataset, index)
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
                    # return the nearest row. If look ahead, return the
                    # first row(the smallest timestamp) that matches the
                    # delta timestamp. If look behind, return the last
                    # row (the largest timestamp)
                    # that matches the delta timestamp.
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
    ) -> IndexFrameCache:
        """Prepare the cache for the given index.

        This function relies on the assumption that the index_dataset
        is ordered by episode_index and timestamp.

        """

        def check_idx(row: dict, idx: int) -> None:
            # Not used because select will change idx.
            if row["index"] != idx:
                raise ValueError(
                    f"Row index {row['index']} does not match the expected "
                    f"index {idx}."
                )

        if cache is None:
            cache = IndexFrameCache()
        cur_row = index_dataset[index]
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


class DeltaTimestampSamplerConfig(
    MultiRowSamplerConfig[DeltaTimestampSampler]
):
    """Configuration class for DeltaTimestampSampler.

    This configuration define the sampling strategy based on delta timestamps
    for each column. It allows specifying the delta timestamps and the
    tolerance for matching timestamps.

    """

    class_type: Type[DeltaTimestampSampler] = DeltaTimestampSampler

    column_delta_ts: dict[str, list[float]]
    """A dictionary where keys are column names and values are lists of
    delta timestamps in seconds. This is used to sample rows based on
    the delta timestamps for each column."""

    tolerance: float = 0.01
    """The tolerance in seconds for matching timestamps.

    The first row that matches the delta_timestamp +/- tolerance will be
    selected. This is useful for ensuring that the sampled rows are close
    to the desired delta timestamps, allowing for some flexibility in
    matching due to potential variations in the data.
    """
