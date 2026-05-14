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

from __future__ import annotations
import operator
from dataclasses import dataclass
from typing import Any, Generator, Iterable

from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode
from robo_orchard_lab.dataset.robot.re_packing.contracts import (
    RODatasetEpisodeSelection,
)

_INDEX_READ_BATCH_SIZE = 1024


@dataclass(frozen=True, slots=True)
class TransformEpisodeChunk:
    episode_index: int
    frame_indices: list[int]


def resolve_transform_episode_chunks(
    dataset: RODataset,
    frame_indices: Iterable[int] | None,
) -> Generator[TransformEpisodeChunk, None, None]:
    closed_episode_indices: set[int] = set()
    current_episode_index: int | None = None
    current_frame_indices: list[int] = []
    current_frame_index_set: set[int] = set()

    for frame_index, row in _iter_selected_index_rows(
        dataset=dataset,
        frame_indices=frame_indices,
    ):
        episode_index = int(row["episode_index"])
        if current_episode_index is None:
            current_episode_index = episode_index
        elif episode_index != current_episode_index:
            if episode_index in closed_episode_indices:
                raise ValueError(
                    "frame_indices must group frames from the same source "
                    f"episode together; episode {episode_index} appears in "
                    "multiple chunks."
                )
            closed_episode_indices.add(current_episode_index)
            yield TransformEpisodeChunk(
                episode_index=current_episode_index,
                frame_indices=sorted(current_frame_indices),
            )
            current_episode_index = episode_index
            current_frame_indices = []
            current_frame_index_set = set()
        if frame_index in current_frame_index_set:
            raise ValueError(
                f"frame_indices contains duplicate frame index {frame_index}."
            )
        current_frame_index_set.add(frame_index)
        current_frame_indices.append(frame_index)

    if current_episode_index is not None:
        yield TransformEpisodeChunk(
            episode_index=current_episode_index,
            frame_indices=sorted(current_frame_indices),
        )


def make_episode_selection(
    *,
    dataset: RODataset,
    episode_index: int,
    selected_frame_indices: list[int],
    source_episode_frame_indices: list[int] | None = None,
    source_episode: Episode | None = None,
) -> RODatasetEpisodeSelection:
    if source_episode_frame_indices is None:
        source_episode_frame_indices = _resolve_source_episode_frame_indices(
            dataset=dataset,
            episode_index=episode_index,
            source_episode=source_episode,
        )
    selected = tuple(selected_frame_indices)
    source = tuple(source_episode_frame_indices)
    source_frame_set = set(source)
    missing = [
        frame_index
        for frame_index in selected
        if frame_index not in source_frame_set
    ]
    if missing:
        raise ValueError(
            f"selected frame {missing[0]} is not part of source episode "
            f"{episode_index}."
        )
    return RODatasetEpisodeSelection(
        selected_frame_indices=selected,
        source_episode_frame_indices=source,
    )


def _resolve_source_episode_frame_indices(
    *,
    dataset: RODataset,
    episode_index: int,
    source_episode: Episode | None = None,
) -> list[int]:
    if source_episode is None:
        source_episode = dataset.get_meta(Episode, episode_index)
    if source_episode is None:
        raise RuntimeError(
            f"Episode metadata not found for episode {episode_index}."
        )
    if (
        source_episode.dataset_begin_index >= 0
        and source_episode.frame_num >= 0
    ):
        begin = source_episode.dataset_begin_index
        return list(range(begin, begin + source_episode.frame_num))
    return _scan_source_episode_frame_indices(
        dataset=dataset,
        episode_index=episode_index,
    )


def _iter_selected_index_rows(
    *,
    dataset: RODataset,
    frame_indices: Iterable[int] | None,
) -> Generator[tuple[int, dict[str, Any]], None, None]:
    frame_num = len(dataset.index_dataset)
    if frame_indices is None:
        for batch_start in range(0, frame_num, _INDEX_READ_BATCH_SIZE):
            batch_indices = list(
                range(
                    batch_start,
                    min(batch_start + _INDEX_READ_BATCH_SIZE, frame_num),
                )
            )
            yield from _read_index_rows(dataset, batch_indices)
        return

    batch_indices: list[int] = []
    for raw_frame_index in frame_indices:
        try:
            frame_index = operator.index(raw_frame_index)
        except TypeError as exc:
            raise TypeError(
                "frame_indices must contain integer indices."
            ) from exc
        if frame_index < 0 or frame_index >= frame_num:
            raise IndexError(
                f"frame index {frame_index} is out of range for dataset "
                f"with {frame_num} frames."
            )

        batch_indices.append(frame_index)
        if len(batch_indices) >= _INDEX_READ_BATCH_SIZE:
            yield from _read_index_rows(dataset, batch_indices)
            batch_indices.clear()

    if batch_indices:
        yield from _read_index_rows(dataset, batch_indices)


def _read_index_rows(
    dataset: RODataset,
    frame_indices: list[int],
) -> Generator[tuple[int, dict[str, Any]], None, None]:
    rows = dataset.index_dataset.__getitems__(frame_indices)
    for frame_index, row in zip(frame_indices, rows, strict=True):
        yield frame_index, row


def _scan_source_episode_frame_indices(
    *,
    dataset: RODataset,
    episode_index: int,
) -> list[int]:
    frame_indices: list[int] = []
    frame_num = len(dataset.index_dataset)
    for batch_start in range(0, frame_num, _INDEX_READ_BATCH_SIZE):
        batch_indices = list(
            range(
                batch_start,
                min(batch_start + _INDEX_READ_BATCH_SIZE, frame_num),
            )
        )
        for frame_index, row in _read_index_rows(dataset, batch_indices):
            if int(row["episode_index"]) == episode_index:
                frame_indices.append(frame_index)
    if not frame_indices:
        raise RuntimeError(
            f"No source frames found for episode {episode_index}."
        )
    return frame_indices
