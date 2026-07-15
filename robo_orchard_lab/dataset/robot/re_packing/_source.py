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
import copy
import operator
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable

import datasets as hg_datasets

from robo_orchard_lab.dataset.datatypes.hg_features import RODictDataFeature
from robo_orchard_lab.dataset.robot.columns import PreservedColumnsKeys
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    EpisodeData,
    EpisodeMeta,
    InstructionData,
    RobotData,
    TaskData,
)

_INDEX_READ_BATCH_SIZE = 1024


@dataclass(frozen=True, slots=True)
class SourceEpisodeChunk:
    """Consecutive selected source frames that belong to one source episode."""

    episode_index: int
    frame_indices: list[int]


@dataclass(frozen=True, slots=True)
class _SourceEpisodeSelection:
    """Source-frame selection state used by the repack owner.

    The selection is intentionally internal to ``re_packing``. Transforms see
    the unified ``EpisodePackaging`` contract, while the runner uses this
    source-side state to decide whether original episode metadata and previous
    episode links can be preserved.
    """

    selected_frame_indices: tuple[int, ...]
    source_episode_frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        selected = tuple(self.selected_frame_indices)
        source = tuple(self.source_episode_frame_indices)
        source_frame_set = set(source)
        missing = sorted(
            frame_index
            for frame_index in selected
            if frame_index not in source_frame_set
        )
        if missing:
            raise ValueError(
                "selected frame indices must be part of the source episode; "
                f"missing={missing!r}."
            )
        object.__setattr__(self, "selected_frame_indices", selected)
        object.__setattr__(self, "source_episode_frame_indices", source)

    @property
    def selected_frame_count(self) -> int:
        """Return how many source frames are visible in the target episode."""

        return len(self.selected_frame_indices)

    @property
    def source_episode_frame_count(self) -> int:
        """Return how many frames exist in the original source episode."""

        return len(self.source_episode_frame_indices)

    @property
    def is_complete_source_episode(self) -> bool:
        """Whether the selection exactly covers the source episode."""

        return self.selected_frame_indices == self.source_episode_frame_indices

    @property
    def is_contiguous_source_slice(self) -> bool:
        """Whether selected frames form one contiguous source-episode slice."""

        if not self.selected_frame_indices:
            return False
        source_positions = {
            frame_index: position
            for position, frame_index in enumerate(
                self.source_episode_frame_indices
            )
        }
        selected_positions = [
            source_positions[index] for index in self.selected_frame_indices
        ]
        return selected_positions == list(
            range(
                selected_positions[0],
                selected_positions[0] + len(selected_positions),
            )
        )


@dataclass(frozen=True, slots=True)
class SourceReader:
    """Read source selection, metadata, and payload frames for repacking.

    ``SourceReader`` is the source-side boundary used by the repack runner.
    It groups selected dataset-level frame indices by source episode, copies
    metadata into packaging DTOs, and yields packaging ``DataFrame`` objects
    in bounded frame batches so transforms can stream data.

    Args:
        dataset (RODataset): Source dataset, already column-projected if the
            public ``repack_dataset(columns=...)`` option was used.
        frame_read_batch_size (int): Number of source frame rows to read per
            frame payload batch. Must be positive.
    """

    dataset: RODataset
    frame_read_batch_size: int = 128
    _keep_columns: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame_read_batch_size <= 0:
            raise ValueError("frame_read_batch_size must be positive.")
        keep_columns = [
            key
            for key in self.dataset.features
            if key not in PreservedColumnsKeys
        ]
        object.__setattr__(self, "_keep_columns", keep_columns)

    def iter_episode_chunks(
        self,
        frame_indices: Iterable[int] | None,
    ) -> Generator[SourceEpisodeChunk, None, None]:
        """Yield selected source frames grouped by source episode.

        The input may be any iterable. Duplicate frame indices are rejected,
        and frames from an episode must not reappear after another episode has
        started. Within each yielded chunk, frame indices are sorted into
        source dataset order.
        """

        yield from _resolve_episode_chunks(
            dataset=self.dataset,
            frame_indices=frame_indices,
        )

    def make_episode_selection(
        self,
        *,
        episode_index: int,
        selected_frame_indices: list[int],
        source_episode_frame_indices: list[int] | None = None,
        source_episode: Episode | None = None,
    ) -> _SourceEpisodeSelection:
        """Return selected-vs-source frame ownership for one episode."""

        return _make_episode_selection(
            dataset=self.dataset,
            episode_index=episode_index,
            selected_frame_indices=selected_frame_indices,
            source_episode_frame_indices=source_episode_frame_indices,
            source_episode=source_episode,
        )

    def copy_episode_meta(
        self,
        *,
        source_episode: Episode,
        first_index_row: dict[str, Any],
        selection: _SourceEpisodeSelection,
        target_episode_index: int,
        target_prev_episode_index: int | None = None,
    ) -> EpisodeMeta:
        """Copy source metadata into a target-local packaging object.

        Complete source episodes preserve episode flags, info, and compatible
        ``prev_episode_index`` links after the caller maps them into target
        episode indices. Partial episodes keep only the target episode index;
        frame count and dataset offsets are recomputed by packaging.
        """

        if selection.is_complete_source_episode:
            episode_data = EpisodeData(
                index=target_episode_index,
                prev_episode_index=target_prev_episode_index,
                truncated=source_episode.truncated,
                success=source_episode.success,
                info=copy.deepcopy(source_episode.info),
            )
        else:
            episode_data = EpisodeData(index=target_episode_index)
        return EpisodeMeta(
            episode=episode_data,
            robot=_get_robot_data(
                self.dataset,
                normalize_optional_meta_index(
                    first_index_row["robot_index"], "robot_index"
                ),
            ),
            task=_get_task_data(
                self.dataset,
                normalize_optional_meta_index(
                    first_index_row["task_index"], "task_index"
                ),
            ),
        )

    def iter_packaging_frames(
        self,
        frame_indices: list[int],
    ) -> Generator[DataFrame, None, None]:
        """Yield selected source frames as packaging payloads.

        The yielded frame payload excludes preserved RODataset metadata columns
        because packaging owns those columns in the target dataset.
        """

        instructions = self._load_frame_instructions(frame_indices)
        for batch_start in range(
            0, len(frame_indices), self.frame_read_batch_size
        ):
            batch_end = min(
                batch_start + self.frame_read_batch_size,
                len(frame_indices),
            )
            yield from self._iter_packaging_frame_batch(
                frame_indices[batch_start:batch_end],
                instructions[batch_start:batch_end],
            )

    def _load_frame_instructions(
        self,
        frame_indices: list[int],
    ) -> list[Instruction | None]:
        index_rows = self.dataset.index_dataset.__getitems__(frame_indices)
        instruction_indices = [
            normalize_optional_meta_index(
                row["instruction_index"], "instruction_index"
            )
            for row in index_rows
        ]
        return self.dataset.get_meta(Instruction, instruction_indices)

    def _iter_packaging_frame_batch(
        self,
        frame_indices: list[int],
        instructions: list[Instruction | None],
    ) -> Generator[DataFrame, None, None]:
        rows = self.dataset.frame_dataset.__getitems__(frame_indices)
        for _frame_index, row, instruction in zip(
            frame_indices,
            rows,
            instructions,
            strict=True,
        ):
            instruction_index = normalize_optional_meta_index(
                row["instruction_index"], "instruction_index"
            )
            if instruction_index is not None and instruction is None:
                raise RuntimeError(
                    f"Instruction metadata not found for {instruction_index}."
                )
            instruction_data = (
                InstructionData(
                    name=instruction.name,
                    json_content=copy.deepcopy(instruction.json_content),
                )
                if instruction is not None
                else None
            )
            yield DataFrame(
                features={key: row[key] for key in self._keep_columns},
                instruction=instruction_data,
                timestamp_ns_min=row["timestamp_min"],
                timestamp_ns_max=row["timestamp_max"],
            )


def make_repack_features(
    source_features: hg_datasets.Features,
) -> hg_datasets.Features:
    """Return the payload feature schema used for repack output.

    Preserved RODataset metadata columns are removed because packaging writes
    them. Adapted ``RODictDataFeature`` instances are reset before reuse so the
    output schema matches the current runtime feature definition instead of a
    loaded dataset adapter state.
    """

    preserved_columns = set(PreservedColumnsKeys)
    features = {
        key: copy.deepcopy(feature)
        for key, feature in source_features.items()
        if key not in preserved_columns
    }
    features = hg_datasets.Features(features)
    # Check whether a feature is adapted for loading. If so, reset it before
    # packaging so the output schema matches the current code version.
    for _, feature in features.items():
        if isinstance(feature, RODictDataFeature):
            try:
                feature.reset()
            except NotImplementedError:
                pass
    return features


def normalize_optional_meta_index(value: Any, column_name: str) -> int | None:
    """Normalize an optional metadata index from a source row.

    ``None`` stays ``None``. Integer-like values are converted with
    ``operator.index``. Boolean values are rejected even though they are
    integer subclasses because metadata indices should not silently accept
    true/false sentinels.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{column_name} must be an integer metadata index.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{column_name} must be an integer metadata index or None."
        ) from exc


def _resolve_episode_chunks(
    *,
    dataset: RODataset,
    frame_indices: Iterable[int] | None,
) -> Generator[SourceEpisodeChunk, None, None]:
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
            yield SourceEpisodeChunk(
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
        yield SourceEpisodeChunk(
            episode_index=current_episode_index,
            frame_indices=sorted(current_frame_indices),
        )


def _make_episode_selection(
    *,
    dataset: RODataset,
    episode_index: int,
    selected_frame_indices: list[int],
    source_episode_frame_indices: list[int] | None = None,
    source_episode: Episode | None = None,
) -> _SourceEpisodeSelection:
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
    return _SourceEpisodeSelection(
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


def _get_robot_data(
    dataset: RODataset,
    robot_index: int | None,
) -> RobotData | None:
    if robot_index is None:
        return None
    robot = dataset.get_meta(Robot, robot_index)
    if robot is None:
        raise RuntimeError(f"Robot metadata not found for {robot_index}.")
    return RobotData.from_orm(robot)


def _get_task_data(
    dataset: RODataset,
    task_index: int | None,
) -> TaskData | None:
    if task_index is None:
        return None
    task = dataset.get_meta(Task, task_index)
    if task is None:
        raise RuntimeError(f"Task metadata not found for {task_index}.")
    return TaskData(
        name=task.name,
        description=task.description,
        info=copy.deepcopy(task.info),
    )
