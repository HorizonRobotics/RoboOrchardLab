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
from typing import Any, Generator

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
from robo_orchard_lab.dataset.robot.re_packing.contracts import (
    RODatasetEpisodeSelection,
)


@dataclass(frozen=True, slots=True)
class SourceFrameCopy:
    frame_index: int
    row: dict[str, Any]
    frame: DataFrame


@dataclass(frozen=True, slots=True)
class SourceCopyReader:
    """Read source episode/frame data used by transform repacking."""

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

    def copy_episode_meta(
        self,
        *,
        source_episode: Episode,
        first_index_row: dict[str, Any],
        selection: RODatasetEpisodeSelection,
        target_episode_index: int,
        target_prev_episode_index: int | None = None,
    ) -> EpisodeMeta:
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

    def iter_frame_copies(
        self,
        frame_indices: list[int],
    ) -> Generator[SourceFrameCopy, None, None]:
        instructions = self._load_frame_instructions(frame_indices)
        for batch_start in range(
            0, len(frame_indices), self.frame_read_batch_size
        ):
            batch_end = min(
                batch_start + self.frame_read_batch_size,
                len(frame_indices),
            )
            yield from self._iter_frame_copy_batch(
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

    def _iter_frame_copy_batch(
        self,
        frame_indices: list[int],
        instructions: list[Instruction | None],
    ) -> Generator[SourceFrameCopy, None, None]:
        rows = self.dataset.frame_dataset.__getitems__(frame_indices)
        for frame_index, row, instruction in zip(
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
            yield SourceFrameCopy(
                frame_index=frame_index,
                row=row,
                frame=DataFrame(
                    features={key: row[key] for key in self._keep_columns},
                    instruction=instruction_data,
                    timestamp_ns_min=row["timestamp_min"],
                    timestamp_ns_max=row["timestamp_max"],
                ),
            )


def make_repack_features(
    source_features: hg_datasets.Features,
) -> hg_datasets.Features:
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


def normalize_optional_meta_index(value: Any, column_name: str) -> int | None:
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
