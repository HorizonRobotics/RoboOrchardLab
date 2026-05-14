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
from typing import Generator, Generic, Iterable, TypeVar

from robo_orchard_lab.dataset.robot.columns import PreservedColumnsKeys
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Instruction, Robot, Task
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    TaskData,
)
from robo_orchard_lab.dataset.robot.re_packing.contracts import (
    RODatasetEpisodeSelection,
)
from robo_orchard_lab.dataset.robot.re_packing.selection import (
    make_episode_selection,
)
from robo_orchard_lab.dataset.robot.re_packing.source_copy import (
    make_repack_features,
    normalize_optional_meta_index,
)


class DefaultRePackingEpisodeHelper(EpisodePackaging):
    """Helper class to re-package an episode from an existing dataset.

    User can inherit this class to customize the episode meta or frame
    generation by optionally overriding:
    - `generate_episode_meta` to customize the episode meta.
    - `generate_frames` to customize the frame data.

    Args:
        dataset (RODataset): The dataset to re-package from.
        episode_frames (list[int]): The list of frame indices that belong to
            the episode to be re-packaged. Note that all frames must belong
            to the same episode.
    """

    def __init__(self, dataset: RODataset, episode_frames: list[int]):
        if len(episode_frames) == 0:
            raise ValueError("episode_frames cannot be empty.")
        self.dataset = dataset
        self.frame_index_list = episode_frames
        self.frame_index_list.sort()

        index_dict = self.dataset.index_dataset[self.frame_index_list]
        episode_list = index_dict["episode_index"]
        if not all(eid == episode_list[0] for eid in episode_list):
            raise ValueError("All frames must belong to the same episode.")
        self._current_episode_index = episode_list[0]
        self._episode_selection = make_episode_selection(
            dataset=self.dataset,
            episode_index=int(self._current_episode_index),
            selected_frame_indices=self.frame_index_list,
        )

    @property
    def episode_selection(self) -> RODatasetEpisodeSelection:
        return self._episode_selection

    def generate_episode_meta(self) -> EpisodeMeta:
        frame_start = self.frame_index_list[0]
        row = self.dataset.index_dataset[frame_start]
        robot_index = normalize_optional_meta_index(
            row["robot_index"], "robot_index"
        )
        task_index = normalize_optional_meta_index(
            row["task_index"], "task_index"
        )
        orm_robot = self.dataset.get_meta(Robot, robot_index)
        assert orm_robot is not None
        orm_task = self.dataset.get_meta(Task, task_index)
        assert orm_task is not None
        robot: RobotData = RobotData.from_orm(orm_robot)
        task = TaskData(
            name=orm_task.name,
            description=orm_task.description,
            info=orm_task.info,
        )
        return EpisodeMeta(episode=EpisodeData(), robot=robot, task=task)

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """Generate frame data for the episode."""

        preserved_columns = set(PreservedColumnsKeys)

        keep_columns = [
            key
            for key in self.dataset.features
            if key not in preserved_columns
        ]

        # Cache all instructions for faster access.
        index_rows = self.dataset.index_dataset[self.frame_index_list]
        all_instruction = self.dataset.get_meta(
            Instruction, index_rows["instruction_index"]
        )

        batch_size = 128
        for batch_start in range(0, len(self.frame_index_list), batch_size):
            batch_end = min(
                batch_start + batch_size, len(self.frame_index_list)
            )
            batch_indices = self.frame_index_list[batch_start:batch_end]
            batch_row = self.dataset.frame_dataset.__getitems__(batch_indices)
            for i, idx in enumerate(batch_indices):
                row = batch_row[i]
                instruction_index = row["instruction_index"]
                orm_instruction = all_instruction[batch_start + i]
                if orm_instruction is None:
                    raise RuntimeError(
                        f"Instruction not found for frame index {idx} "
                        f"with instruction_index {instruction_index}"
                    )
                features = {key: row[key] for key in keep_columns}
                frame = DataFrame(
                    features=features,
                    instruction=InstructionData(
                        name=orm_instruction.name,
                        json_content=orm_instruction.json_content,
                    ),
                    timestamp_ns_min=row["timestamp_min"],
                    timestamp_ns_max=row["timestamp_max"],
                )
                yield frame


RePackingEpisodeType = TypeVar(
    "RePackingEpisodeType", bound=DefaultRePackingEpisodeHelper
)


class RePackingDatasetHelper(Generic[RePackingEpisodeType]):
    """Iterator to generate episodes for re-packaging.

    Args:
        dataset (RODataset): The dataset to re-package from.
        frame_indices (Iterable[int]): An iterable of frame indices to include
            in the re-packaged dataset. Frames from the same episode should be
            grouped together!
    """

    def __init__(
        self,
        dataset: RODataset,
        frame_indices: Iterable[int],
        packing_impl: type[
            RePackingEpisodeType
        ] = DefaultRePackingEpisodeHelper,
    ):
        self.dataset = dataset
        self.frame_indices = frame_indices
        self.packing_impl = packing_impl

    def __iter__(self):
        return self._next_helper()

    def _next_helper(
        self,
    ) -> Generator[RePackingEpisodeType, None, None]:
        current_episode_index = None
        current_episode_frames = []

        def process_batch(batch_frame_indices: list[int]):
            nonlocal current_episode_index, current_episode_frames
            batch_rows = self.dataset.index_dataset.__getitems__(
                batch_frame_indices
            )
            for i, frame_index in enumerate(batch_frame_indices):
                row = batch_rows[i]
                episode_index = row["episode_index"]
                if current_episode_index is None:
                    current_episode_index = episode_index

                if episode_index != current_episode_index:
                    yield self.packing_impl(
                        self.dataset, current_episode_frames
                    )
                    current_episode_index = episode_index
                    current_episode_frames = [frame_index]
                else:
                    current_episode_frames.append(frame_index)

        batch_size = 1024
        batch_frame_indices = []
        for frame_index in self.frame_indices:
            batch_frame_indices.append(frame_index)
            if len(batch_frame_indices) >= batch_size:
                yield from process_batch(batch_frame_indices)
                batch_frame_indices.clear()
        if batch_frame_indices:
            yield from process_batch(batch_frame_indices)
        if current_episode_frames:
            yield self.packing_impl(self.dataset, current_episode_frames)


def helper_repack_dataset(
    *,
    dataset: RODataset,
    target_path: str,
    frame_indices: Iterable[int] | None,
    writer_batch_size: int,
    max_shard_size: str | int,
    force_overwrite: bool,
    packing_impl: type[RePackingEpisodeType],
    fail_fast: bool,
) -> None:
    features = make_repack_features(dataset.features)

    if frame_indices is None:
        frame_indices = range(len(dataset.index_dataset))

    packing = DatasetPackaging(features=features)
    packing.packaging(
        episodes=RePackingDatasetHelper(
            dataset, frame_indices, packing_impl=packing_impl
        ),
        dataset_path=target_path,
        writer_batch_size=writer_batch_size,
        max_shard_size=max_shard_size,
        force_overwrite=force_overwrite,
        fail_fast=fail_fast,
    )
