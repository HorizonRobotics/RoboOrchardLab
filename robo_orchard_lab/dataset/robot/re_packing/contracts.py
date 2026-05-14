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
from dataclasses import dataclass
from typing import Any, Collection, Protocol

import datasets as hg_datasets

from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode
from robo_orchard_lab.dataset.robot.packaging import DataFrame, EpisodeMeta


@dataclass(frozen=True, slots=True)
class RODatasetEpisodeSelection:
    """Frame selection summary for one source episode being repacked."""

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
        return len(self.selected_frame_indices)

    @property
    def source_episode_frame_count(self) -> int:
        return len(self.source_episode_frame_indices)

    @property
    def is_complete_source_episode(self) -> bool:
        return self.selected_frame_indices == self.source_episode_frame_indices

    @property
    def is_contiguous_source_slice(self) -> bool:
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
class RODatasetRepackContext:
    """Run-level context passed to RODataset repack transforms."""

    source_dataset: RODataset
    source_features: hg_datasets.Features
    target_features: hg_datasets.Features


@dataclass(frozen=True, slots=True)
class RODatasetEpisodeContext:
    """Episode-level context passed to RODataset repack transforms."""

    source_dataset: RODataset
    episode_index: int
    source_episode: Episode
    first_index_row: dict[str, Any]
    selection: RODatasetEpisodeSelection


@dataclass(frozen=True, slots=True)
class RODatasetFrameContext:
    """Frame-level context passed to RODataset repack transforms."""

    source_dataset: RODataset
    episode_context: RODatasetEpisodeContext
    episode_index: int
    frame_index: int
    source_frame_row: dict[str, Any]


class RODatasetRepackTransform(Protocol):
    """Transform hook used by ``repack_dataset(..., transforms=...)``."""

    def required_columns(
        self,
        context: RODatasetRepackContext,
    ) -> Collection[str]:
        """Return existing frame feature columns required by this transform."""
        ...

    def prepare(self, context: RODatasetRepackContext) -> None:
        """Validate run-level state and cache transform-local state."""
        ...

    def update_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        """Return target features after applying this transform."""
        ...

    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta,
        context: RODatasetEpisodeContext,
    ) -> EpisodeMeta | None:
        """Transform or skip an episode metadata object."""
        ...

    def transform_frame(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        """Transform a frame. Returning ``None`` is invalid."""
        ...


class IdentityRODatasetRepackTransform:
    """No-op base class for RODataset repack transforms."""

    def required_columns(
        self,
        context: RODatasetRepackContext,
    ) -> Collection[str]:
        return ()

    def prepare(self, context: RODatasetRepackContext) -> None:
        return None

    def update_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        return features

    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta,
        context: RODatasetEpisodeContext,
    ) -> EpisodeMeta | None:
        return episode_meta

    def transform_frame(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        return frame
