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
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from typing import Collection, Generator, Iterable

import datasets as hg_datasets

from robo_orchard_lab.dataset.robot.columns import PreservedColumnsKeys
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeMeta,
    EpisodePackaging,
)
from robo_orchard_lab.dataset.robot.re_packing.contracts import (
    RODatasetEpisodeContext,
    RODatasetEpisodeSelection,
    RODatasetFrameContext,
    RODatasetRepackContext,
    RODatasetRepackTransform,
)
from robo_orchard_lab.dataset.robot.re_packing.selection import (
    TransformEpisodeChunk,
    make_episode_selection,
    resolve_transform_episode_chunks,
)
from robo_orchard_lab.dataset.robot.re_packing.source_copy import (
    SourceCopyReader,
    make_repack_features,
)

_PRESERVED_COLUMNS = set(PreservedColumnsKeys)


@dataclass(frozen=True, slots=True)
class BufferedTransformEpisode(EpisodePackaging):
    """Episode buffer yielded after all selected frames are transformed."""

    episode_meta: EpisodeMeta
    frames: tuple[DataFrame, ...]
    source_episode_index: int
    is_complete_source_episode: bool

    def generate_episode_meta(self) -> EpisodeMeta:
        return self.episode_meta

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        for frame in self.frames:
            yield replace(frame, features=frame.features.copy())


@dataclass(frozen=True, slots=True)
class _WrittenEpisodeState:
    target_episode_index: int
    is_complete_source_episode: bool


@dataclass(frozen=True, slots=True)
class TransformPipeline:
    transforms: list[RODatasetRepackTransform]
    target_features: hg_datasets.Features
    target_feature_keys: frozenset[str]

    @classmethod
    def prepare(
        cls,
        *,
        dataset: RODataset,
        transforms: Iterable[RODatasetRepackTransform],
    ) -> TransformPipeline:
        transform_list = list(transforms)
        target_features = make_repack_features(dataset.features)
        context = RODatasetRepackContext(
            source_dataset=dataset,
            source_features=dataset.features,
            target_features=target_features,
        )
        for transform in transform_list:
            required_columns = transform.required_columns(context)
            cls._validate_required_columns(
                required_columns,
                target_features=context.target_features,
                transform=transform,
            )
            transform.prepare(context)
            updated_features = transform.update_features(
                context.target_features
            )
            target_features = cls._validate_target_features(
                updated_features,
                transform=transform,
            )
            context = replace(context, target_features=target_features)
        return cls(
            transforms=transform_list,
            target_features=target_features,
            target_feature_keys=frozenset(target_features.keys()),
        )

    def transform_episode_meta(
        self,
        episode_meta: EpisodeMeta,
        context: RODatasetEpisodeContext,
    ) -> EpisodeMeta | None:
        for transform in self.transforms:
            updated_episode_meta = transform.transform_episode_meta(
                episode_meta, context
            )
            if updated_episode_meta is None:
                return None
            if not isinstance(updated_episode_meta, EpisodeMeta):
                raise TypeError(
                    f"{type(transform).__name__}.transform_episode_meta() "
                    "must return EpisodeMeta or None."
                )
            episode_meta = updated_episode_meta
        return episode_meta

    def transform_frame(
        self,
        *,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        for transform in self.transforms:
            frame = transform.transform_frame(frame, context)
            if frame is None:
                raise TypeError(
                    f"{type(transform).__name__}.transform_frame() returned "
                    f"None for episode={context.episode_index} "
                    f"frame={context.frame_index}."
                )
            if not isinstance(frame, DataFrame):
                raise TypeError(
                    f"{type(transform).__name__}.transform_frame() must "
                    "return DataFrame."
                )
            self._validate_frame_has_no_preserved_columns(
                frame,
                transform=transform,
                frame_index=context.frame_index,
            )
        self._validate_frame_features_match_target(frame, context)
        return frame

    @staticmethod
    def _validate_required_columns(
        columns: Collection[str],
        *,
        target_features: hg_datasets.Features,
        transform: RODatasetRepackTransform,
    ) -> None:
        if isinstance(columns, str):
            raise TypeError(
                f"{type(transform).__name__}.required_columns() must return a "
                "collection of column names, not a string."
            )
        for column in columns:
            if column in _PRESERVED_COLUMNS:
                raise ValueError(
                    f"{type(transform).__name__} requires reserved column "
                    f"{column!r}."
                )
            if column not in target_features:
                raise ValueError(
                    f"{type(transform).__name__} requires missing column "
                    f"{column!r}."
                )

    @staticmethod
    def _validate_target_features(
        features: hg_datasets.Features,
        *,
        transform: RODatasetRepackTransform,
    ) -> hg_datasets.Features:
        if not isinstance(features, hg_datasets.Features):
            raise TypeError(
                f"{type(transform).__name__}.update_features() must return "
                "datasets.Features."
            )
        reserved = [key for key in features if key in _PRESERVED_COLUMNS]
        if reserved:
            raise ValueError(
                f"{type(transform).__name__}.update_features() returned "
                f"reserved columns {reserved!r}."
            )
        return features

    @staticmethod
    def _validate_frame_has_no_preserved_columns(
        frame: DataFrame,
        *,
        transform: RODatasetRepackTransform,
        frame_index: int,
    ) -> None:
        reserved = [key for key in frame.features if key in _PRESERVED_COLUMNS]
        if reserved:
            raise ValueError(
                f"{type(transform).__name__}.transform_frame() returned "
                f"reserved columns {reserved!r} for frame {frame_index}."
            )

    def _validate_frame_features_match_target(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> None:
        expected = self.target_feature_keys
        actual = set(frame.features.keys())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "Transformed frame features do not match target features for "
                f"episode={context.episode_index} "
                f"frame={context.frame_index}; "
                f"missing={missing!r}, extra={extra!r}."
            )


@dataclass(frozen=True, slots=True)
class TransformRepackRunner:
    dataset: RODataset
    frame_indices: Iterable[int] | None
    pipeline: TransformPipeline
    source_reader: SourceCopyReader

    def __iter__(
        self,
    ) -> Generator[BufferedTransformEpisode, None, None]:
        saw_chunk = False
        yielded_episode = False
        written_episodes: dict[int, _WrittenEpisodeState] = {}
        next_target_episode_index = 0
        for chunk in resolve_transform_episode_chunks(
            self.dataset,
            self.frame_indices,
        ):
            saw_chunk = True
            episode = self.build_episode_buffer(
                chunk,
                target_episode_index=next_target_episode_index,
                written_episodes=written_episodes,
            )
            if episode is not None:
                yielded_episode = True
                yield episode
                written_episodes[episode.source_episode_index] = (
                    _WrittenEpisodeState(
                        target_episode_index=next_target_episode_index,
                        is_complete_source_episode=(
                            episode.is_complete_source_episode
                        ),
                    )
                )
                next_target_episode_index += 1
        if not saw_chunk:
            raise ValueError("No frames selected for transform repack.")
        if not yielded_episode:
            raise ValueError(
                "Transform repack produced no episodes; all selected episodes "
                "were skipped."
            )

    def build_episode_buffer(
        self,
        chunk: TransformEpisodeChunk,
        *,
        target_episode_index: int,
        written_episodes: dict[int, _WrittenEpisodeState],
    ) -> BufferedTransformEpisode | None:
        """Build a complete episode buffer before yielding it to packaging."""

        source_episode = self.dataset.get_meta(Episode, chunk.episode_index)
        if source_episode is None:
            raise RuntimeError(
                "Episode metadata not found for episode "
                f"{chunk.episode_index}."
            )
        first_index_row = self.dataset.index_dataset[chunk.frame_indices[0]]
        selection = make_episode_selection(
            dataset=self.dataset,
            episode_index=chunk.episode_index,
            selected_frame_indices=chunk.frame_indices,
            source_episode=source_episode,
        )
        episode_context = RODatasetEpisodeContext(
            source_dataset=self.dataset,
            episode_index=chunk.episode_index,
            source_episode=source_episode,
            first_index_row=first_index_row,
            selection=selection,
        )
        target_prev_episode_index = _resolve_target_prev_episode_index(
            source_episode=source_episode,
            selection=selection,
            written_episodes=written_episodes,
        )
        episode_meta = self.source_reader.copy_episode_meta(
            source_episode=source_episode,
            first_index_row=first_index_row,
            selection=selection,
            target_episode_index=target_episode_index,
            target_prev_episode_index=target_prev_episode_index,
        )
        episode_meta = self.pipeline.transform_episode_meta(
            episode_meta, episode_context
        )
        if episode_meta is None:
            return None

        frames: list[DataFrame] = []
        for source_frame in self.source_reader.iter_frame_copies(
            chunk.frame_indices
        ):
            frame_context = RODatasetFrameContext(
                source_dataset=self.dataset,
                episode_context=episode_context,
                episode_index=episode_context.episode_index,
                frame_index=source_frame.frame_index,
                source_frame_row=source_frame.row,
            )
            frames.append(
                self.pipeline.transform_frame(
                    frame=source_frame.frame,
                    context=frame_context,
                )
            )
        # Transform mode commits to DatasetPackaging only after the full
        # episode has been transformed and validated.
        return BufferedTransformEpisode(
            episode_meta=episode_meta,
            frames=tuple(frames),
            source_episode_index=chunk.episode_index,
            is_complete_source_episode=selection.is_complete_source_episode,
        )


def _resolve_target_prev_episode_index(
    *,
    source_episode: Episode,
    selection: RODatasetEpisodeSelection,
    written_episodes: dict[int, _WrittenEpisodeState],
) -> int | None:
    if not selection.is_complete_source_episode:
        return None
    source_prev_episode_index = source_episode.prev_episode_index
    if source_prev_episode_index is None:
        return None
    prev_state = written_episodes.get(source_prev_episode_index)
    if prev_state is None or not prev_state.is_complete_source_episode:
        return None
    return prev_state.target_episode_index


def transform_repack_dataset(
    *,
    dataset: RODataset,
    target_path: str,
    frame_indices: Iterable[int] | None,
    writer_batch_size: int,
    max_shard_size: str | int,
    force_overwrite: bool,
    transforms: list[RODatasetRepackTransform],
) -> None:
    pipeline = TransformPipeline.prepare(
        dataset=dataset,
        transforms=transforms,
    )
    runner = TransformRepackRunner(
        dataset=dataset,
        frame_indices=frame_indices,
        pipeline=pipeline,
        source_reader=SourceCopyReader(dataset),
    )
    packing = DatasetPackaging(features=pipeline.target_features)
    staging_path = _make_staging_dataset_path(target_path, force_overwrite)
    try:
        packing.packaging(
            episodes=runner,
            dataset_path=staging_path,
            writer_batch_size=writer_batch_size,
            max_shard_size=max_shard_size,
            force_overwrite=False,
            fail_fast=True,
        )
        _commit_staged_dataset(
            staging_path=staging_path,
            target_path=target_path,
        )
    finally:
        if os.path.exists(staging_path):
            shutil.rmtree(staging_path, ignore_errors=True)


def _make_staging_dataset_path(target_path: str, force_overwrite: bool) -> str:
    target_path = os.path.abspath(os.path.expanduser(target_path))
    if os.path.exists(target_path) and not force_overwrite:
        raise FileExistsError(
            f"The dataset path '{target_path}' already exists. "
            "Please remove it or set force_overwrite=True to overwrite."
        )

    target_folder = os.path.dirname(target_path)
    os.makedirs(target_folder, exist_ok=True)
    staging_path = tempfile.mkdtemp(
        prefix=f".{os.path.basename(target_path)}.tmp-",
        dir=target_folder,
    )
    # DatasetPackaging expects ownership of the output path lifecycle.
    _remove_path(staging_path)
    return staging_path


def _commit_staged_dataset(*, staging_path: str, target_path: str) -> None:
    target_path = os.path.abspath(os.path.expanduser(target_path))
    backup_path: str | None = None
    if os.path.exists(target_path):
        backup_path = tempfile.mkdtemp(
            prefix=f".{os.path.basename(target_path)}.backup-",
            dir=os.path.dirname(target_path),
        )
        shutil.rmtree(backup_path)
        os.rename(target_path, backup_path)

    try:
        os.rename(staging_path, target_path)
    except Exception:
        if backup_path is not None and os.path.exists(backup_path):
            os.rename(backup_path, target_path)
        raise

    if backup_path is not None:
        _remove_path(backup_path)


def _remove_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
