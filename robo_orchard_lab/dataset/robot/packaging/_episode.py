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

"""Episode packaging contracts and transform composition helpers."""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Iterable, Protocol, Sequence

import datasets as hg_datasets
from sqlalchemy.orm import Session

from robo_orchard_lab.dataset.robot.columns import PreservedColumnsKeys
from robo_orchard_lab.dataset.robot.db_orm import Episode
from robo_orchard_lab.dataset.robot.packaging._metadata import (
    EpisodeData,
    EpisodeMetaORM,
    InstructionData,
    RobotData,
    TaskData,
)

if TYPE_CHECKING:
    from robo_orchard_lab.dataset.robot.packaging._writer import (
        DatasetIndexState,
    )

__all__ = [
    "ComposedEpisodePackagingTransform",
    "DataFrame",
    "EpisodeMeta",
    "EpisodePackaging",
    "EpisodePackagingTransform",
    "EpisodePackagingView",
    "IdentityEpisodePackagingTransform",
]


@dataclass
class EpisodeMeta:
    """Metadata for an episode packaging in a RoboOrchard dataset.

    This is the data structure used during the packaging process to
    represent the episode information to be stored in the database.
    """

    episode: EpisodeData
    robot: RobotData | None = None
    task: TaskData | None = None

    def get_transient_orm(
        self, index_state: "DatasetIndexState", session: Session
    ) -> EpisodeMetaORM:
        """Get the transient ORM instance of the episode metadata."""
        expected_index = index_state.last_episode_idx + 1
        episode_index = self.episode.index
        if episode_index is None:
            if self.episode.prev_episode_index is not None:
                raise ValueError(
                    "EpisodeData.prev_episode_index requires "
                    "EpisodeData.index to be set."
                )
            episode_index = expected_index
        elif episode_index != expected_index:
            raise ValueError(
                "EpisodeData.index must match the next target episode "
                f"index {expected_index}, got {episode_index}."
            )

        if self.episode.prev_episode_index is not None:
            prev_episode_index = self.episode.prev_episode_index
            if prev_episode_index < 0 or prev_episode_index >= episode_index:
                raise ValueError(
                    "EpisodeData.prev_episode_index must reference a "
                    "previous target episode; got "
                    f"{prev_episode_index} for episode {episode_index}."
                )

        episode_data = self.episode.__dict__.copy()
        episode_data.pop("index")
        episode = Episode(
            index=episode_index,
            **episode_data,
        )
        robot = (
            self.robot.make_transient_orm(index_state, session=session)
            if self.robot
            else None
        )
        task = (
            self.task.make_transient_orm(index_state, session=session)
            if self.task
            else None
        )

        episode.task_index = task.index if task else None
        episode.robot_index = robot.index if robot else None
        return EpisodeMetaORM(episode=episode, robot=robot, task=task)


@dataclass
class DataFrame:
    """Data for a single frame in a RoboOrchard dataset."""

    features: dict[str, Any]
    instruction: InstructionData | None = None
    timestamp_ns_min: int | None = None
    """The minimum timestamp of the frame in nanoseconds."""
    timestamp_ns_max: int | None = None
    """The maximum timestamp of the frame in nanoseconds."""


class EpisodePackaging(metaclass=ABCMeta):
    """Source interface for one episode payload during dataset packaging.

    Implementations expose episode-level metadata and a lazy frame stream to
    packaging or repack callers. The interface does not own database, Arrow
    writer, sidecar, staging-directory, or cleanup lifecycle; the caller that
    consumes the episode remains responsible for those resources.
    """

    @abstractmethod
    def generate_episode_meta(self) -> EpisodeMeta | None:
        """Generate metadata for the episode if it should be included.

        Should return None if the episode is to be skipped.

        Returns:
            EpisodeMeta | None: Metadata containing the episode, robot,
                and task. If None is returned, the episode will be skipped
                during packaging.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to "
            "generate episode metadata."
        )

    @abstractmethod
    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """Generate frame data for the episode."""
        raise NotImplementedError(
            "This method should be implemented by subclasses to "
            "generate frame data for the episode."
        )


class EpisodePackagingTransform(Protocol):
    """Transform episode payloads without owning packaging resources.

    Use this protocol when a packaging or repack caller needs to rewrite the
    payload schema, metadata, or frame contents while keeping resource
    lifecycle management in the caller. :meth:`prepare_features` is called
    once before writer construction and may cache schema-derived state. A
    transform may skip an entire episode by returning ``None`` from
    :meth:`transform_episode`, but it must not manage database sessions,
    Arrow writers, sidecar writers, temporary directories, or
    finalization/rollback.
    """

    def prepare_features(
        self, features: hg_datasets.Features
    ) -> hg_datasets.Features:
        """Prepare schema-derived state and return this transform's schema.

        Args:
            features (hg_datasets.Features): Current payload feature schema.

        Returns:
            hg_datasets.Features: Payload-only output schema. The result must
                not contain RoboOrchard preserved frame-table columns.
        """
        raise NotImplementedError

    def transform_episode(
        self, episode: EpisodePackaging
    ) -> EpisodePackaging | None:
        """Return a transformed episode or ``None`` to skip it.

        Args:
            episode (EpisodePackaging): Source episode payload. Its metadata
                may be backed by a one-shot source, so transforms should not
                assume repeated upstream metadata reads are cheap or safe.

        Returns:
            EpisodePackaging | None: Transformed episode payload, or ``None``
                when the episode should be omitted.
        """
        raise NotImplementedError


class IdentityEpisodePackagingTransform(EpisodePackagingTransform):
    """No-op transform base for frame-level packaging rewrites.

    Subclasses can override :meth:`prepare_features`,
    :meth:`transform_episode_meta`, or :meth:`transform_frame` while
    inheriting the standard lazy episode wrapper. ``transform_frame`` must
    return a ``DataFrame``; returning ``None`` is treated as a contract
    violation because transforms do not own frame count or frame order
    changes.
    """

    def prepare_features(
        self, features: hg_datasets.Features
    ) -> hg_datasets.Features:
        """Return the input payload feature schema unchanged."""
        return features

    def transform_episode(
        self, episode: EpisodePackaging
    ) -> EpisodePackaging | None:
        """Wrap an episode with lazy metadata and frame transforms."""
        episode_meta = self.transform_episode_meta(
            episode.generate_episode_meta()
        )
        if episode_meta is None:
            return None
        return EpisodePackagingView(
            episode_meta=episode_meta,
            frames=self.transform_frames(episode.generate_frames()),
        )

    def transform_episode_meta(
        self, episode_meta: EpisodeMeta | None
    ) -> EpisodeMeta | None:
        """Return episode metadata unchanged."""
        return episode_meta

    def transform_frames(
        self, frames: Iterable[DataFrame]
    ) -> Generator[DataFrame, None, None]:
        """Yield lazily transformed frames without changing frame count."""
        for frame in frames:
            transformed_frame = self.transform_frame(frame)
            if transformed_frame is None:
                raise TypeError(
                    "IdentityEpisodePackagingTransform.transform_frame "
                    "must return DataFrame, got None."
                )
            yield transformed_frame

    def transform_frame(self, frame: DataFrame) -> DataFrame:
        """Return a frame unchanged."""
        return frame


@dataclass
class EpisodePackagingView(EpisodePackaging):
    """Bind episode metadata to a lazy one-pass frame stream.

    The wrapper is intentionally lightweight: it does not copy frames or make
    the stream replayable. It is used when a caller already owns episode
    metadata and frame-stream construction but needs to present them through
    the shared ``EpisodePackaging`` interface.
    """

    episode_meta: EpisodeMeta | None
    frames: Iterable[DataFrame]

    def generate_episode_meta(self) -> EpisodeMeta | None:
        return self.episode_meta

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        yield from self.frames


class ComposedEpisodePackagingTransform(EpisodePackagingTransform):
    """Prepare and apply packaging transforms for one writer owner.

    The transform owns schema preparation and transform ordering. It does not
    own database, Arrow, sidecar, staging, or cleanup resources; those remain
    with the caller that consumes the returned ``EpisodePackaging``.
    """

    def __init__(
        self,
        transforms: Sequence[EpisodePackagingTransform] | None,
    ) -> None:
        self.transforms = tuple(transforms or ())
        self._target_features: hg_datasets.Features | None = None

    @property
    def target_features(self) -> hg_datasets.Features:
        """Return prepared payload-only target features."""
        if self._target_features is None:
            raise RuntimeError(
                "ComposedEpisodePackagingTransform.prepare_features must be "
                "called before target_features is read."
            )
        return self._target_features.copy()

    def prepare_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        """Prepare transform schemas before writer construction.

        Args:
            features (hg_datasets.Features): Source payload feature schema.

        Returns:
            hg_datasets.Features: Prepared payload-only target features.

        Raises:
            RuntimeError: If the transform has already been prepared.
            ValueError: If a transform returns a schema containing preserved
                frame-table columns.
            TypeError: If a transform returns a non-Features schema.
        """
        if self._target_features is not None:
            raise RuntimeError(
                "ComposedEpisodePackagingTransform is already prepared."
            )
        current_features = features.copy()
        for transform in self.transforms:
            current_features = transform.prepare_features(
                current_features.copy()
            )
            if not isinstance(current_features, hg_datasets.Features):
                raise TypeError(
                    f"{transform!r}.prepare_features must return "
                    "datasets.Features."
                )
            _validate_payload_features(current_features)
            current_features = current_features.copy()
        _validate_payload_features(current_features)
        self._target_features = current_features.copy()
        return self._target_features.copy()

    def transform_episode(
        self, episode: EpisodePackaging
    ) -> EpisodePackaging | None:
        """Apply transforms while preserving source target-link metadata.

        Args:
            episode (EpisodePackaging): Source episode payload.

        Returns:
            EpisodePackaging | None: Transformed episode, or ``None`` when a
                source or transform skips it.

        Raises:
            ValueError: If a transform rewrites ``EpisodeData.index`` or
                ``EpisodeData.prev_episode_index``.
        """
        if self._target_features is None:
            raise RuntimeError(
                "ComposedEpisodePackagingTransform.prepare_features must be "
                "called before transform_episode."
            )
        if not self.transforms:
            return episode

        source_episode_meta = episode.generate_episode_meta()
        if source_episode_meta is None:
            return None
        source_episode_linkage = _target_episode_linkage(source_episode_meta)
        current_episode: EpisodePackaging = EpisodePackagingView(
            episode_meta=source_episode_meta,
            frames=episode.generate_frames(),
        )
        target_episode_meta = source_episode_meta

        for transform in self.transforms:
            transformed_episode = transform.transform_episode(current_episode)
            if transformed_episode is None:
                return None
            target_episode_meta = transformed_episode.generate_episode_meta()
            if target_episode_meta is None:
                return None
            current_episode = EpisodePackagingView(
                episode_meta=target_episode_meta,
                frames=transformed_episode.generate_frames(),
            )
        _validate_target_link_unchanged(
            source_episode_linkage,
            target_episode_meta,
        )
        return current_episode


def _validate_payload_features(features: hg_datasets.Features) -> None:
    """Reject payload schemas that overlap frame-table-owned columns."""
    reserved_columns = sorted(set(features) & set(PreservedColumnsKeys))
    if reserved_columns:
        raise ValueError(
            "Feature schema contains reserved frame-table columns: "
            f"{reserved_columns}."
        )


def _target_episode_linkage(
    episode_meta: EpisodeMeta,
) -> tuple[int | None, int | None]:
    """Snapshot target-link fields before transforms can mutate metadata."""
    return (
        episode_meta.episode.index,
        episode_meta.episode.prev_episode_index,
    )


def _validate_target_link_unchanged(
    source_episode_linkage: tuple[int | None, int | None],
    target_episode_meta: EpisodeMeta,
) -> None:
    """Ensure transforms do not rewrite caller-owned target episode links."""
    if source_episode_linkage == _target_episode_linkage(target_episode_meta):
        return
    raise ValueError(
        "EpisodePackagingTransform must not change target episode linkage "
        "fields EpisodeData.index or EpisodeData.prev_episode_index."
    )
