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
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Generator, Iterable, Sequence

from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode
from robo_orchard_lab.dataset.robot.packaging import (
    ComposedEpisodePackagingTransform,
    DataFrame,
    DatasetPackaging,
    EpisodeMeta,
    EpisodePackaging,
    EpisodePackagingTransform,
    normalize_local_dataset_path,
)
from robo_orchard_lab.dataset.robot.packaging._episode import (
    EpisodePackagingView,
)
from robo_orchard_lab.dataset.robot.re_packing._errors import (
    RepackFrameTransformError,
)
from robo_orchard_lab.dataset.robot.re_packing._source import (
    SourceEpisodeChunk,
    SourceReader,
    _SourceEpisodeSelection,
    make_repack_features,
)


@dataclass(frozen=True, slots=True)
class _PreparedRepackEpisode(EpisodePackaging):
    """Validated replayable episode payload for ``DatasetPackaging``.

    The repack runner drains each transformed episode before packaging so
    an invalid episode is never partially written. ``generate_frames`` replays
    the already validated payloads and copies feature dictionaries to keep
    packaging-side mutations from leaking back into the staged frames.
    """

    episode_meta: EpisodeMeta
    frames: list[DataFrame]
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


@dataclass(slots=True)
class _StagedDatasetWriteSession:
    """Stage a dataset write and publish it only after packaging succeeds.

    The context manager creates a temporary sibling path for
    ``DatasetPackaging`` to own during the write. ``commit`` publishes the
    staged dataset to ``target_path`` and restores an existing target if the
    replacement fails. Leaving the context without ``commit`` removes the
    staging path.
    """

    target_path: str
    force_overwrite: bool
    _staging_path: str | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> _StagedDatasetWriteSession:
        self.target_path = normalize_local_dataset_path(self.target_path)
        self._staging_path = self._make_staging_path()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._staging_path is not None and os.path.exists(
            self._staging_path
        ):
            _remove_path(self._staging_path)

    @property
    def path(self) -> str:
        """Return the active staging path passed to ``DatasetPackaging``."""

        if self._staging_path is None:
            raise RuntimeError("_StagedDatasetWriteSession is not active.")
        return self._staging_path

    def commit(self) -> None:
        """Replace the target path with the staged dataset."""

        staging_path = self.path
        _commit_staged_dataset(
            staging_path=staging_path,
            target_path=self.target_path,
        )
        self._staging_path = None

    def _make_staging_path(self) -> str:
        if os.path.exists(self.target_path) and not self.force_overwrite:
            raise FileExistsError(
                f"The dataset path '{self.target_path}' already exists. "
                "Please remove it or set force_overwrite=True to overwrite."
            )

        target_folder = os.path.dirname(self.target_path)
        os.makedirs(target_folder, exist_ok=True)
        staging_path = tempfile.mkdtemp(
            prefix=f".{os.path.basename(self.target_path)}.tmp-",
            dir=target_folder,
        )
        # DatasetPackaging expects ownership of the output path lifecycle.
        _remove_path(staging_path)
        return staging_path


@dataclass(frozen=True, slots=True)
class RepackEpisodeRunner:
    """Yield validated target episodes to ``DatasetPackaging``.

    The runner reads selected source frames, builds episode metadata, resolves
    complete-episode ``prev_episode_index`` links into target-local indices,
    applies the prepared transform pipeline, and buffers the transformed
    frames for one episode before yielding. Buffering happens at the output
    boundary so a skipped or invalid episode is never partially handed to
    packaging.
    """

    frame_indices: Iterable[int] | None
    pipeline: ComposedEpisodePackagingTransform
    source_reader: SourceReader

    def __iter__(
        self,
    ) -> Generator[_PreparedRepackEpisode, None, None]:
        """Yield transformed episodes or raise if the selection writes none."""

        saw_chunk = False
        yielded_episode = False
        written_episodes: dict[int, _WrittenEpisodeState] = {}
        next_target_episode_index = 0
        for chunk in self.source_reader.iter_episode_chunks(
            self.frame_indices
        ):
            saw_chunk = True
            episode = self._build_episode_buffer(
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
            raise ValueError("No frames selected for repack.")
        if not yielded_episode:
            raise ValueError(
                "Repack produced no episodes; all selected episodes were "
                "skipped."
            )

    def _build_episode_buffer(
        self,
        chunk: SourceEpisodeChunk,
        *,
        target_episode_index: int,
        written_episodes: dict[int, _WrittenEpisodeState],
    ) -> _PreparedRepackEpisode | None:
        """Build and validate one complete target episode buffer.

        Returns ``None`` when a transform intentionally skips the selected
        episode. Otherwise the returned object is ready for
        ``DatasetPackaging`` and contains only payload columns that match the
        final target features.
        """

        dataset = self.source_reader.dataset
        source_episode = dataset.get_meta(Episode, chunk.episode_index)
        if source_episode is None:
            raise RuntimeError(
                "Episode metadata not found for episode "
                f"{chunk.episode_index}."
            )
        first_index_row = dataset.index_dataset[chunk.frame_indices[0]]
        selection = self.source_reader.make_episode_selection(
            episode_index=chunk.episode_index,
            selected_frame_indices=chunk.frame_indices,
            source_episode=source_episode,
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
        episode = self.pipeline.transform_episode(
            EpisodePackagingView(
                episode_meta=episode_meta,
                frames=self.source_reader.iter_packaging_frames(
                    chunk.frame_indices
                ),
            )
        )
        if episode is None:
            return None
        transformed_episode_meta = episode.generate_episode_meta()
        if not isinstance(transformed_episode_meta, EpisodeMeta):
            raise TypeError(
                "Episode transform output must carry EpisodeMeta metadata."
            )

        output_frames = self._drain_and_validate_episode_frames(
            frames=episode.generate_frames(),
            expected_frame_count=len(chunk.frame_indices),
            source_episode_index=chunk.episode_index,
            source_frame_indices=chunk.frame_indices,
        )
        # Repack commits to DatasetPackaging only after the full
        # episode has been transformed and validated.
        return _PreparedRepackEpisode(
            episode_meta=transformed_episode_meta,
            frames=output_frames,
            source_episode_index=chunk.episode_index,
            is_complete_source_episode=selection.is_complete_source_episode,
        )

    def _drain_and_validate_episode_frames(
        self,
        *,
        frames: Iterable[DataFrame],
        expected_frame_count: int,
        source_episode_index: int,
        source_frame_indices: Sequence[int],
    ) -> list[DataFrame]:
        """Drain transform output and enforce one-to-one row semantics."""

        output_frames: list[DataFrame] = []
        frame_iterator = iter(frames)
        frame_offset = 0
        while True:
            try:
                frame = next(frame_iterator)
            except StopIteration:
                break
            except RepackFrameTransformError:
                raise
            except Exception as exc:
                source_frame_index = (
                    source_frame_indices[frame_offset]
                    if frame_offset < len(source_frame_indices)
                    else None
                )
                raise RepackFrameTransformError(
                    source_episode_index=source_episode_index,
                    source_frame_index=source_frame_index,
                    frame_offset=frame_offset,
                    original_error=exc,
                ) from exc

            if not isinstance(frame, DataFrame):
                raise TypeError(
                    "EpisodePackagingTransform must yield DataFrame frames."
                )
            self._validate_frame_features(
                frame,
                source_episode_index=source_episode_index,
                frame_offset=frame_offset,
            )
            output_frames.append(frame)
            frame_offset += 1
        if len(output_frames) != expected_frame_count:
            raise ValueError(
                "Episode transform changed row count for "
                f"episode={source_episode_index}; expected "
                f"{expected_frame_count}, got {len(output_frames)}."
            )
        return output_frames

    def _validate_frame_features(
        self,
        frame: DataFrame,
        *,
        source_episode_index: int,
        frame_offset: int,
    ) -> None:
        expected = set(self.pipeline.target_features)
        actual = set(frame.features)
        if actual == expected:
            return
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Transformed frame features do not match target features for "
            f"episode={source_episode_index} frame_offset={frame_offset}; "
            f"missing={missing!r}, extra={extra!r}."
        )


def repack_dataset(
    source_dataset: RODataset,
    target_path: str,
    frame_indices: Iterable[int] | None = None,
    columns: str | list[str] | None = None,
    writer_batch_size: int = 1024,
    max_shard_size: str | int = "8GB",
    force_overwrite: bool = False,
    transforms: Sequence[EpisodePackagingTransform] | None = None,
) -> None:
    """Write a selected or transformed copy of a RODataset.

    This is the public entry point for RODataset-to-RODataset repacking. The
    selected frames are read from ``source_dataset``, grouped by source
    episode, optionally transformed in user-provided order with
    ``EpisodePackagingTransform``, and then written through the canonical
    dataset packaging path. If ``target_path`` already exists, it is replaced
    only after the full staged dataset has been packaged successfully.

    ``re_packing`` owns source selection, owner-provisional target episode
    indices, previous-episode link remapping, and episode-level atomicity.
    Transforms only see the generic ``EpisodePackaging`` source interface and
    may rewrite payload metadata/frames or skip an entire episode; they do not
    own resource lifecycle or target-link assignment.

    Args:
        source_dataset (RODataset): Source dataset to read from.
        target_path (str): Destination path for the new dataset.
        frame_indices (Iterable[int] | None): An iterable of frame indices
            to include in the re-packaged dataset. Frames from the same
            source episode must stay grouped together. If None, all frames are
            included. Default is None.
        columns (str | list[str] | None): The columns to include in the
            re-packaged dataset. If None, all columns will be included. If
            a string is provided, it will be treated as a single column name.
            Default is None.
        writer_batch_size (int): The batch size for writing the arrow file.
            This may affect the performance of packaging or reading the
            dataset later. Default is 1024.
        max_shard_size (str | int | None): The maximum size of each shard.
            If None, no sharding will be applied. This can be a string
            like '10GB' or an integer representing the size in bytes.
            Default is '8GB'.
        force_overwrite (bool): Whether to overwrite the target path if it
            already exists. An existing target is replaced only after the
            repacked dataset is fully packaged. Default is False.
        transforms (Sequence[EpisodePackagingTransform] | None, optional):
            Resource-free transform pipeline to apply in order. If None,
            repack runs as an identity copy through the canonical runner.
            Default is None.

    Raises:
        FileExistsError: If ``target_path`` exists and ``force_overwrite`` is
            False.
        RepackFrameTransformError: If draining a selected source frame stream
            fails during source materialization or frame-level transform
            execution. The original exception is available through
            ``__cause__`` and ``original_error``.
        ValueError: If the selected frames are empty, duplicated, split across
            episode chunks, all selected episodes are skipped, or a transform
            changes caller-owned target episode linkage.
        TypeError: If a transform returns a value outside the packaging
            transform contract.
    """
    if columns is not None:
        dataset = source_dataset.select_columns(columns, include_index=True)
    else:
        dataset = source_dataset

    pipeline = ComposedEpisodePackagingTransform(transforms)
    target_features = pipeline.prepare_features(
        make_repack_features(dataset.features)
    )
    runner = RepackEpisodeRunner(
        frame_indices=frame_indices,
        pipeline=pipeline,
        source_reader=SourceReader(dataset),
    )
    packing = DatasetPackaging(features=target_features)
    with _StagedDatasetWriteSession(
        target_path=target_path,
        force_overwrite=force_overwrite,
    ) as output:
        packing.packaging(
            episodes=runner,
            dataset_path=output.path,
            writer_batch_size=writer_batch_size,
            max_shard_size=max_shard_size,
            force_overwrite=False,
            fail_fast=True,
        )
        output.commit()


def _resolve_target_prev_episode_index(
    *,
    source_episode: Episode,
    selection: _SourceEpisodeSelection,
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


def _commit_staged_dataset(*, staging_path: str, target_path: str) -> None:
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
