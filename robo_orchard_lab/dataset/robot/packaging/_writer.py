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

"""Dataset writer implementation for RoboOrchard packaging."""

from __future__ import annotations
import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import datasets as hg_datasets
import fsspec
from datasets.exceptions import DatasetGenerationError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, make_transient

from robo_orchard_lab.dataset.packaging_paths import (
    DatasetPackagingPaths,
    _create_coordination_lock,
)
from robo_orchard_lab.dataset.robot.columns import (
    PreservedColumnsKeys,
    PreservedIndexColumns,
    PreservedIndexColumnsKeys,
)
from robo_orchard_lab.dataset.robot.dataset_db_engine import (
    create_engine,
    create_tables,
    get_local_db_url,
)
from robo_orchard_lab.dataset.robot.db_orm import Instruction
from robo_orchard_lab.dataset.robot.packaging._episode import (
    ComposedEpisodePackagingTransform,
    DataFrame,
    EpisodePackaging,
    EpisodePackagingTransform,
)
from robo_orchard_lab.dataset.robot.packaging._metadata import (
    EpisodeMetaORM,
    InstructionData,
)
from robo_orchard_lab.utils.filesystem import (
    remove_path,
    rename_noreplace,
)

__all__ = [
    "DatasetIndexState",
    "DatasetPackaging",
    "dataset_format_version",
]

dataset_format_version = "0.1.0"
logger = logging.getLogger(__name__)


class DatasetPackaging:
    """Package episode sources into a local RoboOrchard dataset.

    ``DatasetPackaging`` owns the Arrow writer, metadata database creation,
    target frame/episode indices, and output cleanup for direct packaging.
    Callers provide payload-only frame features and an iterable of
    ``EpisodePackaging`` sources. Optional ``transforms`` are constructor
    inputs because they can change the payload feature schema before the
    writer is built.

    The optional transform sequence is resource-free from this class's point
    of view: transforms may rewrite metadata and frames or skip whole
    episodes, but database, Arrow, sidecar, staging, and rollback lifecycle
    remain owned by the packaging caller.

    Each call builds under a disposable sibling workspace and publishes only a
    complete dataset with a no-replace rename. Cooperative writers for one
    normalized target share a cache-resident advisory lock. The target parent,
    target, workspace, and lock must not be replaced externally during an
    active call. ``force_overwrite=True`` may remove the existing target and
    its reserved stale workspace before building; direct packaging does not
    promise rollback after that removal.

    Each call serializes same-target writers with a stable lock in the user
    cache, builds Hugging Face output inside a temporary sibling workspace,
    and publishes it with an atomic same-filesystem no-replace rename. The
    workspace and its HF locks are removed before return, so the dataset and
    its parent retain no packaging lock files. ``force_overwrite=True`` still
    removes an existing target before generation and does not restore it after
    a later failure. Writers on different hosts must share ``XDG_CACHE_HOME``
    to coordinate writes to the same shared-filesystem target.

    The internal Hugging Face builder uses a fixed config ID so HF does not
    fingerprint the generator closure and its potentially streaming or
    unpickleable ``episodes`` iterable. This is safe only because every call
    owns a fresh ``hf_cache_dir`` inside its target-scoped workspace and
    same-target calls are serialized. If that cache becomes persistent or
    shared, its config ID must instead distinguish all cache-relevant inputs.

    Args:
        features (hg_datasets.Features): Payload frame features provided by
            source episodes before RoboOrchard preserved frame-table columns
            are injected.
        database_driver (str): The database driver to use for the meta
            database. Default is "duckdb".
        check_timestamp (bool, optional): Whether to check the
            timestamp of the frames. Timestamps are required for time-based
            queries and operations. If True, it will raise an error
            if the timestamp is not set or if timestamp_min is greater
            than timestamp_max. Default is False.
        transforms (Sequence[EpisodePackagingTransform] | None, optional):
            Resource-free transforms applied to each episode payload before
            frame writing. Transforms run in the provided order, may update
            the payload feature schema, and may skip entire episodes. They do
            not own database, Arrow writer, sidecar, staging-directory, or
            cleanup lifecycle. Default is None.

    """

    def __init__(
        self,
        features: hg_datasets.Features,
        database_driver: str = "duckdb",
        check_timestamp: bool = False,
        *,
        transforms: Sequence[EpisodePackagingTransform] | None = None,
    ):
        self._transform_pipeline = ComposedEpisodePackagingTransform(
            transforms
        )
        self._payload_features = self._transform_pipeline.prepare_features(
            features
        )
        self._payload_feature_keys = set(self._payload_features)
        self._features = self._check_and_update_features(
            self._payload_features
        )
        self._database_driver = database_driver
        self._index_state: DatasetIndexState = DatasetIndexState()
        self._instruction_cache: InstructionCache = InstructionCache()
        self._check_timestamp = check_timestamp

    @property
    def features(self) -> hg_datasets.Features:
        """Return writer features including RoboOrchard preserved columns."""
        return self._features

    def _check_and_update_features(
        self, features: hg_datasets.Features
    ) -> hg_datasets.Features:
        """Validate payload features and add frame-table index columns."""
        index_keys = PreservedIndexColumnsKeys
        for key in index_keys:
            if key in features:
                raise ValueError(
                    f"Feature '{key}' is reserved for internal use "
                    "and cannot be used in the dataset features."
                )
        # Add index fields to the features
        ret = features.copy()
        for key in index_keys:
            if key not in ret:
                ret[key] = hg_datasets.Value(dtype="int64")
        return ret

    def _extend_frame_with_index(
        self,
        frame: DataFrame,
        episode_meta: EpisodeMetaORM,
        instruction: Instruction | None,
    ):
        """Validate one payload frame and append target-local index columns."""
        features = frame.features
        for key in PreservedColumnsKeys:
            if key in features:
                raise ValueError(
                    f"key '{key}' is reserved for internal use "
                    "and cannot be used in the frame features."
                )
        feature_keys = set(features)
        if feature_keys != self._payload_feature_keys:
            missing_keys = sorted(self._payload_feature_keys - feature_keys)
            extra_keys = sorted(feature_keys - self._payload_feature_keys)
            raise ValueError(
                "Frame features must match the packaging payload schema. "
                f"Missing keys: {missing_keys}. Extra keys: {extra_keys}."
            )

        index_columns = PreservedIndexColumns(
            index=self._index_state.last_frame_idx + 1,
            frame_index=self._index_state.last_episode_frame_idx + 1,
            episode_index=episode_meta.episode.index,
            robot_index=episode_meta.robot.index
            if episode_meta.robot
            else None,
            task_index=episode_meta.task.index if episode_meta.task else None,
            instruction_index=instruction.index if instruction else None,
            timestamp_min=frame.timestamp_ns_min,
            timestamp_max=frame.timestamp_ns_max,
        )

        features.update(index_columns.__dict__)

    def _add_instruction(
        self, instruction_data: InstructionData, engine: Engine
    ) -> Instruction:
        """Add an instruction to the database and update the index state."""
        cached_instruction = self._instruction_cache.get(instruction_data)
        if cached_instruction is not None:
            # If the instruction is already cached, return it
            return cached_instruction

        with Session(engine, expire_on_commit=False) as session:
            instruction = instruction_data.make_transient_orm(
                self._index_state, session=session
            )
            if instruction.index > self._index_state.last_instruction_idx:
                session.add(instruction)
                session.commit()
                make_transient(instruction)
                self._instruction_cache.add(instruction)
            return instruction

    def _make_packaging_generator(
        self,
        episodes: Iterable[EpisodePackaging],
        db_path: str,
        *,
        fail_fast: bool,
    ):
        if os.path.exists(db_path):
            raise FileExistsError(
                f"The meta database path '{db_path}' already exists."
            )
        url = get_local_db_url(
            drivername=self._database_driver, db_path=db_path
        )
        engine = create_engine(url=url, echo=False)
        create_tables(engine=engine)

        def frame_generator(episode: EpisodePackaging):
            try:
                for frame in episode.generate_frames():
                    if self._check_timestamp:
                        if (
                            frame.timestamp_ns_min is None
                            or frame.timestamp_ns_max is None
                        ):
                            raise ValueError(
                                "Frame must have both timestamp_ns_min "
                                "and timestamp_ns_max set."
                            )
                        if frame.timestamp_ns_min > frame.timestamp_ns_max:
                            raise ValueError(
                                "timestamp_ns_min cannot be greater than "
                                "timestamp_ns_max."
                            )

                    yield frame
            except Exception as e:
                if fail_fast:
                    raise
                warnings.warn(
                    f"Failed to generate frames for {episode}. "
                    f"Skipping this episode. Error: "
                )
                import traceback

                traceback.print_exception(e)
                return

        try:
            for episode in episodes:
                try:
                    episode = self._transform_pipeline.transform_episode(
                        episode
                    )
                    if episode is None:
                        continue
                except Exception as e:
                    if fail_fast:
                        raise
                    warnings.warn(
                        f"Failed to transform episode {episode}. "
                        f"Skipping this episode. Error: "
                    )
                    import traceback

                    traceback.print_exception(e)
                    continue

                try:
                    episode_meta = episode.generate_episode_meta()
                    if episode_meta is None:
                        continue
                except Exception as e:
                    if fail_fast:
                        raise
                    warnings.warn(
                        f"Failed to generate episode metadata for {episode}. "
                        f"Skipping this episode.  Error: "
                    )
                    import traceback

                    traceback.print_exception(e)
                    continue

                with Session(engine) as session:
                    episode_meta_orm = episode_meta.get_transient_orm(
                        self._index_state, session
                    )
                self._index_state.last_episode_frame_idx = -1
                episode_meta_orm.episode.dataset_begin_index = (
                    self._index_state.last_frame_idx + 1
                )
                # clear _instruction_cache for each episode
                self._instruction_cache.clear()
                for frame in frame_generator(episode):
                    instruction_orm = (
                        self._add_instruction(
                            frame.instruction,
                            engine=engine,
                        )
                        if frame.instruction
                        else None
                    )
                    self._extend_frame_with_index(
                        frame, episode_meta_orm, instruction_orm
                    )
                    # encode_example here.
                    # yield self._features.encode_example(frame.features)
                    yield frame.features
                    # update status
                    if instruction_orm is not None:
                        self._index_state.last_instruction_idx = max(
                            self._index_state.last_instruction_idx,
                            instruction_orm.index,
                        )
                    self._index_state.last_episode_frame_idx += 1
                    self._index_state.last_frame_idx += 1

                # Update the index state with the episode metadata
                with Session(engine, expire_on_commit=False) as session:
                    if episode_meta_orm.robot:
                        if (
                            episode_meta_orm.robot.index
                            > self._index_state.last_robot_idx
                        ):
                            session.add(episode_meta_orm.robot)
                        self._index_state.last_robot_idx = max(
                            self._index_state.last_robot_idx,
                            episode_meta_orm.robot.index,
                        )
                    if episode_meta_orm.task:
                        if (
                            episode_meta_orm.task.index
                            > self._index_state.last_task_idx
                        ):
                            session.add(episode_meta_orm.task)
                        self._index_state.last_task_idx = max(
                            self._index_state.last_task_idx,
                            episode_meta_orm.task.index,
                        )
                    self._index_state.last_episode_idx = max(
                        self._index_state.last_episode_idx,
                        episode_meta_orm.episode.index,
                    )
                    # Update the episode metadata in the database
                    episode_meta_orm.episode.frame_num = (
                        self._index_state.last_episode_frame_idx + 1
                    )
                    session.add(episode_meta_orm.episode)
                    session.commit()
        finally:
            engine.dispose()

    def _complete_arrow_cache_as_dataset(
        self,
        dataset_path: str,
        builder: hg_datasets.DatasetBuilder,
        split: hg_datasets.Split | None,
    ):
        split_infos = builder.info.splits
        total_examples = (
            None
            if split_infos is None
            else sum(item.num_examples for item in split_infos.values())
        )
        if total_examples == 0:
            split_name = str(split or hg_datasets.Split.TRAIN)
            empty_dataset = hg_datasets.Dataset.from_dict(
                {name: [] for name in self._features},
                features=self._features,
                split=split_name,
            )
            dataset_dict = hg_datasets.DatasetDict({split_name: empty_dataset})
        else:
            dataset_dict = builder.as_dataset(split=split)
        assert isinstance(dataset_dict, hg_datasets.DatasetDict)
        assert len(dataset_dict) == 1
        for k, v in dataset_dict.items():
            dataset: hg_datasets.Dataset = v
            split_name = str(k)
            break
        ori_arrow_prefix = f"{builder.name}-{split_name}"
        fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(dataset_path)[0]
        arrow_files: list[str] = fs.glob(
            os.path.join(dataset_path, f"{builder.name}-{split_name}*"),
            maxdepth=1,
        )  # type: ignore

        # rename file names to match the expected format
        # e.g. data-00000-of-00001.arrow
        arrow_files = [
            os.path.relpath(f, dataset_path)
            for f in sorted(
                _rename_arrow_files(
                    arrow_files, fs=fs, source_prefix=ori_arrow_prefix
                )
            )
        ]

        # add the dataset state file as it is required by datasets
        state = {
            key: dataset.__dict__[key]
            for key in [
                "_fingerprint",
                "_format_columns",
                "_format_kwargs",
                "_format_type",
                "_output_all_columns",
            ]
        }

        state["_split"] = (
            str(dataset.split) if dataset.split is not None else dataset.split
        )
        state["_data_files"] = [{"filename": f} for f in arrow_files]
        for k in state["_format_kwargs"].keys():
            try:
                json.dumps(state["_format_kwargs"][k])
            except TypeError as e:
                raise TypeError(
                    str(e) + f"\nThe format kwargs must be JSON serializable, "
                    f"but key '{k}' isn't."
                ) from None
        from datasets import config as hg_datasets_config

        # append robo_orchard_state
        state["robo_orchard_state"] = {}
        state["robo_orchard_state"]["dataset_format_version"] = (
            dataset_format_version
        )
        with fs.open(
            os.path.join(
                dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
            ),
            "w",
            encoding="utf-8",
        ) as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)

    def packaging(
        self,
        episodes: Iterable[EpisodePackaging],
        dataset_path: str | os.PathLike[str],
        dataset_info: hg_datasets.DatasetInfo | None = None,
        writer_batch_size: int = 1024,
        max_shard_size: str | int = "8GB",
        split: hg_datasets.Split | None = None,
        force_overwrite: bool = False,
        fail_fast: bool = False,
    ):
        """Package the dataset and save it to the specified path.

        Args:
            episodes (Iterable[EpisodePackaging]): An iterable of episode
                packaging instances.
            dataset_path (str | os.PathLike[str]): Local filesystem path to
                save the packaged dataset. URI paths such as `s3://...` are
                not supported.
            dataset_info (hg_datasets.DatasetInfo | None): Information about
                the dataset, such as description, citation, and homepage.
                If None, use the default dataset info.
            writer_batch_size (int): The batch size for writing the arrow file.
                This may affect the performance of packaging or reading the
                dataset later. Default is 1024.
            max_shard_size (str | int): The maximum size of each shard. This
                can be a string like '10GB' or an integer representing the
                size in bytes. Default is "8GB".
            split (hg_datasets.Split | None): The split of the dataset.
                If None, use "train" as the default split.
            force_overwrite (bool): If True, remove the existing dataset and
                this target's reserved stale workspace before building. If
                False, raise an error if either exists. Direct packaging does
                not restore a removed target after a later failure. Default
                is False.
            fail_fast (bool): If True, immediately raise the first episode
                metadata or frame generation error instead of converting it to
                a warning and skipping that episode. Default is False.
        """

        packaging_paths = DatasetPackagingPaths.resolve(dataset_path)
        # The final target can be rejected before waiting, but the workspace
        # is owned only after this writer acquires the shared target lock.
        packaging_paths.validate_preconditions(
            force_overwrite=force_overwrite,
            include_workspace=False,
        )
        dataset_path = packaging_paths.dataset_dir
        os.makedirs(
            os.path.dirname(packaging_paths.coordination_lock_path),
            exist_ok=True,
        )
        with _create_coordination_lock(packaging_paths.coordination_lock_path):
            packaging_paths.validate_locked_target_identity()
            # The first check is a cheap preflight. This second check is the
            # live authorization after serializing writers for this target.
            packaging_paths.validate_preconditions(
                force_overwrite=force_overwrite
            )
            target_existed = os.path.lexists(dataset_path)
            if target_existed:
                warnings.warn(
                    f"The dataset path '{dataset_path}' already exists. "
                    "It will be overwritten."
                )

            packaging_succeeded = False
            try:
                if force_overwrite:
                    for stale_path in packaging_paths.output_roots:
                        if os.path.lexists(stale_path):
                            remove_path(stale_path)

                self._index_state = DatasetIndexState()
                self._instruction_cache.clear()

                # HF derives its incomplete and lock paths from output_dir.
                # Staging that output inside this workspace keeps all of those
                # transient artifacts out of the user-provided output tree.
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                os.makedirs(packaging_paths.workspace_dir)
                db_path = os.path.join(
                    packaging_paths.workspace_dir,
                    f"meta_db.{self._database_driver}",
                )

                def generator():
                    yield from self._make_packaging_generator(
                        episodes,
                        db_path=db_path,
                        fail_fast=fail_fast,
                    )

                from datasets.packaged_modules.generator.generator import (
                    Generator,
                )

                builder = Generator(
                    generator=generator,
                    features=self._features,
                    writer_batch_size=writer_batch_size,
                    info=dataset_info,
                    cache_dir=packaging_paths.hf_cache_dir,
                    # The cache is isolated inside this run's workspace. A
                    # fixed id only prevents HF from fingerprinting episodes.
                    config_id="robo_orchard",
                )
                builder.download_and_prepare(
                    output_dir=packaging_paths.builder_output_dir,
                    max_shard_size=max_shard_size,
                    file_format="arrow",
                )
                db_new_path = os.path.join(
                    packaging_paths.builder_output_dir,
                    f"meta_db.{self._database_driver}",
                )
                os.rename(db_path, db_new_path)

                self._complete_arrow_cache_as_dataset(
                    dataset_path=packaging_paths.builder_output_dir,
                    builder=builder,
                    split=split,
                )
                if os.path.lexists(dataset_path):
                    raise FileExistsError(
                        "The dataset target appeared while packaging was in "
                        f"progress: {dataset_path!r}. Refusing to replace an "
                        "artifact not owned by this writer."
                    )
                rename_noreplace(
                    packaging_paths.builder_output_dir,
                    dataset_path,
                )
                packaging_succeeded = True
            except Exception as e:
                if fail_fast and isinstance(e, DatasetGenerationError):
                    cause = e.__cause__
                    if cause is not None:
                        raise cause
                raise
            finally:
                self._cleanup_workspace(
                    packaging_paths.workspace_dir,
                    suppress_errors=not packaging_succeeded,
                )

    def _cleanup_workspace(
        self,
        path: str,
        *,
        suppress_errors: bool,
    ) -> None:
        """Remove an owned path without masking an active packaging error."""

        try:
            remove_path(path)
        except OSError as exc:
            if suppress_errors:
                logger.warning(
                    "Failed to clean packaging path %r: %s",
                    path,
                    exc,
                    exc_info=True,
                )
            else:
                raise


class InstructionCache:
    def __init__(self):
        self._cache: dict[bytes, Instruction] = {}

    def _get_key(self, instruction: InstructionData) -> bytes:
        """Generate a unique key for the instruction data."""
        return pickle.dumps(instruction)

    def get(self, instruction_data: InstructionData) -> Instruction | None:
        """Get an instruction from the cache using its packing data."""
        return self._cache.get(self._get_key(instruction_data), None)

    def add(self, instruction: Instruction):
        """Add an instruction to the cache."""
        self._cache[
            self._get_key(
                InstructionData(
                    name=instruction.name,
                    json_content=instruction.json_content,
                )
            )
        ] = instruction

    def clear(self):
        self._cache.clear()


@dataclass
class DatasetIndexState:
    """State for dataset index information."""

    last_episode_idx: int = -1
    """The index of the last episode in the dataset."""
    last_robot_idx: int = -1
    """The index of the last robot in the dataset."""
    last_task_idx: int = -1
    """The index of the last task in the dataset."""
    last_instruction_idx: int = -1
    """The index of the last instruction in the dataset."""
    last_frame_idx: int = -1
    """The index of the last frame in the dataset."""
    last_episode_frame_idx: int = -1
    """The index of the last frame in the last episode in the dataset."""


def _patch_dataset_state_file(dataset_path: str):
    """Patch the dataset state file to include the robo_orchard_state."""
    from datasets import config as hg_datasets_config

    fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(dataset_path)[0]

    state_file_path = os.path.join(
        dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
    )
    if not fs.exists(state_file_path):
        raise FileNotFoundError(
            f"The dataset state file '{state_file_path}' does not exist."
        )
    # read existing state file
    with fs.open(state_file_path, "r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    # append robo_orchard_state
    state["robo_orchard_state"] = {}
    state["robo_orchard_state"]["dataset_format_version"] = (
        dataset_format_version
    )

    with fs.open(state_file_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)


def _rename_arrow_files(
    files: list[str], fs: fsspec.AbstractFileSystem, source_prefix: str
) -> list[str]:
    """Rename arrow files to match the expected format.

    In some cases, the arrow files generated by datasets may exceed the
    100000 shards limit, which will cause the file names to be inconsistent
    with the expected format. This function renames the arrow files to
    match the expected format.
    """
    if len(files) == 0:
        return []
    num_shards = len(files)
    digits = max(5, len(str(num_shards)))
    ret = []
    for f in files:
        dir_name = os.path.dirname(f)
        base_name = os.path.basename(f)
        if len(base_name) == len(source_prefix) + 6:
            assert base_name == f"{source_prefix}.arrow"
            shard_idx = 0
        else:
            split_info = base_name[len(source_prefix) + 1 : -6]  # noqa: E203
            split_info = split_info.split("-of-")
            assert len(split_info) == 2
            shard_idx = int(split_info[0])
            num_shards_in_file = int(split_info[1])
            assert num_shards_in_file == num_shards
        new_base_name = (
            f"data-{shard_idx:0{digits}d}-of-{num_shards:0{digits}d}.arrow"  # noqa: E501
        )
        new_f = os.path.join(dir_name, new_base_name)
        fs.move(f, new_f)
        ret.append(new_f)
    return ret
