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
import bisect
import inspect
import json
import os
import shutil
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
    TypeVar,
    overload,
)

import fsspec
import torch
from datasets import (
    Dataset as HFDataset,
    DatasetInfo,
    Features,
)
from datasets.arrow_dataset import Column, SplitDict
from pydantic import Field, model_validator
from robo_orchard_core.utils.config import ClassType, Config
from sqlalchemy import URL, Engine, Select, select
from sqlalchemy.orm import Session, make_transient
from sqlalchemy.sql import func
from typing_extensions import Self

# import all datatypes and features
from robo_orchard_lab.dataset.datatypes import *  # noqa: F403,F401
from robo_orchard_lab.dataset.robot.columns import PreservedIndexColumnsKeys
from robo_orchard_lab.dataset.robot.dataset_db_engine import (
    create_engine,
    get_local_db_md5,
    get_local_db_url,
    try_upgrade_database,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DatasetItem,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.row_sampler import (
    CachedIndexDataset,
    MultiRowSampler,
    MultiRowSamplerConfig,
)

__all__ = [
    "RODataset",
    "RODatasetImageDecodeOptions",
    "RODatasetInfo",
    "ROMultiRowDataset",
    "ConcatRODataset",
    "RODatasetItem",
    "_complete_dataset_info",
    "get_row_num_from_dataset_info",
]

MetaType = TypeVar("MetaType", Episode, Instruction, Robot, Task)
"""A type variable for metadata types in the RoboOrchard dataset."""
MetaIndexKeyType = Literal[
    "episode_index", "task_index", "robot_index", "instruction_index"
]
TorchDataset: TypeAlias = torch.utils.data.Dataset

DatasetType = TypeVar("DatasetType", bound=TorchDataset)


@dataclass
class RODatasetInfo:
    num_rows: int
    columns_without_transform: list[str]
    dataset_transform: Callable[[dict], dict]


class RODatasetImageDecodeOptions(Config):
    """Configure opt-in ImageEncoded feature materialization.

    The reader applies these options only to top-level columns whose stored
    feature is :class:`BatchCameraDataEncodedFeature`. ``columns=None``
    selects every such column. Supplying an explicit column list narrows the
    selection and validates it against the stored feature schema.

    This config does not enable video decoding or change user transforms.
    Omit it from a reader to preserve encoded camera values.

    Args:
        backend (Literal["pil", "cv2"]): Image decode backend. Default is
            ``"pil"``.
        columns (tuple[str, ...] | None): Explicit ImageEncoded columns, or
            None to discover all ImageEncoded columns. Default is None.
        invert_rgb (bool): Whether newly decoded RGB/BGR metadata should be
            inverted. Already decoded values are never modified. Default is
            False.
    """

    backend: Literal["pil", "cv2"] = "pil"
    columns: tuple[str, ...] | None = None
    invert_rgb: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.columns is not None and not self.columns:
            raise ValueError(
                "RODatasetImageDecodeOptions.columns must contain at least "
                "one column when specified."
            )
        if self.columns is not None and len(set(self.columns)) != len(
            self.columns
        ):
            raise ValueError(
                "RODatasetImageDecodeOptions.columns contains duplicate "
                "column names."
            )


class RODataset(TorchDataset):
    """The RoboOrchard dataset for robot data.

    We use a tabular dataset to store the frame-level information, and a
    separate database to store the episode-level information. The huggingface
    datasets (pyarrow_dataset) is used as table format, and SQLAlchemy with
    DuckDB are used to manage the database.

    A dataset owns its metadata database resource. Call :meth:`close` when it
    is no longer needed, or use a ``with`` block for deterministic cleanup.
    Views created in the same process by methods such as :meth:`select` share
    that resource and therefore share one closed state. ``close()`` is
    idempotent and closing any related view invalidates metadata access through
    all of them. Serialization recreates one process-local resource for each
    original shared reader lifecycle.


    Note:
        The dataset loading process will automatically check the version of the
        meta database, and upgrade it to the latest version if necessary.
        Please call the `RODataset.upgrade_meta(dataset_path)` method to
        upgrade the database permanently!

    Args:
        dataset_path (str): The path to the dataset directory.
            It should contain a `dataset.arrow` file and a `meta_db.*` file.
        storage_options (dict | None, optional): Additional Key/value pairs to
            be passed on to the file-system backend, if any. This is passed
            to the `datasets.Dataset.load_from_disk` method. Defaults to None.
        meta_index2meta (bool, optional): Whether to convert the index-based
            metadata to actual metadata objects when accessing the dataset.
            If True, the `episode`, `task`, `robot`, and `instruction` fields
            will be added and the corresponding index fields will be removed.
            Defaults to False.
        image_decode_options (RODatasetImageDecodeOptions | Mapping, optional):
            Opt-in materialization for stored
            ``BatchCameraDataEncodedFeature`` columns. ``None`` preserves
            encoded camera values. Defaults to None.

    """

    frame_dataset: HFDataset
    """The Hugging Face Dataset object containing the frame data.

    The dataset should not be shuffled or modified after loading, as the
    metadata database relies on the index columns to retrieve the episode-level
    information.

    """

    index_dataset: HFDataset
    """The same as `frame_dataset`, but only contains the preserved index columns.

    The dataset should not be shuffled or modified after loading, as the
    metadata database relies on the index columns to retrieve the episode-level
    information.
    """  # noqa: E501

    meta_index2meta: bool
    """Whether to convert the index-based metadata to actual metadata
    objects when accessing the dataset."""

    def __init__(
        self,
        dataset_path: str,
        storage_options: dict | None = None,
        meta_index2meta: bool = False,
        *,
        image_decode_options: (
            RODatasetImageDecodeOptions | Mapping[str, Any] | None
        ) = None,
    ):
        dataset_path = os.path.expanduser(dataset_path)

        from robo_orchard_lab.dataset.robot._hf_dataset import (
            load_from_disk,
        )

        try:
            self.frame_dataset = load_from_disk(
                dataset_path, storage_options=storage_options
            )
        except (TypeError, ValueError) as e:
            warnings.warn(
                "Failed to load dataset using wrapped version. "
                "Falling back to use `datasets.load_from_disk`. "
                f"Original error: {e}",
                stacklevel=2,
            )
            self.frame_dataset = HFDataset.load_from_disk(
                dataset_path, storage_options=storage_options
            )

        self.index_dataset = self.frame_dataset.select_columns(
            column_names=list(PreservedIndexColumnsKeys)
        )
        self.meta_index2meta = meta_index2meta
        self._set_image_decode_options(image_decode_options)
        # recover state dict
        from datasets import config as hg_datasets_config

        state_file = os.path.join(
            dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
        )
        with open(state_file, "r") as f:
            state: dict = json.load(f)
        self._dataset_format_version = state.get("robo_orchard_state", {}).get(
            "dataset_format_version", None
        )
        # load db
        self.db_engine = self._load_db(dataset_path)

        self._transform: Callable[[dict], dict] | None = None

    @staticmethod
    def from_dataset(
        frame_dataset: HFDataset,
        meta_db_engine: Engine,
        meta_index2meta: bool = False,
        *,
        image_decode_options: (
            RODatasetImageDecodeOptions | Mapping[str, Any] | None
        ) = None,
    ) -> RODataset:
        """Create a RODataset from a frame dataset and meta db engine.

        The frame dataset must not contain an indices mapping, and explicit
        image decode columns are validated against its stored features before
        this method takes ownership of ``meta_db_engine``. The caller remains
        responsible for providing the preserved index columns and a valid
        metadata database. A returned dataset owns ``meta_db_engine`` and
        disposes it when the dataset or any related view is closed.
        """

        if frame_dataset._indices is not None:
            raise ValueError(
                "frame_dataset should not have any indices. "
                "Please reset the indices before creating RODataset."
            )

        ret = RODataset.__new__(RODataset)
        ret.frame_dataset = frame_dataset
        ret.index_dataset = frame_dataset.select_columns(
            column_names=list(PreservedIndexColumnsKeys)
        )
        ret.meta_index2meta = meta_index2meta
        ret._dataset_format_version = None
        ret._transform = None
        # Validate reader policy before taking ownership of the caller's
        # engine.
        ret._set_image_decode_options(image_decode_options)
        ret._db_resource = _RODatasetDBResource(meta_db_engine)
        return ret

    def _get_info_dict(self) -> RODatasetInfo:
        dict_info = {}
        dict_info["num_rows"] = len(self)
        dict_info["columns_without_transform"] = list(
            self.frame_dataset.column_names
        )
        if self.meta_index2meta:
            # change all columns with index to meta
            for col in (
                "episode",
                "task",
                "robot",
                "instruction",
            ):
                col_index = f"{col}_index"
                dict_info["columns_without_transform"].remove(col_index)
                dict_info["columns_without_transform"].append(col)
        dict_info["dataset_transform"] = self._transform

        return RODatasetInfo(**dict_info)

    def __repr__(self) -> str:
        dict_info = self._get_info_dict()
        return f"RODataset({dict_info})"

    def _get_state_(self) -> dict:
        """Get all internal state of the dataset.

        This method is used to share the internal state of the dataset
        within single process, not for pickling!
        """
        return self.__dict__.copy()

    def __getstate__(self) -> dict:
        """Get the state of the dataset for pickling."""
        state = self._get_state_()
        state.pop("_image_decoder", None)
        state.pop("_image_decode_columns", None)
        if self._db_resource.closed:
            raise RuntimeError("Cannot pickle a closed RODataset.")
        return state

    def __setstate__(self, state: dict):
        """Set the state of the dataset from a pickled state."""
        state = state.copy()
        if state.pop("_closed", False):
            raise ValueError("Cannot restore a closed RODataset state.")

        image_decode_options = state.pop("_image_decode_options", None)
        state.pop("_image_decoder", None)
        state.pop("_image_decode_columns", None)

        db_resource = state.pop("_db_resource", None)
        if db_resource is None:
            db_engine_url = state.pop("db_engine_url", None)
            engine = state.pop("db_engine", None)
            if engine is None:
                engine = state.pop("_db_engine", None)
            if db_engine_url is not None:
                engine = create_engine(db_engine_url, readonly=True)
            if engine is None:
                raise KeyError(
                    "db_engine_url not found in state. "
                    "Cannot restore db_engine."
                )
            db_resource = _RODatasetDBResource(engine)
        elif db_resource.closed:
            raise ValueError("Cannot restore a closed RODataset resource.")

        state["_db_resource"] = db_resource
        self.__dict__.update(state)
        self._set_image_decode_options(image_decode_options)

    @property
    def image_decode_options(self) -> RODatasetImageDecodeOptions | None:
        """Return the configured ImageEncoded materialization policy."""

        return self._image_decode_options

    def _set_image_decode_options(
        self,
        options: RODatasetImageDecodeOptions | Mapping[str, Any] | None,
    ) -> None:
        """Validate image decode intent and build its stateless decoder."""

        if options is None:
            normalized_options = None
        elif isinstance(options, RODatasetImageDecodeOptions):
            normalized_options = options
        elif isinstance(options, Mapping):
            normalized_options = RODatasetImageDecodeOptions.model_validate(
                dict(options)
            )
        else:
            raise TypeError(
                "image_decode_options must be "
                "RODatasetImageDecodeOptions, a mapping, or None; got "
                f"{type(options).__name__}."
            )

        self._image_decode_options = normalized_options
        self._image_decode_columns = ()
        self._image_decoder = None
        if normalized_options is None:
            return

        available_features = self.frame_dataset.features
        if normalized_options.columns is None:
            columns = tuple(
                column
                for column, feature in available_features.items()
                if isinstance(feature, BatchCameraDataEncodedFeature)  # noqa: F405
            )
        else:
            missing_columns = [
                column
                for column in normalized_options.columns
                if column not in available_features
            ]
            if missing_columns:
                raise KeyError(
                    "Image decode columns are missing from the dataset: "
                    + ", ".join(missing_columns)
                )
            columns = normalized_options.columns
            invalid_columns = [
                column
                for column in columns
                if not isinstance(
                    available_features[column],
                    BatchCameraDataEncodedFeature,  # noqa: F405
                )
            ]
            if invalid_columns:
                invalid_details = ", ".join(
                    f"{column}={type(available_features[column]).__name__}"
                    for column in invalid_columns
                )
                raise ValueError(
                    "Image decode columns must use "
                    "BatchCameraDataEncodedFeature; got "
                    f"{invalid_details}."
                )

        self._image_decode_columns = columns
        if not columns:
            return

        from robo_orchard_lab.transforms.image.decode import (
            ImageDecodeConfig,
        )

        self._image_decoder = ImageDecodeConfig(
            input_columns=columns,
            backend=normalized_options.backend,
            invert_rgb=normalized_options.invert_rgb,
        )()

    def _materialize_storage_features(
        self,
        raw_row: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Materialize configured storage features before later stages."""

        row = dict(raw_row)
        if self._image_decoder is None:
            return row
        encoded_values = {
            column: row[column]
            for column in self._image_decode_columns
            if column in row
        }
        if encoded_values:
            row.update(self._image_decoder.transform(**encoded_values))
        return row

    def _materialize_storage_feature_rows(
        self,
        raw_rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Materialize one ordered batch before later read stages.

        Subclasses may override this seam to combine storage work across rows
        while preserving input order and cardinality.

        Args:
            raw_rows (Sequence[Mapping[str, Any]]): Raw frame-table rows.

        Returns:
            list[dict[str, Any]]: Materialized rows in input order.
        """

        return [self._materialize_storage_features(row) for row in raw_rows]

    def _requires_storage_materialization(self) -> bool:
        """Return whether row reads must apply the storage-feature hook."""

        return self._image_decoder is not None

    @property
    def db_engine(self) -> Engine:
        """Return the active metadata database engine.

        A closed reader cannot reopen its metadata connection pool. Construct
        a new reader instead of accessing metadata after :meth:`close`.
        """

        return self._db_resource.active_engine

    @db_engine.setter
    def db_engine(self, engine: Engine) -> None:
        """Initialize the metadata engine before a reader owns a resource.

        Raises:
            RuntimeError: If this reader already belongs to a metadata
                resource lifecycle.
        """

        db_resource = getattr(self, "_db_resource", None)
        if db_resource is not None:
            raise RuntimeError(
                "Cannot replace the metadata engine after RODataset resource "
                "initialization."
            )
        self._db_resource = _RODatasetDBResource(engine)

    def close(self) -> None:
        """Release the metadata database engine owned by this dataset.

        The operation is idempotent. Closing a dataset invalidates the
        database resource used for metadata lookups; create a new reader when
        further database access is required.
        """

        self._db_resource.close()

    def __enter__(self) -> Self:
        """Enter a dataset resource context and return this reader."""

        _ = self.db_engine
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the metadata database resource at context exit."""

        self.close()

    def _load_db(self, dataset_path: str) -> Engine:
        return create_engine(
            url=_get_dataset_db_url(dataset_path), readonly=True
        )

    @staticmethod
    def upgrade_meta(
        dataset_path: str,
    ):
        """Upgrade the meta database to the latest version.

        Note:
            This method is not thread-safe. It is the caller's responsibility
            to ensure that no other process is accessing the database
            during the upgrade process.

        Args:
            dataset_path (str): The path to the dataset directory.
                It should contain a `dataset.arrow` file and a `meta_db.*`
                file.
        """
        db_url = _get_dataset_db_url(dataset_path)
        _, file_content_md5 = get_local_db_md5(db_url)
        new_url = try_upgrade_database(db_url)
        # overwrite the old db file with the new db file
        new_db_path = new_url.database
        assert new_db_path is not None
        assert os.path.exists(new_db_path)
        # move old db file to f"{old_path}.{md5}"
        old_db_path = db_url.database
        assert old_db_path is not None
        backup_db_path = f"{old_db_path}.{file_content_md5}"
        shutil.move(old_db_path, backup_db_path)
        shutil.move(new_db_path, old_db_path)

    @property
    def features(self) -> Features:
        """Get the features of the dataset.

        Features are the schema of the original dataset without transforms.
        """
        return self.frame_dataset.features

    @property
    def transform(self) -> Callable[[dict], dict] | None:
        return self._transform

    @property
    def episode_num(self) -> int:
        """Get the number of episodes in the dataset."""
        with Session(self.db_engine) as session:
            result = session.execute(select(func.count(Episode.index)))
            episode_num = result.scalar_one()
        return episode_num

    @property
    def task_num(self) -> int:
        """Get the number of tasks in the dataset."""
        with Session(self.db_engine) as session:
            result = session.execute(select(func.count(Task.index)))
            task_num = result.scalar_one()
        return task_num

    @property
    def robot_num(self) -> int:
        """Get the number of robots in the dataset."""
        with Session(self.db_engine) as session:
            result = session.execute(select(func.count(Robot.index)))
            robot_num = result.scalar_one()
        return robot_num

    def set_transform(self, transform: Callable[[dict], dict] | None):
        self._transform = transform

    @contextmanager
    def transform_context(self, transform: Callable[[dict], dict] | None):
        """Context manager to temporarily set a transform for the dataset.

        This is useful for applying a transform only within a specific
        context, without permanently changing the dataset's transform.
        """

        old_transform = self._transform
        self.set_transform(transform)
        try:
            yield self
        finally:
            self.set_transform(old_transform)

    def rename_columns(
        self,
        column_mapping: dict[str, str],
        new_fingerprint: str | None = None,
    ) -> Self:
        """Rename several columns in the dataset.

        Args:
            column_mapping (dict[str, str]): A dictionary mapping the old
                column name to the new column name. The dictionary should
                contain exactly one key-value pair.
            new_column_name (str): The new name of the column.

        Returns:
            Self: A new instance of type(self) with the renamed column.
        """

        # check that the old and new columns does not contain index columns
        reserved_keys = set(PreservedIndexColumnsKeys)
        for k, v in column_mapping.items():
            if k in reserved_keys:
                raise ValueError(
                    f"Cannot rename index column {k}. "
                    f"Index columns are: {PreservedIndexColumnsKeys}"
                )
            if v in reserved_keys:
                raise ValueError(
                    f"Cannot rename to index column {v}. "
                    f"Index columns are: {PreservedIndexColumnsKeys}"
                )

        state_dict = self._get_state_()
        state_dict["frame_dataset"] = self.frame_dataset.rename_columns(
            column_mapping=column_mapping,
            new_fingerprint=new_fingerprint,
        )
        ret = type(self).__new__(type(self))
        ret.__dict__.update(state_dict)
        if (
            ret._image_decode_options is not None
            and ret._image_decode_options.columns is not None
        ):
            ret._image_decode_options = ret._image_decode_options.model_copy(
                update={
                    "columns": tuple(
                        column_mapping.get(column, column)
                        for column in ret._image_decode_options.columns
                    )
                }
            )
        ret._set_image_decode_options(ret._image_decode_options)
        return ret

    def select_columns(
        self,
        column_names: str | list[str],
        new_fingerprint: str | None = None,
        include_index: bool = True,
    ) -> Self:
        """Select one or more columns from the dataset.

        Args:
            column_names (str | list[str]): The name(s) of the column(s) to
                select. If a single column name is provided, it can be a
                string.
            new_fingerprint (str | None, optional): The new fingerprint of
                the frame dataset after transform. If `None`, the new
                fingerprint is computed using a hash of the previous
                fingerprint, and the transform arguments. This argument is
                used in the `select_columns` method of the Hugging Face
                Dataset to ensure that the dataset is properly cached and
                can be loaded efficiently
                in the future. Defaults to None.
            include_index (bool, optional): Whether to include the index
                columns in the selected columns. If True, the index columns
                will be included in the selected columns. Defaults to True.

        Returns:
            Self: A new instance of type(self) with the selected columns.

        """
        if not isinstance(column_names, (list, tuple)):
            column_names = [column_names]
        if isinstance(column_names, tuple):
            column_names = list(column_names)
        if include_index:
            column_names = list(
                set(column_names) | set(PreservedIndexColumnsKeys)
            )

        state_dict = self._get_state_()
        state_dict["frame_dataset"] = self.frame_dataset.select_columns(
            column_names=column_names, new_fingerprint=new_fingerprint
        )

        ret = type(self).__new__(type(self))
        ret.__dict__.update(state_dict)
        if (
            ret._image_decode_options is not None
            and ret._image_decode_options.columns is not None
        ):
            selected_columns = tuple(
                column
                for column in ret._image_decode_options.columns
                if column in ret.frame_dataset.features
            )
            ret._image_decode_options = (
                ret._image_decode_options.model_copy(
                    update={"columns": selected_columns}
                )
                if selected_columns
                else None
            )
        ret._set_image_decode_options(ret._image_decode_options)
        return ret

    def save_to_disk(
        self,
        dataset_path: str,
        max_shard_size: str | int = "2000MB",
        num_shards: int | None = None,
        num_proc: int | None = None,
        storage_options: dict | None = None,
        batch_size: int | None = None,
    ):
        """Saves a dataset to filesystem.

        Args:
            dataset_path (str): The path to the dataset directory where
                the dataset will be saved.
            max_shard_size (str | int , optional): The maximum size of
                each shard. Defaults to "2000MB". This can be a string
                (e.g., "2000MB") or an integer (e.g., 2000 * 1024 * 1024
                for 2000MB).
            num_shards (int | None, optional): The number of shards to create.
                Number of shards to write. By default the number of shards
                depends on `max_shard_size` and `num_proc`.
            num_proc (int | None, optional): The number of processes to use
                for saving the dataset. Defaults to None.
            storage_options (dict | None, optional): Additional Key/value pairs
                to be passed on to the file-system backend, if any. Defaults
                to None.
            batch_size (int | None, optional): The batch size to use when
                saving the dataset. If None, the default batch size from
                Hugging Face Datasets will be used. Defaults to None.
        """
        from datasets import config as hg_datasets_config

        old_batch_size = hg_datasets_config.DEFAULT_MAX_BATCH_SIZE
        if batch_size is not None:
            hg_datasets_config.DEFAULT_MAX_BATCH_SIZE = batch_size

        self.frame_dataset.save_to_disk(
            dataset_path=dataset_path,
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
            storage_options=storage_options,
        )

        # save dataset info if needed
        need_dataset_info_save, info = _complete_dataset_info(
            dataset_path, arrow_dataset=self.frame_dataset
        )
        if need_dataset_info_save:
            info_path = os.path.join(
                dataset_path, hg_datasets_config.DATASET_INFO_FILENAME
            )  # noqa: E501
            info = asdict(info)
            with open(info_path, "w") as f:
                sorted_keys_dataset_info = {k: info[k] for k in sorted(info)}
                json.dump(sorted_keys_dataset_info, f, indent=2)

        hg_datasets_config.DEFAULT_MAX_BATCH_SIZE = old_batch_size
        src_meta_db_path = self.db_engine.url.database
        assert src_meta_db_path is not None
        fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(dataset_path)[0]
        # copy the meta db file to the new dataset path
        dst_meta_db_path = os.path.join(
            dataset_path, os.path.basename(src_meta_db_path)
        )
        if fs.exists(dst_meta_db_path):
            fs.rm(dst_meta_db_path)
        fs.copy(src_meta_db_path, dst_meta_db_path)

    def select(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: str | None = None,
        writer_batch_size: int = 1000,
        new_fingerprint: str | None = None,
    ) -> Self:
        """Select a subset of the dataset based on indices.

        This method is similar to the `select` method in Hugging Face
        Datasets.

        Warning:
            The returned new RODataset will only contain the selected frames,
            and the episode-level metadata does not change. Therefore, the
            episode-level metadata may not correspond to the selected frames!


        Args:
            indices (Iterable): The indices of the frames to select.
                This can be a list of integers or a slice object.
            keep_in_memory (bool, optional): Whether to keep the indices
                mapping in memory instead of writing to a cache file. If
                indices is too large, it is recommended to set this
                to False to avoid memory issues. Defaults to False.
            indices_cache_file_name (str | None, optional): The name of the
                cache file to store the indices mapping. If `None`, the
                indices mapping will not be cached. This argument should
                be set to a valid file path if `keep_in_memory` is False.
                Defaults to None.
            writer_batch_size (int , optional): The batch size to use
                when writing the indices mapping to the cache file. Higher
                batch size can improve performance, but may also increase
                memory usage. Defaults to 1000.
            new_fingerprint (str | None, optional): The new fingerprint of
                the frame dataset after transform. If `None`, the new
                fingerprint is computed using a hash of the previous
                fingerprint, and the transform arguments.
        """
        state_dict = self._get_state_()
        for k, v in state_dict.items():
            if isinstance(v, HFDataset):
                state_dict[k] = v.select(
                    indices=indices,
                    keep_in_memory=keep_in_memory,
                    indices_cache_file_name=indices_cache_file_name,
                    writer_batch_size=writer_batch_size,
                    new_fingerprint=new_fingerprint,
                )
        ret = type(self).__new__(type(self))
        ret.__dict__.update(state_dict)
        return ret

    def _meta_index2meta(self, src: dict[str, Any]) -> dict:
        """Convert the index-based metadata in `src` to actual metadata objects."""  # noqa: E501
        dst = src.copy()
        dict_to_update = {}
        if "episode_index" in dst:
            dict_to_update["episode"] = self.get_meta(
                Episode, dst.pop("episode_index", None)
            )
        if "task_index" in dst:
            dict_to_update["task"] = self.get_meta(
                Task, dst.pop("task_index", None)
            )
        if "robot_index" in dst:
            dict_to_update["robot"] = self.get_meta(
                Robot, dst.pop("robot_index", None)
            )
        if "instruction_index" in dst:
            dict_to_update["instruction"] = self.get_meta(
                Instruction, dst.pop("instruction_index", None)
            )
        dst.update(dict_to_update)
        return dst

    @overload
    def __getitem__(self, index: int | slice | list[int]) -> dict: ...

    @overload
    def __getitem__(self, index: str) -> list[Any]: ...

    def __getitem__(self, index: int | slice | list[int] | str) -> dict | list:
        """Get the frame data at the specified index.

        The returned data will be transformed if a transform is registered
        using the `set_transform` method. If no transform is registered,
        the raw frame data will be returned.

        Args:
            index (int | slice | list[int] | str): The index of the frame
                data to retrieve. If `index` is a slice, it returns
                a dict with values of list type.
                If `index` is a string, it is treated as a column name
                and returns the data for that column. Note that string
                index will load the entire column data, which may
                consume a lot of memory if the column is large.

        Returns:
            dict | list: The frame data at the specified index.
                if `index` is a string, returns the data for that column.
                Otherwise, returns a dict with the frame data.
        """

        ret: dict | list = self.__getitem_no_transform__(index)
        if self._transform is not None:
            # apply transform if available
            if isinstance(ret, list):
                raise TypeError(
                    "Transform is not supported for list type data. "
                    "Please use list or int index type instead of "
                    "string index."
                )
            ret = self._transform(ret)
        return ret

    @overload
    def __getitem_no_transform__(
        self, index: int | slice | list[int]
    ) -> dict: ...

    @overload
    def __getitem_no_transform__(self, index: str) -> list[Any]: ...

    def __getitem_no_transform__(
        self, index: int | slice | list[int] | str
    ) -> dict | list:
        """Get the frame data at the specified index.

        Args:
            index (int | slice | list[int] | str): The index of the frame
                data to retrieve. If `index` is a slice, it returns
                a dict with values of list type.
                If `index` is a string, it is treated as a column name
                and returns the data for that column. Note that string
                index will load the entire column data, which may
                consume a lot of memory if the column is large.

        Returns:
            dict | list: The frame data at the specified index.
                if `index` is a string, returns the data for that column.
                Otherwise, returns a dict with the frame data.
        """

        ret = self._get_frame_data_without_meta_conversion(index)
        if self.meta_index2meta:
            if isinstance(ret, dict):
                ret = self.convert_meta_index2meta(data=ret)
            else:
                ret = self.convert_meta_index2meta(data=ret, column_name=index)  # type: ignore # noqa: E501
        return ret

    def _get_frame_data_without_meta_conversion(
        self,
        index: int | slice | list[int] | str,
    ) -> dict | list:
        """Read and materialize frame data while preserving raw index fields.

        This internal stage lets multi-row sampling reuse the current row's
        preserved index columns before optional metadata expansion removes
        them. User transforms are applied by later public access stages.
        """

        ret: dict | list = self.frame_dataset[index]
        if (
            not isinstance(index, str)
            and self._requires_storage_materialization()
        ):
            if isinstance(index, int):
                if not isinstance(ret, dict):
                    raise TypeError("Integer row access must return a dict.")
                return self._materialize_storage_features(ret)

            if not isinstance(ret, dict):
                raise TypeError("Batch row access must return a dict.")
            row_count = len(next(iter(ret.values()))) if len(ret) > 0 else 0
            if row_count > 0:
                raw_rows = [
                    {column: values[offset] for column, values in ret.items()}
                    for offset in range(row_count)
                ]
                rows = self._materialize_storage_feature_rows(raw_rows)
                ret = {
                    column: [row[column] for row in rows] for column in rows[0]
                }
        return ret

    def make_iter(self) -> Iterable[dict]:
        """Create an iterator over the dataset."""
        for i in range(len(self)):
            yield self[i]

    @overload
    def convert_meta_index2meta(
        self, data: dict[str, Any]
    ) -> dict[str, Any]: ...

    @overload
    def convert_meta_index2meta(
        self, data: list[Any], column_name: MetaIndexKeyType
    ) -> list[Any]: ...

    def convert_meta_index2meta(
        self,
        data: dict[str, Any] | list,
        column_name: MetaIndexKeyType | None = None,
    ) -> dict | list:
        """Convert the metadata index in `data` to actual metadata objects.

        The `data` can be either a row sample containing multiple columns
        (dict), or a slice of a single column (list).

        Args:
            data (dict | list): The data to convert. If `data` is a dict
                with index-based metadata, it will be converted to actual
                metadata objects. If `data` is a list, it is supposed to be
                a slice of a single column with index-based metadata, and
                it will be converted to actual metadata objects.
            column_name (MetaIndexKeyType | None, optional): The name of the
                column to convert. If `data` is a list, this argument must be
                provided to convert the index-based metadata to actual
                metadata objects. Defaults to None.

        """

        ret = data
        if isinstance(data, list) and column_name is None:
            raise KeyError(
                "If data is a list, column_name must be provided to convert "
                "the index-based metadata to actual metadata objects."
            )

        if isinstance(ret, dict):
            ret = self._meta_index2meta(ret)
        else:
            ret_dict = {f"{column_name}": ret}
            ret_dict = self._meta_index2meta(ret_dict)
            assert len(ret_dict) == 1, "Expected only one key in the dict"
            ret = ret_dict[next(iter(ret_dict))]
        return ret

    def __len__(self) -> int:
        """Get the number of frames in the dataset."""
        return len(self.frame_dataset)

    @overload
    def get_meta(
        self, meta_type: type[MetaType], index: int | None
    ) -> MetaType | None: ...

    @overload
    def get_meta(
        self, meta_type: type[MetaType], index: list[int | None] | Column
    ) -> list[MetaType | None]: ...

    def get_meta(
        self,
        meta_type: type[MetaType],
        index: int | None | list[int | None] | Column,
    ) -> MetaType | None | list[MetaType | None]:
        """Get metadata of a specific type.

        This method retrieves metadata from the database using index.
        Possible metadata types include `Episode`, `Instruction`, `Robot`,
        and `Task`.

        Args:
            meta_type (type[MetaType]): The type of metadata to retrieve.
            index (int | None | list[int | None]): The index of the metadata
                to retrieve. If None, returns None.

        Returns:
            MetaType | None | list[MetaType | None]: The metadata object or
                None if not found. If `index` is a list, returns a list of
                metadata objects or None for each index.

        """
        if index is None:
            return None

        if isinstance(index, (list, Column)):
            # get all not None value
            non_none_index = set([i for i in index if i is not None])
            if len(non_none_index) == 0:
                return [None for _ in index]
            # If index is a list, retrieve multiple metadata objects
            stmt = select(meta_type).where(meta_type.index.in_(non_none_index))
            with Session(self.db_engine) as session:
                ret = session.scalars(stmt).all()
                # make transient to avoid session issues
                for item in ret:
                    make_transient(item)
                # fill None for missing indices
                ret_dict: dict[int | None, Any] = {
                    item.index: item for item in ret
                }
                ret_dict[None] = None
                return [ret_dict.get(i, None) for i in index]
        else:
            with Session(self.db_engine) as session:
                ret = session.get(meta_type, index)
                if ret is not None:
                    make_transient(ret)
                return ret

    def iterate_meta_by_statement(self, stmt: Select) -> Iterable[Any]:
        """Create an iterator over the meta information in the dataset.

        Args:
            stmt (Select): The SQLAlchemy Select statement to execute.

        Yields:
            Iterable[Any]: An iterator over the metadata objects returned
                by the query.

        """
        with Session(self.db_engine) as session:
            for row in session.execute(stmt):
                yield row

    def iterate_meta(
        self,
        meta_type: type[MetaType],
        ordered: bool = True,
        transient: bool = True,
    ) -> Iterable[MetaType]:
        """Create an iterator over the meta information in the dataset.

        Args:
            meta_type (type[MetaType]): The type of metadata to iterate over.
            ordered (bool, optional): Whether to iterate in order of index.
                Defaults to True.
            transient (bool, optional): Whether to make the metadata objects
                transient to avoid session issues. Defaults to True.
        """
        with Session(self.db_engine) as session:
            stmt = select(meta_type)
            if ordered:
                stmt = stmt.order_by(meta_type.index)
            for data in session.scalars(stmt).all():
                if transient:
                    make_transient(data)
                yield data

    @property
    def dataset_format_version(self) -> str | None:
        """Get the dataset format version of loaded dataset."""
        return self._dataset_format_version

    def __getitems__(self, keys: list[int]) -> list:
        """Get a batch using list of index.

        Unlike __getitem__, this method returns a list of rows
        where each row is a dictionary of column names and their
        corresponding values for that index.

        Args:
            keys (list[int]): A list of indices to retrieve from the dataset.

        """
        batch = self.__getitem_no_transform__(keys)
        n_examples = len(batch[next(iter(batch))])
        if self._transform is not None:
            ret = [
                self._transform(
                    {col: array[i] for col, array in batch.items()}
                )
                for i in range(n_examples)
            ]
        else:
            ret = [
                {col: array[i] for col, array in batch.items()}
                for i in range(n_examples)
            ]
        return ret


class ROMultiRowDataset(RODataset):
    """A dataset that returns multiple rows for each index.

    This class extends `RODataset` to support multi-row sampling.
    It provides a method to sample multiple rows based on the index dataset.


    If column is in the row_sampler, it will sample multiple rows
    for that column based on the index dataset, and the column in
    the returned row will be a list of rows. If the column is not
    in the row_sampler, it will return a single row for that column.

    Args:
        dataset_path (str): The path to the dataset directory.
        row_sampler (MultiRowSamplerConfig): The configuration for the
            multi-row sampler. It defines how to sample multiple
            rows based on the index dataset.
        storage_options (dict | None, optional): Additional Key/value pairs
            to be passed on to the file-system backend, if any.
            Defaults to None.
        meta_index2meta (bool, optional): Whether to convert the index-based
            metadata to actual metadata objects when accessing the dataset.
            Defaults to False.
    """

    def __init__(
        self,
        dataset_path: str,
        row_sampler: MultiRowSamplerConfig,
        storage_options: dict | None = None,
        meta_index2meta: bool = False,
        *,
        image_decode_options: (
            RODatasetImageDecodeOptions | Mapping[str, Any] | None
        ) = None,
    ):
        super().__init__(
            dataset_path,
            storage_options,
            meta_index2meta,
            image_decode_options=image_decode_options,
        )
        self._column_datasets = {}
        self._set_row_sampler(row_sampler())

    @property
    def row_sampler(self) -> MultiRowSampler:
        return self._row_sampler

    @row_sampler.setter
    def row_sampler(self, row_sampler: MultiRowSampler) -> None:
        self._set_row_sampler(row_sampler)

    def _set_row_sampler(self, row_sampler: MultiRowSampler) -> None:
        """Set the row sampler for the dataset."""
        self._row_sampler: MultiRowSampler = row_sampler
        for col_name in self._row_sampler.column_rows_keys:
            if col_name not in self._column_datasets:
                if col_name not in self.frame_dataset.column_names:
                    raise KeyError(
                        f"Column {col_name} not found in dataset. "
                        f"Available columns: {self.frame_dataset.column_names}"
                    )
            self._column_datasets[col_name] = (
                self.frame_dataset.select_columns(column_names=col_name)
            )

    @staticmethod
    def from_dataset(
        dataset: RODataset,
        row_sampler: MultiRowSamplerConfig,
    ) -> ROMultiRowDataset:
        """Create a ROMultiRowDataset from an existing RODataset.

        The returned dataset shares the source dataset's metadata database
        lifecycle. Closing either object invalidates metadata access through
        both objects.

        Args:
            dataset (RODataset): The base dataset to extend.
            row_sampler (MultiRowSamplerConfig): The configuration for the
                multi-row sampler.

        """
        parent_state_dict = dataset._get_state_()
        ret = ROMultiRowDataset.__new__(ROMultiRowDataset)
        ret.__dict__.update(parent_state_dict)
        ret._column_datasets = {}
        ret._set_row_sampler(row_sampler())
        return ret

    def __getitem_no_transform__(self, index: int | slice | list[int]) -> dict:
        cached_index_dataset = CachedIndexDataset(self.index_dataset)

        def materialize_sampled_column(
            col_name: str,
            per_row_indices: list[list[int | None]],
            current_values: dict[int, Any],
        ) -> list[list[Any | None]]:
            """Read each non-current referenced row once per batch."""

            referenced_indices = list(
                dict.fromkeys(
                    sampled_index
                    for row_indices in per_row_indices
                    for sampled_index in row_indices
                    if sampled_index is not None
                    and sampled_index not in current_values
                )
            )
            values_by_index = dict(current_values)
            if referenced_indices:
                referenced_values = self._column_datasets[col_name][
                    referenced_indices
                ][col_name]
                if col_name in self._image_decode_columns:
                    referenced_values = [
                        self._materialize_storage_features({col_name: value})[
                            col_name
                        ]
                        for value in referenced_values
                    ]
                values_by_index.update(
                    zip(
                        referenced_indices,
                        referenced_values,
                        strict=True,
                    )
                )
            return [
                [
                    None
                    if sampled_index is None
                    else values_by_index[sampled_index]
                    for sampled_index in row_indices
                ]
                for row_indices in per_row_indices
            ]

        if isinstance(index, int):
            row_data = self._get_frame_data_without_meta_conversion(index)
            if not isinstance(row_data, dict):
                raise TypeError("Integer row access must return a dict.")
            cur_row = (
                self.convert_meta_index2meta(data=row_data)
                if self.meta_index2meta
                else row_data
            )
            # update column that needs multi-row sampling
            for col_name, idx_rows in (
                self._row_sampler._sample_row_idx_with_row_data(
                    cached_index_dataset,
                    index,
                    row_data,
                )
            ).items():
                current_values = (
                    {index: cur_row[col_name]} if col_name in cur_row else {}
                )
                cur_row[col_name] = materialize_sampled_column(
                    col_name,
                    [idx_rows],
                    current_values,
                )[0]
            return cur_row
        else:
            if isinstance(index, slice):
                index = [
                    i for i in range(index.start, index.stop, index.step or 1)
                ]
            assert isinstance(index, list), (
                "Index must be an int, slice, or list of ints."
            )
            row_data = self._get_frame_data_without_meta_conversion(index)
            if not isinstance(row_data, dict):
                raise TypeError("Batch row access must return a dict.")
            cur_rows = (
                self.convert_meta_index2meta(data=row_data)
                if self.meta_index2meta
                else row_data
            )
            # update column that needs multi-row sampling

            # first collect all column rows
            sampled_indices_by_column: dict[str, list[list[int | None]]] = (
                self._row_sampler._sample_row_idx_batch_with_row_data(
                    cached_index_dataset,
                    index,
                    row_data,
                )
            )
            sampled_values_by_column: dict[str, list[list[Any | None]]] = {}
            for k, v in sampled_indices_by_column.items():
                current_values = (
                    {
                        current_index: cur_rows[k][offset]
                        for offset, current_index in enumerate(index)
                    }
                    if k in cur_rows
                    else {}
                )
                sampled_values_by_column[k] = materialize_sampled_column(
                    k,
                    v,
                    current_values,
                )

            for k in cur_rows:
                cur_rows[k] = sampled_values_by_column.get(k, cur_rows[k])
            return cur_rows

    def __getitem__(self, index: int | slice | list[int]) -> dict:
        ret = self.__getitem_no_transform__(index)
        if self._transform is not None:
            ret = self._transform(ret)
        return ret


class ConcatRODataset(TorchDataset):
    """Concatenate multiple RODataset instances into a single dataset.

    This class extends `RODataset` to support concatenation of multiple
    datasets. It provides a unified interface to access data from all
    concatenated datasets. Every input must use the same stored features and
    row materialization policy, including image decode options.

    Args:
        datasets (list[RODataset]): A list of RODataset instances to
            concatenate.
    """

    dataset_index_key: str = "dataset_index"

    def __init__(self, datasets: list[RODataset]):
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided.")

        features = datasets[0].features
        for ds in datasets[1:]:
            if ds.features != features:
                raise ValueError(
                    "All datasets must have the same features "
                    "to be concatenated."
                )
            if ds.meta_index2meta != datasets[0].meta_index2meta:
                raise ValueError(
                    "All datasets must have the same meta_index2meta "
                    "to be concatenated."
                )
            if ds.image_decode_options != datasets[0].image_decode_options:
                raise ValueError(
                    "All datasets must have the same image_decode_options "
                    "to be concatenated."
                )
            if ds.transform != datasets[0].transform:
                raise ValueError(
                    "All datasets must have the same transform "
                    "to be concatenated."
                )
            if self.dataset_index_key in ds.features:
                raise KeyError(
                    f"{self.dataset_index_key} is a reserved key in "
                    "ConcatRODataset. Please rename the column in the "
                    "original dataset."
                )
        self.datasets = datasets
        self.cumulative_sizes = self._compute_cumulative_sizes()

    @property
    def meta_index2meta(self) -> bool:
        return self.datasets[0].meta_index2meta

    @meta_index2meta.setter
    def meta_index2meta(self, value: bool):
        for ds in self.datasets:
            ds.meta_index2meta = value

    @property
    def features(self) -> Features:
        """Get the features of the concatenated dataset.

        Features are the schema of the original dataset without transforms.
        """
        return self.datasets[0].features

    def _compute_cumulative_sizes(self) -> list[int]:
        """Compute the cumulative sizes of the concatenated datasets."""
        cumulative_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            cumulative_sizes.append(total)
        return cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    @property
    def transform(self) -> Callable[[dict], dict] | None:
        return self.datasets[0].transform

    def set_transform(self, transform: Callable[[dict], dict] | None):
        for ds in self.datasets:
            ds.set_transform(transform)

    @contextmanager
    def transform_context(self, transform: Callable[[dict], dict] | None):
        """Context manager to temporarily set a transform for the dataset.

        This is useful for applying a transform only within a specific
        context, without permanently changing the dataset's transform.
        """

        old_transform = self.transform
        self.set_transform(transform)
        try:
            yield self
        finally:
            self.set_transform(old_transform)

    def _map_index_to_dataset(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Index out of range.")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitem__(self, index: int | slice | list[int]) -> dict:
        if isinstance(index, slice):
            index = [
                i for i in range(index.start, index.stop, index.step or 1)
            ]
        # group index by dataset
        if isinstance(index, int):
            dataset_idx, sample_idx = self._map_index_to_dataset(index)
            ret = self.datasets[dataset_idx][sample_idx]
            ret[self.dataset_index_key] = dataset_idx
            return ret

        elif isinstance(index, list):
            grouped_index: dict[int, list[int]] = {}
            # each tuple: dataset_idx, offset_in_dataset
            grouped_index2origin: list[tuple[int, int]] = []

            for idx in index:
                dataset_idx, sample_idx = self._map_index_to_dataset(idx)
                if dataset_idx not in grouped_index:
                    grouped_index[dataset_idx] = []
                offset = len(grouped_index[dataset_idx])
                grouped_index[dataset_idx].append(sample_idx)
                grouped_index2origin.append((dataset_idx, offset))
            # get data from each dataset
            batch_rows = {}
            for dataset_idx, sample_indices in grouped_index.items():
                batch_rows[dataset_idx] = self.datasets[dataset_idx][
                    sample_indices
                ]

            # reorder the batch rows to match the original index order
            ret = {}
            ret[self.dataset_index_key] = []

            for dataset_idx, offset in grouped_index2origin:
                multi_row_dict: dict = batch_rows[dataset_idx]
                for k in multi_row_dict.keys():
                    if k not in ret:
                        ret[k] = []
                    ret[k].append(multi_row_dict[k][offset])
                ret[self.dataset_index_key].append(dataset_idx)
            return ret
        else:
            raise TypeError("Index must be an int, slice, or list of ints.")

    def __getitems__(self, index: list[int]) -> list:
        grouped_index: dict[int, list[int]] = {}
        # each tuple: dataset_idx, sample_idx
        grouped_index2origin: list[tuple[int, int]] = []

        for idx in index:
            dataset_idx, sample_idx = self._map_index_to_dataset(idx)
            if dataset_idx not in grouped_index:
                grouped_index[dataset_idx] = []
            offset = len(grouped_index[dataset_idx])
            grouped_index[dataset_idx].append(sample_idx)
            grouped_index2origin.append((dataset_idx, offset))
        # get data from each dataset
        batch_rows = {}
        for dataset_idx, sample_indices in grouped_index.items():
            batch_rows[dataset_idx] = self.datasets[dataset_idx].__getitems__(
                sample_indices
            )
        ret = []
        for dataset_idx, offset in grouped_index2origin:
            row = batch_rows[dataset_idx][offset]
            row[self.dataset_index_key] = dataset_idx
            ret.append(row)
        return ret


class RODatasetItem(DatasetItem[RODataset]):
    """Lazily construct a configured ``RODataset``-compatible reader.

    ``image_decode_options`` is the common opt-in ImageEncoded materialization
    policy. ``reader_init_kwargs`` is reserved for reader-specific constructor
    keywords such as an internal sidecar reader's ``sidecar_options``. Common
    construction fields remain owned by this item and cannot be overridden
    through that second keyword path.
    """

    class_type: ClassType[RODataset] = RODataset
    dataset_path: str
    storage_options: dict | None = None
    meta_index2meta: bool = False
    image_decode_options: RODatasetImageDecodeOptions | None = None
    transform: Callable | None = None
    reader_init_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reader_contract(self) -> Self:
        """Reject invalid reader types and constructor arguments early."""

        self._reader_init_kwargs()
        return self

    def _reader_init_kwargs(self) -> dict[str, Any]:
        """Return validated constructor arguments for ``class_type``."""

        reader_type = self.class_type
        if not isinstance(reader_type, type) or not issubclass(
            reader_type, RODataset
        ):
            raise ValueError("class_type must be an RODataset subclass.")

        reserved_keys = {
            "dataset_path",
            "storage_options",
            "meta_index2meta",
            "image_decode_options",
            "transform",
        }
        overridden_keys = sorted(
            reserved_keys.intersection(self.reader_init_kwargs)
        )
        if overridden_keys:
            raise ValueError(
                "reader_init_kwargs cannot override RODatasetItem fields: "
                f"{', '.join(overridden_keys)}"
            )

        kwargs = {
            "dataset_path": self.dataset_path,
            "storage_options": self.storage_options,
            "meta_index2meta": self.meta_index2meta,
            **self.reader_init_kwargs,
        }
        if self.image_decode_options is not None:
            kwargs["image_decode_options"] = self.image_decode_options
        try:
            inspect.signature(reader_type).bind(**kwargs)
        except TypeError as exc:
            raise ValueError(
                f"Invalid constructor arguments for {reader_type.__name__}: "
                f"{exc}"
            ) from exc
        return kwargs

    def get_dataset_row_num(self) -> int:
        """Get the number of rows in the dataset."""
        reader_init_kwargs = self._reader_init_kwargs()
        rows = get_row_num_from_dataset_info(
            dataset_path=self.dataset_path,
        )
        if rows is not None:
            return rows

        with self.class_type(**reader_init_kwargs) as dataset:
            return len(dataset)

    def _create_dataset(self) -> RODataset:
        """Create a dataset from the dataset item configuration."""
        dataset = self.class_type(**self._reader_init_kwargs())
        dataset.set_transform(self.transform)
        return dataset


@dataclass(slots=True)
class _RODatasetDBResource:
    """Share one terminal metadata-engine lifecycle across dataset views."""

    engine: Engine
    closed: bool = False

    @property
    def active_engine(self) -> Engine:
        """Return the engine or reject access after the shared close."""

        if self.closed:
            raise RuntimeError(
                "RODataset is closed; construct a new reader for metadata "
                "access."
            )
        return self.engine

    def close(self) -> None:
        """Dispose the shared engine exactly once."""

        if self.closed:
            return
        self.engine.dispose()
        self.closed = True

    def __getstate__(self) -> dict:
        """Serialize the resource URL while preserving graph-level aliases."""

        if self.closed:
            raise RuntimeError("Cannot pickle a closed RODataset resource.")
        return {"db_engine_url": self.engine.url}

    def __setstate__(self, state: dict) -> None:
        """Recreate a process-local engine for one unpickled resource graph."""

        self.engine = create_engine(state["db_engine_url"], readonly=True)
        self.closed = False


def _get_dataset_db_url(dataset_path: str) -> URL:
    fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(dataset_path)[0]
    file_list = fs.ls(dataset_path, detail=False)
    db_candidate = [
        f for f in file_list if os.path.basename(f).startswith("meta_db.")
    ]
    if len(db_candidate) == 0:
        raise ValueError(
            f"No meta db file found in {dataset_path}. "
            "Please ensure the dataset has been properly packaged."
        )
    if len(db_candidate) > 1:
        raise ValueError(
            f"Multiple meta db files found in {dataset_path}: {db_candidate}"  # noqa: E501
        )
    db_path = db_candidate[0]
    # get drivername from file extension
    _, ext = os.path.splitext(db_path)
    drivername = ext[1:]
    return get_local_db_url(drivername=drivername, db_path=db_path)


def _complete_dataset_info(
    dataset_path: str, arrow_dataset: HFDataset | None
) -> tuple[bool, DatasetInfo]:
    import datasets.config as hg_datasets_config
    from datasets import SplitInfo

    def _get_all_arrow_files_total_size(dataset_path: str) -> int:
        dataset_state_path = os.path.join(
            dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
        )
        if not os.path.exists(dataset_state_path):
            raise FileNotFoundError(
                f"Dataset state file not found in {dataset_path}."
            )
        with open(dataset_state_path, "r", encoding="utf-8") as f:
            dataset_state = json.load(f)

        arrow_files: list[str] = [
            t["filename"] for t in dataset_state["_data_files"]
        ]
        total_size = 0
        for arrow_file in arrow_files:
            if not os.path.exists(os.path.join(dataset_path, arrow_file)):
                raise FileNotFoundError(
                    f"Arrow file {arrow_file} not found in {dataset_path}."
                )
            total_size += os.path.getsize(
                os.path.join(dataset_path, arrow_file)
            )
        return total_size

    dataset_info_path = os.path.join(
        dataset_path, hg_datasets_config.DATASET_INFO_FILENAME
    )
    if not os.path.exists(dataset_info_path):
        raise FileNotFoundError(
            f"Dataset info file not found in {dataset_path}."
        )
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = DatasetInfo.from_dict(json.load(f))

    need_update = (
        dataset_info.dataset_size is None or dataset_info.splits is None
    )
    if not need_update:
        return False, dataset_info

    total_size = _get_all_arrow_files_total_size(dataset_path)
    if arrow_dataset is not None:
        total_rows = len(arrow_dataset)
    else:
        arrow_dataset = HFDataset.load_from_disk(dataset_path)
        total_rows = len(arrow_dataset)
    if dataset_info.dataset_size is None:
        dataset_info.dataset_size = total_size
    if dataset_info.splits is None:
        dataset_info.splits = SplitDict()
        dataset_info.splits["train"] = SplitInfo(
            name="train",
            num_bytes=total_size,
            num_examples=total_rows,
        )
    return True, dataset_info


def get_row_num_from_dataset_info(dataset_path: str) -> int | None:
    """Get the number of rows in the dataset from the dataset info file.

    The row number is a total count of all rows in all splits. If the dataset
    info file does not contain the split information, this function will
    return None.

    Note:
        This method assumes that the dataset info file is stored in the
        dataset directory, and the split information is available in the
        dataset info file.

    Args:
        dataset_path (str): The path to the dataset directory.

    Returns:
        int | None: The number of rows in the dataset if available,
            otherwise None.

    Raises:
        FileNotFoundError: If the dataset info file is not found in the
            specified dataset path.

    """
    import datasets.config as hg_datasets_config

    dataset_info_path = os.path.join(
        dataset_path, hg_datasets_config.DATASET_INFO_FILENAME
    )
    if not os.path.exists(dataset_info_path):
        raise FileNotFoundError(
            f"Dataset info file not found in {dataset_path}."
        )
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = DatasetInfo.from_dict(json.load(f))

    if dataset_info.splits is None:
        return None

    return dataset_info.splits.total_num_examples
