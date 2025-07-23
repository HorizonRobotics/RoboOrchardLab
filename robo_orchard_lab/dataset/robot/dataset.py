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

import json
import os
from typing import Any, TypeAlias, TypeVar

import fsspec
import torch
from datasets import Dataset as HFDataset
from sqlalchemy import URL, Engine
from sqlalchemy.orm import Session, make_transient

from robo_orchard_lab.dataset.datatypes import *
from robo_orchard_lab.dataset.robot.columns import (
    PreservedIndexColumnsKeys,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.engine import create_engine
from robo_orchard_lab.dataset.robot.row_sampler import (
    MultiRowSampler,
    MultiRowSamplerConfig,
)

__all__ = ["RODataset", "ROMultiRowDataset"]

MetaType = TypeVar("MetaType", Episode, Instruction, Robot, Task)
"""A type variable for metadata types in the RoboOrchard dataset."""

TorchDataset: TypeAlias = torch.utils.data.Dataset


class RODataset(TorchDataset):
    """The RoboOrchard dataset for robot data.

    We use a tabular dataset to store the frame-level information, and a
    separate database to store the episode-level information. The huggingface
    datasets (pyarrow_dataset) is used as table format, and SQLAlchemy with
    DuckDB are used to manage the database.

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
            Defaults to True.

    """

    frame_dataset: HFDataset
    """The Hugging Face Dataset object containing the frame data."""
    db_engine: Engine
    """The SQLAlchemy engine for the meta database"""
    index_dataset: HFDataset
    """The same as `frame_dataset`, but only contains the preserved index columns."""  # noqa: E501
    meta_index2meta: bool
    """Whether to convert the index-based metadata to actual metadata
    objects when accessing the dataset."""

    def __init__(
        self,
        dataset_path: str,
        storage_options: dict | None = None,
        meta_index2meta: bool = True,
    ):
        dataset_path = os.path.expanduser(dataset_path)
        self.frame_dataset = HFDataset.load_from_disk(
            dataset_path, storage_options=storage_options
        )
        self.index_dataset = self.frame_dataset.select_columns(
            column_names=list(PreservedIndexColumnsKeys)
        )
        self.meta_index2meta = meta_index2meta
        # recover state dict
        from datasets import config as hg_datasets_config

        state_file = os.path.join(
            dataset_path, hg_datasets_config.DATASET_STATE_JSON_FILENAME
        )
        state: dict = json.load(open(state_file, "r"))
        self._dataset_format_version = state.get("robo_orchard_state", {}).get(
            "dataset_format_version", None
        )
        # load db
        self._load_db(dataset_path)

    def _load_db(self, dataset_path: str):
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
        self.db_engine = create_engine(
            url=URL.create(drivername=drivername, database=db_path),
            readonly=True,
        )

    def _meta_index2meta(self, src: dict[str, Any]) -> dict:
        """Convert the index-based metadata in `src` to actual metadata objects."""  # noqa: E501
        dst = src.copy()
        episode = self.get_meta(Episode, dst.pop("episode_index", None))
        task = self.get_meta(Task, dst.pop("task_index", None))
        robot = self.get_meta(Robot, dst.pop("robot_index", None))
        instruction = self.get_meta(
            Instruction, dst.pop("instruction_index", None)
        )
        dst.update(
            {
                "episode": episode,
                "task": task,
                "robot": robot,
                "instruction": instruction,
            }
        )
        return dst

    def __getitem__(self, index) -> dict:
        """Get the frame data at the specified index.

        Args:
            index (int): The index of the frame data to retrieve.

        Returns:
            dict: The frame data at the specified index.
        """

        ret: dict = self.frame_dataset[index]
        if self.meta_index2meta:
            ret = self._meta_index2meta(ret)
        return ret

    def __len__(self) -> int:
        """Get the number of frames in the dataset."""
        return len(self.frame_dataset)

    def get_meta(
        self, meta_type: type[MetaType], index: int | None
    ) -> MetaType | None:
        """Get metadata of a specific type.

        This method retrieves metadata from the database using index.
        Possible metadata types include `Episode`, `Instruction`, `Robot`,
        and `Task`.

        Args:
            meta_type (type[MetaType]): The type of metadata to retrieve.
            index (int | None): The index of the metadata to retrieve.
                If None, returns None.
        """
        if index is None:
            return None

        with Session(self.db_engine) as session:
            ret = session.get(meta_type, index)
            if ret is not None:
                make_transient(ret)
            return ret

    @property
    def dataset_format_version(self) -> str | None:
        """Get the dataset format version of loaded dataset."""
        return self._dataset_format_version


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
            Defaults to True.
    """

    def __init__(
        self,
        dataset_path: str,
        row_sampler: MultiRowSamplerConfig,
        storage_options: dict | None = None,
        meta_index2meta: bool = True,
    ):
        super().__init__(dataset_path, storage_options, meta_index2meta)
        self._row_sampler: MultiRowSampler = row_sampler()
        self._column_datasets = {
            col_name: self.frame_dataset.select_columns(column_names=col_name)
            for col_name in self._row_sampler.column_rows_keys
        }

    def __getitem__(self, index) -> dict:
        cur_row = super().__getitem__(index)
        for col_name, idx_rows in self._row_sampler.sample_row_idx(
            self.index_dataset, index
        ).items():
            col_dataset = self._column_datasets[col_name]
            cur_row[col_name] = [
                col_dataset[idx][col_name] if idx is not None else None
                for idx in idx_rows
            ]
        return cur_row
