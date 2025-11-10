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

import os
from typing import Type, TypeVar

import numpy as np
import pyarrow as pa
from datasets import (
    Dataset as HFDataset,
    Features,
)
from datasets.arrow_dataset import _concatenate_map_style_datasets
from datasets.arrow_writer import ArrowWriter
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)
from robo_orchard_lab.dataset.robot.db_orm.md5 import MD5ObjCache

MD5TableType = TypeVar("MD5TableType", bound=Instruction | Robot | Task)


def _merge_md5_table(
    orm_type: Type[MD5TableType],
    src_session: Session,
    dst_session: Session,
    src_index_mapping: np.ndarray,
    batch_size: int = 500,
) -> None:
    dst_max_index = (
        dst_session.query(orm_type.index)
        .order_by(orm_type.index.desc())
        .first()
    )
    dst_max_index = dst_max_index[0] if dst_max_index is not None else -1
    dst_next_index = dst_max_index + 1  # next available index in dst
    check_column_names = orm_type.md5_content_fields()

    bar = tqdm(
        unit="rows",
        total=src_session.query(orm_type).count(),
        desc=f" Merging {orm_type.__tablename__} ",
    )

    with bar:

        def batch_processor(
            src_batch: list[MD5TableType], dst_session: Session
        ) -> None:
            nonlocal dst_next_index

            # find existing md5 in dst_session
            batch_md5 = [obj.md5 for obj in src_batch]
            cache = MD5ObjCache[orm_type](
                check_column_names=check_column_names
            )

            cache.extend(
                dst_session.scalars(
                    select(orm_type).where(orm_type.md5.in_(batch_md5))
                ).all()
            )

            for src_obj in src_batch:
                if (existing_obj := cache.find(src_obj)) is not None:
                    # already exists, use existing index
                    src_index_mapping[src_obj.index] = existing_obj.index
                    continue
                # create new dst_obj and add to dst_session
                dst_obj_dict = {}
                orm_type.column_copy(src_obj, dst_obj_dict)
                dst_obj = orm_type(**dst_obj_dict)
                dst_obj.index = dst_next_index
                src_index_mapping[src_obj.index] = dst_next_index
                dst_next_index += 1
                dst_session.add(dst_obj)

            dst_session.commit()  # commit the batch to get indexes assigned
            bar.update(len(src_batch))

        batch = []
        for src_obj in src_session.scalars(select(orm_type)):
            # cache src_obj data as batch
            batch.append(src_obj)
            if len(batch) < batch_size:
                continue
            else:
                batch_processor(batch, dst_session)
                batch.clear()

        if len(batch) > 0:
            batch_processor(batch, dst_session)
            batch.clear()


def _merge_episode_table(
    src_session: Session,
    dst_session: Session,
    robot_mapping: np.ndarray,
    task_mapping: np.ndarray,
    src_index_mapping: np.ndarray,
    batch_size: int = 500,
):
    # get next available episode index in dst_session
    src_max_episode_index = (
        dst_session.query(Episode.index).order_by(Episode.index.desc()).first()
    )
    src_max_episode_index = (
        src_max_episode_index[0] if src_max_episode_index is not None else -1
    )
    next_dst_episode_index = src_max_episode_index + 1

    bar = tqdm(
        unit="rows",
        total=src_session.query(Episode).count(),
        desc=f" Merging {Episode.__tablename__} ",
    )
    with bar:
        for i, src_episode in enumerate(src_session.scalars(select(Episode))):
            dst_episode_dict = {}
            Episode.column_copy(src_episode, dst_episode_dict)
            new_episode = Episode(**dst_episode_dict)
            # remap foreign keys
            if new_episode.task_index is not None:
                new_episode.task_index = int(
                    task_mapping[new_episode.task_index]
                )
            if new_episode.robot_index is not None:
                new_episode.robot_index = int(
                    robot_mapping[new_episode.robot_index]
                )
            new_episode.index = next_dst_episode_index
            src_index_mapping[src_episode.index] = next_dst_episode_index
            next_dst_episode_index += 1
            dst_session.add(new_episode)

            if (i + 1) % batch_size == 0:
                dst_session.commit()
            bar.update(1)
        dst_session.commit()


def _merge_meta_db(
    src_session: Session, dst_session: Session, cache_dir: str
) -> dict[str, np.ndarray]:
    """Merge meta database from src_session to dst_session.

    Args:
        src_session (Session): The source database session.
        dst_session (Session): The destination database session.

    Returns:
        dict[str, np.ndarray]: A mapping from table name to index mapping
            array. Each index mapping array maps from source index to
            destination index.
    """

    def prepare_index_mapping(
        cache_dir: str, orm_type: Type[Instruction | Robot | Task | Episode]
    ) -> np.ndarray:
        max_src_index = (
            src_session.query(orm_type.index)
            .order_by(orm_type.index.desc())
            .first()
        )
        max_src_index = max_src_index[0] if max_src_index is not None else -1
        max_src_index += 1
        # create mmap numpy array to store the mapping
        # from src index to dst index
        mapping_file = os.path.join(
            cache_dir, f"{orm_type.__tablename__}_index_mapping.dat"
        )
        if os.path.exists(mapping_file):
            os.remove(mapping_file)

        index_mapping = np.memmap(
            filename=mapping_file,
            dtype=np.int64,
            mode="w+",
            shape=(max_src_index,),
        )
        index_mapping[:] = -1  # initialize to -1
        return index_mapping

    src_index_mapping: dict[str, np.ndarray] = {}

    for orm_type in (Instruction, Robot, Task):
        index_mapping = prepare_index_mapping(
            cache_dir=cache_dir, orm_type=orm_type
        )
        src_index_mapping[orm_type.__tablename__] = index_mapping
        _merge_md5_table(
            orm_type=orm_type,
            src_session=src_session,
            dst_session=dst_session,
            src_index_mapping=index_mapping,
        )
    # episode_index mapping
    episode_index_mapping = prepare_index_mapping(
        cache_dir=cache_dir, orm_type=Episode
    )
    _merge_episode_table(
        src_session=src_session,
        dst_session=dst_session,
        robot_mapping=src_index_mapping[Robot.__tablename__],
        task_mapping=src_index_mapping[Task.__tablename__],
        src_index_mapping=episode_index_mapping,
    )
    src_index_mapping[Episode.__tablename__] = episode_index_mapping

    return src_index_mapping


def _remap_meta_index(
    frame_dataset: HFDataset,
    index_mapping: dict[str, np.ndarray],
    target_index_start: int,
    cached_index_path: str,
) -> HFDataset:
    """Remap meta index columns in the frame dataset.

    All meta index columns (instruction_index, task_index, robot_index,
    episode_index) will be remapped according to the provided index_mapping.
    The `index` column will be offset by `target_index_start`.

    The remapped columns will be saved to a new Arrow file at
    `cached_index_path`.

    Args:
        frame_dataset (HFDataset): The frame dataset to remap.
        index_mapping (dict[str, np.ndarray]): A mapping from meta table
            name to index mapping array.
        target_index_start (int): The starting index for the `index` column
            offset.
        cached_index_path (str): The path to save the remapped Arrow file.

    Returns:
        HFDataset: The remapped frame dataset.

    """

    mapped_arrays = []

    # remap meta index
    mapping_columns = [
        ("instruction_index", "instruction"),
        ("task_index", "task"),
        ("robot_index", "robot"),
        ("episode_index", "episode"),
    ]
    for column_name, src_name in mapping_columns:
        column_idx_mapping = index_mapping[src_name]
        origin_column = frame_dataset[column_name]
        new_column: np.ndarray = column_idx_mapping[origin_column]
        # check that all indices are valid
        if np.any(new_column < 0):
            raise ValueError(f"Invalid index found in column {column_name}.")
        mapped_arrays.append(pa.array(new_column))

    # Offset `index` by target_index_start
    origin_index_column = frame_dataset["index"]
    new_index_column = (
        np.array(origin_index_column, dtype=np.int64) + target_index_start
    )
    mapped_arrays.append(pa.array(new_index_column))
    mapping_columns.append(("index", None))  # type: ignore
    mapped_column_names = [k for k, _ in mapping_columns]

    # Write to Arrow file
    table = pa.Table.from_arrays(mapped_arrays, names=mapped_column_names)
    feature = Features(
        {k: frame_dataset.features[k] for k in mapped_column_names}
    )
    instruction_column_writer = ArrowWriter(
        path=cached_index_path, features=feature
    )
    instruction_column_writer.write_table(table)
    instruction_column_writer.finalize()
    instruction_column_writer.close()

    mapped_idx_dataset = HFDataset.from_file(cached_index_path)
    mapped_dataset = _concatenate_map_style_datasets(
        [
            frame_dataset.select_columns(
                [
                    k
                    for k in frame_dataset.features.keys()
                    if k not in mapped_column_names
                ]
            ),
            mapped_idx_dataset,
        ],
        axis=1,
    )

    return mapped_dataset
