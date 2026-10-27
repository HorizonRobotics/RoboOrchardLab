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

from pathlib import Path

import datasets as hg_datasets
import pytest
from sqlalchemy.orm import Session

from robo_orchard_lab.dataset.robot.columns import PreservedIndexColumnsKeys
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.dataset_db_engine import (
    create_engine,
    create_tables,
    get_local_db_url,
)
from robo_orchard_lab.dataset.robot.db_orm import Episode
from robo_orchard_lab.dataset.robot.filter import RODatasetFilter


@pytest.mark.parametrize("episode_index_begin", [0, 100])
def test_unfiltered_rows_use_stored_episode_ids(
    tmp_path: Path,
    episode_index_begin: int,
) -> None:
    engine = create_engine(
        get_local_db_url(str(tmp_path / "meta_db.sqlite"), "sqlite")
    )
    try:
        create_tables(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    Episode(
                        index=episode_index_begin,
                        dataset_begin_index=0,
                        frame_num=2,
                    ),
                    Episode(
                        index=episode_index_begin + 1,
                        dataset_begin_index=2,
                        frame_num=3,
                    ),
                ]
            )
            session.commit()

        columns: dict[str, list[int | None]] = {
            column: [None] * 5 for column in PreservedIndexColumnsKeys
        }
        columns.update(
            index=list(range(5)),
            episode_index=[episode_index_begin] * 2
            + [episode_index_begin + 1] * 3,
            frame_index=[0, 1, 0, 1, 2],
        )
        frames = hg_datasets.Dataset.from_dict(
            columns,
            features=hg_datasets.Features(
                {
                    column: hg_datasets.Value("int64")
                    for column in PreservedIndexColumnsKeys
                }
            ),
        )
        with RODataset.from_dataset(frames, engine) as dataset:
            assert dataset.episode_num == 2
            dataset_filter = RODatasetFilter()
            assert dataset_filter.get_filtered_episode_indices(dataset) == {
                episode_index_begin,
                episode_index_begin + 1,
            }
            selected = dataset_filter.apply(dataset)
            assert len(selected) == 5
            assert sorted(selected[i]["index"] for i in range(5)) == list(
                range(5)
            )
    finally:
        engine.dispose()
