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

import os

import pytest

from robo_orchard_lab.dataset.robot._table_manager import get_table_update_plan
from robo_orchard_lab.dataset.robot.dataset import (
    RODataset,
)
from robo_orchard_lab.dataset.robot.dataset_db_engine import (
    create_engine,
    get_local_db_url,
)
from robo_orchard_lab.dataset.robot.db_orm.base import DatasetORMBase
from robo_orchard_lab.dataset.robot.db_orm.table_info import TableInfo


@pytest.fixture()
def robotwin_ro_dataset_path(ROBO_ORCHARD_TEST_WORKSPACE: str) -> str:
    return os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/robotwin/ro_dataset",
    )


@pytest.fixture(
    params=[
        "robotwin_ro_dataset_path",
    ],
    ids=["robotwin_ro_V_None"],
)
def old_version_dataset(request) -> str:
    """Provide a database engine from different backends."""
    return request.getfixturevalue(request.param)


class TestDatasetUpgrade:
    def test_load_old_version(self, old_version_dataset: str):
        dataset = RODataset(
            dataset_path=old_version_dataset, meta_index2meta=True
        )
        print(len(dataset))
        print(dataset.frame_dataset.features)
        row = dataset[2]
        print(row.keys())
        print("episode: ", row["episode"])
        print("joints: ", row["joints"])

    def test_get_table_update_plan(self, robotwin_ro_dataset_path):
        db_path = os.path.join(robotwin_ro_dataset_path, "meta_db.duckdb")
        engine = create_engine(
            url=get_local_db_url(db_path, "duckdb"),
            readonly=True,
            auto_upgrade=False,
        )
        plan = get_table_update_plan(engine)
        assert plan is not None
        print(plan)
        all_cls_changes_in_plan = [change.orm_class for change in plan.actions]
        for mapper in DatasetORMBase.registry.mappers:
            if mapper.class_ is TableInfo:
                continue
            assert mapper.class_ in all_cls_changes_in_plan
