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
from pathlib import Path

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from robo_orchard_lab.dataset.robot._table_manager import get_table_update_plan
from robo_orchard_lab.dataset.robot.dataset import (
    RODataset,
)
from robo_orchard_lab.dataset.robot.dataset_db_engine import (
    DatasetMetaDBHandler,
    create_engine,
    get_local_db_url,
    need_upgrade,
)
from robo_orchard_lab.dataset.robot.db_orm.base import DatasetORMBase
from robo_orchard_lab.dataset.robot.db_orm.table_info import TableInfo
from robo_orchard_lab.dataset.robot.db_orm.task import Task


@pytest.fixture()
def robotwin_ro_dataset_path(ROBO_ORCHARD_TEST_WORKSPACE: str) -> str:
    return os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/robotwin/ro_dataset",
    )


@pytest.fixture()
def libero_old_v0_dataset_path(
    ROBO_ORCHARD_TEST_WORKSPACE: str,
) -> str:
    return os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/old_version/libero_dataset_v0",
    )


@pytest.fixture()
def legacy_task_info_v001_dataset_path(
    ROBO_ORCHARD_TEST_WORKSPACE: str,
) -> str:
    return os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/old_version/"
        "legacy_dataset_task_info_v001",
    )


@pytest.fixture(
    params=[
        "robotwin_ro_dataset_path",
        "libero_old_v0_dataset_path",
        "legacy_task_info_v001_dataset_path",
    ],
    ids=[
        "robotwin_ro_V_None",
        "libero_ro_v0",
        "legacy_task_info_v001",
    ],
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
        row = dataset[min(2, len(dataset) - 1)]
        print(row.keys())
        print("episode: ", row["episode"])
        if "joints" in row:
            print("joints: ", row["joints"])

    def test_load_legacy_task_info_v001(
        self, legacy_task_info_v001_dataset_path: str, tmp_path: Path
    ):
        db_path = os.path.join(
            legacy_task_info_v001_dataset_path, "meta_db.sqlite"
        )
        db_url = get_local_db_url(db_path, "sqlite")
        assert need_upgrade(db_url)

        engine = create_engine(
            url=db_url,
            readonly=True,
            auto_upgrade=False,
        )
        try:
            task_columns = {
                column["name"]
                for column in inspect(engine).get_columns("task")
            }
            assert "info" not in task_columns

            with Session(engine) as session:
                task_info = session.scalar(
                    select(TableInfo).where(TableInfo.table_name == "task")
                )
                assert task_info is not None
                assert task_info.table_version == "0.0.1"

            plan = get_table_update_plan(engine)
            assert plan is not None
            upgraded_db_path = str(tmp_path / "meta_db.upgraded.sqlite")
            DatasetMetaDBHandler(db_path, "sqlite").upgrade_to_cache(
                src_engine=engine,
                target_db_path=upgraded_db_path,
                plan=plan,
            )
        finally:
            engine.dispose()

        upgraded_engine = create_engine(
            url=get_local_db_url(upgraded_db_path, "sqlite"),
            readonly=True,
            auto_upgrade=False,
        )
        try:
            assert need_upgrade(upgraded_engine) is False
            with Session(upgraded_engine) as session:
                tasks = list(
                    session.scalars(select(Task).order_by(Task.index))
                )
        finally:
            upgraded_engine.dispose()

        assert [task.info for task in tasks] == [None, None]
        assert all(task.md5 == task.calculate_md5() for task in tasks)

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
