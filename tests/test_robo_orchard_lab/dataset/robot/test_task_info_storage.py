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

import pytest
from packaging.version import Version
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from robo_orchard_lab.dataset.robot.dataset_db_engine import (
    create_engine,
    create_tables,
    get_local_db_url,
)
from robo_orchard_lab.dataset.robot.db_orm.task import (
    Task,
    TaskDeprecatedVersion1,
)
from robo_orchard_lab.dataset.robot.db_orm.upgrade import TableUpgradeRegistry
from robo_orchard_lab.dataset.robot.packaging import (
    DatasetIndexState,
    TaskData,
)


def _create_sqlite_engine(tmp_path: Path):
    db_url = get_local_db_url(
        db_path=str(tmp_path / "task_info.sqlite"),
        drivername="sqlite",
    )
    engine = create_engine(db_url, readonly=False)
    create_tables(engine)
    return engine


def test_task_table_has_reserved_info_column(tmp_path: Path) -> None:
    engine = _create_sqlite_engine(tmp_path)
    try:
        task_columns = {
            column["name"] for column in inspect(engine).get_columns("task")
        }
        assert Task.__version__ == "0.0.2"
        assert "info" in task_columns
    finally:
        engine.dispose()


def test_task_data_uses_info_as_task_identity(tmp_path: Path) -> None:
    engine = _create_sqlite_engine(tmp_path)
    try:
        index_state = DatasetIndexState()
        with Session(engine) as session:
            task = TaskData(
                name="pick_cube",
                description="Pick up the cube.",
            ).make_transient_orm(index_state, session=session)

            assert task.info is None
            session.add(task)
            session.commit()
            index_state.last_task_idx = task.index

            same_task = TaskData(
                name="pick_cube",
                description="Pick up the cube.",
                info={},
            ).make_transient_orm(index_state, session=session)

            assert same_task.index == task.index
            assert same_task.info is None

            info_task = TaskData(
                name="pick_cube",
                description="Pick up the cube.",
                info={"dataset_specific": "metadata"},
            ).make_transient_orm(index_state, session=session)

            assert info_task.index == 1
            assert info_task.info == {"dataset_specific": "metadata"}
            session.add(info_task)
            session.commit()
            index_state.last_task_idx = info_task.index

            same_info_task = TaskData(
                name="pick_cube",
                description="Pick up the cube.",
                info={"dataset_specific": "metadata"},
            ).make_transient_orm(index_state, session=session)

            assert same_info_task.index == info_task.index
            assert same_info_task.info == {"dataset_specific": "metadata"}
    finally:
        engine.dispose()


def test_task_md5_uses_info_as_semantic_identity() -> None:
    without_info = Task(
        index=0,
        name="pick_cube",
        description="Pick up the cube.",
        info=None,
    )
    with_info = Task(
        index=1,
        name="pick_cube",
        description="Pick up the cube.",
        info={"reserved": True},
    )
    same_info_with_different_key_order = Task(
        index=2,
        name="pick_cube",
        description="Pick up the cube.",
        info={"z": 1, "a": 2},
    )
    same_info_with_different_key_order_2 = Task(
        index=3,
        name="pick_cube",
        description="Pick up the cube.",
        info={"a": 2, "z": 1},
    )

    assert Task.md5_content_fields() == ["name", "description", "info"]
    assert without_info.update_md5() != with_info.update_md5()
    assert (
        same_info_with_different_key_order.update_md5()
        == same_info_with_different_key_order_2.update_md5()
    )


def test_task_normalizes_empty_info_at_orm_boundary(
    tmp_path: Path,
) -> None:
    none_info = Task(
        index=0,
        name="pick_cube",
        description="Pick up the cube.",
        info=None,
    )
    empty_info = Task(
        index=1,
        name="pick_cube",
        description="Pick up the cube.",
        info={},
    )

    assert empty_info.info is None
    assert none_info.update_md5() == empty_info.update_md5()

    engine = _create_sqlite_engine(tmp_path)
    try:
        with Session(engine) as session:
            session.add(none_info)
            session.commit()

            same_task = Task.query_by_content_with_md5(
                session,
                name="pick_cube",
                description="Pick up the cube.",
                info={},
            )

            assert same_task is not None
            assert same_task.index == none_info.index
            assert same_task.info is None
    finally:
        engine.dispose()


def test_task_info_rejects_non_string_json_key() -> None:
    index_state = DatasetIndexState()

    with pytest.raises(TypeError, match="JSON dict key"):
        TaskData(
            name="pick_cube",
            description="Pick up the cube.",
            info={1: "bad"},  # type: ignore[dict-item]
        ).make_transient_orm(index_state, session=None)


def test_task_info_rejects_non_finite_float() -> None:
    index_state = DatasetIndexState()

    with pytest.raises(ValueError, match="finite"):
        TaskData(
            name="pick_cube",
            description="Pick up the cube.",
            info={"score": float("nan")},
        ).make_transient_orm(index_state, session=None)


def test_task_query_by_content_with_md5_checks_reserved_info(
    tmp_path: Path,
) -> None:
    engine = _create_sqlite_engine(tmp_path)
    try:
        with Session(engine) as session:
            task = Task(
                index=0,
                name="pick_cube",
                description="Pick up the cube.",
                info=None,
            )
            task.update_md5()
            session.add(task)
            session.commit()

            assert (
                Task.query_by_content_with_md5(
                    session,
                    name="pick_cube",
                    description="Pick up the cube.",
                    info=None,
                ).index
                == 0
            )
            assert (
                Task.query_by_content_with_md5(
                    session,
                    name="pick_cube",
                    description="Pick up the cube.",
                    info={"reserved": True},
                )
                is None
            )
    finally:
        engine.dispose()


def test_task_upgrade_from_0_0_1_adds_empty_info_and_recomputes_md5(
    tmp_path: Path,
) -> None:
    engine = _create_sqlite_engine(tmp_path)
    try:
        with Session(engine) as session:
            upgrade = TableUpgradeRegistry.get_upgrade_func(
                table_name=Task.__tablename__,
                current_version=Version("0.0.1"),
                target_version=Version("0.0.2"),
            )
            old_row = {
                "index": 0,
                "name": "pick_cube",
                "description": "Pick up the cube.",
                "md5": b"legacy-md5-value",
            }

            upgraded_row = upgrade(session, old_row)

            expected_task = Task(
                index=0,
                name="pick_cube",
                description="Pick up the cube.",
                info=None,
            )
            expected_task.update_md5()
            assert upgraded_row == {
                "index": 0,
                "name": "pick_cube",
                "description": "Pick up the cube.",
                "info": None,
                "md5": expected_task.md5,
            }
            assert (
                TableUpgradeRegistry.get_version_orm(
                    Task.__tablename__, Version("0.0.1")
                )
                is TaskDeprecatedVersion1
            )
    finally:
        engine.dispose()
