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


import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from robo_orchard_lab.dataset.robot.db_orm import (
    DatasetORMBase,
    Episode,
    Instruction,
    Robot,
    RobotDescriptionFormat,
    Task,
)
from robo_orchard_lab.dataset.robot.engine import (
    create_tables,
    create_temp_engine,
)


@pytest.fixture(scope="session")
def ut_database_engine_duckdb(tmp_local_folder: str):
    """Create a temporary SQLite database engine for unit tests."""
    # Create a temporary SQLite database engine
    with create_temp_engine(
        dir=tmp_local_folder,
        prefix="ut_database_engine_duckdb",
        drivername="duckdb",
    ) as engine:
        create_tables(engine, base=DatasetORMBase)
        yield engine


@pytest.fixture(scope="session")
def ut_database_engine_sqlite(tmp_local_folder: str):
    """Create a temporary SQLite database engine for unit tests."""
    # Create a temporary SQLite database engine
    with create_temp_engine(
        dir=tmp_local_folder,
        prefix="ut_database_engine_sqlite",
        drivername="sqlite",
    ) as engine:
        create_tables(engine, base=DatasetORMBase)
        yield engine


def prepare_data(engine: Engine):
    with Session(engine) as session:
        # Create a Robot instance
        robot = Robot(
            index=0,
            name="robot_1",
            content="robot_1_urdf_content",
            content_format=RobotDescriptionFormat.URDF,
        )
        robot.update_md5()
        session.add(robot)
        # Create a Task instance
        task = Task(index=0, name="task_1", description="task_1_description")
        session.add(task)
        task.update_md5()

        # Create an Instruction instance
        instruction = Instruction(
            index=0,
            name="instruction_1",
            json_content={
                "instruction": "instruction_1_content",
                "robot": robot.name,
                "task": task.name,
            },
        )
        instruction.update_md5()
        session.add(instruction)

        # Create an Instruction instance wiout task and robot
        instruction = Instruction(
            index=1,
            name="instruction_1",
            json_content={
                "instruction": "instruction_1_content",
                "robot": robot.name,
                "task": task.name,
            },
        )
        instruction.update_md5()
        session.add(instruction)
        # Create an Episode instance
        episode = Episode(index=0, frame_num=3, dataset_begin_index=0)
        session.add(episode)
        # we have to commit the session because duckdb does not support
        # self foreign key constraints in the same transaction
        session.commit()

        # Create an Episode instance
        episode = Episode(
            index=1,
            frame_num=2,
            robot_index=0,
            task_index=0,
            prev_episode_index=0,
            dataset_begin_index=9,
        )
        session.add(episode)
        session.commit()


@pytest.fixture(
    scope="session",
    params=[
        "ut_database_engine_duckdb",
        "ut_database_engine_sqlite",
    ],
    ids=["duckdb", "sqlite"],
)
def db_engine(request) -> Engine:
    """Provide a database engine from different backends."""
    ret: Engine = request.getfixturevalue(request.param)
    prepare_data(ret)
    return ret


class TestInstruction:
    """Test the Instruction class."""

    def test_get_instruction(self, db_engine: Engine):
        """Test creating an Instruction instance."""
        with Session(db_engine) as session:
            stmt = select(Instruction).where(Instruction.index == 0)
            data = session.execute(stmt).scalar_one()
            assert data.index == 0
            print(data.json_content)

    def test_query_content_with_md5(self, db_engine: Engine):
        """Test querying an Instruction instance by content and MD5."""
        with Session(db_engine) as session:
            instruction = Instruction.query_by_content_with_md5(
                session,
                name="instruction_1",
                json_content={
                    "instruction": "instruction_1_content",
                    "robot": "robot_1",
                    "task": "task_1",
                },
            )
            assert instruction is not None
            assert instruction.index == 0

            # Test querying with different content order
            instruction = Instruction.query_by_content_with_md5(
                session,
                name="instruction_1",
                json_content={
                    "instruction": "instruction_1_content",
                    "task": "task_1",
                    "robot": "robot_1",
                },
            )
            assert instruction is not None
            assert instruction.index == 0

            # Test querying with different content
            instruction = Instruction.query_by_content_with_md5(
                session,
                name="instruction_1",
                json_content={
                    "instruction": "instruction_1_content",
                    "robot": "robot_1",
                    "task": None,
                },
            )
            assert instruction is None


class TestEpisode:
    """Test the Episode class."""

    def test_get_episode(self, db_engine: Engine):
        """Test creating an Episode instance."""
        with Session(db_engine) as session:
            stmt = select(Episode).where(Episode.index == 0)
            data = session.execute(stmt).scalar_one()
            assert data.index == 0
            assert data.frame_num is not None
            assert data.prev_episode is None
            next_episode = data.next_episode
            assert next_episode is not None
            assert len(next_episode) == 1

    def test_episode_fk(self, db_engine: Engine):
        """Test creating an Episode instance."""
        with Session(db_engine) as session:
            stmt = select(Episode).where(Episode.index == 1)
            data = session.execute(stmt).scalar_one()
            assert data.robot_index == 0
            assert data.task_index == 0
            next_episode = data.next_episode
            prev_episode = data.prev_episode
            assert prev_episode is not None
            assert isinstance(prev_episode, Episode)

            assert len(next_episode) == 0
            assert data.task is not None
