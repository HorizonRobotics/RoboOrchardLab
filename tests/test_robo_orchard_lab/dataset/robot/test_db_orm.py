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

import hashlib
import json
from copy import deepcopy
from pathlib import Path

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
from robo_orchard_lab.dataset.robot.metadata_schema import (
    MetadataSchemaError,
    ROInstructionContent,
    parse_registered_metadata_schema,
)
from robo_orchard_lab.dataset.robot.packaging import InstructionData
from robo_orchard_lab.dataset.robot.packaging._writer import DatasetIndexState


def test_create_temp_engine_preserves_parent_with_tmp_in_name(
    tmp_path: Path,
) -> None:
    """Only the generated filename suffix changes for temporary engines."""

    database_parent = tmp_path / "sidecar.tmp"
    database_parent.mkdir()

    with create_temp_engine(
        dir=str(database_parent),
        prefix="sidecar-",
        drivername="sqlite",
    ) as engine:
        assert engine.url.database is not None
        database_path = Path(engine.url.database)
        assert database_path.parent == database_parent
        assert database_path.suffix == ".sqlite"
        create_tables(engine, base=DatasetORMBase)


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

    def test_canonical_identity_storage_and_query(self, tmp_path):
        payload = {
            "schema_id": "robo_orchard.instruction_content",
            "schema_version": "1.1",
            "descriptions": [" z ", "a", "a"],
            "subtask_type": "frame",
            "subtasks": [
                {
                    "frame_index_begin": 3,
                    "frame_index_end": 5,
                    "descriptions": ["b"],
                },
                {
                    "frame_index_begin": 0,
                    "frame_index_end": 4,
                    "descriptions": ["z", "a"],
                    "extras": {"nested": {"b": 2, "a": 1}},
                },
            ],
        }
        variant = deepcopy(payload)
        variant["descriptions"] = ["a", "z"]
        variant["subtasks"] = [
            deepcopy(payload["subtasks"][1]),
            deepcopy(payload["subtasks"][0]),
            deepcopy(payload["subtasks"][0]),
        ]
        variant["subtasks"][0]["descriptions"] = ["a", "z", "a"]
        variant["subtasks"][0]["extras"]["nested"] = {"a": 1, "b": 2}
        with create_temp_engine(
            dir=str(tmp_path), drivername="sqlite"
        ) as engine:
            create_tables(engine, base=DatasetORMBase)
            item = Instruction(index=12, name="new", json_content=payload)
            assert item.json_content is payload
            item.update_md5()
            assert (
                item.json_content
                == ROInstructionContent.model_validate(variant).to_json_dict()
            )
            with Session(engine) as session:
                session.add(item)
                session.commit()
                found = Instruction.query_by_content_with_md5(
                    session, "new", item.json_content
                )
                assert found.index == 12
                assert (
                    Instruction.query_by_content_with_md5(
                        session, "new", variant
                    )
                    is None
                )
                before = deepcopy(variant)
                duplicate = InstructionData("new", variant).make_transient_orm(
                    DatasetIndexState(last_instruction_idx=12), session
                )
                assert duplicate.index == 12
                assert duplicate.json_content == item.json_content
                assert variant == before
                assert (
                    Instruction.query_by_content_with_md5(
                        session, "other", variant
                    )
                    is None
                )
        # Same-boundary different records are ordered content, not a set.
        records = [
            {
                "frame_index_begin": 0,
                "frame_index_end": 2,
                "descriptions": [goal],
            }
            for goal in ["a", "b"]
        ]
        first = Instruction(
            name="x", json_content={**payload, "subtasks": records}
        )
        second = Instruction(
            name="x", json_content={**payload, "subtasks": records[::-1]}
        )
        assert first.update_md5() != second.update_md5()
        old = Instruction(
            name="x",
            json_content=ROInstructionContent(
                schema_version="1.0"
            ).to_json_dict(),
        )
        new = Instruction(
            name="x", json_content=ROInstructionContent().to_json_dict()
        )
        assert old.update_md5() != new.update_md5()

    @pytest.fixture
    def mutable_instruction_payload(self):
        return ROInstructionContent(
            descriptions=["a", "z"],
            subtask_type="frame",
            subtasks=[
                {
                    "frame_index_begin": 0,
                    "frame_index_end": 3,
                    "descriptions": ["first"],
                    "extras": {"source": {"values": [1]}},
                }
            ],
            extras={"nested": {"values": [1]}},
        ).to_json_dict()

    @pytest.mark.parametrize(
        "mutation", ["clean", "descriptions", "subtasks", "extras"]
    )
    def test_calculate_md5_is_pure_after_nested_changes(
        self, tmp_path, mutable_instruction_payload, mutation
    ):
        producer = mutable_instruction_payload
        producer_before = deepcopy(producer)
        with create_temp_engine(
            dir=str(tmp_path), drivername="sqlite"
        ) as engine:
            create_tables(engine, base=DatasetORMBase)
            with Session(engine) as session:
                item = Instruction(index=1, name="goal", json_content=producer)
                item.update_md5()
                stored_content = deepcopy(item.json_content)
                stored_md5 = item.md5
                session.add(item)
                session.commit()
            with Session(engine) as session:
                item = session.get(Instruction, 1)
                content = item.json_content
                if mutation == "descriptions":
                    content["descriptions"].append(" middle ")
                elif mutation == "subtasks":
                    content["subtasks"][0]["descriptions"].append(" another ")
                elif mutation == "extras":
                    content["extras"]["nested"]["values"].append(2)
                before = deepcopy(content)
                subtask = content["subtasks"][0]
                assert item not in session.dirty
                assert not session.is_modified(item, include_collections=True)
                digest = item.calculate_md5()
                assert (
                    digest
                    == hashlib.md5(
                        ("goal" + json.dumps(before, sort_keys=True)).encode()
                    ).digest()
                )
                assert item.json_content is content
                assert item.json_content["subtasks"][0] is subtask
                assert item.json_content == before
                assert item.md5 == stored_md5
                assert item not in session.dirty
                assert not session.is_modified(item, include_collections=True)
                assert (digest == stored_md5) == (mutation == "clean")
                session.commit()
            with Session(engine) as session:
                stored = session.get(Instruction, 1)
                assert stored.json_content == stored_content
                assert stored.md5 == stored_md5
            assert producer == producer_before

    @pytest.mark.parametrize(
        "mutation",
        [
            "canonical_descriptions",
            "noncanonical_descriptions",
            "canonical_subtasks",
            "noncanonical_subtasks",
            "nested_extras",
        ],
    )
    def test_update_md5_persists_nested_content_and_digest_together(
        self, tmp_path, mutable_instruction_payload, mutation
    ):
        producer = mutable_instruction_payload
        producer_before = deepcopy(producer)
        with create_temp_engine(
            dir=str(tmp_path), drivername="sqlite"
        ) as engine:
            create_tables(engine, base=DatasetORMBase)
            with Session(engine) as session:
                item = Instruction(index=1, name="goal", json_content=producer)
                item.update_md5()
                old_md5 = item.md5
                session.add(item)
                session.commit()
            with Session(engine) as session:
                item = session.get(Instruction, 1)
                content = item.json_content
                if mutation == "canonical_descriptions":
                    content["descriptions"].insert(1, "middle")
                elif mutation == "noncanonical_descriptions":
                    content["descriptions"].extend([" middle ", "a", ""])
                elif mutation in (
                    "canonical_subtasks",
                    "noncanonical_subtasks",
                ):
                    content["subtasks"].append(
                        {
                            "frame_index_begin": 4,
                            "frame_index_end": 6,
                            "descriptions": ["later"],
                        }
                    )
                    if mutation == "noncanonical_subtasks":
                        content["subtasks"].reverse()
                        content["subtasks"][0]["descriptions"][:] = [
                            " z ",
                            "later",
                            "later",
                            "",
                        ]
                else:
                    content["extras"]["nested"]["values"].append(2)
                    content["subtasks"][0]["extras"]["source"][
                        "values"
                    ].append(2)
                assert (
                    item not in session.dirty
                )  # Plain JSON misses nested edits.
                expected = ROInstructionContent.model_validate(
                    deepcopy(content)
                ).to_json_dict()
                digest = item.update_md5()
                assert item.json_content == expected
                assert digest == item.md5
                assert digest != old_md5
                session.commit()
            with Session(engine) as session:
                stored = session.get(Instruction, 1)
                assert stored.json_content == expected
                expected_digest = hashlib.md5(
                    (
                        "goal"
                        + json.dumps(stored.json_content, sort_keys=True)
                    ).encode("utf-8")
                ).digest()
                assert stored.md5 == expected_digest == digest
                assert (
                    Instruction.query_by_content_with_md5(
                        session, "goal", expected
                    ).index
                    == 1
                )
            assert producer == producer_before

    @pytest.mark.parametrize("content", [None, {}, {"goal": ["one"]}])
    def test_hash_methods_preserve_untagged_content(self, tmp_path, content):
        original = deepcopy(content)
        with create_temp_engine(
            dir=str(tmp_path), drivername="sqlite"
        ) as engine:
            create_tables(engine, base=DatasetORMBase)
            with Session(engine) as session:
                item = Instruction(
                    index=1, name="legacy", json_content=content
                )
                digest = item.update_md5()
                session.add(item)
                session.commit()
            with Session(engine) as session:
                item = session.get(Instruction, 1)
                assert item.calculate_md5() == digest
                assert item not in session.dirty
                assert item.json_content == original
                assert item.update_md5() == digest
                session.commit()
            with Session(engine) as session:
                stored = session.get(Instruction, 1)
                assert stored.json_content == original
                assert stored.md5 == digest

    @pytest.mark.parametrize("invalid", ["partial", "unknown", "invalid"])
    def test_hash_is_raw_and_only_known_write_rejects_inplace_changes(
        self, tmp_path, mutable_instruction_payload, invalid
    ):
        with create_temp_engine(
            dir=str(tmp_path), drivername="sqlite"
        ) as engine:
            create_tables(engine, base=DatasetORMBase)
            with Session(engine) as session:
                item = Instruction(
                    index=1,
                    name="goal",
                    json_content=mutable_instruction_payload,
                )
                item.update_md5()
                session.add(item)
                session.commit()
                content = item.json_content
                original_md5 = item.md5
                if invalid == "partial":
                    del content["schema_version"]
                elif invalid == "unknown":
                    content["schema_version"] = "9.9"
                else:
                    content["subtasks"][0]["descriptions"] = []
                before = deepcopy(content)
                assert item not in session.dirty
                digest = item.calculate_md5()
                assert (
                    digest
                    == hashlib.md5(
                        ("goal" + json.dumps(before, sort_keys=True)).encode()
                    ).digest()
                )
                assert item not in session.dirty
                assert item.json_content is content
                assert item.json_content == before
                assert item.md5 == original_md5
                if invalid == "invalid":
                    with pytest.raises(MetadataSchemaError):
                        item.update_md5()
                    assert item.md5 == original_md5
                else:
                    assert item.update_md5() == digest
                    assert item.json_content == before
                with pytest.raises(MetadataSchemaError):
                    parse_registered_metadata_schema(before)
                session.rollback()

    def test_fixed_raw_legacy_sqlite_identity(self, tmp_path):
        raw = {
            "schema_id": "robo_orchard.instruction_content",
            "schema_version": "1.0",
            "descriptions": [" Z ", "A", " Z "],
            "subtasks": [
                {
                    "frame_index_begin": 8,
                    "frame_index_end": 10,
                    "descriptions": [],
                },
                {
                    "frame_index_begin": 1,
                    "frame_index_end": 3,
                    "descriptions": [" b ", "a"],
                },
            ],
        }
        digest = bytes.fromhex("e2da3b864f36ff1031639a75c23fa539")
        with create_temp_engine(
            dir=str(tmp_path), drivername="sqlite"
        ) as engine:
            create_tables(engine, base=DatasetORMBase)
            with engine.begin() as conn:
                conn.execute(
                    Instruction.__table__.insert().values(
                        index=7, name="legacy", json_content=raw, md5=digest
                    )
                )
            with Session(engine) as session:
                item = session.get(Instruction, 7)
                assert item.json_content == raw
                assert item.md5 == digest
                assert item.calculate_md5() == digest
                assert item not in session.dirty
                assert item.update_md5() == digest
                assert item.json_content == raw
                assert (
                    parse_registered_metadata_schema(
                        item.json_content
                    ).to_json_dict()
                    != raw
                )
                assert (
                    Instruction.query_by_content_with_md5(
                        session, "legacy", raw
                    ).index
                    == 7
                )
                session.commit()
            with Session(engine) as session:
                item = session.get(Instruction, 7)
                assert item.json_content == raw
                assert item.md5 == digest

    @pytest.mark.parametrize(
        "payload",
        [
            {"schema_id": "robo_orchard.instruction_content"},
            {
                "schema_id": "robo_orchard.instruction_content",
                "schema_version": "9.9",
            },
            {
                "schema_id": "robo_orchard.instruction_content",
                "schema_version": "1.1",
                "subtasks": [{}],
            },
        ],
    )
    def test_raw_construction_and_explicit_known_write_validation(
        self, payload
    ):
        item = Instruction(name="bad", json_content=payload)
        assert item.json_content is payload
        item.calculate_md5()
        with pytest.raises(MetadataSchemaError):
            parse_registered_metadata_schema(payload)
        if payload.get("schema_version") == "1.1":
            with pytest.raises(MetadataSchemaError):
                item.update_md5()
        else:
            item.update_md5()
            assert item.json_content == payload

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            [],
            ["x", {"value": 1}],
            "text",
            7,
            True,
            {"schema_id": "robo_orchard.instruction_content"},
            {
                "schema_id": "robo_orchard.instruction_content",
                "schema_version": "9.9",
            },
            {
                "schema_id": "robo_orchard.instruction_content",
                "schema_version": "1.0",
                "subtasks": [{}],
            },
            {"schema_id": "other.schema", "schema_version": "1.1"},
        ],
    )
    def test_opaque_write_dto_preserves_json_and_raw_identity(
        self, tmp_path, payload
    ):
        before = deepcopy(payload)
        with create_temp_engine(
            dir=str(tmp_path), drivername="sqlite"
        ) as engine:
            create_tables(engine, base=DatasetORMBase)
            with Session(engine) as session:
                dto = InstructionData("opaque", payload)
                state = DatasetIndexState()
                item = dto.make_transient_orm(state, session)
                session.add(item)
                session.commit()
                duplicate = dto.make_transient_orm(state, session)
                assert duplicate.index == item.index
                assert duplicate.json_content == before == payload
                assert Instruction.__table__.c.json_content.nullable
                assert Instruction.__version__ == "0.0.1"

    def test_robot_preview_query_and_existing_task_query(
        self, db_engine, monkeypatch
    ):
        def no_update(*args, **kwargs):
            pytest.fail("content query invoked a write operation")

        monkeypatch.setattr(Robot, "update_md5", no_update)
        with Session(db_engine) as session:
            robot = session.get(Robot, 0)
            task = session.get(Task, 0)
            task_digest = task.md5
            assert (
                Robot.query_by_content_with_md5(
                    session,
                    name=robot.name,
                    content=robot.content,
                    content_format=robot.content_format,
                ).index
                == robot.index
            )
            assert (
                Task.query_by_content_with_md5(
                    session,
                    name=task.name,
                    description=task.description,
                    info=task.info,
                ).index
                == task.index
            )
            assert task.md5 == task_digest and task not in session.dirty


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
