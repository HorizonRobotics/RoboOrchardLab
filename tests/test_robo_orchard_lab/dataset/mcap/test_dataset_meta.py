# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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
import json
from typing import Any, cast

from pydantic import BaseModel

from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    dataset_meta,
)
from robo_orchard_lab.dataset.experimental.mcap.writer import (
    create_default_to_mcap_message_factory,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    RobotDescriptionFormat,
    Task,
)


def _make_episode(**kwargs: Any) -> Episode:
    episode_cls = cast(Any, Episode)
    return episode_cls(**kwargs)


def test_dataset_meta_public_interface_hides_payload_models() -> None:
    assert all(
        not name.endswith("McapMetadata") for name in dataset_meta.__all__
    )
    assert "FromOrmToMcapMetadata" not in dataset_meta.__all__


def test_converters_validate_from_orm_and_dump_md5_as_hex() -> None:
    robot = Robot(
        index=1,
        name="robot",
        content="<robot/>",
        content_format=RobotDescriptionFormat.URDF,
        md5=bytes.fromhex("000102030405060708090a0b0c0d0e0f"),
    )
    task = Task(
        index=2,
        name="pick",
        description="Pick the cube.",
        info={"family": "libero"},
        md5=bytes.fromhex("101112131415161718191a1b1c1d1e1f"),
    )
    instruction = Instruction(
        index=3,
        name="instruction",
        json_content={"descriptions": ["pick"]},
        md5=bytes.fromhex("202122232425262728292a2b2c2d2e2f"),
    )
    episode = _make_episode(
        index=4,
        robot_index=1,
        task_index=2,
        prev_episode_index=None,
        dataset_begin_index=10,
        frame_num=5,
        truncated=False,
        success=True,
        info={"fps": 30},
    )

    assert json.loads(
        dataset_meta.FromRobotConfig()().convert(robot).model_dump_json()
    ) == {
        "index": 1,
        "name": "robot",
        "content": "<robot/>",
        "content_format": "urdf",
        "md5": "000102030405060708090a0b0c0d0e0f",
    }
    assert json.loads(
        dataset_meta.FromTaskConfig()().convert(task).model_dump_json()
    ) == {
        "index": 2,
        "name": "pick",
        "description": "Pick the cube.",
        "info": {"family": "libero"},
        "md5": "101112131415161718191a1b1c1d1e1f",
    }
    assert json.loads(
        dataset_meta.FromInstructionConfig()()
        .convert(instruction)
        .model_dump_json()
    ) == {
        "index": 3,
        "name": "instruction",
        "json_content": {"descriptions": ["pick"]},
        "md5": "202122232425262728292a2b2c2d2e2f",
    }
    episode_payload = dataset_meta.FromEpisodeConfig()().convert(episode)
    assert episode_payload.model_dump() == {
        "index": 4,
        "robot_index": 1,
        "task_index": 2,
        "prev_episode_index": None,
        "dataset_begin_index": 10,
        "frame_num": 5,
        "truncated": False,
        "success": True,
        "info": {"fps": 30},
    }


def test_episode_converter_returns_metadata_without_timestamp() -> None:
    episode = _make_episode(
        index=7,
        robot_index=None,
        task_index=None,
        prev_episode_index=6,
        dataset_begin_index=100,
        frame_num=20,
        truncated=None,
        success=None,
        info={"source": "fixture"},
    )

    converted = dataset_meta.FromEpisodeConfig()().convert(episode)

    assert isinstance(converted, BaseModel)
    assert converted.model_dump() == {
        "index": 7,
        "robot_index": None,
        "task_index": None,
        "prev_episode_index": 6,
        "dataset_begin_index": 100,
        "frame_num": 20,
        "truncated": None,
        "success": None,
        "info": {"source": "fixture"},
    }
    assert "timestamp" not in converted.model_fields
    assert "log_time" not in converted.model_fields


def test_default_factory_converts_orm_metadata_same_topic() -> None:
    episode = _make_episode(
        index=8,
        robot_index=1,
        task_index=2,
        prev_episode_index=None,
        dataset_begin_index=12,
        frame_num=3,
        truncated=True,
        success=False,
        info=None,
    )
    factory = create_default_to_mcap_message_factory()
    converter = factory.create_converter("/metadata/episode", episode)

    assert converter is not None
    topic_map = converter.convert(
        StampedMessage(data=episode, log_time=123, pub_time=456)
    )

    assert list(topic_map) == ["/metadata/episode"]
    [message] = topic_map["/metadata/episode"]
    assert isinstance(message.data, BaseModel)
    assert message.data.model_dump() == (
        dataset_meta.FromEpisodeConfig()().convert(episode).model_dump()
    )
    assert message.log_time == 123
    assert message.pub_time == 456
