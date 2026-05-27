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
from __future__ import annotations
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, field_serializer
from robo_orchard_core.utils.config import ClassType

from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    RobotDescriptionFormat,
    Task,
)

__all__ = [
    "FromEpisode",
    "FromEpisodeConfig",
    "FromInstruction",
    "FromInstructionConfig",
    "FromRobot",
    "FromRobotConfig",
    "FromTask",
    "FromTaskConfig",
]


TOrm = TypeVar("TOrm")
TModel = TypeVar("TModel", bound=BaseModel)


class _OrmMcapMetadata(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    @field_serializer("md5", when_used="json", check_fields=False)
    def _serialize_md5(self, value: bytes | None) -> str | None:
        return None if value is None else value.hex()


class _RobotMcapMetadata(_OrmMcapMetadata):
    index: int
    name: str
    content: str | None
    content_format: RobotDescriptionFormat | None
    md5: bytes


class _TaskMcapMetadata(_OrmMcapMetadata):
    index: int
    name: str
    description: str | None
    info: dict[str, Any] | None
    md5: bytes


class _InstructionMcapMetadata(_OrmMcapMetadata):
    index: int
    name: str | None
    json_content: dict[str, Any] | None
    md5: bytes


class _EpisodeMcapMetadata(_OrmMcapMetadata):
    index: int
    robot_index: int | None
    task_index: int | None
    prev_episode_index: int | None
    dataset_begin_index: int
    frame_num: int
    truncated: bool | None
    success: bool | None
    info: dict[str, Any] | None


class _FromOrmToMcapMetadata(
    MessageConverterStateless[TOrm, TModel],
    Generic[TOrm, TModel],
):
    """Convert a SQLAlchemy ORM row into its Pydantic MCAP payload."""

    def __init__(self, model_type: type[TModel]) -> None:
        self._model_type = model_type

    def convert(self, data: TOrm) -> TModel:
        return self._model_type.model_validate(data)


class FromRobot(_FromOrmToMcapMetadata[Robot, _RobotMcapMetadata]):
    cfg: FromRobotConfig

    def __init__(self, cfg: FromRobotConfig):
        self.cfg = cfg
        super().__init__(_RobotMcapMetadata)


class FromTask(_FromOrmToMcapMetadata[Task, _TaskMcapMetadata]):
    cfg: FromTaskConfig

    def __init__(self, cfg: FromTaskConfig):
        self.cfg = cfg
        super().__init__(_TaskMcapMetadata)


class FromInstruction(
    _FromOrmToMcapMetadata[Instruction, _InstructionMcapMetadata]
):
    cfg: FromInstructionConfig

    def __init__(self, cfg: FromInstructionConfig):
        self.cfg = cfg
        super().__init__(_InstructionMcapMetadata)


class FromEpisode(_FromOrmToMcapMetadata[Episode, _EpisodeMcapMetadata]):
    cfg: FromEpisodeConfig

    def __init__(self, cfg: FromEpisodeConfig):
        self.cfg = cfg
        super().__init__(_EpisodeMcapMetadata)


class FromRobotConfig(MessageConverterConfig[FromRobot]):
    class_type: ClassType[FromRobot] = FromRobot


class FromTaskConfig(MessageConverterConfig[FromTask]):
    class_type: ClassType[FromTask] = FromTask


class FromInstructionConfig(MessageConverterConfig[FromInstruction]):
    class_type: ClassType[FromInstruction] = FromInstruction


class FromEpisodeConfig(MessageConverterConfig[FromEpisode]):
    class_type: ClassType[FromEpisode] = FromEpisode
