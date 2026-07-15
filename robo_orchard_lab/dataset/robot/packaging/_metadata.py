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

"""Metadata payload objects used while packaging RoboOrchard datasets."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session, make_transient
from typing_extensions import deprecated

from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    RobotDescriptionFormat,
    Task,
)

if TYPE_CHECKING:
    from robo_orchard_lab.dataset.robot.packaging._writer import (
        DatasetIndexState,
    )

__all__ = [
    "EpisodeData",
    "EpisodeMetaORM",
    "InstructionData",
    "RobotData",
    "TaskData",
]


@dataclass
class EpisodeData:
    """Data for an episode information which is used for packaging."""

    frame_num: int | None = None
    """The total number of frames in the episode.

    No need to set this field during packaging, it will be updated
    automatically during packaging.
    """
    prev_episode_index: int | None = None
    """The index of the previous episode in the dataset."""
    dataset_begin_index: int | None = None
    """The index of the first dataset item in this episode.

    No need to set this field during packaging, it will be updated
    automatically during packaging.
    """

    truncated: bool | None = None
    """Whether the episode was truncated."""

    success: bool | None = None
    """Whether the episode was successful."""

    info: dict[str, Any] | None = None
    """Additional information about the episode."""

    index: int | None = None
    """The target episode index.

    If None, packaging assigns the next contiguous target episode index. If
    set, it must match the next target episode index. This must be set when
    prev_episode_index is set so the previous episode reference is explicitly
    in target-index space.
    """


@dataclass
class RobotData:
    """Data for a robot information which is used for packaging."""

    name: str
    """The name of the robot."""

    content: str | None
    content_format: RobotDescriptionFormat | None

    @classmethod
    def from_orm(cls, orm_robot: Robot) -> RobotData:
        """Create a RobotData instance from an ORM Robot instance."""
        return cls(
            name=orm_robot.name,
            content=orm_robot.content,
            content_format=orm_robot.content_format,
        )

    def make_transient_orm(
        self, index_state: "DatasetIndexState", session: Session | None
    ) -> Robot:
        """Create a transient ORM instance of the robot.

        If session is provided, it will check if the robot already exists
        in the database using its name and URDF content. If it exists, it
        will return the existing robot instance, otherwise it will create a
        new transient instance with the next index.

        If session is None, it will create a new transient instance with the
        next index without checking the database.

        """

        def make_new():
            ret = Robot(index=index_state.last_robot_idx + 1, **self.__dict__)
            ret.update_md5()
            return ret

        if session is not None:
            ret = Robot.query_by_content_with_md5(session, **self.__dict__)
            if ret is not None:
                # Make sure the robot is not transient
                make_transient(ret)
                return ret
            else:
                return make_new()
        else:
            return make_new()

    @property
    @deprecated("Use 'content' and 'content_format' instead.")  # type: ignore
    def urdf_content(self) -> str | None:
        """The URDF content of the robot."""
        if self.content_format == RobotDescriptionFormat.URDF:
            return self.content
        else:
            return None

    @urdf_content.setter
    @deprecated("Use 'content' and 'content_format' instead.")  # type: ignore
    def urdf_content(self, value: str | None):
        """Set the URDF content of the robot."""
        self.content = value
        self.content_format = RobotDescriptionFormat.URDF


@dataclass
class TaskData:
    """Data for a task information which is used for packaging."""

    name: str
    """The name of the task."""
    description: str | None = None
    """The description of the task."""
    info: dict[str, Any] | None = None
    """Additional task information.

    ``info`` is part of the task identity. Empty dictionaries are normalized
    to ``None`` so empty info and missing info do not create distinct tasks.
    Non-empty values must be strict JSON objects: keys must be strings, values
    must be JSON-compatible, and floats must be finite. Invalid values are
    rejected before ORM storage or task identity hashing.
    """

    def make_transient_orm(
        self, index_state: "DatasetIndexState", session: Session | None
    ) -> Task:
        """Create a transient ORM instance of the task.

        If session is provided, it will check if the task already exists
        in the database using its semantic identity. If it exists, it will
        return the existing task instance, otherwise it will create a new
        transient instance with the next index.

        If session is None, it will create a new transient instance with the
        next index without checking the database.

        """
        info = Task.normalize_info(self.info)

        def make_new():
            ret = Task(
                index=index_state.last_task_idx + 1,
                name=self.name,
                description=self.description,
                info=info,
            )
            ret.update_md5()
            return ret

        if session is not None:
            ret = Task.query_by_content_with_md5(
                session,
                name=self.name,
                description=self.description,
                info=info,
            )
            if ret is not None:
                make_transient(ret)
                return ret
            else:
                return make_new()
        else:
            return make_new()


@dataclass
class EpisodeMetaORM:
    """Metadata for an episode in a RoboOrchard dataset."""

    episode: Episode
    robot: Robot | None = None
    task: Task | None = None


@dataclass
class InstructionData:
    """Data for an instruction information which is used for packaging."""

    name: str | None
    """The name of the instruction."""
    json_content: dict[str, Any] | None
    """The content of the instruction, typically a dictionary with keys like
    'instruction', 'robot', and 'task'.
    """

    def make_transient_orm(
        self,
        index_state: "DatasetIndexState",
        session: Session | None = None,
    ) -> Instruction:
        """Create a transient ORM instance of the instruction.

        If session is provided, it will check if the instruction already exists
        in the database using its name and JSON content. If it exists, it
        will return the existing instruction instance, otherwise it will
        create a new transient instance with the next index.

        If session is None, it will create a new transient instance with the
        next index without checking the database.

        """

        def make_new():
            ret = Instruction(
                index=index_state.last_instruction_idx + 1, **self.__dict__
            )
            ret.update_md5()
            return ret

        if session is not None:
            ret = Instruction.query_by_content_with_md5(
                session, **self.__dict__
            )
            if ret is not None:
                make_transient(ret)
                return ret
            else:
                return make_new()
        else:
            return make_new()
