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
from __future__ import annotations

from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import BIGINT, INTEGER

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    register_table_mapper,
)
from robo_orchard_lab.dataset.robot.db_orm.robot import Robot
from robo_orchard_lab.dataset.robot.db_orm.task import Task

__all__ = ["Episode"]


@register_table_mapper
class Episode(DatasetORMBase):
    """ORM model for an episode in a RoboOrchard dataset."""

    __tablename__ = "episode"

    index: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=False
    )
    """The unique index of the episode."""

    dataset_begin_index: Mapped[int] = mapped_column(BIGINT)
    """The index of the first dataset item in this episode.

    Can be -1 if not available. User should fix it after all
    processings are done.
    """

    frame_num: Mapped[int] = mapped_column(INTEGER)

    robot_index: Mapped[int | None] = mapped_column(
        INTEGER, ForeignKey(Robot.index), default=None, index=True
    )
    task_index: Mapped[int | None] = mapped_column(
        INTEGER, ForeignKey(Task.index), default=None, index=True
    )
    prev_episode_index: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey(f"{__tablename__}.index"), default=None, index=True
    )
    """The episode index of the previous episode.
    This is used to link episodes together in a sequence.
    """

    @declared_attr
    def robot(cls) -> Mapped[Robot | None]:
        return relationship(
            "Robot",
            backref=cls.__tablename__,
            foreign_keys=[cls.robot_index],  # type: ignore
        )

    @declared_attr
    def task(cls) -> Mapped[Task | None]:
        return relationship(
            "Task",
            backref=cls.__tablename__,
            foreign_keys=[cls.task_index],  # type: ignore
        )

    @declared_attr
    def prev_episode(cls) -> Mapped[Episode | None]:
        return relationship(
            "Episode",
            back_populates="next_episode",
            remote_side=[cls.index],  # type: ignore
        )

    @declared_attr
    def next_episode(cls) -> Mapped[list[Episode]]:
        return relationship(
            "Episode",
            back_populates="prev_episode",
        )
