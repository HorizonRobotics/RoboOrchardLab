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
import hashlib

from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import BLOB, INTEGER, Text

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    register_table_mapper,
)
from robo_orchard_lab.dataset.robot.db_orm.md5 import MD5FieldMixin

__all__ = ["Task"]


@register_table_mapper
class Task(DatasetORMBase, MD5FieldMixin["Task"]):
    """ORM model for a task in a RoboOrchard dataset."""

    __tablename__ = "task"

    index: Mapped[int] = mapped_column(
        INTEGER, primary_key=True, autoincrement=False
    )
    name: Mapped[str] = mapped_column(Text, index=True)
    description: Mapped[str | None] = mapped_column(Text)

    md5: Mapped[bytes] = mapped_column(BLOB(length=16), index=True)

    @classmethod
    def md5_content_fields(cls) -> list[str]:
        exclude_keys = ["index", "md5"]
        ret = []
        for key in cls.__table__.columns.keys():
            if key not in exclude_keys:
                ret.append(key)
        return ret

    def update_md5(self) -> bytes:
        """Generate a unique MD5 hash for the instruction content.

        The MD5 hash is generated from the JSON content and name.
        """
        content_str = self.description if self.description else ""
        combined_str = f"{self.name}{content_str}".encode("utf-8")
        ret = hashlib.md5(combined_str).digest()
        if self.md5 != ret:
            self.md5 = ret
        return self.md5

    @staticmethod
    def query_by_content_with_md5(
        session: Session, name: str, description: str | None
    ) -> Task | None:
        """Query a robot by its name and URDF content."""

        return MD5FieldMixin[Task].query_by_content_with_md5(
            session, Task, name=name, description=description
        )
