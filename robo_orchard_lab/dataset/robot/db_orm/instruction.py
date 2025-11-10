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
import json

from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import BIGINT, BLOB, JSON, TEXT

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    register_table_mapper,
)
from robo_orchard_lab.dataset.robot.db_orm.md5 import MD5FieldMixin

__all__ = ["Instruction"]


@register_table_mapper
class Instruction(DatasetORMBase, MD5FieldMixin["Instruction"]):
    __tablename__ = "instruction"

    index: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=False
    )
    name: Mapped[str | None] = mapped_column(TEXT)

    json_content: Mapped[dict | None] = mapped_column(JSON)

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
        content_str = (
            json.dumps(self.json_content, sort_keys=True)
            if self.json_content
            else ""
        )
        combined_str = f"{self.name}{content_str}".encode("utf-8")
        ret = hashlib.md5(combined_str).digest()
        if self.md5 != ret:
            self.md5 = ret
        return self.md5

    @staticmethod
    def query_by_content_with_md5(
        session: Session, name: str | None, json_content: dict | None
    ) -> Instruction | None:
        """Query a robot by its name and URDF content."""

        return MD5FieldMixin[Instruction].query_by_content_with_md5(
            session=session,
            cls=Instruction,
            name=name,
            json_content=json_content,
        )
