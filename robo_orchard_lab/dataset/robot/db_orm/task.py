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
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Mapped, Session, mapped_column, validates
from sqlalchemy.types import BLOB, INTEGER, JSON, Text

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    DeprecatedDatasetORMBase,
    register_table_mapper,
)
from robo_orchard_lab.dataset.robot.db_orm.md5 import MD5FieldMixin
from robo_orchard_lab.dataset.robot.db_orm.upgrade import (
    TableUpgradeRegistry,
    Version,
)
from robo_orchard_lab.dataset.robot.metadata_schema import (
    validate_json_value,
)

__all__ = ["Task"]


@register_table_mapper
class Task(DatasetORMBase, MD5FieldMixin["Task"]):
    """ORM model for a task in a RoboOrchard dataset."""

    __tablename__ = "task"
    __version__ = "0.0.2"

    index: Mapped[int] = mapped_column(
        INTEGER, primary_key=True, autoincrement=False
    )
    name: Mapped[str] = mapped_column(Text, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    info: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    md5: Mapped[bytes] = mapped_column(BLOB(length=16), index=True)

    @classmethod
    def md5_content_fields(cls) -> list[str]:
        exclude_keys = ["index", "md5"]
        ret = []
        for key in cls.__table__.columns.keys():
            if key not in exclude_keys:
                ret.append(key)
        return ret

    @staticmethod
    def normalize_info(
        info: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Normalize task info before storage, hashing, or lookup."""
        if info is None or info == {}:
            return None
        if not isinstance(info, dict):
            raise TypeError("task info must be a JSON object or None.")
        validate_json_value(info)
        return info

    @validates("info")
    def _normalize_info_assignment(
        self, _key: str, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return self.normalize_info(value)

    def calculate_md5(self) -> bytes:
        """Generate a unique MD5 hash for the task identity.

        ``info`` is part of the task semantic identity.
        """
        self.info = self.normalize_info(self.info)
        content = {
            field_name: getattr(self, field_name)
            for field_name in self.md5_content_fields()
        }
        content_bytes = json.dumps(
            content,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.md5(content_bytes).digest()

    @staticmethod
    def query_by_content_with_md5(
        session: Session,
        name: str,
        description: str | None,
        info: dict[str, Any] | None = None,
    ) -> Task | None:
        """Query a task by its semantic content identity."""

        normalized_info = Task.normalize_info(info)
        task = Task(
            name=name,
            description=description,
            info=normalized_info,
        )
        task.update_md5()
        stmt = select(Task).where(Task.md5 == task.md5)
        for result in session.execute(stmt).scalars():
            if (
                result.name == name
                and result.description == description
                and Task.normalize_info(result.info) == normalized_info
            ):
                return result
        return None


class TaskDeprecatedVersion1(
    DeprecatedDatasetORMBase, MD5FieldMixin["TaskDeprecatedVersion1"]
):
    """Deprecated ORM model for Task version 0.0.1."""

    __tablename__ = Task.__tablename__
    __version__ = "0.0.1"

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

    def calculate_md5(self) -> bytes:
        """Generate the legacy Task 0.0.1 MD5 hash."""
        content_str = self.description if self.description else ""
        combined_str = f"{self.name}{content_str}".encode("utf-8")
        return hashlib.md5(combined_str).digest()


@TableUpgradeRegistry.register_upgrade(
    table_name=Task.__tablename__,
    from_version=None,
    to_version=Version("0.0.1"),
    from_orm_type=TaskDeprecatedVersion1,
)
def upgrade_task_to_0_0_1(session: Session, row: dict) -> dict:
    """Upgrade a Task row to version 0.0.1.

    Since this is the initial version, no changes are made.

    Args:
        session (Session): The database session.
        row (dict): The Task row to upgrade.

    Returns:
        dict: The upgraded Task row.
    """
    return row


@TableUpgradeRegistry.register_upgrade(
    table_name=Task.__tablename__,
    from_version=Version("0.0.1"),
    to_version=Version("0.0.2"),
    from_orm_type=TaskDeprecatedVersion1,
)
def upgrade_task_to_0_0_2(session: Session, row: dict) -> dict:
    """Upgrade a Task row to version 0.0.2."""
    row["info"] = None
    task = Task(
        index=row["index"],
        name=row["name"],
        description=row["description"],
        info=None,
    )
    task.update_md5()
    row["md5"] = task.md5
    return row
