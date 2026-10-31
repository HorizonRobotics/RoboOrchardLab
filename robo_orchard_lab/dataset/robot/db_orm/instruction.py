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

from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.types import BIGINT, BLOB, JSON, TEXT

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    register_table_mapper,
)
from robo_orchard_lab.dataset.robot.db_orm.md5 import MD5FieldMixin
from robo_orchard_lab.dataset.robot.db_orm.upgrade import (
    TableUpgradeRegistry,
    Version,
)
from robo_orchard_lab.dataset.robot.metadata_schema import (
    parse_registered_metadata_schema,
)

__all__ = ["Instruction"]


@register_table_mapper
class Instruction(DatasetORMBase, MD5FieldMixin["Instruction"]):
    """Store arbitrary instruction JSON with raw read and hash identity.

    Construction, assignment and query preserve stored JSON. Explicit writes
    through update_md5 or InstructionData format known instruction 1.1 only.
    """

    __tablename__ = "instruction"
    __version__ = "0.0.1"

    index: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=False
    )
    name: Mapped[str | None] = mapped_column(TEXT)

    json_content: Mapped[Any | None] = mapped_column(JSON)

    md5: Mapped[bytes] = mapped_column(BLOB(length=16), index=True)

    @staticmethod
    def normalize_content(json_content: Any) -> Any:
        """Format exact known instruction 1.1 for an explicit new write.

        Other JSON, including unknown/partial tags and raw 1.0, is opaque.
        Malformed exact 1.1 fails under the current interval schema; historical
        point content is not implicitly migrated. Explicit parsers stay strict.
        """
        if (
            not isinstance(json_content, dict)
            or json_content.get("schema_id")
            != "robo_orchard.instruction_content"
            or json_content.get("schema_version") != "1.1"
        ):
            return json_content
        return parse_registered_metadata_schema(
            json_content,
            allowed_schema_ids={"robo_orchard.instruction_content"},
        ).to_json_dict()

    @classmethod
    def md5_content_fields(cls) -> list[str]:
        exclude_keys = ["index", "md5"]
        ret = []
        for key in cls.__table__.columns.keys():
            if key not in exclude_keys:
                ret.append(key)
        return ret

    def calculate_md5(self) -> bytes:
        """Hash current raw JSON without parsing or changing ORM state.

        Preserve the original name + sorted JSON algorithm and empty-value
        rule, including arbitrary historical schema-looking values.
        """
        content_str = (
            json.dumps(self.json_content, sort_keys=True)
            if self.json_content
            else ""
        )
        return hashlib.md5(
            f"{self.name}{content_str}".encode("utf-8")
        ).digest()

    def update_md5(self) -> bytes:
        """Format known 1.1 content and synchronize its digest for writing.

        Call after nested in-place JSON changes as well as after assignment.
        Explicit dirty marking persists even already-canonical nested edits;
        plain JSON has no automatic deep mutation tracking. Other JSON stays
        opaque, including raw 1.0. Read/hash/query never invoke this operation.
        """
        self.json_content = self.normalize_content(self.json_content)
        flag_modified(self, "json_content")
        return super().update_md5()

    @staticmethod
    def query_by_content_with_md5(
        session: Session, name: str | None, json_content: Any
    ) -> Instruction | None:
        """Query exact raw JSON identity without formatting or parsing it."""

        return MD5FieldMixin[Instruction].query_by_content_with_md5(
            session=session,
            cls=Instruction,
            name=name,
            json_content=json_content,
        )


@TableUpgradeRegistry.register_upgrade(
    table_name=Instruction.__tablename__,
    from_version=None,
    to_version=Version("0.0.1"),
    from_orm_type=Instruction,
)
def upgrade_instruction_to_0_0_1(session: Session, row: dict) -> dict:
    """Upgrade an Instruction row to version 0.0.1.

    Since this is the initial version, no changes are made.

    Args:
        session (Session): The database session.
        row (dict): The Instruction row to upgrade.

    Returns:
        dict: The upgraded Instruction row.
    """
    return row
