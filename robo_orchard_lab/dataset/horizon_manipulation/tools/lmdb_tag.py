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

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from robo_orchard_lab.dataset.horizon_manipulation.tools.lmdb_pack_log import (
    append_error_tag_history,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import BaseIndexData
from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb

logger = logging.getLogger(__name__)


def _load_tag_config(config_path: Path) -> dict[str, list[str]]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Tag config must be a JSON object.")

    config = {}
    for lmdb_path, uuids in payload.items():
        if not isinstance(lmdb_path, str) or not lmdb_path.strip():
            raise ValueError("Each tag config key must be a non-empty string.")
        if not isinstance(uuids, list) or not all(
            isinstance(uuid, str) for uuid in uuids
        ):
            raise ValueError(
                f"Tag config value for {lmdb_path} must be a list of strings."
            )
        config[lmdb_path] = uuids
    return config


def tag_error_uuids(
    lmdb_path: Path,
    error_uuid_list: list[str],
    *,
    lmdb_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Set ``error=True`` for matching index records in one LMDB package.

    Args:
        lmdb_path (Path): LMDB package root containing an ``index`` directory.
        error_uuid_list (list[str]): UUIDs to mark as error episodes.
        lmdb_kwargs (dict[str, Any], optional): Extra kwargs passed to LMDB.

    Returns:
        list[str]: UUIDs that were found and updated.
    """
    lmdb_kwargs = lmdb_kwargs or {}
    index_lmdb = Lmdb(
        str(lmdb_path / "index"),
        writable=True,
        commit_step=1,
        **lmdb_kwargs,
    )
    affected_uuids = []
    error_uuid_set = set(error_uuid_list)
    try:
        for episode_idx in index_lmdb.keys():
            if episode_idx == "__len__":
                continue
            meta = BaseIndexData.model_validate(index_lmdb.get(episode_idx))
            if meta.uuid in error_uuid_set:
                logger.info("Find error data: %s", meta.uuid)
                meta.error = True
                affected_uuids.append(meta.uuid)
                index_lmdb.write(episode_idx, meta.model_dump())
    finally:
        index_lmdb.close()

    if affected_uuids:
        append_error_tag_history(lmdb_path, affected_uuids)
    return affected_uuids


def tag_error_uuids_from_json(config_path: Path) -> dict[str, list[str]]:
    """Apply error tags from a JSON mapping of LMDB path to UUID list."""
    config = _load_tag_config(config_path)
    result = {}
    for lmdb_path, uuids in config.items():
        affected_uuids = tag_error_uuids(Path(lmdb_path), uuids)
        result[lmdb_path] = affected_uuids
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set error tags for LMDB packages from a JSON mapping of "
            "lmdb_path to UUID list."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help='JSON file: {"/path/to/lmdb": ["uuid", ...], ...}',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = tag_error_uuids_from_json(args.config)
    total = sum(len(uuids) for uuids in result.values())
    print(
        f"Updated {total} error-tagged episodes in {len(result)} package(s)."
    )


if __name__ == "__main__":
    main()
