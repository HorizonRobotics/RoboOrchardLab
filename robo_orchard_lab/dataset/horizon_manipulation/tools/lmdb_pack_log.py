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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from robo_orchard_lab.dataset.horizon_manipulation.tools.utils import (
    normalize_embodiment_items,
    normalize_embodiment_name,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import BaseIndexData
from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb

SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_NAME = "pack_log.json"
LMDB_DIR_NAMES = ("index", "meta", "image", "depth")
UNKNOWN_VALUE = "unknown"
INDEX_DATA_FILE = "data.mdb"


def _local_datetime_from_timestamp(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp).astimezone()


def _format_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _normal_value(value: Any) -> str:
    if value is None:
        return UNKNOWN_VALUE
    value = str(value).strip()
    if not value:
        return UNKNOWN_VALUE
    return value


def _normal_embodiment_value(value: Any) -> str:
    if isinstance(value, list):
        normalized = normalize_embodiment_items(value)
    else:
        normalized = normalize_embodiment_name(value)
    return normalized if normalized else UNKNOWN_VALUE


def _record_embodiment(record: BaseIndexData) -> str:
    embodiment = _normal_embodiment_value(record.embodiment)
    if embodiment != UNKNOWN_VALUE:
        return embodiment

    metas = getattr(record, "metas", None)
    if not isinstance(metas, dict):
        return UNKNOWN_VALUE
    return _normal_embodiment_value(metas.get("embodiment"))


def _new_counter() -> dict[str, int]:
    return {"num_episodes": 0, "num_frames": 0}


def _sorted_dict(
    data: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {key: data[key] for key in sorted(data)}


def collect_index_records(lmdb_path: Path) -> list[BaseIndexData]:
    """Read episode records from the package index LMDB.

    Args:
        lmdb_path (Path): LMDB package root containing an ``index`` directory.

    Returns:
        list[BaseIndexData]: Validated index records.
    """
    index_path = lmdb_path / "index"
    if not index_path.is_dir():
        raise FileNotFoundError(
            f"index LMDB directory not found: {index_path}"
        )

    index_lmdb = Lmdb(str(index_path), writable=False)
    try:
        records = []
        for episode_idx in index_lmdb.keys():
            if episode_idx == "__len__":
                continue
            raw_record = index_lmdb.get(episode_idx)
            records.append(BaseIndexData.model_validate(raw_record))
        return records
    finally:
        index_lmdb.close()


def build_pack_log(
    records: list[BaseIndexData],
    *,
    created_at: datetime,
    updated_at: datetime,
) -> dict[str, Any]:
    """Build the parseable pack log document from index records.

    Args:
        records (list[BaseIndexData]): Episode index records.
        created_at (datetime): Timestamp used for the initial pack event.
        updated_at (datetime): Timestamp used for the generated document.

    Returns:
        dict[str, Any]: JSON-serializable pack log document.
    """
    by_user_name = defaultdict(_new_counter)
    by_task_name = defaultdict(_new_counter)
    by_embodiment = defaultdict(_new_counter)
    error_counter = _new_counter()

    user_names = set()
    task_names = set()
    embodiments = set()
    num_frames = 0

    for record in records:
        user_name = _normal_value(
            record.user or getattr(record, "user_name", None)
        )
        task_name = _normal_value(record.task_name)
        embodiment = _record_embodiment(record)
        frames = int(record.num_steps)

        user_names.add(user_name)
        task_names.add(task_name)
        embodiments.add(embodiment)
        num_frames += frames

        for breakdown, key in (
            (by_user_name, user_name),
            (by_task_name, task_name),
            (by_embodiment, embodiment),
        ):
            breakdown[key]["num_episodes"] += 1
            breakdown[key]["num_frames"] += frames

        if record.error:
            error_counter["num_episodes"] += 1
            error_counter["num_frames"] += frames

    summary = {
        "num_episodes": len(records),
        "num_frames": num_frames,
        "user_names": sorted(user_names),
        "task_names": sorted(task_names),
        "embodiments": sorted(embodiments),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _format_datetime(created_at),
        "updated_at": _format_datetime(updated_at),
        "summary": summary,
        "breakdown": {
            "by_user_name": _sorted_dict(dict(by_user_name)),
            "by_task_name": _sorted_dict(dict(by_task_name)),
            "by_embodiment": _sorted_dict(dict(by_embodiment)),
        },
        "tags": {
            "error": error_counter,
        },
        "history": [
            {
                "timestamp": _format_datetime(created_at),
                "type": "initial_pack",
                "message": (
                    "Created LMDB package. Timestamp reconstructed from "
                    "LMDB directory mtimes."
                ),
                "summary_after": {
                    "num_episodes": summary["num_episodes"],
                    "num_frames": summary["num_frames"],
                },
            }
        ],
    }


def infer_created_at(lmdb_path: Path) -> datetime:
    """Infer initial package time from LMDB directory modification times."""
    mtimes = []
    for dirname in LMDB_DIR_NAMES:
        path = lmdb_path / dirname
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    if not mtimes:
        raise FileNotFoundError(
            f"No LMDB directories found under package root: {lmdb_path}"
        )
    return _local_datetime_from_timestamp(min(mtimes))


def infer_index_updated_at(lmdb_path: Path) -> datetime:
    """Infer the latest index update time from the LMDB data file."""
    index_path = lmdb_path / "index"
    index_data_path = index_path / INDEX_DATA_FILE
    if index_data_path.exists():
        return _local_datetime_from_timestamp(index_data_path.stat().st_mtime)
    if index_path.exists():
        return _local_datetime_from_timestamp(index_path.stat().st_mtime)
    return datetime.now().astimezone()


def _read_pack_log(output_path: Path) -> dict[str, Any] | None:
    if not output_path.exists():
        return None
    with output_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_pack_log_json(output_path: Path, pack_log: dict[str, Any]) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(pack_log, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _created_at_from_existing_log(
    existing_log: dict[str, Any] | None,
    lmdb_path: Path,
) -> datetime:
    if existing_log is None:
        return infer_created_at(lmdb_path)
    created_at = existing_log.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        return infer_created_at(lmdb_path)
    return datetime.fromisoformat(created_at)


def _history_from_existing_log(existing_log: dict[str, Any] | None) -> list:
    if not isinstance(existing_log, dict):
        return []
    history = existing_log.get("history", [])
    return history if isinstance(history, list) else []


def write_pack_log(
    lmdb_path: Path,
    *,
    output_name: str = DEFAULT_OUTPUT_NAME,
    overwrite: bool = False,
) -> Path:
    """Generate and write ``pack_log.json`` for an LMDB package."""
    output_path = lmdb_path / output_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. "
            "Re-run with --overwrite to replace."
        )

    records = collect_index_records(lmdb_path)
    pack_log = build_pack_log(
        records,
        created_at=infer_created_at(lmdb_path),
        updated_at=infer_index_updated_at(lmdb_path),
    )

    _write_pack_log_json(output_path, pack_log)
    return output_path


def append_error_tag_history(
    lmdb_path: Path,
    affected_uuids: list[str],
    *,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    """Update pack log statistics and append an error-tag history event.

    Args:
        lmdb_path (Path): LMDB package root containing ``index``.
        affected_uuids (list[str]): Episode UUIDs whose error tag was set.
        output_name (str, optional): Pack log filename. Default is
            ``pack_log.json``.

    Returns:
        Path: Written pack log path.
    """
    output_path = lmdb_path / output_name
    existing_log = _read_pack_log(output_path)
    existing_history = _history_from_existing_log(existing_log)

    updated_at = infer_index_updated_at(lmdb_path)
    records = collect_index_records(lmdb_path)
    pack_log = build_pack_log(
        records,
        created_at=_created_at_from_existing_log(existing_log, lmdb_path),
        updated_at=updated_at,
    )
    if existing_history:
        pack_log["history"] = existing_history

    error_summary = pack_log["tags"]["error"]
    pack_log["history"].append(
        {
            "timestamp": _format_datetime(updated_at),
            "type": "tag_update",
            "message": "Marked error episodes.",
            "tag": "error",
            "value": True,
            "affected_uuids": sorted(set(affected_uuids)),
            "summary_after": {
                "error_episodes": error_summary["num_episodes"],
                "error_frames": error_summary["num_frames"],
            },
        }
    )

    _write_pack_log_json(output_path, pack_log)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a parseable pack_log.json for an LMDB package."
    )
    parser.add_argument(
        "--lmdb-path",
        type=Path,
        required=True,
        help=(
            "LMDB package root containing index/meta/image/depth directories."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output JSON filename. Default is {DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing pack log file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_pack_log(
        args.lmdb_path,
        output_name=args.output_name,
        overwrite=args.overwrite,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
