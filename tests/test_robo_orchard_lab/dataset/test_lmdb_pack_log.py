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

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from robo_orchard_lab.dataset.horizon_manipulation.tools.lmdb_pack_log import (
    append_error_tag_history,
    build_pack_log,
    infer_created_at,
    infer_index_updated_at,
    write_pack_log,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import BaseIndexData


def _record(
    uuid: str,
    *,
    num_steps: int,
    task_name: str | None = None,
    user: str | None = None,
    embodiment: str | None = None,
    metas: dict | None = None,
    error: bool = False,
) -> BaseIndexData:
    data = {
        "uuid": uuid,
        "num_steps": num_steps,
        "error": error,
    }
    if task_name is not None:
        data["task_name"] = task_name
    if user is not None:
        data["user"] = user
    if embodiment is not None:
        data["embodiment"] = embodiment
    if metas is not None:
        data["metas"] = metas
    return BaseIndexData.model_validate(data)


def _iso_from_mtime(mtime: int) -> str:
    return (
        datetime.fromtimestamp(mtime)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _touch_index_data_mdb(lmdb_path: Path, mtime: int) -> Path:
    index_path = lmdb_path / "index"
    index_path.mkdir(exist_ok=True)
    index_data_path = index_path / "data.mdb"
    index_data_path.touch()
    os.utime(index_data_path, (mtime, mtime))
    return index_data_path


def test_build_pack_log_aggregates_summary_breakdowns_and_error_tag():
    created_at = datetime(2026, 6, 29, 20, 14, tzinfo=timezone.utc)
    updated_at = datetime(2026, 7, 9, 12, 30, tzinfo=timezone.utc)
    records = [
        _record(
            "task_a/alice/episode_0",
            num_steps=10,
            task_name="task_a",
            user="alice",
            embodiment="piper_x",
        ),
        _record(
            "task_a/bob/episode_1",
            num_steps=20,
            task_name="task_a",
            user="bob",
            embodiment="piper_x",
            error=True,
        ),
        _record(
            "task_b/alice/episode_2",
            num_steps=30,
            task_name="task_b",
            user="alice",
            embodiment="hexfellow",
            error=True,
        ),
    ]

    pack_log = build_pack_log(
        records,
        created_at=created_at,
        updated_at=updated_at,
    )

    assert pack_log["schema_version"] == "1.0"
    assert pack_log["summary"] == {
        "num_episodes": 3,
        "num_frames": 60,
        "user_names": ["alice", "bob"],
        "task_names": ["task_a", "task_b"],
        "embodiments": ["hexfellow", "piper_x"],
    }
    assert pack_log["breakdown"]["by_user_name"] == {
        "alice": {"num_episodes": 2, "num_frames": 40},
        "bob": {"num_episodes": 1, "num_frames": 20},
    }
    assert pack_log["breakdown"]["by_task_name"] == {
        "task_a": {"num_episodes": 2, "num_frames": 30},
        "task_b": {"num_episodes": 1, "num_frames": 30},
    }
    assert pack_log["breakdown"]["by_embodiment"] == {
        "hexfellow": {"num_episodes": 1, "num_frames": 30},
        "piper_x": {"num_episodes": 2, "num_frames": 30},
    }
    assert pack_log["tags"]["error"] == {
        "num_episodes": 2,
        "num_frames": 50,
    }
    assert pack_log["history"] == [
        {
            "timestamp": "2026-06-29T20:14:00+00:00",
            "type": "initial_pack",
            "message": (
                "Created LMDB package. Timestamp reconstructed from "
                "LMDB directory mtimes."
            ),
            "summary_after": {
                "num_episodes": 3,
                "num_frames": 60,
            },
        }
    ]


def test_build_pack_log_uses_unknown_for_missing_grouping_values():
    created_at = datetime(2026, 6, 29, 20, 14, tzinfo=timezone.utc)
    records = [_record("task/unknown/episode_0", num_steps=7)]

    pack_log = build_pack_log(
        records,
        created_at=created_at,
        updated_at=created_at,
    )

    assert pack_log["summary"]["user_names"] == ["unknown"]
    assert pack_log["summary"]["task_names"] == ["unknown"]
    assert pack_log["summary"]["embodiments"] == ["unknown"]
    assert pack_log["breakdown"]["by_user_name"] == {
        "unknown": {"num_episodes": 1, "num_frames": 7}
    }


def test_build_pack_log_uses_embodiment_from_metas_fallback():
    created_at = datetime(2026, 6, 29, 20, 14, tzinfo=timezone.utc)
    records = [
        _record(
            "task/user/episode_0",
            num_steps=7,
            task_name="task",
            user="user",
            metas={"embodiment": ["piper_x"]},
        )
    ]

    pack_log = build_pack_log(
        records,
        created_at=created_at,
        updated_at=created_at,
    )

    assert pack_log["summary"]["embodiments"] == ["piper_x"]
    assert pack_log["breakdown"]["by_embodiment"] == {
        "piper_x": {"num_episodes": 1, "num_frames": 7}
    }


def test_append_error_tag_history_refreshes_stats_and_preserves_history(
    tmp_path: Path,
    monkeypatch,
):
    index_updated_mtime = 1_783_584_000
    _touch_index_data_mdb(tmp_path, index_updated_mtime)
    existing_log = {
        "schema_version": "1.0",
        "created_at": "2026-06-29T20:14:00+00:00",
        "updated_at": "2026-07-09T12:00:00+00:00",
        "history": [
            {
                "timestamp": "2026-06-29T20:14:00+00:00",
                "type": "initial_pack",
                "message": "Created LMDB package.",
                "summary_after": {
                    "num_episodes": 2,
                    "num_frames": 30,
                },
            }
        ],
    }
    (tmp_path / "pack_log.json").write_text(
        json.dumps(existing_log),
        encoding="utf-8",
    )
    records = [
        _record(
            "task/user/episode_0",
            num_steps=10,
            task_name="task",
            user="user",
            embodiment="piper_x",
            error=True,
        ),
        _record(
            "task/user/episode_1",
            num_steps=20,
            task_name="task",
            user="user",
            embodiment="piper_x",
        ),
    ]
    monkeypatch.setattr(
        "robo_orchard_lab.dataset.horizon_manipulation.tools."
        "lmdb_pack_log.collect_index_records",
        lambda _lmdb_path: records,
    )

    output_path = append_error_tag_history(
        tmp_path,
        ["task/user/episode_0"],
    )

    assert output_path == tmp_path / "pack_log.json"
    pack_log = json.loads(output_path.read_text())
    expected_updated_at = _iso_from_mtime(index_updated_mtime)
    assert pack_log["created_at"] == existing_log["created_at"]
    assert pack_log["updated_at"] == expected_updated_at
    assert pack_log["summary"]["num_episodes"] == 2
    assert pack_log["summary"]["num_frames"] == 30
    assert pack_log["tags"]["error"] == {
        "num_episodes": 1,
        "num_frames": 10,
    }
    assert [item["type"] for item in pack_log["history"]] == [
        "initial_pack",
        "tag_update",
    ]
    tag_event = pack_log["history"][-1]
    assert tag_event["timestamp"] == expected_updated_at
    assert tag_event["tag"] == "error"
    assert tag_event["value"] is True
    assert tag_event["affected_uuids"] == ["task/user/episode_0"]
    assert tag_event["summary_after"] == {
        "error_episodes": 1,
        "error_frames": 10,
    }


def test_infer_index_updated_at_uses_index_data_mdb_mtime(tmp_path: Path):
    mtime = 1_783_584_000
    _touch_index_data_mdb(tmp_path, mtime)

    updated_at = infer_index_updated_at(tmp_path)

    assert updated_at.timestamp() == mtime
    assert updated_at.tzinfo is not None


def test_write_pack_log_uses_index_data_mtime_for_updated_at(
    tmp_path: Path,
    monkeypatch,
):
    for dirname in ["index", "meta", "image", "depth"]:
        (tmp_path / dirname).mkdir()
    created_mtime = 1_782_739_200
    updated_mtime = 1_783_584_000
    os.utime(tmp_path / "meta", (created_mtime, created_mtime))
    _touch_index_data_mdb(tmp_path, updated_mtime)
    monkeypatch.setattr(
        "robo_orchard_lab.dataset.horizon_manipulation.tools."
        "lmdb_pack_log.collect_index_records",
        lambda _lmdb_path: [
            _record(
                "task/user/episode_0",
                num_steps=7,
                task_name="task",
                user="user",
                embodiment="piper_x",
            )
        ],
    )

    output_path = write_pack_log(tmp_path)

    pack_log = json.loads(output_path.read_text())
    assert pack_log["updated_at"] == _iso_from_mtime(updated_mtime)


def test_infer_created_at_uses_earliest_lmdb_directory_mtime(tmp_path: Path):
    index = tmp_path / "index"
    meta = tmp_path / "meta"
    image = tmp_path / "image"
    for path in [index, meta, image]:
        path.mkdir()

    old_mtime = 1_782_739_200
    newer_mtime = 1_782_742_800
    for path, mtime in [
        (index, newer_mtime),
        (meta, old_mtime),
        (image, newer_mtime),
    ]:
        os.utime(path, (mtime, mtime))

    created_at = infer_created_at(tmp_path)

    assert created_at.timestamp() == old_mtime
    assert created_at.tzinfo is not None
