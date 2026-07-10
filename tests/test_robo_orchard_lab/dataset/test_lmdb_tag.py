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
from pathlib import Path

import pytest

from robo_orchard_lab.dataset.horizon_manipulation.tools.lmdb_tag import (
    tag_error_uuids,
    tag_error_uuids_from_json,
)


class FakeLmdb:
    records = {}

    def __init__(self, *_args, **_kwargs):
        pass

    def keys(self):
        return [*self.records.keys(), "__len__"]

    def get(self, episode_idx):
        return self.records[episode_idx]

    def write(self, episode_idx, record):
        self.records[episode_idx] = record

    def close(self):
        pass


def test_tag_error_uuids_updates_matching_index_records(
    tmp_path: Path,
    monkeypatch,
):
    FakeLmdb.records = {
        0: {
            "uuid": "task/user/episode_0",
            "num_steps": 10,
            "task_name": "task",
            "user": "user",
            "error": False,
        },
        1: {
            "uuid": "task/user/episode_1",
            "num_steps": 20,
            "task_name": "task",
            "user": "user",
            "error": False,
        },
    }
    log_updates = []
    monkeypatch.setattr(
        "robo_orchard_lab.dataset.horizon_manipulation.tools.lmdb_tag.Lmdb",
        FakeLmdb,
    )
    monkeypatch.setattr(
        "robo_orchard_lab.dataset.horizon_manipulation.tools.lmdb_tag."
        "append_error_tag_history",
        lambda lmdb_path, affected_uuids: log_updates.append(
            (lmdb_path, affected_uuids)
        ),
    )

    affected_uuids = tag_error_uuids(
        tmp_path,
        ["task/user/episode_0", "task/user/missing"],
    )

    assert affected_uuids == ["task/user/episode_0"]
    assert FakeLmdb.records[0]["error"] is True
    assert FakeLmdb.records[1]["error"] is False
    assert log_updates == [(tmp_path, ["task/user/episode_0"])]


def test_tag_error_uuids_from_json_applies_each_package(
    tmp_path: Path,
    monkeypatch,
):
    config_path = tmp_path / "tags.json"
    lmdb_a = tmp_path / "lmdb_a"
    lmdb_b = tmp_path / "lmdb_b"
    config_path.write_text(
        json.dumps(
            {
                str(lmdb_a): ["uuid-a"],
                str(lmdb_b): ["uuid-b", "uuid-c"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "robo_orchard_lab.dataset.horizon_manipulation.tools.lmdb_tag."
        "tag_error_uuids",
        lambda lmdb_path, uuids: [f"matched:{uuid}" for uuid in uuids],
    )

    result = tag_error_uuids_from_json(config_path)

    assert result == {
        str(lmdb_a): ["matched:uuid-a"],
        str(lmdb_b): ["matched:uuid-b", "matched:uuid-c"],
    }


def test_tag_error_uuids_from_json_rejects_invalid_uuid_list(tmp_path: Path):
    config_path = tmp_path / "tags.json"
    config_path.write_text(
        json.dumps({str(tmp_path / "lmdb"): "not-a-list"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a list of strings"):
        tag_error_uuids_from_json(config_path)
