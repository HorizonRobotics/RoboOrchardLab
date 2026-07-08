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
import io
import json
import math
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from robo_orchard_lab.dataset.horizon_manipulation.tools import (
    app as app_module,
)
from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
    EPISODE_CACHE,
    PROJECT_ROOT,
    FilterOptions,
    app,
    build_summary,
    build_summary_with_filters,
    date_prefix_matches,
    format_duration_hours,
    get_cache_file_path,
    get_cache_key,
    get_cached_episode_records,
    get_env_path,
    get_robo_orchard_lab_dir,
    has_cached_episode_records,
    infer_day_from_episode_dir,
    iter_episode_dirs,
    load_cache_from_disk,
    maybe_build_initial_loading_summary,
    merge_records_for_date_prefixes,
    refresh_cached_episode_records_for_date_prefixes,
    scan_episode_records_parallel,
    seconds_until_next_refresh,
)


def create_episode(path: Path, mtime: int, nanoseconds: int = 0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    (path / f"{path.name}_0.mcap").write_bytes(b"mcap")
    (path / "episode_meta.json").write_text("{}\n", encoding="utf-8")
    (path / "metadata.yaml").write_text(
        f"rosbag2_bagfile_information:\n  duration:\n    nanoseconds: {nanoseconds}\n",  # noqa: E501
        encoding="utf-8",
    )

    os.utime(path, (mtime, mtime))


def test_get_env_path_supports_relative_paths(monkeypatch):
    monkeypatch.setenv("RELATIVE_PATH_TEST", "relative/path")
    assert (
        get_env_path("RELATIVE_PATH_TEST", PROJECT_ROOT / "fallback")
        == PROJECT_ROOT / "relative/path"
    )


def test_get_env_path_keeps_absolute_paths(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ABSOLUTE_PATH_TEST", str(tmp_path / "absolute-path"))
    assert (
        get_env_path("ABSOLUTE_PATH_TEST", PROJECT_ROOT / "fallback")
        == tmp_path / "absolute-path"
    )


def test_get_robo_orchard_lab_dir_returns_absolute_path(monkeypatch):
    monkeypatch.setenv("ROBO_ORCHARD_LAB_DIR", "../../../../robo_orchard_lab")

    expected_path = (PROJECT_ROOT / "../../../../robo_orchard_lab").resolve()
    assert get_robo_orchard_lab_dir() == str(expected_path)


def test_infer_day_from_episode_dir_prefers_episode_id_date(tmp_path: Path):
    episode_dir = tmp_path / "alice" / "pick" / "episode_2026_03_12-10_00_00"
    create_episode(episode_dir, 1710115200, 0)

    assert infer_day_from_episode_dir(episode_dir) == "2026-03-12"


def test_infer_day_from_episode_dir_falls_back_to_mtime_for_nonstandard_name(
    tmp_path: Path,
):
    episode_dir = tmp_path / "alice" / "pick" / "custom_episode_name"
    create_episode(episode_dir, 1710115200, 0)

    assert infer_day_from_episode_dir(episode_dir) == "2024-03-11"


def test_scan_and_summary(tmp_path: Path):
    create_episode(
        tmp_path / "alice" / "pick" / "ep001", 1710115200, 3_600_000_000_000
    )  # 2024-03-11
    create_episode(
        tmp_path / "alice" / "pick" / "ep002", 1710115200, 1_800_000_000_000
    )
    create_episode(
        tmp_path / "alice" / "place" / "ep003", 1710201600, 7_200_000_000_000
    )  # 2024-03-12
    create_episode(
        tmp_path / "bob" / "pick" / "ep101", 1710201600, 1_800_000_000_000
    )

    records = scan_episode_records_parallel(tmp_path)
    summary = build_summary(records, tmp_path)

    assert summary["total_episodes"] == 4
    assert summary["totals"]["by_day"]["2024-03-11"] == 2
    assert summary["totals"]["by_day"]["2024-03-12"] == 2
    assert summary["totals"]["by_user"]["alice"] == 3
    assert summary["totals"]["by_task"]["pick"] == 3
    assert math.isclose(summary["total_hours"], 4.0, rel_tol=1e-9)
    assert math.isclose(
        summary["totals"]["hours_by_day"]["2024-03-11"], 1.5, rel_tol=1e-9
    )
    assert math.isclose(
        summary["totals"]["hours_by_user"]["alice"], 3.5, rel_tol=1e-9
    )
    assert summary["total_duration_text"] == "4 hours 0 mins"
    assert (
        summary["totals"]["duration_text_by_day"]["2024-03-11"]
        == "1 hours 30 mins"
    )
    assert len(summary["episodes"]) == 4
    assert summary["episodes"][0]["episode_id"] == "ep101"
    assert summary["episodes"][0]["duration_text"] == "30 mins"

    alice = next(
        user for user in summary["users"] if user["user_name"] == "alice"
    )
    pick = next(task for task in alice["tasks"] if task["task_name"] == "pick")
    assert pick["day_counts"]["2024-03-11"] == 2
    assert math.isclose(pick["day_hours"]["2024-03-11"], 1.5, rel_tol=1e-9)
    assert pick["day_duration_text"]["2024-03-11"] == "1 hours 30 mins"


def test_filter_by_user_task_and_date_prefix(tmp_path: Path):
    create_episode(
        tmp_path / "alice" / "pick" / "episode_2024_03_11-10_00_00",
        1710115200,
        3_600_000_000_000,
    )
    create_episode(
        tmp_path / "alice" / "place" / "episode_2024_03_12-11_30_00",
        1710201600,
        1_800_000_000_000,
    )
    create_episode(
        tmp_path / "bob" / "pick" / "episode_2024_03_12-14_45_00",
        1710201600,
        1_800_000_000_000,
    )

    records = scan_episode_records_parallel(tmp_path)

    user_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(user_name="alice")
    )
    assert user_summary["total_episodes"] == 2
    assert set(user_summary["totals"]["by_user"].keys()) == {"alice"}

    task_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(task_name="pick")
    )
    assert task_summary["total_episodes"] == 2
    assert task_summary["totals"]["by_task"] == {"pick": 2}
    assert math.isclose(
        task_summary["totals"]["hours_by_task"]["pick"], 1.5, rel_tol=1e-9
    )

    date_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(date_prefix="2024_03_12")
    )
    assert date_summary["total_episodes"] == 2
    assert date_summary["totals"]["by_day"] == {"2024-03-12": 2}

    month_prefix_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(date_prefix="2024_03")
    )
    assert month_prefix_summary["total_episodes"] == 3

    multi_user_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(user_name="alice,bob")
    )
    assert multi_user_summary["total_episodes"] == 3

    multi_task_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(task_name="place,pick")
    )
    assert multi_task_summary["total_episodes"] == 3

    multi_date_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(date_prefix="2024_03_11,2024_03_12")
    )
    assert multi_date_summary["total_episodes"] == 3

    partial_multi_date_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(date_prefix="2024_03_11,2024_03")
    )
    assert partial_multi_date_summary["total_episodes"] == 3

    combo_summary = build_summary_with_filters(
        records,
        tmp_path,
        FilterOptions(
            user_name="bob", task_name="pick", date_prefix="2024_03"
        ),
    )
    assert combo_summary["total_episodes"] == 1
    assert combo_summary["users"][0]["user_name"] == "bob"


def test_filter_by_regex_date_prefix(tmp_path: Path):
    create_episode(
        tmp_path / "alice" / "pick" / "episode_2024_03_11-10_00_00",
        1710115200,
        3_600_000_000_000,
    )
    create_episode(
        tmp_path / "alice" / "place" / "episode_2024_03_12-11_30_00",
        1710201600,
        1_800_000_000_000,
    )
    create_episode(
        tmp_path / "bob" / "pick" / "episode_2024_03_13-14_45_00",
        1710288000,
        1_800_000_000_000,
    )

    records = scan_episode_records_parallel(tmp_path)
    regex_summary = build_summary_with_filters(
        records, tmp_path, FilterOptions(date_prefix=r"2024_03_1[12]")
    )

    assert regex_summary["total_episodes"] == 2
    assert regex_summary["totals"]["by_day"] == {
        "2024-03-11": 1,
        "2024-03-12": 1,
    }


def test_multi_embodiment_episode_is_treated_as_one_combination(
    tmp_path: Path,
):
    episode_dir = (
        tmp_path / "jinqi.huo" / "fold_t_shirt" / "episode_2026_06_30-18_01_34"
    )
    create_episode(episode_dir, 1782838894, 90_000_000_000)
    (episode_dir / "episode_meta.json").write_text(
        json.dumps(
            {
                "metas": {
                    "embodiment": [
                        "piper_x",
                        "hexfellow",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    records = scan_episode_records_parallel(tmp_path)

    assert len(records) == 1
    assert records[0].embodiment == "hexfellow_piper_x"
    assert app_module.parse_record_embodiments("hexfellow,piper_x") == [
        "hexfellow_piper_x"
    ]

    summary = build_summary(records, tmp_path)
    assert summary["totals"]["by_embodiment"] == {"hexfellow_piper_x": 1}
    assert summary["episodes"][0]["embodiment"] == "hexfellow_piper_x"

    combo_filtered = build_summary_with_filters(
        records, tmp_path, FilterOptions(embodiment="hexfellow_piper_x")
    )
    single_label_filtered = build_summary_with_filters(
        records, tmp_path, FilterOptions(embodiment="piper_x")
    )
    assert combo_filtered["total_episodes"] == 1
    assert single_label_filtered["total_episodes"] == 0

    groups = app_module.split_records_by_combination(records)
    assert list(groups.keys()) == [
        ("jinqi.huo", "fold_t_shirt", "hexfellow_piper_x")
    ]

    selection = app_module.derive_submit_selection(records, FilterOptions())
    assert selection["embodiment"] == "hexfellow_piper_x"


def test_date_prefix_matcher_falls_back_for_invalid_regex():
    assert date_prefix_matches("2024_03_[-11_30_00", "2024_03_[")
    assert not date_prefix_matches("2024_04_[-11_30_00", "2024_03_[")


def test_iter_episode_dirs_skips_incomplete_episode_directories(
    tmp_path: Path,
):
    valid_episode = tmp_path / "alice" / "pick" / "episode_2026_03_13-10_00_00"
    create_episode(valid_episode, 1771000000, 90_000_000_000)

    missing_meta = tmp_path / "alice" / "pick" / "episode_2026_03_13-10_05_00"
    create_episode(missing_meta, 1771000100, 90_000_000_000)
    (missing_meta / "episode_meta.json").unlink()

    missing_yaml = tmp_path / "alice" / "pick" / "episode_2026_03_13-10_10_00"
    create_episode(missing_yaml, 1771000200, 90_000_000_000)
    (missing_yaml / "metadata.yaml").unlink()

    missing_mcap = tmp_path / "alice" / "pick" / "episode_2026_03_13-10_15_00"
    create_episode(missing_mcap, 1771000300, 90_000_000_000)
    for file_path in missing_mcap.glob("*.mcap"):
        file_path.unlink()

    episode_dirs = iter_episode_dirs(tmp_path)

    assert [episode_dir.name for _, _, episode_dir in episode_dirs] == [
        "episode_2026_03_13-10_00_00"
    ]


def test_iter_episode_dirs_filters_by_date_prefix_during_traversal(
    tmp_path: Path,
):
    create_episode(
        tmp_path / "alice" / "pick" / "episode_2026_03_12-10_00_00",
        1771000000,
        90_000_000_000,
    )
    create_episode(
        tmp_path / "alice" / "pick" / "episode_2026_03_13-10_00_00",
        1771086400,
        90_000_000_000,
    )

    episode_dirs = iter_episode_dirs(tmp_path, date_prefixes=["2026_03_12"])

    assert [episode_dir.name for _, _, episode_dir in episode_dirs] == [
        "episode_2026_03_12-10_00_00"
    ]


def test_iter_episode_dirs_skips_validation_for_non_matching_date_prefixes(
    tmp_path: Path, monkeypatch
):
    matching_episode = (
        tmp_path / "alice" / "pick" / "episode_2026_03_12-10_00_00"
    )
    create_episode(matching_episode, 1771000000, 90_000_000_000)

    skipped_episode = (
        tmp_path / "alice" / "pick" / "episode_2026_03_13-10_00_00"
    )
    create_episode(skipped_episode, 1771086400, 90_000_000_000)

    checked_episode_names: list[str] = []
    original_is_valid_episode_dir = app_module.is_valid_episode_dir

    def tracking_is_valid_episode_dir(episode_dir: Path) -> bool:
        checked_episode_names.append(episode_dir.name)
        return original_is_valid_episode_dir(episode_dir)

    monkeypatch.setattr(
        app_module, "is_valid_episode_dir", tracking_is_valid_episode_dir
    )

    episode_dirs = iter_episode_dirs(tmp_path, date_prefixes=["2026_03_12"])

    assert [episode_dir.name for _, _, episode_dir in episode_dirs] == [
        "episode_2026_03_12-10_00_00"
    ]
    assert checked_episode_names == ["episode_2026_03_12-10_00_00"]


def test_iter_episode_dirs_reports_progress_for_scan_task(tmp_path: Path):
    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        create_scan_task,
        get_scan_task,
    )

    for task_idx in range(2):
        create_episode(
            tmp_path
            / "alice"
            / f"task_{task_idx}"
            / f"episode_2026_03_13-10_0{task_idx}_00",
            1771000000 + task_idx,
            90_000_000_000,
        )

    task_id = create_scan_task(tmp_path)
    episode_dirs = iter_episode_dirs(tmp_path, task_id=task_id)
    task_state = get_scan_task(task_id)

    assert len(episode_dirs) == 2
    assert task_state is not None
    assert task_state["status"] == "running"
    assert task_state["progress"] == 100
    assert task_state["processed"] == 2
    assert task_state["total"] == 2
    assert "task directories checked" in task_state["message"]


def test_iter_episode_dirs_reports_listing_stage_message(
    tmp_path: Path, monkeypatch
):
    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        create_scan_task,
        get_scan_task,
    )

    create_episode(
        tmp_path / "alice" / "pick" / "episode_2026_03_13-10_00_00",
        1771000000,
        90_000_000_000,
    )
    create_episode(
        tmp_path / "alice" / "pick" / "episode_2026_03_13-10_00_01",
        1771000001,
        90_000_000_000,
    )

    task_id = create_scan_task(tmp_path)
    seen_messages: list[str] = []
    original_update_scan_task = app_module.update_scan_task

    def tracking_update_scan_task(inner_task_id: str, **kwargs: Any) -> None:
        message = kwargs.get("message")
        if inner_task_id == task_id and isinstance(message, str):
            seen_messages.append(message)
        original_update_scan_task(inner_task_id, **kwargs)

    monkeypatch.setattr(
        app_module, "update_scan_task", tracking_update_scan_task
    )

    episode_dirs = iter_episode_dirs(tmp_path, task_id=task_id)

    assert len(episode_dirs) == 2
    assert any(
        message.startswith("Listing episode directories:")
        for message in seen_messages
    )
    assert any(
        message.startswith("Validating episodes in")
        for message in seen_messages
    )

    task_state = get_scan_task(task_id)
    assert task_state is not None


def test_iter_episode_dirs_validation_progress_uses_matching_units(
    tmp_path: Path, monkeypatch
):
    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        create_scan_task,
    )

    for index in range(3):
        create_episode(
            tmp_path / "alice" / "pick" / f"episode_2026_03_13-10_00_0{index}",
            1771000000 + index,
            90_000_000_000,
        )

    task_id = create_scan_task(tmp_path)
    captured_updates: list[dict[str, Any]] = []
    original_update_scan_task = app_module.update_scan_task

    def tracking_update_scan_task(inner_task_id: str, **kwargs: Any) -> None:
        if inner_task_id == task_id and isinstance(kwargs.get("message"), str):
            captured_updates.append(dict(kwargs))
        original_update_scan_task(inner_task_id, **kwargs)

    monkeypatch.setattr(
        app_module, "update_scan_task", tracking_update_scan_task
    )

    iter_episode_dirs(tmp_path, task_id=task_id)

    validating_updates = [
        update
        for update in captured_updates
        if str(update.get("message", "")).startswith("Validating episodes in")
    ]

    assert validating_updates
    assert [update["processed"] for update in validating_updates] == [1, 2, 3]
    assert all(update["total"] == 3 for update in validating_updates)


def test_iter_episode_dirs_can_be_cancelled(tmp_path: Path, monkeypatch):
    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        create_scan_task,
        get_scan_task,
    )

    for task_idx in range(3):
        create_episode(
            tmp_path
            / "alice"
            / f"task_{task_idx}"
            / f"episode_2026_03_13-10_1{task_idx}_00",
            1771000100 + task_idx,
            90_000_000_000,
        )

    task_id = create_scan_task(tmp_path)
    original_is_valid_episode_dir = app_module.is_valid_episode_dir

    def cancelling_is_valid_episode_dir(episode_dir: Path) -> bool:
        app_module.update_scan_task(task_id, cancel_requested=True)
        return original_is_valid_episode_dir(episode_dir)

    monkeypatch.setattr(
        app_module, "is_valid_episode_dir", cancelling_is_valid_episode_dir
    )

    try:
        iter_episode_dirs(tmp_path, task_id=task_id)
        AssertionError(False, "Expected ScanCancelledError")
    except Exception as exc:
        assert exc.__class__.__name__ == "ScanCancelledError"

    task_state = get_scan_task(task_id)
    assert task_state is not None
    assert task_state["cancel_requested"] is True


def test_parallel_scan_skips_incomplete_episode_directories(tmp_path: Path):
    valid_episode = tmp_path / "alice" / "pick" / "episode_2026_03_13-10_00_00"
    create_episode(valid_episode, 1771000000, 90_000_000_000)

    invalid_episode = (
        tmp_path / "alice" / "pick" / "episode_2026_03_13-10_20_00"
    )
    create_episode(invalid_episode, 1771000400, 90_000_000_000)
    (invalid_episode / "episode_meta.json").unlink()

    records = scan_episode_records_parallel(tmp_path)

    assert [record.episode_id for record in records] == [
        "episode_2026_03_13-10_00_00"
    ]


def test_api_summary_supports_custom_data_root(tmp_path: Path):
    create_episode(
        tmp_path / "carol" / "stack" / "episode_2024_03_13-09_08_07",
        1710288000,
        5_400_000_000_000,
    )

    client = app.test_client()
    response = client.get(
        "/api/summary",
        query_string={
            "data_root": str(tmp_path),
            "user_name": "carol",
            "date_prefix": "2024_03",
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["base_path"] == str(tmp_path)
    assert data["filters"]["data_root"] == str(tmp_path)
    assert data["filters"]["user_name"] == "carol"
    assert data["total_episodes"] == 1
    assert math.isclose(data["total_hours"], 1.5, rel_tol=1e-9)
    assert data["users"][0]["user_name"] == "carol"
    assert data["episodes"][0]["task_name"] == "stack"
    assert data["episodes"][0]["download_url"].startswith("/api/download?")


def test_has_cached_episode_records_checks_memory_and_disk(tmp_path: Path):
    EPISODE_CACHE.clear()
    assert has_cached_episode_records(tmp_path) is False

    records, _ = get_cached_episode_records(tmp_path, refresh=True)
    assert records == []
    assert has_cached_episode_records(tmp_path) is True

    EPISODE_CACHE.clear()
    assert has_cached_episode_records(tmp_path) is True


def test_maybe_build_initial_loading_summary_without_cache(tmp_path: Path):
    EPISODE_CACHE.clear()
    cache_file = get_cache_file_path(get_cache_key(tmp_path))
    if cache_file.exists():
        cache_file.unlink()

    summary = maybe_build_initial_loading_summary({"data_root": str(tmp_path)})

    assert summary is not None
    assert summary["loading"]["is_loading"] is True
    assert summary["loading"]["scan_task_id"]
    assert summary["filters"]["data_root"] == str(tmp_path)
    assert summary["cache"]["cache_source"] == "loading"


def test_maybe_build_initial_loading_summary_skips_when_cache_exists(
    tmp_path: Path,
):
    EPISODE_CACHE.clear()
    get_cached_episode_records(tmp_path, refresh=True)

    summary = maybe_build_initial_loading_summary({"data_root": str(tmp_path)})

    assert summary is None


def test_index_renders_initial_loading_state_when_cache_missing(
    tmp_path: Path,
):
    EPISODE_CACHE.clear()
    cache_file = get_cache_file_path(get_cache_key(tmp_path))
    if cache_file.exists():
        cache_file.unlink()

    client = app.test_client()
    response = client.get("/", query_string={"data_root": str(tmp_path)})

    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "page-loading-state" in body
    assert '"is_loading": true' in body
    assert "Scanning Data" in body


def test_api_summary_paginates_episodes_by_default(tmp_path: Path):
    for index in range(25):
        create_episode(
            tmp_path
            / "alice"
            / "pick"
            / f"episode_2026_03_12-10_00_{index:02d}",
            1771000000 + index,
            90_000_000_000,
        )

    client = app.test_client()
    response = client.get(
        "/api/summary", query_string={"data_root": str(tmp_path)}
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["total_episodes"] == 25
    assert len(data["episodes"]) == 20
    assert data["pagination"]["page"] == 1
    assert data["pagination"]["page_size"] == 20
    assert data["pagination"]["total_pages"] == 2
    assert data["pagination"]["total_items"] == 25
    assert data["pagination"]["has_next"] is True
    assert data["pagination"]["start_index"] == 1
    assert data["pagination"]["end_index"] == 20


def test_api_summary_supports_episode_pagination_navigation(tmp_path: Path):
    for index in range(25):
        create_episode(
            tmp_path
            / "alice"
            / "pick"
            / f"episode_2026_03_12-10_00_{index:02d}",
            1771000000 + index,
            90_000_000_000,
        )

    client = app.test_client()
    response = client.get(
        "/api/summary",
        query_string={"data_root": str(tmp_path), "page": 2},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["total_episodes"] == 25
    assert len(data["episodes"]) == 5
    assert data["pagination"]["page"] == 2
    assert data["pagination"]["has_prev"] is True
    assert data["pagination"]["has_next"] is False
    assert data["pagination"]["start_index"] == 21
    assert data["pagination"]["end_index"] == 25


def test_api_summary_prefers_existing_cache_for_custom_data_root(
    tmp_path: Path,
):
    EPISODE_CACHE.clear()
    cache_key = get_cache_key(tmp_path)
    cache_file = get_cache_file_path(cache_key)
    if cache_file.exists():
        cache_file.unlink()

    create_episode(
        tmp_path / "iris" / "pick" / "episode_2026_03_12-10_00_00",
        1771000000,
        3_600_000_000_000,
    )
    records, _ = get_cached_episode_records(tmp_path, refresh=True)
    assert len(records) == 1

    create_episode(
        tmp_path / "iris" / "pick" / "episode_2026_03_12-10_10_00",
        1771000100,
        3_600_000_000_000,
    )

    EPISODE_CACHE.clear()
    client = app.test_client()
    response = client.get(
        "/api/summary", query_string={"data_root": str(tmp_path)}
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["cache"]["cache_hit"] is True
    assert data["cache"]["cache_source"] == "disk"
    assert data["total_episodes"] == 1
    assert [episode["episode_id"] for episode in data["episodes"]] == [
        "episode_2026_03_12-10_00_00"
    ]


def test_date_prefix_matches_episode_id_time_prefix(tmp_path: Path):
    create_episode(
        tmp_path / "zoe" / "arrange" / "episode_2026_03_03-17_00_40",
        1770000000,
        90_000_000_000,
    )

    records = scan_episode_records_parallel(tmp_path)

    assert (
        build_summary_with_filters(
            records, tmp_path, FilterOptions(date_prefix="2026")
        )["total_episodes"]
        == 1
    )
    assert (
        build_summary_with_filters(
            records, tmp_path, FilterOptions(date_prefix="2026_03")
        )["total_episodes"]
        == 1
    )
    assert (
        build_summary_with_filters(
            records, tmp_path, FilterOptions(date_prefix="2026_03_03")
        )["total_episodes"]
        == 1
    )
    assert (
        build_summary_with_filters(
            records, tmp_path, FilterOptions(date_prefix="2026_03_03-17")
        )["total_episodes"]
        == 1
    )
    assert (
        build_summary_with_filters(
            records, tmp_path, FilterOptions(date_prefix="2025_03")
        )["total_episodes"]
        == 0
    )


def test_download_episode_zip_returns_archive(tmp_path: Path):
    episode_dir = tmp_path / "ella" / "sort" / "ep777"
    create_episode(episode_dir, 1710288000, 5_400_000_000_000)
    (episode_dir / "notes.txt").write_text("hello zip\n", encoding="utf-8")

    client = app.test_client()
    response = client.get(
        "/api/download", query_string={"episode_path": str(episode_dir)}
    )

    assert response.status_code == 200
    assert response.mimetype == "application/zip"
    assert "ep777.zip" in response.headers["Content-Disposition"]

    archive = zipfile.ZipFile(io.BytesIO(response.data))
    assert set(archive.namelist()) == {
        "ep777_0.mcap",
        "episode_meta.json",
        "metadata.yaml",
        "notes.txt",
    }


def test_download_status_returns_file_count(tmp_path: Path):
    episode_dir = tmp_path / "frank" / "fold" / "ep888"
    create_episode(episode_dir, 1710288000, 5_400_000_000_000)
    (episode_dir / "notes.txt").write_text("progress\n", encoding="utf-8")

    client = app.test_client()
    response = client.get(
        "/api/download-status", query_string={"episode_path": str(episode_dir)}
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "preparing"
    assert data["file_count"] == 4
    assert "Packaging" in data["message"]


def test_scan_task_api_completes_and_reports_progress(tmp_path: Path):
    create_episode(
        tmp_path / "gina" / "stack" / "episode_2026_03_03-17_00_40",
        1770000000,
        90_000_000_000,
    )

    client = app.test_client()
    response = client.post(
        "/api/scan-tasks", json={"data_root": str(tmp_path)}
    )
    assert response.status_code == 202
    task = response.get_json()["tasks"][0]
    assert task["status"] in {"pending", "running", "completed"}

    task_id = task["task_id"]
    last = None
    for _ in range(30):
        last_response = client.get(f"/api/scan-tasks/{task_id}")
        assert last_response.status_code == 200
        last = last_response.get_json()
        if last["status"] == "completed":
            break
        time.sleep(0.05)

    assert last is not None
    assert last["status"] == "completed"
    assert last["progress"] == 100


def test_scan_task_can_be_cancelled(tmp_path: Path):
    for idx in range(10):
        create_episode(
            tmp_path
            / "henry"
            / "stack"
            / f"episode_2026_03_03-17_00_{idx:02d}",
            1770000000 + idx,
            90_000_000_000,
        )

    client = app.test_client()
    response = client.post(
        "/api/scan-tasks", json={"data_root": str(tmp_path)}
    )
    assert response.status_code == 202
    task_id = response.get_json()["tasks"][0]["task_id"]

    cancel_response = client.post(f"/api/scan-tasks/{task_id}/cancel")
    assert cancel_response.status_code == 200

    last = None
    for _ in range(30):
        last_response = client.get(f"/api/scan-tasks/{task_id}")
        assert last_response.status_code == 200
        last = last_response.get_json()
        if last["status"] == "cancelled":
            break
        time.sleep(0.05)

    assert last is not None
    assert last["status"] == "cancelled"


def test_scan_task_cancel_stops_slow_running_scan(tmp_path: Path, monkeypatch):
    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        read_duration_hours as original_read_duration_hours,
    )

    for idx in range(20):
        create_episode(
            tmp_path
            / "nina"
            / "stack"
            / f"episode_2026_03_12-13_00_{idx:02d}",
            1771001000 + idx,
            90_000_000_000,
        )

    def slow_read_duration_hours(episode_dir: Path) -> float:
        time.sleep(0.1)
        return original_read_duration_hours(episode_dir)

    monkeypatch.setattr(
        app_module, "read_duration_hours", slow_read_duration_hours
    )

    client = app.test_client()
    response = client.post(
        "/api/scan-tasks", json={"data_root": str(tmp_path)}
    )
    assert response.status_code == 202
    task_id = response.get_json()["tasks"][0]["task_id"]

    time.sleep(0.05)
    cancel_response = client.post(f"/api/scan-tasks/{task_id}/cancel")
    assert cancel_response.status_code == 200

    last = None
    for _ in range(40):
        last_response = client.get(f"/api/scan-tasks/{task_id}")
        assert last_response.status_code == 200
        last = last_response.get_json()
        if last["status"] == "cancelled":
            break
        time.sleep(0.05)

    assert last is not None
    assert last["status"] == "cancelled"
    assert last["processed"] < 20


def test_threaded_scan_with_progress_matches_parallel_scan(tmp_path: Path):
    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        create_scan_task,
        scan_episode_records_with_progress,
    )

    for user_idx in range(3):
        for task_idx in range(2):
            for episode_idx in range(4):
                create_episode(
                    tmp_path
                    / f"user_{user_idx}"
                    / f"task_{task_idx}"
                    / f"episode_2026_03_{user_idx + 1:02d}-10_00_{episode_idx:02d}",  # noqa: E501
                    1770000000 + user_idx * 100 + task_idx * 10 + episode_idx,
                    30_000_000_000 * (episode_idx + 1),
                )

    parallel_records = scan_episode_records_parallel(tmp_path)

    task_id = create_scan_task(tmp_path)
    threaded_records = scan_episode_records_with_progress(tmp_path, task_id)

    assert len(threaded_records) == len(parallel_records)

    parallel_keys = sorted(
        (
            record.user_name,
            record.task_name,
            record.episode_id,
            record.duration_hours,
        )
        for record in parallel_records
    )
    threaded_keys = sorted(
        (
            record.user_name,
            record.task_name,
            record.episode_id,
            record.duration_hours,
        )
        for record in threaded_records
    )
    assert threaded_keys == parallel_keys

    parallel_summary = build_summary(parallel_records, tmp_path)
    threaded_summary = build_summary(threaded_records, tmp_path)
    assert (
        threaded_summary["total_episodes"]
        == parallel_summary["total_episodes"]
    )
    assert threaded_summary["totals"] == parallel_summary["totals"]


def test_parallel_scan_is_stable_without_progress(tmp_path: Path):
    for user_idx in range(2):
        for task_idx in range(3):
            for episode_idx in range(3):
                create_episode(
                    tmp_path
                    / f"user_{user_idx}"
                    / f"task_{task_idx}"
                    / f"episode_2026_03_12-11_{task_idx:02d}_{episode_idx:02d}",  # noqa: E501
                    1771000000 + user_idx * 100 + task_idx * 10 + episode_idx,
                    45_000_000_000 * (episode_idx + 1),
                )

    first_parallel_records = scan_episode_records_parallel(tmp_path)
    second_parallel_records = scan_episode_records_parallel(tmp_path)

    assert sorted(
        (
            record.user_name,
            record.task_name,
            record.episode_id,
            record.duration_hours,
        )
        for record in first_parallel_records
    ) == sorted(
        (
            record.user_name,
            record.task_name,
            record.episode_id,
            record.duration_hours,
        )
        for record in second_parallel_records
    )


def test_cache_miss_uses_parallel_scan(tmp_path: Path, monkeypatch):
    EPISODE_CACHE.clear()
    cache_key = get_cache_key(tmp_path)
    cache_file = get_cache_file_path(cache_key)
    if cache_file.exists():
        cache_file.unlink()

    create_episode(
        tmp_path / "maya" / "pick" / "episode_2026_03_12-12_00_00",
        1771000200,
        90_000_000_000,
    )

    called = {"parallel": 0}

    def fake_parallel_scan(base_path: Path):
        called["parallel"] += 1
        return scan_episode_records_parallel(tmp_path)

    monkeypatch.setattr(
        app_module, "scan_episode_records_parallel", fake_parallel_scan
    )

    records, cache_info = get_cached_episode_records(tmp_path, refresh=False)

    assert called["parallel"] == 1
    assert len(records) == 1
    assert cache_info["cache_hit"] is False
    assert cache_info["cache_source"] == "refresh"


def test_merge_records_for_date_prefixes_keeps_other_days(tmp_path: Path):
    existing_first = app_module.EpisodeRecord(
        user_name="alice",
        task_name="pick",
        embodiment="",
        episode_id="episode_2026_03_12-10_00_00",
        day="2026-03-12",
        path=str(tmp_path / "alice" / "pick" / "episode_2026_03_12-10_00_00"),
        duration_hours=1.0,
    )
    existing_second = app_module.EpisodeRecord(
        user_name="alice",
        task_name="pick",
        embodiment="",
        episode_id="episode_2026_03_13-10_00_00",
        day="2026-03-13",
        path=str(tmp_path / "alice" / "pick" / "episode_2026_03_13-10_00_00"),
        duration_hours=1.0,
    )
    refreshed = app_module.EpisodeRecord(
        user_name="alice",
        task_name="pick",
        embodiment="",
        episode_id="episode_2026_03_12-11_00_00",
        day="2026-03-12",
        path=str(tmp_path / "alice" / "pick" / "episode_2026_03_12-11_00_00"),
        duration_hours=2.0,
    )

    merged = merge_records_for_date_prefixes(
        [existing_first, existing_second], [refreshed], ["2026_03_12"]
    )

    assert sorted(record.episode_id for record in merged) == [
        "episode_2026_03_12-11_00_00",
        "episode_2026_03_13-10_00_00",
    ]


def test_partial_refresh_updates_requested_date_only(tmp_path: Path):
    EPISODE_CACHE.clear()
    create_episode(
        tmp_path / "alice" / "pick" / "episode_2026_03_12-10_00_00",
        1771000000,
        3_600_000_000_000,
    )
    create_episode(
        tmp_path / "alice" / "pick" / "episode_2026_03_13-10_00_00",
        1771086400,
        3_600_000_000_000,
    )

    initial_records, _ = get_cached_episode_records(tmp_path, refresh=True)
    assert len(initial_records) == 2

    updated_episode = (
        tmp_path / "alice" / "pick" / "episode_2026_03_12-11_00_00"
    )
    create_episode(updated_episode, 1771000100, 7_200_000_000_000)
    old_episode = tmp_path / "alice" / "pick" / "episode_2026_03_12-10_00_00"
    for file_path in old_episode.glob("*"):
        if file_path.is_file():
            file_path.unlink()
    old_episode.rmdir()

    task_id = app_module.create_scan_task(tmp_path)
    records = refresh_cached_episode_records_for_date_prefixes(
        tmp_path, ["2026_03_12"], task_id
    )

    assert sorted(record.episode_id for record in records) == [
        "episode_2026_03_12-11_00_00",
        "episode_2026_03_13-10_00_00",
    ]
    cached_records, cache_info = get_cached_episode_records(
        tmp_path, refresh=False
    )
    assert sorted(record.episode_id for record in cached_records) == [
        "episode_2026_03_12-11_00_00",
        "episode_2026_03_13-10_00_00",
    ]
    assert cache_info["cache_hit"] is True


def test_partial_refresh_api_requires_date_prefix(tmp_path: Path):
    client = app.test_client()
    response = client.post(
        "/api/scan-tasks",
        json={
            "data_root": str(tmp_path),
            "refresh_mode": "partial",
            "date_prefix": "",
        },
    )

    assert response.status_code == 400
    assert "date_prefix is required" in response.get_data(as_text=True)


def test_prepare_submit_job_creates_initial_config(
    tmp_path: Path, monkeypatch
):
    create_episode(
        tmp_path / "zoe" / "pick_box" / "episode_2026_03_12-10_00_00",
        1771000000,
        90_000_000_000,
    )
    create_episode(
        tmp_path / "amy" / "place_box" / "episode_2026_03_13-11_00_00",
        1771086400,
        90_000_000_000,
    )
    get_cached_episode_records(tmp_path, refresh=True)

    robo_dir = tmp_path / "robo_orchard_lab"
    template_dir = robo_dir / "dataset" / "horizon_manipulation" / "tools"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "submit_check.json").write_text(
        json.dumps(
            {
                "job_name": "demo",
                "to_upload": "/tmp/original",
                "cmd": [
                    "input_path=/tmp/input",
                    "date_prefix=old",
                    "user_name=old",
                    "task_name=old",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ROBO_ORCHARD_LAB_DIR", str(robo_dir))

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/prepare",
        json={
            "source": "check",
            "data_root": str(tmp_path),
            "filters": {
                "user_name": "zoe,amy",
                "task_name": "pick_box,place_box",
                "date_prefix": "2026_03_12,2026_03_13",
            },
        },
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["selection"]["user_names"] == ["amy", "zoe"]
    assert data["selection"]["task_names"] == ["pick_box", "place_box"]
    assert data["selection"]["date_prefixes"] == ["2026_03_12", "2026_03_13"]
    assert data["config"]["to_upload"] == [str(robo_dir)]
    assert (
        data["config"]["job_name"]
        == "data-check-amy-zoe-piper-pick_box-place_box-2026_03_12-2026_03_13"
    )
    assert "input_path=" + str(tmp_path) in data["config"]["cmd"]
    assert "user_name=amy,zoe" in data["config"]["cmd"]
    assert "task_name=pick_box,place_box" in data["config"]["cmd"]
    assert "date_prefix=2026_03_12,2026_03_13" in data["config"]["cmd"]


def test_prepare_submit_job_preserves_user_requested_date_prefix(
    tmp_path: Path, monkeypatch
):
    create_episode(
        tmp_path / "zoe" / "pick_box" / "episode_2026_02_24-11_33_59",
        1771000000,
        90_000_000_000,
    )
    create_episode(
        tmp_path / "zoe" / "pick_box" / "episode_2026_02_24-11_35_27",
        1771000100,
        90_000_000_000,
    )
    get_cached_episode_records(tmp_path, refresh=True)

    robo_dir = tmp_path / "robo_orchard_lab"
    template_dir = robo_dir / "dataset" / "horizon_manipulation" / "tools"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "submit_check.json").write_text(
        json.dumps(
            {
                "job_name": "demo",
                "to_upload": "/tmp/original",
                "cmd": [
                    "input_path=/tmp/input",
                    "date_prefix=old",
                    "user_name=old",
                    "task_name=old",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ROBO_ORCHARD_LAB_DIR", str(robo_dir))

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/prepare",
        json={
            "source": "check",
            "data_root": str(tmp_path),
            "filters": {
                "user_name": "zoe",
                "task_name": "pick_box",
                "date_prefix": "2026_02_24-11",
            },
        },
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["selection"]["date_prefixes"] == ["2026_02_24-11"]
    assert (
        data["config"]["job_name"]
        == "data-check-zoe-piper-pick_box-2026_02_24-11"
    )
    assert "input_path=" + str(tmp_path) in data["config"]["cmd"]
    assert "date_prefix=2026_02_24-11" in data["config"]["cmd"]


def test_prepare_submit_job_uses_all_days_when_date_prefix_not_provided(
    tmp_path: Path, monkeypatch
):
    create_episode(
        tmp_path / "zoe" / "pick_box" / "episode_2026_02_24-11_33_59",
        1771000000,
        90_000_000_000,
    )
    create_episode(
        tmp_path / "zoe" / "pick_box" / "episode_2026_02_25-11_35_27",
        1771086400,
        90_000_000_000,
    )
    get_cached_episode_records(tmp_path, refresh=True)

    robo_dir = tmp_path / "robo_orchard_lab"
    template_dir = robo_dir / "dataset" / "horizon_manipulation" / "tools"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "submit_check.json").write_text(
        json.dumps(
            {
                "job_name": "demo",
                "to_upload": "/tmp/original",
                "cmd": [
                    "input_path=/tmp/input",
                    "date_prefix=old",
                    "user_name=old",
                    "task_name=old",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ROBO_ORCHARD_LAB_DIR", str(robo_dir))

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/prepare",
        json={
            "source": "check",
            "data_root": str(tmp_path),
            "filters": {
                "user_name": "zoe",
                "task_name": "pick_box",
                "date_prefix": "",
            },
        },
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["selection"]["date_prefixes"] == ["2026_02_24", "2026_02_25"]
    assert "input_path=" + str(tmp_path) in data["config"]["cmd"]
    assert "date_prefix=2026_02_24,2026_02_25" in data["config"]["cmd"]


def test_prepare_submit_job_applies_submit_config_patch(
    tmp_path: Path, monkeypatch
):
    create_episode(
        tmp_path / "zoe" / "pick_box" / "episode_2026_03_12-10_00_00",
        1771000000,
        90_000_000_000,
    )
    get_cached_episode_records(tmp_path, refresh=True)

    robo_dir = tmp_path / "robo_orchard_lab"
    template_dir = robo_dir / "dataset" / "horizon_manipulation" / "tools"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "submit_check.json").write_text(
        json.dumps(
            {
                "job_name": "demo",
                "to_upload": "/tmp/original",
                "labels": {"team": "origin", "priority": "normal"},
                "cmd": [
                    "input_path=/tmp/input",
                    "date_prefix=old",
                    "user_name=old",
                    "task_name=old",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    patch_path = tmp_path / "submit_config_patch.json"
    patch_path.write_text(
        json.dumps(
            {
                "job_name": "patched-job-name",
                "labels": {"priority": "high", "owner": "qa"},
                "cmd": ["patched-cmd"],
                "extra_flag": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ROBO_ORCHARD_LAB_DIR", str(robo_dir))
    monkeypatch.setenv("SUBMIT_CONFIG_PATCH_PATH", str(patch_path))

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/prepare",
        json={
            "source": "check",
            "data_root": str(tmp_path),
            "filters": {
                "user_name": "zoe",
                "task_name": "pick_box",
                "date_prefix": "2026_03_12",
            },
        },
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["config"]["job_name"] == "patched-job-name"
    assert data["config"]["labels"] == {
        "team": "origin",
        "priority": "high",
        "owner": "qa",
    }
    assert data["config"]["cmd"] == ["patched-cmd"]
    assert data["config"]["extra_flag"] is True


def test_submit_job_api_runs_submit_command(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "submit_configs"
    monkeypatch.setenv("SUBMIT_CONFIG_DIR", str(config_dir))

    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        get_submit_config_path,
    )

    config_path = get_submit_config_path("cfg001")
    config_path.write_text(
        json.dumps({"job_name": "demo"}, ensure_ascii=False), encoding="utf-8"
    )

    class FakeStream:
        def __init__(self, lines: list[str]):
            self._lines = list(lines)

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

        def close(self):
            return None

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStream(["submitted ok\n"])
            self.stderr = FakeStream([])
            self.returncode = None
            self._poll_calls = 0

        def poll(self):
            self._poll_calls += 1
            if self._poll_calls < 2:
                return None
            self.returncode = 0
            return self.returncode

    def fake_popen(cmd, stdout, stderr, text, bufsize, env, cwd=None):
        assert cmd[:3] == [
            "RoboOrchardJob-AIDISubmit",
            "submit_from_config",
            "--config",
        ]
        assert cmd[3] == str(config_path)
        assert stdout is not None
        assert stderr is not None
        assert text is True
        assert bufsize == 1
        assert isinstance(env, dict)
        return FakeProcess()

    monkeypatch.setattr(app_module.subprocess, "Popen", fake_popen)

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/cfg001/submit",
        json={"config": {"job_name": "updated-demo"}},
    )

    assert response.status_code == 202
    data = response.get_json()
    assert data["status"] in {"pending", "running"}
    assert data["command"][:3] == [
        "RoboOrchardJob-AIDISubmit",
        "submit_from_config",
        "--config",
    ]
    assert (
        "RoboOrchardJob-AIDISubmit submit_from_config --config"
        in data["command_text"]
    )

    task_id = data["task_id"]
    deadline = time.time() + 1.5
    task_data = None
    while time.time() < deadline:
        task_response = client.get(f"/api/submit-jobs/tasks/{task_id}")
        assert task_response.status_code == 200
        task_data = task_response.get_json()
        if task_data["status"] == "submitted":
            break
        time.sleep(0.05)

    assert task_data is not None
    assert task_data["status"] == "submitted"
    assert task_data["returncode"] == 0
    assert task_data["stdout"] == "submitted ok\n"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["job_name"] == "updated-demo"


def test_submit_job_api_returns_logs_on_nonzero_exit(
    tmp_path: Path, monkeypatch
):
    config_dir = tmp_path / "submit_configs"
    monkeypatch.setenv("SUBMIT_CONFIG_DIR", str(config_dir))

    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        get_submit_config_path,
    )

    config_path = get_submit_config_path("cfg-failed")
    config_path.write_text(
        json.dumps({"job_name": "demo"}, ensure_ascii=False), encoding="utf-8"
    )

    class FakeStream:
        def __init__(self, lines: list[str]):
            self._lines = list(lines)

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

        def close(self):
            return None

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStream(["step1\n", "step2\n"])
            self.stderr = FakeStream(["fatal error\n"])
            self.returncode = None
            self._poll_calls = 0

        def poll(self):
            self._poll_calls += 1
            if self._poll_calls < 3:
                return None
            self.returncode = 2
            return self.returncode

    def fake_popen(cmd, stdout, stderr, text, bufsize, env, cwd=None):
        assert cmd[3] == str(config_path)
        assert isinstance(env, dict)
        return FakeProcess()

    monkeypatch.setattr(app_module.subprocess, "Popen", fake_popen)

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/cfg-failed/submit",
        json={"config": {"job_name": "updated-demo"}},
    )

    assert response.status_code == 202
    data = response.get_json()
    task_id = data["task_id"]

    deadline = time.time() + 1.5
    task_data = None
    while time.time() < deadline:
        task_response = client.get(f"/api/submit-jobs/tasks/{task_id}")
        assert task_response.status_code == 200
        task_data = task_response.get_json()
        if task_data["status"] == "failed":
            break
        time.sleep(0.05)

    assert task_data is not None
    assert task_data["status"] == "failed"
    assert task_data["returncode"] == 2
    assert task_data["stdout"] == "step1\nstep2\n"
    assert task_data["stderr"] == "fatal error\n"
    assert task_data["error_message"] == ""
    assert task_data["command_text"].endswith(str(config_path))


def test_submit_job_api_returns_logs_when_command_cannot_start(
    tmp_path: Path, monkeypatch
):
    config_dir = tmp_path / "submit_configs"
    monkeypatch.setenv("SUBMIT_CONFIG_DIR", str(config_dir))

    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        get_submit_config_path,
    )

    config_path = get_submit_config_path("cfg-oserror")
    config_path.write_text(
        json.dumps({"job_name": "demo"}, ensure_ascii=False), encoding="utf-8"
    )

    def fake_popen(cmd, stdout, stderr, text, bufsize, env, cwd=None):
        raise FileNotFoundError("RoboOrchardJob-AIDISubmit: command not found")

    monkeypatch.setattr(app_module.subprocess, "Popen", fake_popen)

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/cfg-oserror/submit",
        json={"config": {"job_name": "updated-demo"}},
    )

    assert response.status_code == 202
    data = response.get_json()
    task_id = data["task_id"]

    deadline = time.time() + 1.0
    task_data = None
    while time.time() < deadline:
        task_response = client.get(f"/api/submit-jobs/tasks/{task_id}")
        assert task_response.status_code == 200
        task_data = task_response.get_json()
        if task_data["status"] == "failed":
            break
        time.sleep(0.05)

    assert task_data is not None
    assert task_data["status"] == "failed"
    assert task_data["returncode"] is None
    assert task_data["stdout"] == ""
    assert task_data["stderr"] == ""
    assert "command not found" in task_data["error_message"]
    assert task_data["command"][0] == "RoboOrchardJob-AIDISubmit"


def test_submit_job_task_api_exposes_running_logs(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "submit_configs"
    monkeypatch.setenv("SUBMIT_CONFIG_DIR", str(config_dir))

    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        get_submit_config_path,
    )

    config_path = get_submit_config_path("cfg-running")
    config_path.write_text(
        json.dumps({"job_name": "demo"}, ensure_ascii=False), encoding="utf-8"
    )

    poll_gate = {"allow_finish": False}

    class FakeStream:
        def __init__(self, lines: list[str]):
            self._lines = list(lines)

        def readline(self):
            time.sleep(0.02)
            return self._lines.pop(0) if self._lines else ""

        def close(self):
            return None

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStream(["boot\n", "running\n"])
            self.stderr = FakeStream(["warn\n"])
            self.returncode = None

        def poll(self):
            if not poll_gate["allow_finish"]:
                return None
            self.returncode = 0
            return self.returncode

    monkeypatch.setattr(
        app_module.subprocess,
        "Popen",
        lambda cmd,
        stdout,
        stderr,
        text,
        bufsize,
        env,
        cwd=None: FakeProcess(),
    )

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/cfg-running/submit",
        json={"config": {"job_name": "updated-demo"}},
    )
    assert response.status_code == 202
    task_id = response.get_json()["task_id"]

    time.sleep(0.2)
    running_response = client.get(f"/api/submit-jobs/tasks/{task_id}")
    assert running_response.status_code == 200
    running_data = running_response.get_json()
    assert running_data["status"] in {"pending", "running"}
    assert "boot\n" in running_data["stdout"]
    assert "warn\n" in running_data["stderr"]

    poll_gate["allow_finish"] = True
    deadline = time.time() + 1.0
    done_data = running_data
    while time.time() < deadline:
        done_response = client.get(f"/api/submit-jobs/tasks/{task_id}")
        done_data = done_response.get_json()
        if done_data["status"] == "submitted":
            break
        time.sleep(0.05)

    assert done_data["status"] == "submitted"


def test_submit_job_clears_proxy_env_when_enabled(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "submit_configs"
    monkeypatch.setenv("SUBMIT_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("SUBMIT_JOB_CLEAR_PROXY", "true")
    monkeypatch.setenv("HTTP_PROXY", "http://10.103.80.5:20272")
    monkeypatch.setenv("HTTPS_PROXY", "http://10.103.80.5:20272")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    from robo_orchard_lab.dataset.horizon_manipulation.tools.app import (
        get_submit_config_path,
    )

    config_path = get_submit_config_path("cfg-clear-proxy")
    config_path.write_text(
        json.dumps({"job_name": "demo"}, ensure_ascii=False), encoding="utf-8"
    )

    captured_env: dict[str, str] = {}

    class FakeStream:
        def readline(self):
            return ""

        def close(self):
            return None

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStream()
            self.stderr = FakeStream()
            self.returncode = 0

        def poll(self):
            return 0

    def fake_popen(cmd, stdout, stderr, text, bufsize, env, cwd=None):
        captured_env.update(env)
        return FakeProcess()

    monkeypatch.setattr(app_module.subprocess, "Popen", fake_popen)

    client = app.test_client()
    response = client.post(
        "/api/submit-jobs/cfg-clear-proxy/submit",
        json={"config": {"job_name": "updated-demo"}},
    )
    assert response.status_code == 202

    task_id = response.get_json()["task_id"]
    deadline = time.time() + 1.0
    while time.time() < deadline:
        task_response = client.get(f"/api/submit-jobs/tasks/{task_id}")
        assert task_response.status_code == 200
        if task_response.get_json()["status"] == "submitted":
            break
        time.sleep(0.05)

    assert "HTTP_PROXY" not in captured_env
    assert "HTTPS_PROXY" not in captured_env
    assert "NO_PROXY" not in captured_env


def test_cached_episode_records_only_refresh_on_demand(tmp_path: Path):
    EPISODE_CACHE.clear()
    create_episode(
        tmp_path / "alice" / "pick" / "ep001", 1710115200, 3_600_000_000_000
    )

    records, cache_info = get_cached_episode_records(tmp_path, refresh=False)
    assert len(records) == 1
    assert cache_info["cache_hit"] is False

    create_episode(
        tmp_path / "alice" / "pick" / "ep002", 1710115200, 3_600_000_000_000
    )

    cached_records, cached_info = get_cached_episode_records(
        tmp_path, refresh=False
    )
    assert len(cached_records) == 1
    assert cached_info["cache_hit"] is True

    refreshed_records, refreshed_info = get_cached_episode_records(
        tmp_path, refresh=True
    )
    assert len(refreshed_records) == 2
    assert refreshed_info["cache_hit"] is False


def test_persistent_cache_survives_memory_clear(tmp_path: Path):
    EPISODE_CACHE.clear()
    create_episode(
        tmp_path / "dora" / "lift" / "ep001", 1710115200, 3_600_000_000_000
    )

    records, cache_info = get_cached_episode_records(tmp_path, refresh=True)
    assert len(records) == 1
    assert cache_info["cache_source"] == "refresh"

    cache_key = get_cache_key(tmp_path)
    cache_file = get_cache_file_path(cache_key)
    assert cache_file.exists()

    EPISODE_CACHE.clear()
    loaded = load_cache_from_disk(cache_key)
    assert loaded is not None
    assert len(loaded["records"]) == 1

    records_after_clear, reused_info = get_cached_episode_records(
        tmp_path, refresh=False
    )
    assert len(records_after_clear) == 1
    assert reused_info["cache_hit"] is True
    assert reused_info["cache_source"] == "disk"


def test_seconds_until_next_refresh_targets_2am():
    before_two = datetime.fromisoformat("2026-03-12T01:30:00")
    after_two = datetime.fromisoformat("2026-03-12T03:15:00")

    assert seconds_until_next_refresh(before_two) == 1800
    assert seconds_until_next_refresh(after_two) == 81900


def test_format_duration_hours_for_minute_level_data():
    assert format_duration_hours(0) == "0 secs"
    assert format_duration_hours(0.5 / 60) == "30 secs"
    assert format_duration_hours(1 / 60) == "1 mins"
    assert format_duration_hours(1.5) == "1 hours 30 mins"
    assert (
        format_duration_hours(1 + 5 / 60 + 2 / 3600) == "1 hours 5 mins 2 secs"
    )
