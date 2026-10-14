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


"""Coverage for LIBERO task selection, fan-out and result aggregation."""

import pytest

from projects.holobrain_0_5.libero_eval import (
    DEFAULT_TRIALS,
    LIBERO_SUITES,
    _benchmark_root,
    _parse_args,
    _worker_command,
    allocate_tasks_to_workers,
    build_task_list,
    resolve_trials_per_task,
    summarize_plus_categories,
    summarize_results,
)

__all__ = []

COUNTS = {suite: 10 for suite in LIBERO_SUITES}
BASE_ARGV = [
    "libero_eval",
    "--libero-root",
    "/libero",
    "--model-dir",
    "/model",
    "--output",
    "/out",
]


def _parse(monkeypatch, *extra):
    monkeypatch.setattr("sys.argv", [*BASE_ARGV, *extra])
    return _parse_args()


def test_libero_defaults_to_fifty_trials():
    assert resolve_trials_per_task("libero", None) == 50


def test_libero_plus_defaults_to_one_trial():
    """LIBERO-Plus already varies each task, so one pass per variation."""
    assert resolve_trials_per_task("libero_plus", None) == 1


def test_explicit_trial_count_overrides_the_default():
    for benchmark in DEFAULT_TRIALS:
        assert resolve_trials_per_task(benchmark, 3) == 3


def test_default_selection_covers_every_task_of_four_suites():
    tasks = build_task_list(LIBERO_SUITES, None, COUNTS)

    assert len(tasks) == 40
    assert {suite for suite, _ in tasks} == set(LIBERO_SUITES)
    assert sorted(t for s, t in tasks if s == "libero_goal") == list(range(10))


def test_explicit_task_ids_apply_to_every_selected_suite():
    tasks = build_task_list(["libero_goal", "libero_10"], [0, 2], COUNTS)

    assert tasks == [
        ("libero_goal", 0),
        ("libero_goal", 2),
        ("libero_10", 0),
        ("libero_10", 2),
    ]


def test_out_of_range_task_id_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        build_task_list(["libero_goal"], [10], COUNTS)


def test_unknown_suite_is_rejected():
    with pytest.raises(ValueError, match="Unknown suite"):
        build_task_list(["libero_nope"], None, COUNTS)


def test_allocation_covers_every_task_exactly_once():
    tasks = build_task_list(LIBERO_SUITES, None, COUNTS)

    allocated = allocate_tasks_to_workers(tasks, num_gpus=8)

    assert len(allocated) == 8
    dealt = [task for _, worker_tasks in allocated for task in worker_tasks]
    assert sorted(dealt) == sorted(tasks)
    assert {gpu for gpu, _ in allocated} == set(range(8))


def test_allocation_balances_and_honors_processes_per_gpu():
    tasks = build_task_list(LIBERO_SUITES, None, COUNTS)

    allocated = allocate_tasks_to_workers(tasks, 8, processes_per_gpu=2)

    assert len(allocated) == 16
    assert {len(worker_tasks) for _, worker_tasks in allocated} == {
        40 // 16 + 1,
        40 // 16,
    }


def test_allocation_never_creates_idle_workers():
    """More slots than tasks would make a worker launch with nothing to do."""
    allocated = allocate_tasks_to_workers([("libero_goal", 0)], 8)

    assert len(allocated) == 1


def test_allocation_rejects_degenerate_topology():
    with pytest.raises(ValueError):
        allocate_tasks_to_workers([("libero_goal", 0)], 0)
    with pytest.raises(ValueError):
        allocate_tasks_to_workers([("libero_goal", 0)], 1, 0)


def test_summary_averages_per_suite_and_overall():
    results = {
        ("libero_goal", 0): 1.0,
        ("libero_goal", 1): 0.0,
        ("libero_10", 0): 0.5,
    }

    summary = summarize_results(results, ["libero_goal", "libero_10"])

    overall = summary["overall_summary"]
    assert overall["libero_goal"] == {
        "average_success_rate": 0.5,
        "num_tasks": 2,
    }
    assert overall["average_success_rate_across_suites"] == 0.5
    assert summary["suites_detail"]["libero_goal"]["task_1"] == 0.0


def test_summary_ignores_suites_without_results():
    """A suite whose workers all crashed must not skew the mean."""
    summary = summarize_results({("libero_goal", 0): 1.0}, LIBERO_SUITES)

    overall = summary["overall_summary"]
    assert set(overall) == {
        "libero_goal",
        "average_success_rate_across_suites",
    }
    assert overall["average_success_rate_across_suites"] == 1.0


def test_summary_of_nothing_is_zero():
    summary = summarize_results({}, LIBERO_SUITES)

    assert (
        summary["overall_summary"]["average_success_rate_across_suites"] == 0.0
    )


def test_cli_defaults_to_all_suites_and_fifty_trials(monkeypatch):
    args = _parse(monkeypatch)

    assert args.benchmark == "libero"
    assert args.task_suites == LIBERO_SUITES
    assert args.task_ids is None
    assert args.num_trials_per_task == 50


def test_cli_libero_plus_defaults_to_one_trial(monkeypatch):
    args = _parse(monkeypatch, "--benchmark", "libero_plus")

    assert args.num_trials_per_task == 1
    assert args.task_suites == LIBERO_SUITES


def test_cli_explicit_trials_win(monkeypatch):
    args = _parse(monkeypatch, "--num-trials-per-task", "2")

    assert args.num_trials_per_task == 2


def test_worker_command_carries_the_resolved_selection(monkeypatch):
    args = _parse(monkeypatch, "--save-video")

    command = _worker_command(args, "libero_goal", [0, 3])

    assert "--worker" in command
    assert "--save-video" in command
    assert command[command.index("--task-suites") + 1] == "libero_goal"
    ids = command.index("--task-ids")
    assert command[ids + 1 : ids + 3] == ["0", "3"]
    assert command[command.index("--num-trials-per-task") + 1] == "50"
    assert "--libero-plus-root" not in command


def test_worker_command_forwards_the_plus_root(monkeypatch):
    args = _parse(
        monkeypatch,
        "--benchmark",
        "libero_plus",
        "--libero-plus-root",
        "/plus",
    )

    command = _worker_command(args, "libero_goal", [1])

    assert command[command.index("--libero-plus-root") + 1] == "/plus"
    assert command[command.index("--benchmark") + 1] == "libero_plus"
    assert command[command.index("--num-trials-per-task") + 1] == "1"


def test_plus_run_needs_no_libero_checkout(monkeypatch):
    """A LIBERO-Plus job must not be forced to stage a plain LIBERO tree."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "libero_eval",
            "--model-dir",
            "/model",
            "--output",
            "/out",
            "--benchmark",
            "libero_plus",
            "--libero-plus-root",
            "/plus",
        ],
    )
    args = _parse_args()

    assert args.libero_root is None
    assert _benchmark_root(args) == "/plus"
    assert "--libero-root" not in _worker_command(args, "libero_goal", [0])


def test_missing_root_names_the_flag_to_pass(monkeypatch):
    args = _parse(monkeypatch)
    args.libero_root = None

    with pytest.raises(SystemExit, match="--libero-root is required"):
        _benchmark_root(args)


def test_plus_without_its_root_is_rejected(monkeypatch):
    args = _parse(monkeypatch, "--benchmark", "libero_plus")

    with pytest.raises(SystemExit, match="--libero-plus-root is required"):
        _benchmark_root(args)


def test_plus_summary_omits_per_task_detail():
    """Thousands of LIBERO-Plus variations would drown the log."""
    summary = summarize_results(
        {("libero_goal", 0): 1.0}, LIBERO_SUITES, detail=False
    )

    assert "suites_detail" not in summary
    assert "overall_summary" in summary


def test_plus_categories_group_by_perturbation_axis():
    results = {
        ("libero_goal", 0): 1.0,
        ("libero_goal", 1): 0.0,
        ("libero_goal", 2): 0.5,
    }
    task_names = {
        ("libero_goal", 0): "a",
        ("libero_goal", 1): "b",
        ("libero_goal", 2): "c",
    }
    categories = {
        "libero_goal": [
            {"name": "a", "category": "Camera Viewpoints"},
            {"name": "b", "category": "Camera Viewpoints"},
            {"name": "c", "category": "Sensor Noise"},
        ]
    }

    import json as _json
    import tempfile
    from pathlib import Path as _Path

    root = _Path(tempfile.mkdtemp())
    path = root / "libero/libero/benchmark"
    path.mkdir(parents=True)
    (path / "task_classification.json").write_text(_json.dumps(categories))

    summary = summarize_plus_categories(results, task_names, str(root))

    detail = summary["categories_detail"]
    assert detail["Camera Viewpoints"] == {
        "average_success_rate": 0.5,
        "num_tasks": 2,
    }
    assert detail["Sensor Noise"]["num_tasks"] == 1
    # Category mean weights axes equally; the all-task mean does not.
    assert summary["average_success_rate_across_categories"] == 0.5
    assert summary["average_success_rate_all_tasks"] == 0.5
    assert summary["num_tasks"] == 3
    assert list(detail) == ["Camera Viewpoints", "Sensor Noise"]


def test_unclassified_plus_tasks_still_counted(tmp_path):
    """A missing classification file must not silently drop results."""
    summary = summarize_plus_categories(
        {("libero_goal", 0): 1.0}, {("libero_goal", 0): "a"}, str(tmp_path)
    )

    assert summary["categories_detail"]["Unclassified"]["num_tasks"] == 1
    assert summary["num_tasks"] == 1
