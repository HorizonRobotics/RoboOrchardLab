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


"""Coverage for the RoboTwin evaluation report."""

import pytest

from projects.holobrain_0_5.robotwin_eval import (
    allocate_tasks_to_workers,
    list_robotwin_tasks,
    select_tasks,
    summarize_results,
)


def test_matches_the_reference_report_shape():
    """RoboTwin reports percentages per task, e.g. 45/50 shows as 90.0."""
    results = {
        "adjust_bottle": {
            "successes": 45,
            "episodes": 50,
            "success_rate": 0.9,
        },
        "beat_block_hammer": {
            "successes": 30,
            "episodes": 50,
            "success_rate": 0.6,
        },
    }
    s = summarize_results(results, 50)
    assert set(s) == {
        "success_rate",
        "num_tested",
        "test_num_per_task",
        "num_tasks_in_mean",
        "mean",
    }
    assert s["success_rate"] == {
        "adjust_bottle": 90.0,
        "beat_block_hammer": 60.0,
    }
    assert s["num_tested"] == {"adjust_bottle": 50, "beat_block_hammer": 50}
    assert s["test_num_per_task"] == 50
    assert s["num_tasks_in_mean"] == 2
    assert s["mean"] == 75.0


def test_mean_averages_task_percentages_not_raw_counts():
    """Uneven episode counts must not weight the mean by task length."""
    results = {
        "a": {"successes": 1, "episodes": 2, "success_rate": 0.5},
        "b": {"successes": 50, "episodes": 100, "success_rate": 0.5},
    }

    assert summarize_results(results, 50)["mean"] == 50.0


def test_empty_results_do_not_divide_by_zero():
    assert summarize_results({}, 50)["mean"] == 0.0


def test_allocation_spreads_over_every_gpu_before_doubling_up():
    """GPU-major slots keep a short task list off a single GPU."""
    allocated = allocate_tasks_to_workers(list("abcd"), 8, processes_per_gpu=2)

    assert [gpu for gpu, _ in allocated] == [0, 1, 2, 3]


def test_allocation_covers_every_task_once():
    tasks = [f"task_{i}" for i in range(50)]

    allocated = allocate_tasks_to_workers(tasks, 8, processes_per_gpu=2)

    assert len(allocated) == 16
    dealt = [t for _, worker in allocated for t in worker]
    assert sorted(dealt) == sorted(tasks)
    assert {gpu for gpu, _ in allocated} == set(range(8))


def test_allocation_of_nothing_is_empty():
    assert allocate_tasks_to_workers([], 8) == []


def test_allocation_rejects_degenerate_topology():
    with pytest.raises(ValueError):
        allocate_tasks_to_workers(["a"], 0)
    with pytest.raises(ValueError):
        allocate_tasks_to_workers(["a"], 1, 0)


def _checkout(tmp_path, *names):
    env_dir = tmp_path / "envs"
    env_dir.mkdir()
    for name in names:
        (env_dir / f"{name}.py").touch()
    return tmp_path


def test_discovery_skips_private_modules(tmp_path):
    """``_base_task.py`` and ``__init__.py`` are machinery, not tasks."""
    root = _checkout(tmp_path, "lift_pot", "adjust_bottle", "_base_task")
    (root / "envs" / "__init__.py").touch()
    (root / "envs" / "notes.txt").touch()

    assert list_robotwin_tasks(root) == ["adjust_bottle", "lift_pot"]


def test_no_requested_tasks_means_every_task():
    assert select_tasks(["b", "a"], None, "demo_clean") == ["a", "b"]


def test_requested_tasks_keep_their_order():
    assert select_tasks(["a", "b", "c"], ["c", "a"], "demo_clean") == [
        "c",
        "a",
    ]


def test_unknown_task_is_dropped_not_raised():
    """One stale name must not cost a 50-task sweep."""
    assert select_tasks(["a"], ["a", "typo"], "demo_clean") == ["a"]


def test_ur5_wsg_drops_tasks_its_planner_cannot_solve():
    available = ["lift_pot", "handover_block", "put_object_cabinet"]

    assert select_tasks(available, None, "demo_clean_ur5-wsg") == ["lift_pot"]
    assert select_tasks(available, None, "demo_clean") == sorted(available)
