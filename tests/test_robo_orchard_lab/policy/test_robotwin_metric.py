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
from __future__ import annotations
from types import SimpleNamespace

import pytest

import robo_orchard_lab.policy.evaluator as evaluator_pkg
from robo_orchard_lab.policy.evaluator.benchmark import (
    BenchmarkEpisode,
    BenchmarkEpisodeRecord,
    BenchmarkEvaluator,
    BenchmarkResult,
)
from robo_orchard_lab.policy.evaluator.benchmark.robotwin import (
    SEM_TASKS_16,
    SuccessRateInfo,
    SuccessRateMetric,
)


class _FakeWorkerMetric(SuccessRateMetric):
    def __init__(
        self,
        task_name: str,
        seed: int,
        success: bool = True,
    ) -> None:
        super().__init__()
        info = {"seed": seed, "success": success}
        self.info = {
            task_name: SuccessRateInfo(
                task_name=task_name,
                success_count=1 if success else 0,
                total_count=1,
                info_list=[info],
            )
        }
        self.last_update_info = {
            "task_name": task_name,
            "seed": seed,
            "success": success,
        }


def test_success_rate_metric_keeps_merge_explicit() -> None:
    metric = SuccessRateMetric()
    metric.merge([_FakeWorkerMetric(task_name="task_a", seed=1)])

    assert metric.compute()["average_success_rate"] == 1.0


def test_success_rate_metric_state_round_trip() -> None:
    metric = SuccessRateMetric()
    metric.merge([_FakeWorkerMetric(task_name="task_a", seed=1)])

    restored = SuccessRateMetric()
    restored.load_state(metric.get_state())

    assert restored.compute() == metric.compute()


def test_success_rate_metric_update_records_robotwin_seed_info() -> None:
    metric = SuccessRateMetric()
    step_return = SimpleNamespace(
        truncated=False,
        terminated=True,
        rewards=True,
        info={
            "task": "task_a",
            "seed": 100,
            "start_seed": 1,
            "resolved_start_seed": 101,
            "offset_seed": 3,
        },
    )

    metric.update(action=None, step_return=step_return)
    computed = metric.compute()

    assert computed["average_success_rate"] == 1.0
    assert computed["last_update"] == {
        "task_name": "task_a",
        "seed": 100,
        "start_seed": 1,
        "resolved_start_seed": 101,
        "offset_seed": 3,
        "success": True,
    }


def test_success_rate_metric_records_max_steps_failure() -> None:
    metric = SuccessRateMetric()
    step_return = SimpleNamespace(
        truncated=False,
        terminated=False,
        rewards=False,
        info={
            "task": "task_a",
            "seed": 100,
            "start_seed": 1,
            "resolved_start_seed": 101,
            "offset_seed": 3,
        },
    )

    metric.update(action=None, step_return=step_return)
    computed = metric.compute()

    assert computed["average_success_rate"] == 0.0
    assert computed["last_update"]["success"] is False


def test_success_rate_metric_rejects_missing_step_info() -> None:
    metric = SuccessRateMetric()
    step_return = SimpleNamespace(
        truncated=True,
        terminated=False,
        rewards=False,
        info=None,
    )

    with pytest.raises(ValueError, match="step_return.info"):
        metric.update(action=None, step_return=step_return)


def test_sem_tasks_16_keeps_expected_task_subset() -> None:
    assert len(SEM_TASKS_16) == 16
    assert "adjust_bottle" in SEM_TASKS_16
    assert "stack_bowls_three" in SEM_TASKS_16


def test_package_root_exports_only_stable_benchmark_entrypoints() -> None:
    assert evaluator_pkg.BenchmarkEpisode is BenchmarkEpisode
    assert evaluator_pkg.BenchmarkEpisodeRecord is BenchmarkEpisodeRecord
    assert evaluator_pkg.BenchmarkEvaluator is BenchmarkEvaluator
    assert evaluator_pkg.BenchmarkResult is BenchmarkResult
    assert "BenchmarkAttemptRequest" not in evaluator_pkg.__dict__
