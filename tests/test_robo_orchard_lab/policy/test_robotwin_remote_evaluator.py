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
import queue
import time
from types import SimpleNamespace
from unittest.mock import patch

from robo_orchard_lab.policy.evaluator.robotwin import (
    RoboTwinRemoteEvaluator,
    RoboTwinTaskQueueItem,
    SuccessRateInfo,
    SuccessRateMetric,
    TaskInfo,
    TaskStatus,
)


class _FakeWorkerMetric:
    def __init__(
        self, task_name: str, seed: int, success: bool = True
    ) -> None:
        self.info = {
            task_name: SuccessRateInfo(
                task_name=task_name,
                success_count=1 if success else 0,
                total_count=1,
                info_list=[{"seed": seed, "success": success}],
            )
        }
        self.last_update_info = {
            "task_name": task_name,
            "seed": seed,
            "success": success,
        }

    def reset(self, **kwargs) -> None:
        self.info = {}
        self.last_update_info = None


class _FakeEvaluator:
    def __init__(self, behavior: str = "ok", sleep_s: float = 0.0) -> None:
        self.behavior = behavior
        self.sleep_s = sleep_s
        self.metrics = None
        self.current_episode = None
        self.setup_calls = []
        self.reset_calls = []

    def async_setup(self, env_cfg, policy_or_cfg, metrics, device=None):
        self.setup_calls.append(
            {
                "env_cfg": env_cfg,
                "policy_or_cfg": policy_or_cfg,
                "device": device,
            }
        )
        self.metrics = metrics

        class _DoneFuture:
            def result(self):
                return None

        return _DoneFuture()

    def reset_metrics(self):
        if self.metrics is not None:
            self.metrics.reset()

    def reset_env(self, **kwargs):
        self.reset_calls.append(kwargs)
        self.current_episode = kwargs
        return None, {"seed": kwargs["seed"]}

    def evaluate_episode(self, max_steps, env_reset_kwargs=None):
        if env_reset_kwargs is not None:
            self.reset_env(**env_reset_kwargs)
        assert self.current_episode is not None
        if self.behavior == "hang":
            time.sleep(self.sleep_s)
        task_name = self.current_episode["task_name"]
        seed = self.current_episode["seed"]
        self.metrics = _FakeWorkerMetric(task_name=task_name, seed=seed)
        return {"last_update": self.metrics.last_update_info}

    def get_metrics(self):
        return self.metrics


def test_task_info_counts_only_completed_episodes():
    task_queue: queue.Queue[RoboTwinTaskQueueItem] = queue.Queue()
    task_queue.put(RoboTwinTaskQueueItem(task_name="task_a", seed=1))
    info = TaskInfo(
        tasks={"task_a": TaskStatus()},
        task_queue=task_queue,
        episode_per_task=2,
    )

    item = info.get_task_item()
    assert item == RoboTwinTaskQueueItem(task_name="task_a", seed=1)
    assert info.tasks["task_a"].episode_count == 0

    assert info.requeue_task_item(item) is True
    same_item = info.get_task_item()
    assert same_item == RoboTwinTaskQueueItem(task_name="task_a", seed=1)

    assert info.complete_task_item(same_item, next_seed=2) is True
    assert info.tasks["task_a"].episode_count == 1

    next_item = info.get_task_item()
    assert next_item == RoboTwinTaskQueueItem(
        task_name="task_a",
        seed=2,
        episode_idx=1,
    )
    assert info.complete_task_item(next_item, next_seed=3) is False
    assert info.tasks["task_a"].episode_count == 2
    assert info.has_pending_work() is False


def test_robotwin_remote_evaluator_replaces_timed_out_worker(monkeypatch):
    killed = []
    created_replacements = []

    evaluator = RoboTwinRemoteEvaluator.__new__(RoboTwinRemoteEvaluator)
    evaluator.cfg = SimpleNamespace(
        episode_num=2,
        worker_poll_interval_s=0.005,
        episode_timeout_s=0.02,
        task_names=["task_a", "task_b"],
        config_type="demo_clean",
        seed=0,
        format_datatypes=False,
        artifact_root_dir=None,
    )
    evaluator.metric = SuccessRateMetric()
    evaluator.evaluators = [
        _FakeEvaluator(behavior="hang", sleep_s=0.1),
        _FakeEvaluator(),
    ]

    def _create_replacement_evaluator():
        replacement = _FakeEvaluator()
        created_replacements.append(replacement)
        return replacement

    evaluator._remote_cfg = _create_replacement_evaluator

    monkeypatch.setattr(
        evaluator,
        "_generate_env_cfg",
        lambda task_name: {"task_name": task_name},
    )
    monkeypatch.setattr(
        evaluator,
        "_kill_evaluator",
        lambda timed_out_evaluator: killed.append(timed_out_evaluator),
    )

    remaining_tasks = ["task_a", "task_b"]
    assert evaluator._evaluate_tasks(
        remaining_task=remaining_tasks,
        policy_or_cfg="policy",
        need_setup=True,
        device="cpu",
    )

    assert len(killed) >= 1
    assert len(created_replacements) == len(killed)
    assert len(remaining_tasks) == 0
    assert evaluator.metric.info["task_a"].total_count == 2
    assert evaluator.metric.info["task_b"].total_count == 2
    assert evaluator.metric.info["task_a"].success_count == 2
    assert evaluator.metric.info["task_b"].success_count == 2


def test_task_info_marks_task_started_once():
    task_queue: queue.Queue[RoboTwinTaskQueueItem] = queue.Queue()
    task_queue.put(RoboTwinTaskQueueItem(task_name="task_a", seed=1))
    info = TaskInfo(
        tasks={"task_a": TaskStatus()},
        task_queue=task_queue,
        episode_per_task=2,
    )

    first_item = info.get_task_item()
    assert first_item == RoboTwinTaskQueueItem(task_name="task_a", seed=1)
    assert info.mark_task_started("task_a") is True
    assert info.mark_task_started("task_a") is False

    assert info.requeue_task_item(first_item) is True
    second_item = info.get_task_item()
    assert second_item == RoboTwinTaskQueueItem(task_name="task_a", seed=1)
    assert info.mark_task_started("task_a") is False


def test_robotwin_remote_evaluator_prints_task_start_and_final_summary():
    evaluator = RoboTwinRemoteEvaluator.__new__(RoboTwinRemoteEvaluator)
    evaluator.cfg = SimpleNamespace(episode_num=1)
    evaluator.metric = SuccessRateMetric()

    task_queue: queue.Queue[RoboTwinTaskQueueItem] = queue.Queue()
    task_info = TaskInfo(
        tasks={"task_a": TaskStatus()},
        task_queue=task_queue,
        episode_per_task=1,
    )
    workers = [
        SimpleNamespace(
            generation=0,
            item=RoboTwinTaskQueueItem(task_name="task_a", seed=1),
            thread=object(),
            started_at=1.0,
        )
    ]
    result_queue: queue.Queue = queue.Queue()
    worker_metric = _FakeWorkerMetric(task_name="task_a", seed=1, success=True)
    result_queue.put(
        SimpleNamespace(
            worker_idx=0,
            generation=0,
            item=RoboTwinTaskQueueItem(task_name="task_a", seed=1),
            next_seed=2,
            metric_info={"last_update": worker_metric.last_update_info},
            metrics=worker_metric,
            error=None,
        )
    )

    with patch("builtins.print") as mock_print:
        evaluator._log_task_started_if_needed(
            task_info=task_info,
            item=RoboTwinTaskQueueItem(task_name="task_a", seed=1),
        )
        evaluator._drain_worker_results(
            result_queue=result_queue,
            workers=workers,
            task_info=task_info,
            policy_or_cfg="policy",
            device="cpu",
        )

    printed_lines = [
        " ".join(str(arg) for arg in call.args)
        for call in mock_print.call_args_list
    ]
    assert any("task_a: started" in line for line in printed_lines)
    assert any(
        "task_a: Success rate: 1/1 => 100.0%" in line for line in printed_lines
    )


def test_robotwin_remote_evaluator_passes_seed_video_path():
    evaluator = RoboTwinRemoteEvaluator.__new__(RoboTwinRemoteEvaluator)
    evaluator.cfg = SimpleNamespace(
        config_type="demo_clean",
        artifact_root_dir="/job_data",
    )

    fake_evaluator = _FakeEvaluator()
    result_queue: queue.Queue = queue.Queue()

    evaluator._task_eval_thread(
        worker_idx=0,
        generation=0,
        evaluator=fake_evaluator,
        item=RoboTwinTaskQueueItem(
            task_name="task_a",
            seed=123,
            episode_idx=4,
        ),
        result_queue=result_queue,
    )

    assert fake_evaluator.reset_calls == [
        {
            "seed": 123,
            "task_name": "task_a",
            "clear_cache": True,
            "return_obs": True,
            "video_dir": "/job_data/task_a/demo_clean",
            "video_episode_idx": 4,
        }
    ]
