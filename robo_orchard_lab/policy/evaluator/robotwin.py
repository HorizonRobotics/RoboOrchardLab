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
import copy
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import ray
import torch
from pydantic import Field
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
)
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.ray import RayRemoteClassConfig

from robo_orchard_lab.envs.robotwin import (
    RoboTwinEnvCfg,
    RoboTwinEnvStepReturn,
)
from robo_orchard_lab.envs.robotwin.env import (
    EVAL_INSTRUCTION_NUM,
    config_robotwin_path,
    create_task_from_name,
    in_robotwin_workspace,
)
from robo_orchard_lab.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_lab.policy.evaluator import (
    PolicyEvaluatorConfig,
    PolicyEvaluatorRemote,
)

logger = LoggerManager().get_child(__name__)


SEM_TASKS_16 = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "dump_bin_bigbin",
    "handover_mic",
    "lift_pot",
    "move_pillbottle_pad",
    "open_laptop",
    "open_microwave",
    "place_cans_plasticbox",
    "place_dual_shoes",
    "place_empty_cup",
    "rotate_qrcode",
    "stack_blocks_three",
    "stack_bowls_three",
)


@dataclass
class RoboTwinTaskQueueItem:
    task_name: str
    offset_seed: int = 0
    episode_id: int = 0


@dataclass
class TaskStatus:
    episode_count: int = 0


@dataclass
class TaskInfo:
    tasks: dict[str, TaskStatus]
    task_queue: queue.Queue[RoboTwinTaskQueueItem]
    episode_per_task: int
    started_tasks: set[str] | None = None

    lock: threading.RLock = threading.RLock()

    def __post_init__(self) -> None:
        if self.started_tasks is None:
            self.started_tasks = set()

    def get_task_item(self) -> RoboTwinTaskQueueItem | None:
        """Get a task item from the queue.

        If all tasks have reached the episode_per_task limit,
        return None.
        """
        with self.lock:
            if all(
                info.episode_count >= self.episode_per_task
                for info in self.tasks.values()
            ):
                return None
            while not self.task_queue.empty():
                ret = self.task_queue.get()
                if (
                    self.tasks[ret.task_name].episode_count
                    < self.episode_per_task
                ):
                    return ret
            return None

    def complete_task_item(
        self,
        item: RoboTwinTaskQueueItem,
        next_offset_seed: int,
    ) -> bool:
        """Mark an episode complete and enqueue the next offset if needed.

        Returns:
            bool: True if the next offset was enqueued, False otherwise.
        """
        with self.lock:
            name = item.task_name
            self.tasks[name].episode_count += 1
            if self.tasks[name].episode_count < self.episode_per_task:
                self.task_queue.put(
                    RoboTwinTaskQueueItem(
                        task_name=name,
                        offset_seed=next_offset_seed,
                        episode_id=item.episode_id + 1,
                    )
                )
                return True
            else:
                return False

    def requeue_task_item(
        self,
        item: RoboTwinTaskQueueItem,
    ) -> bool:
        """Put an unfinished task item back into the queue."""
        with self.lock:
            if (
                self.tasks[item.task_name].episode_count
                >= self.episode_per_task
            ):
                return False
            self.task_queue.put(item)
            return True

    def has_pending_work(self) -> bool:
        """Return whether there are remaining episodes to evaluate."""
        with self.lock:
            return any(
                info.episode_count < self.episode_per_task
                for info in self.tasks.values()
            )

    def mark_task_started(self, task_name: str) -> bool:
        """Return whether the task is seen for the first time."""
        with self.lock:
            assert self.started_tasks is not None
            if task_name in self.started_tasks:
                return False
            self.started_tasks.add(task_name)
            return True


@dataclass
class WorkerEvalResult:
    worker_idx: int
    generation: int
    item: RoboTwinTaskQueueItem
    next_offset_seed: int | None = None
    metric_info: dict[str, Any] | None = None
    metrics: SuccessRateMetric | None = None
    error: BaseException | None = None


@dataclass
class WorkerSlot:
    evaluator: PolicyEvaluatorRemote
    generation: int = 0
    item: RoboTwinTaskQueueItem | None = None
    thread: threading.Thread | None = None
    started_at: float | None = None

    @property
    def busy(self) -> bool:
        return self.item is not None


@dataclass
class SuccessRateInfo:
    task_name: str
    success_count: int
    total_count: int
    info_list: list[dict[str, Any]]

    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return float(self.success_count) / self.total_count

    def merge(
        self,
        infos: Iterable[SuccessRateInfo],
    ):
        if not infos:
            raise ValueError("No SuccessRateInfo to merge.")
        ret = self
        for info in infos:
            if ret.task_name != info.task_name:
                raise ValueError(
                    "Cannot merge SuccessRateInfo with different task names."
                )
            ret.success_count += info.success_count
            ret.total_count += info.total_count
            ret.info_list.extend(info.info_list)

    def summary(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "success_count": self.success_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate(),
        }


class SuccessRateMetric:
    info: dict[str, SuccessRateInfo]

    last_update_info: dict | None

    def __init__(self):
        self.info = {}
        self.last_update_info = None

    def reset(self, **kwargs):
        self.info.clear()
        self.last_update_info = None

    def update(self, action: Any, step_return: RoboTwinEnvStepReturn):
        assert step_return.truncated or step_return.terminated, (
            "Episode not done yet. Either truncated or "
            "terminated should be True."
        )
        assert step_return.info is not None
        task_name: str = step_return.info["task"]
        success: bool = step_return.rewards
        if task_name not in self.info:
            self.info[task_name] = SuccessRateInfo(
                task_name=task_name,
                success_count=0,
                total_count=0,
                info_list=[],
            )
        task_info = self.info[task_name]
        task_info.total_count += 1
        if success:
            task_info.success_count += 1
        seed_info = {
            "task_name": task_name,
            "seed": step_return.info["seed"],
            "start_seed": step_return.info["start_seed"],
            "resolved_start_seed": step_return.info["resolved_start_seed"],
            "offset_seed": step_return.info["offset_seed"],
            "success": success,
        }
        task_info.info_list.append(seed_info)
        self.last_update_info = seed_info

    def compute(self) -> dict:
        success_rates: list[float] = []
        summarys = []
        for _, info in self.info.items():
            success_rates.append(info.success_rate())
            summarys.append(info.summary())

        average_success_rate = (
            sum(success_rates) / len(success_rates) if success_rates else 0.0
        )
        return {
            "tasks": summarys,
            "average_success_rate": average_success_rate,
            "last_update": self.last_update_info,
        }

    def to(self, *args, **kwargs):
        pass

    def merge(
        self,
        metrics: Iterable[SuccessRateMetric],
    ):
        ret = self
        for metric in metrics:
            for task_name, info in metric.info.items():
                if task_name not in ret.info:
                    ret.info[task_name] = info
                else:
                    ret.info[task_name].merge([info])


class RoboTwinRemoteEvaluator:
    """Evaluate policies on RoboTwin tasks using remote ray evaluators.

    This evaluator runs multiple remote evaluators in parallel to
    evaluate a list of RoboTwin tasks, collecting success rate metrics
    for each task.

    Compared to official RoboTwin evaluation scripts, this evaluator
    runs evaluations in parallel using Ray, which can significantly
    speed up the evaluation process when multiple CPUs/GPUs are available:

    - Tasks are distributed among multiple remote evaluators.
    - Episodes for each task are run in parallel across the remote evaluators
      while the seed for each episode is incremented to ensure diversity.

    """

    cfg: RoboTwinRemoteEvaluatorCfg

    evaluators: list[PolicyEvaluatorRemote]
    metric: SuccessRateMetric

    def __init__(
        self,
        cfg: RoboTwinRemoteEvaluatorCfg,
    ) -> None:
        self.cfg = cfg

        remote_cfg = PolicyEvaluatorConfig().as_remote(
            remote_class_config=cfg.remote_class_config,
            ray_init_config=cfg.ray_init_config,
        )
        self._remote_cfg = remote_cfg

        self.evaluators = [remote_cfg() for _ in range(cfg.num_parallel_envs)]
        self.metric = SuccessRateMetric()

    def evaluate(
        self,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        device: str | torch.device | None = None,
    ) -> dict:
        """Evaluate the policy on all tasks and return the metrics."""
        remaining_task = copy.deepcopy(self.cfg.task_names)
        need_setup = True
        while self._evaluate_tasks(
            remaining_task=remaining_task,
            policy_or_cfg=policy_or_cfg,
            need_setup=need_setup,
            device=device,
        ):
            need_setup = False

        return self.metric.compute()

    def _evaluate_tasks(
        self,
        remaining_task: list[str],
        policy_or_cfg: PolicyConfig | PolicyMixin,
        need_setup: bool = True,
        device: str | torch.device | None = None,
    ) -> bool:
        if len(remaining_task) == 0:
            return False

        # prepare current tasks
        current_tasks: list[str] = []
        for _ in range(min(len(self.evaluators), len(remaining_task))):
            cur_task = remaining_task.pop(0)
            current_tasks.append(cur_task)

        # prepare task_info
        task_queue: queue.Queue[RoboTwinTaskQueueItem] = queue.Queue()
        for task_name in current_tasks:
            task_queue.put(
                RoboTwinTaskQueueItem(
                    task_name=task_name,
                    offset_seed=0,
                    episode_id=0,
                )
            )
        task_info = TaskInfo(
            tasks={task_name: TaskStatus() for task_name in current_tasks},
            task_queue=task_queue,
            episode_per_task=self.cfg.episode_num,
        )

        if need_setup:
            future_setups = []
            for evaluator in self.evaluators:
                cur_task = current_tasks[0]
                future = self._setup_evaluator_async(
                    evaluator=evaluator,
                    task_name=cur_task,
                    policy_or_cfg=policy_or_cfg,
                    device=device,
                )
                future_setups.append(future)
            for future in future_setups:
                future.result()

        result_queue: queue.Queue[WorkerEvalResult] = queue.Queue()
        workers = [
            WorkerSlot(evaluator=evaluator) for evaluator in self.evaluators
        ]

        while task_info.has_pending_work() or any(
            worker.busy for worker in workers
        ):
            for worker_idx, worker in enumerate(workers):
                if worker.busy:
                    continue
                item = task_info.get_task_item()
                if item is None:
                    continue
                self._log_task_started_if_needed(
                    task_info=task_info, item=item
                )
                thread = threading.Thread(
                    target=self._task_eval_thread,
                    args=(
                        worker_idx,
                        worker.generation,
                        worker.evaluator,
                        item,
                        result_queue,
                    ),
                    daemon=True,
                )
                worker.item = item
                worker.thread = thread
                worker.started_at = time.monotonic()
                thread.start()

            self._drain_worker_results(
                result_queue=result_queue,
                workers=workers,
                task_info=task_info,
                policy_or_cfg=policy_or_cfg,
                device=device,
            )
            self._handle_worker_timeouts(
                workers=workers,
                task_info=task_info,
                policy_or_cfg=policy_or_cfg,
                device=device,
            )

            if task_info.has_pending_work() and not any(
                worker.busy for worker in workers
            ):
                continue

            time.sleep(self.cfg.worker_poll_interval_s)

        return True

    def _task_eval_thread(
        self,
        worker_idx: int,
        generation: int,
        evaluator: PolicyEvaluatorRemote,
        item: RoboTwinTaskQueueItem,
        result_queue: queue.Queue[WorkerEvalResult],
    ):
        try:
            evaluator.reset_metrics()
            env_reset_kwargs = {
                "offset_seed": item.offset_seed,
                "task_name": item.task_name,
                "episode_id": item.episode_id,
                "clear_cache": True,
                "return_obs": True,
                "video_dir": self._get_episode_video_dir(
                    task_name=item.task_name,
                ),
            }
            metric_info = evaluator.evaluate_episode(
                max_steps=1500,
                env_reset_kwargs=env_reset_kwargs,
            )
            current_metrics = evaluator.get_metrics()
            assert isinstance(current_metrics, SuccessRateMetric)
            assert current_metrics.last_update_info is not None
            actual_offset_seed = current_metrics.last_update_info[
                "offset_seed"
            ]
            result_queue.put(
                WorkerEvalResult(
                    worker_idx=worker_idx,
                    generation=generation,
                    item=item,
                    next_offset_seed=actual_offset_seed + 1,
                    metric_info=metric_info,
                    metrics=current_metrics,
                )
            )
        except BaseException as exc:
            result_queue.put(
                WorkerEvalResult(
                    worker_idx=worker_idx,
                    generation=generation,
                    item=item,
                    error=exc,
                )
            )

    def _setup_evaluator_async(
        self,
        evaluator: PolicyEvaluatorRemote,
        task_name: str,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        device: str | torch.device | None = None,
    ):
        env_cfg = self._generate_env_cfg(task_name)
        return evaluator.async_setup(
            env_cfg=env_cfg,
            policy_or_cfg=policy_or_cfg,
            metrics=SuccessRateMetric(),
            device=device,
        )

    def _replace_evaluator(
        self,
        worker: WorkerSlot,
        task_name: str,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        device: str | torch.device | None = None,
    ) -> PolicyEvaluatorRemote:
        self._kill_evaluator(worker.evaluator)
        worker.generation += 1
        worker.evaluator = self._remote_cfg()
        self._setup_evaluator_async(
            evaluator=worker.evaluator,
            task_name=task_name,
            policy_or_cfg=policy_or_cfg,
            device=device,
        ).result()
        return worker.evaluator

    def _kill_evaluator(self, evaluator: PolicyEvaluatorRemote) -> None:
        remote = getattr(evaluator, "_remote", None)
        if remote is None:
            return
        try:
            ray.kill(remote, no_restart=True)
        except Exception:
            logger.exception("Failed to kill timed-out remote evaluator.")

    def _log_task_started_if_needed(
        self,
        task_info: TaskInfo,
        item: RoboTwinTaskQueueItem,
    ) -> None:
        if task_info.mark_task_started(item.task_name):
            print(f"{item.task_name}: started", flush=True)

    def _drain_worker_results(
        self,
        result_queue: queue.Queue[WorkerEvalResult],
        workers: list[WorkerSlot],
        task_info: TaskInfo,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        device: str | torch.device | None = None,
    ) -> None:
        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                return

            worker = workers[result.worker_idx]
            if result.generation != worker.generation:
                continue
            if worker.item is None:
                continue

            worker.item = None
            worker.thread = None
            worker.started_at = None

            if result.error is not None:
                task_info.requeue_task_item(result.item)
                logger.error(
                    "Worker failed on task %s offset_seed %s.",
                    result.item.task_name,
                    result.item.offset_seed,
                    exc_info=(
                        type(result.error),
                        result.error,
                        result.error.__traceback__,
                    ),
                )
                self._replace_evaluator(
                    worker=worker,
                    task_name=result.item.task_name,
                    policy_or_cfg=policy_or_cfg,
                    device=device,
                )
                continue

            assert result.metrics is not None
            assert result.metric_info is not None
            assert result.next_offset_seed is not None
            self.metric.merge([result.metrics])
            result.metrics.reset()
            has_next = task_info.complete_task_item(
                result.item, result.next_offset_seed
            )
            success_info = self.metric.info[result.item.task_name]
            print(
                "Episode done: ",
                result.metric_info["last_update"],
                " Task status: ",
                success_info.summary(),
                flush=True,
            )
            if not has_next:
                summary = success_info.summary()
                print(
                    (
                        f"{result.item.task_name}: Success rate: "
                        f"{summary['success_count']}/"
                        f"{summary['total_count']} => "
                        f"{summary['success_rate'] * 100:.1f}%"
                    ),
                    flush=True,
                )

    def _handle_worker_timeouts(
        self,
        workers: list[WorkerSlot],
        task_info: TaskInfo,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        device: str | torch.device | None = None,
    ) -> None:
        timeout = self.cfg.episode_timeout_s
        if timeout is None:
            return

        now = time.monotonic()
        for worker in workers:
            if not worker.busy or worker.started_at is None:
                continue
            if now - worker.started_at <= timeout:
                continue

            assert worker.item is not None
            timed_out_item = worker.item
            task_info.requeue_task_item(timed_out_item)
            logger.warning(
                (
                    "Worker timed out on task %s offset_seed %s after %.1fs. "
                    "Replacing worker."
                ),
                timed_out_item.task_name,
                timed_out_item.offset_seed,
                timeout,
            )
            worker.item = None
            worker.thread = None
            worker.started_at = None
            self._replace_evaluator(
                worker=worker,
                task_name=timed_out_item.task_name,
                policy_or_cfg=policy_or_cfg,
                device=device,
            )

    def _generate_env_cfg(self, task_name: str) -> RoboTwinEnvCfg:
        task_config_path = os.path.join(
            config_robotwin_path(),
            "task_config",
            f"{self.cfg.config_type}.yml",
        )
        if not os.path.exists(task_config_path):
            raise FileNotFoundError(
                f"Task config file not found: {task_config_path}"
            )

        return RoboTwinEnvCfg(
            task_name=task_name,
            check_expert=True,
            check_task_init=False,
            eval_mode=True,
            max_instruction_num=EVAL_INSTRUCTION_NUM,
            format_datatypes=self.cfg.format_datatypes,
            action_type=self.cfg.action_type,
            task_config_path=task_config_path,
            seed=self.cfg.seed,
        )

    def _get_episode_video_dir(
        self,
        task_name: str,
    ) -> str | None:
        if self.cfg.artifact_root_dir is None:
            return None
        return os.path.join(
            self.cfg.artifact_root_dir,
            task_name,
            self.cfg.config_type,
        )


class RoboTwinRemoteEvaluatorCfg(ClassConfig[RoboTwinRemoteEvaluator]):
    class_type: ClassType[RoboTwinRemoteEvaluator] = RoboTwinRemoteEvaluator

    task_names: list[str]
    episode_num: int = 100
    """The number of evaluation episodes for each task."""

    config_type: Literal["demo_clean", "demo_randomized"] = "demo_clean"
    """The type of RoboTwin configuration to use:
        - "demo_clean": Use the clean demo configuration, known as easy tasks.
        - "demo_randomized": Use the randomized demo configuration,
        known as hard tasks.
    """

    seed: int = 0
    """Caller-facing evaluation start seed passed into RoboTwinEnvCfg."""

    format_datatypes: bool = True
    """Whether to convert RoboTwin env observations to orchard datatypes."""

    action_type: Literal["qpos", "ee"] = "qpos"
    """The RoboTwin env action representation used during evaluation."""

    num_parallel_envs: int = Field(ge=1, default=1)
    """Number of parallel environments to run in the remote evaluator."""

    episode_timeout_s: float | None = Field(default=600.0, gt=0)
    """Timeout for one evaluation episode before replacing the worker."""

    worker_poll_interval_s: float = Field(default=0.1, gt=0)
    """Polling interval for collecting worker results and timeout checks."""

    remote_class_config: RayRemoteClassConfig = RayRemoteClassConfig(
        num_cpus=8, num_gpus=1, memory=16 * 1024**3
    )
    """The configuration for the remote class."""
    ray_init_config: dict[str, Any] | None = None
    artifact_root_dir: str | None = None

    def __post_init__(self):
        if self.num_parallel_envs > len(self.task_names):
            warning_msg = (
                f"num_parallel_envs ({self.num_parallel_envs}) is greater "
                f"than the number of task_names ({len(self.task_names)}). "
                f"Setting num_parallel_envs to {len(self.task_names)}."
            )
            logger.warning(warning_msg)
            self.num_parallel_envs = len(self.task_names)

        # check task_name to be valid
        with in_robotwin_workspace():
            for task_name in self.task_names:
                create_task_from_name(task_name)

    def __call__(self, *args, **kwargs):
        return self.create_instance_by_cfg(*args, **kwargs)
