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
from dataclasses import dataclass
from typing import Any, Iterable, Literal

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
    EVAL_SEED_BASE,
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
    seed: int | None


@dataclass
class TaskStatus:
    episode_count: int = 0


@dataclass
class TaskInfo:
    tasks: dict[str, TaskStatus]
    task_queue: queue.Queue[RoboTwinTaskQueueItem]
    episode_per_task: int

    lock: threading.RLock = threading.RLock()

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
            ret = self.task_queue.get()
            self.tasks[ret.task_name].episode_count += 1
            return ret

    def push_task_item(
        self,
        item: RoboTwinTaskQueueItem,
    ) -> bool:
        """Push a task item back to the queue.

        The item should be pushed back only if the task has not
        reached the episode_per_task limit.

        User should make sure that the item information is updated
        correctly before pushing back.

        """
        with self.lock:
            name = item.task_name
            if self.tasks[name].episode_count < self.episode_per_task:
                self.task_queue.put(item)
                return True
            else:
                return False


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
        task_info.info_list.append(
            {"seed": step_return.info["seed"], "success": success}
        )
        self.last_update_info = {
            "task_name": task_name,
            "seed": step_return.info["seed"],
            "success": success,
        }

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
                    seed=self._robotwin_eval_start_seed,
                )
            )
        task_info = TaskInfo(
            tasks={task_name: TaskStatus() for task_name in current_tasks},
            task_queue=task_queue,
            episode_per_task=self.cfg.episode_num,
        )

        if need_setup:
            # the first time to setup.
            # the evaluator number is guaranteed to be not greater than
            # the task number.
            future_setups = []
            for evaluator in self.evaluators:
                cur_task = current_tasks[0]
                env_cfg = self._generate_env_cfg(cur_task)
                future = evaluator.async_setup(
                    env_cfg=env_cfg,
                    policy_or_cfg=policy_or_cfg,
                    metrics=SuccessRateMetric(),
                    device=device,
                )
                future_setups.append(future)
            for future in future_setups:
                future.result()

        threads: list[threading.Thread] = []
        for evaluator in self.evaluators:
            # reset metrics before evaluation
            thread = threading.Thread(
                target=self._task_eval_thread,
                args=(evaluator, task_info),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        # collect metrics
        return True

    def _task_eval_thread(
        self,
        evaluator: PolicyEvaluatorRemote,
        info: TaskInfo,
    ):
        evaluator.reset_metrics()
        while (item := info.get_task_item()) is not None:
            env_reset_kwargs = {
                "seed": item.seed,
                "task_name": item.task_name,
                "clear_cache": True,
                "return_obs": False,
            }
            _, env_info = evaluator.reset_env(**env_reset_kwargs)
            item.seed = env_info["seed"] + 1
            info.push_task_item(item)
            metric_info = evaluator.evaluate_episode(
                max_steps=1500,
                env_reset_kwargs={
                    "clear_cache": True,
                    "seed": None,
                },
            )
            # update metrics
            last_update = metric_info["last_update"]
            current_metrics = evaluator.get_metrics()
            assert isinstance(current_metrics, SuccessRateMetric)
            self.metric.merge([current_metrics])
            evaluator.reset_metrics()
            success_info = self.metric.info[item.task_name]
            print(
                "Episode done: ",
                last_update,
                " Task status: ",
                success_info.summary(),
                flush=True,
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
            format_datatypes=True,
            task_config_path=task_config_path,
            seed=self._robotwin_eval_start_seed,
        )

    @property
    def _robotwin_eval_start_seed(self) -> int:
        return EVAL_SEED_BASE * (1 + self.cfg.seed)


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

    num_parallel_envs: int = Field(ge=1, default=1)
    """Number of parallel environments to run in the remote evaluator."""

    remote_class_config: RayRemoteClassConfig = RayRemoteClassConfig(
        num_cpus=8, num_gpus=1, memory=16 * 1024**3
    )
    """The configuration for the remote class."""
    ray_init_config: dict[str, Any] | None = None

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
