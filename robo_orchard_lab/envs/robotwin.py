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
import functools
import importlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias

import gymnasium as gym
import numpy as np
import yaml
from robo_orchard_core.envs.env_base import EnvBase, EnvBaseCfg, EnvStepReturn
from robo_orchard_core.utils.logging import LoggerManager
from typing_extensions import Literal

if TYPE_CHECKING:
    from envs._base_task import Base_Task


logger = LoggerManager().get_child(__name__)

InstructionType: TypeAlias = Literal["seen", "unseen"]


@dataclass
class RoboTwinEnvStepReturn(EnvStepReturn[dict[str, Any] | None, bool]):
    observations: dict[str, Any] | None
    terminated: bool
    rewards: bool
    """The rewards is a boolean indicating whether the task was successful."""
    truncated: bool
    """Whether the episode was truncated due to reaching the step limit."""


class RoboTwinEnv(EnvBase[dict[str, Any] | None, bool]):
    """RoboTwin environment.

    This class provides RoboTwin environment with robo_orchard_core interface.
    To make it work, you need to install RoboTwin and set the `RoboTwin_PATH`
    environment variable to the path of the RoboTwin package.


    After initialization, you need to call `reset()` to create the environment.

    """

    def __init__(self, cfg: RoboTwinEnvCfg):
        self.cfg = cfg

    def _check_and_update_seed(self):
        instructions = None
        task = None
        from description.utils.generate_episode_instructions import (
            generate_episode_descriptions,
        )

        if self.cfg.check_expert:
            logger.info(
                "Checking expert trajectory for the task. "
                "This may take a while... "
                "You can set `check_expert=False` to skip this step."
            )
            task, success = self._check_expert_traj()
            while not success:
                logger.warning(
                    f"Task {self.cfg.task_name} with seed {self.cfg.seed} "
                    "failed to complete using expert trajectory. "
                    "Using a new seed."
                )
                self.cfg.seed += 1
                task, success = self._check_expert_traj()
            assert task is not None
            instructions = generate_episode_descriptions(
                self.cfg.task_name,
                [task.info["info"]],
                max_descriptions=self.cfg.max_instruction_num,
            )[0]
        else:
            if self.cfg.check_task_init:
                logger.info(
                    "Checking expert trajectory for the task. "
                    "This may take a while... "
                    "You can set `check_task_init=False` to skip this step."
                )
                task, success = self._check_expert_traj()
                if task is None:
                    raise RuntimeError(
                        f"Failed to create task {self.cfg.task_name} "
                        f"with seed {self.cfg.seed}. Please try a different "
                        "seed or check the task configuration."
                    )
                instructions = generate_episode_descriptions(
                    self.cfg.task_name,
                    [task.info["info"]],
                    max_descriptions=self.cfg.max_instruction_num,
                )[0]
            else:
                task = self._create_task()
                instructions = None

        return task, instructions

    @property
    def current_seed(self) -> int:
        """The current seed of the environment."""
        return self.cfg.seed

    @property
    def instructions(self) -> dict | None | str:
        """The instructions for the environment.

        This property is only valid if the environment is initialized
        with `check_expert=True` or `check_task_init=True`.

        If in eval_mode, return the instruction from the task, usually
        a string, otherwise the returned instruction is a dictionary
        containing multiple instructions, with maximum number specified
        by `max_instruction_num`.

        """
        if self.cfg.eval_mode:
            assert self._instructions is not None
            # random pick one in unseen instructions
            eval_instruction_type: InstructionType = "unseen"
            return np.random.choice(self._instructions[eval_instruction_type])

        else:
            return self._instructions

    def _create_task(self) -> Base_Task:
        with in_robotwin_workspace():
            task = create_task_from_name(self.cfg.task_name)
            task_config = self.cfg.get_task_config()
            task.setup_demo(**task_config)  # type: ignore
            return task

    def _check_expert_traj(self) -> tuple[Base_Task | None, bool]:
        """Check whether current config can success if using expert trajectory.

        Returns:
            tuple[Base_Task | None, bool]: A tuple containing the task and a
                boolean indicating whether the task was successful.

        """
        with in_robotwin_workspace():
            task = create_task_from_name(self.cfg.task_name)
            config = self.cfg.get_task_config()
            config["render_freq"] = 0
            try:
                task.setup_demo(**config)  # type: ignore
                task.play_once()  # type: ignore
            except Exception as e:
                logger.error(
                    f"Failed to play the task config {self.cfg} "
                    f"with error: {e}"
                )
                return task, False
            finally:
                task.close_env()

        success: bool = task.plan_success and task.check_success()  # type: ignore
        return task, success

    def step(self, action: list[float] | np.ndarray) -> RoboTwinEnvStepReturn:
        """Take a step in the environment.

        Args:
            action (list[float] | np.ndarray): The action to take in the
                environment. Actually it is the joint positions of the
                robot. The action should be 1-D array with length matching
                the task.

        Returns:
            RoboTwinEnvStepReturn | None: The observation and environment
                state after taking the action, or None if the episode
                has ended.


        """
        if isinstance(action, np.ndarray):
            if action.ndim != 1:
                raise ValueError(
                    "Action should be a 1-D array, "
                    f"but got {action.ndim} dimensions."
                )
        # the take_action method will do internal check if reach step limit
        # or task is successful. Either case, the task will not take further
        # actions.
        self._task.take_action(action)

        # when reach step limit, truncated is True
        if self._task.take_action_cnt >= self._task.step_lim:  # type: ignore
            truncated = True
        else:
            truncated = False

        # robotwin env does not have a concept of done.
        # when a task is evaluated as success, the task does not
        # take further actions anymore. We consider the episode
        # is done when the task is successful.
        if self._task.eval_success:
            terminated = True
        else:
            terminated = False

        return RoboTwinEnvStepReturn(
            observations=self._task.get_obs(),
            rewards=self._task.eval_success,
            terminated=terminated,
            truncated=truncated,
            info=self._task.info,
        )

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        seed: int | None = None,
        clear_cache: bool = False,
    ) -> tuple[dict[str, Any] | None, dict]:
        """Reset the environment.

        If the environment has not been reset before, or the seed is
        different from the previous one, the environment will be re-created
        and check the seed.

        Warning:
            RoboTwin does not use local RandomGenerator, when the environment
            is re-created, the seed will be set to the one in the config
            for both numpy and torch. This may affect the randomness of other
            parts of the code!
            This is a BUG in RoboTwin!

        """

        self.close(clear_cache=clear_cache)
        if not hasattr(self, "_task") or (
            seed is not None and seed != self.cfg.seed
        ):
            task, instructions = self._check_and_update_seed()
            assert task is not None
            self._task = task
            self._instructions = instructions

        with in_robotwin_workspace():
            task_config = self.cfg.get_task_config()
            self._task.setup_demo(**task_config)  # type: ignore

        return self._task.get_obs(), self._task.info

    def close(self, clear_cache: bool = True):
        """Close the environment."""
        if hasattr(self, "_task") and self._task is not None:
            self._task.close_env(clear_cache=clear_cache)
            if self._task.render_freq > 0:
                self._task.viewer.close()

    @property
    def num_envs(self) -> int:
        # always 1 because RoboTwin does not support multi-envs
        return 1

    @property
    def action_space(self) -> gym.Space:
        """The action space of the environment.

        Actually RoboTwin does not implement the action space!
        Call this method will raise an error!

        Returns:
            gym.Space: The action space of the environment.
        """
        return self._task.action_space

    @property
    def observation_space(self) -> gym.Space:
        """The observation space of the environment.

        Actually RoboTwin does not implement the observation space!
        Call this method will raise an error!

        Returns:
            gym.Space: The observation space of the environment.
        """
        return self._task.observation_space

    def unwrapped_env(self) -> Base_Task:
        """Get the original RoboTwin environment."""
        return self._task


class RoboTwinEnvCfg(EnvBaseCfg[RoboTwinEnv]):
    """Configuration for the RoboTwin environment."""

    class_type: type[RoboTwinEnv] = RoboTwinEnv

    task_name: str
    """The name of the task to run, e.g., 'place_object_scale'."""

    seed: int = 0
    """The random seed for the environment.

    If eval_mode is True, the seed will be changed in `__post_init__` to
    `100000 * (1 + seed)` to match the implementation in RoboTwin.
    """

    episode_id: int = 0
    """The episode ID for the environment, used for logging and tracking."""

    check_expert: bool = False
    """Whether to check the expert trajectory for the task.

    If true, the environment will attempt to run the task with given seed
    to check if the task can be completed successfully using the expert
    trajectory. If fails, new seed will be generated and used.

    This field is used to make sure that the environment can be recorded
    successfully using the expert trajectory for imitation learning.

    """

    check_task_init: bool = True
    """Whether to check the task initialization.

    If true, the environment will call `play_once()` to execute the task
    with expert trajectory to check if the task can be initialized
    successfully.

    This field should be set to True because some task attributes that
    required for interaction may be initialized in the `play_once()` method,
    such as `place_object_scale` task.

    This should be a BUG in RoboTwin and will significantly affect the
    performance of the environment initialization.

    """

    eval_mode: bool = False
    """Whether for evaluation.

    If true, the environment will use unseen texture_type.
    """

    max_instruction_num: int = 10
    """The maximum number of instructions to generate for the env."""

    def __post_init__(self):
        task_config_path = self.task_config_path
        if not os.path.exists(task_config_path):
            raise FileNotFoundError(
                f"Task configuration file {task_config_path} does not exist."
            )

        # check that check_expert or check_task_init can not be both True
        if self.check_expert and self.check_task_init:
            raise ValueError(
                "check_expert and check_task_init can not be both True."
            )

        if self.eval_mode and self.check_expert is False:
            logger.info(
                "Set check_expert from False to True for eval_mode."
                "This is to make sure the environment can successfully "
                "be initialized and completed using expert trajectory."
            )
            self.check_expert = True
            self.check_task_init = False

        if self.eval_mode and self.seed < 100000:
            new_seed = 100000 * (1 + self.seed)
            logger.info(
                f"Eval mode is on, changing seed from {self.seed} "
                f"to {new_seed}"
            )
            self.seed = new_seed

    @property
    def task_config_path(self) -> str:
        """Path to the task configuration file."""
        robo_twin_root = config_robotwin_path()
        ret = os.path.join(
            robo_twin_root, "task_config", f"{self.task_name}.yml"
        )
        if os.path.exists(ret):
            return ret
        else:
            # for RoboTwin 2.0.
            return os.path.join(
                robo_twin_root, "task_config", "_config_template.yml"
            )

    @property
    def embodiment_config_path(self) -> str:
        """Path to the embodiment configuration file."""
        robo_twin_root = config_robotwin_path()
        return os.path.join(
            robo_twin_root, "task_config", "_embodiment_config.yml"
        )

    @property
    def camera_config_path(self) -> str:
        """Path to the camera configuration file."""
        robo_twin_root = config_robotwin_path()
        return os.path.join(
            robo_twin_root, "task_config", "_camera_config.yml"
        )

    def get_task_config(self) -> dict[str, Any]:
        """Get the configuration for the task."""

        with (
            open(self.task_config_path, "r", encoding="utf-8") as f,
        ):
            task_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            return self._update_task_config(task_config)

    def _update_task_config(self, task_args: dict[str, Any]) -> dict[str, Any]:
        """Update the task configuration.

        The function reads additional configuration files for task arguments
        such as embodiment and camera settings, and updates the task arguments
        accordingly. The returned dictionary is used for `setup_demo()`.

        """
        embodiment_type: list[str] = task_args.get("embodiment")  # type: ignore
        with open(self.embodiment_config_path, "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(embodiment_type: str) -> str:
            robot_file = embodiment_types[embodiment_type]["file_path"]
            if robot_file is None:
                raise ValueError("No embodiment files")
            return robot_file

        def get_embodiment_config(robot_file):
            robot_config_file = os.path.join(robot_file, "config.yml")
            with open(robot_config_file, "r", encoding="utf-8") as f:
                embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
            return embodiment_args

        with open(self.camera_config_path, "r", encoding="utf-8") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = task_args["camera"]["head_camera_type"]
        task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
        task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

        if len(embodiment_type) == 1:
            task_args["left_robot_file"] = get_embodiment_file(
                embodiment_type[0]
            )
            task_args["right_robot_file"] = get_embodiment_file(
                embodiment_type[0]
            )
            task_args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            task_args["left_robot_file"] = get_embodiment_file(
                embodiment_type[0]
            )
            task_args["right_robot_file"] = get_embodiment_file(
                embodiment_type[1]
            )
            task_args["embodiment_dis"] = embodiment_type[2]
            task_args["dual_arm_embodied"] = False
        else:
            raise RuntimeError("embodiment items should be 1 or 3")

        task_args["left_embodiment_config"] = get_embodiment_config(
            task_args["left_robot_file"]
        )
        task_args["right_embodiment_config"] = get_embodiment_config(
            task_args["right_robot_file"]
        )
        if len(embodiment_type) == 1:
            embodiment_name = str(embodiment_type[0])
        else:
            embodiment_name = (
                str(embodiment_type[0]) + "+" + str(embodiment_type[1])
            )
        task_args["embodiment_name"] = embodiment_name

        # update attributes in self

        task_args["seed"] = self.seed
        task_args["now_ep_num"] = self.episode_id
        task_args["eval_mode"] = self.eval_mode

        return task_args


@functools.lru_cache(maxsize=1)
def config_robotwin_path() -> str:
    robo_twin_path = os.environ.get("RoboTwin_PATH", default=None)
    if robo_twin_path is None:
        raise ValueError(
            "RoboTwin_PATH environment variable is not set. "
            "Please set it to the path of the RoboTwin package."
        )
    if robo_twin_path not in sys.path:
        sys.path.append(robo_twin_path)
    return robo_twin_path


@contextmanager
def in_robotwin_workspace():
    """Context manager to temporarily change the `cwd` to the RoboTwin root."""
    robotwin_root = config_robotwin_path()
    original_cwd = os.getcwd()
    os.chdir(robotwin_root)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def create_task_from_name(task_name: str) -> Base_Task:
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as _:
        raise ImportError(
            f"Failed to import environment class {task_name} from "
            f"module {envs_module.__name__}. "
            "Please ensure the class name matches the task name and "
            "is defined in the module."
        )
    return env_instance
