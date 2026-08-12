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
import copy
import importlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeAlias

import gymnasium as gym
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.logging import LoggerManager
from typing_extensions import Literal

from robo_orchard_lab.dataset.datatypes import BatchFrameTransformGraph
from robo_orchard_lab.dataset.experimental.mcap.messages import (
    StampedMessage,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Robot,
)
from robo_orchard_lab.envs.base import (
    EnvBase,
    EnvBaseCfg,
    EnvStepReturn,
    EnvToMcapProtocol,
)
from robo_orchard_lab.envs.robotwin._runtime import (
    _RoboTwinRuntimeLayout,
    _transform_joint_vector_to_eef,
    build_joints_to_eef_transform,
    build_obs_robots,
    derive_runtime_layout,
    dispose_task_runtime,
    get_joint_state_names,
    get_robot_base_tf_graph,
    read_robot_urdfs,
)
from robo_orchard_lab.envs.robotwin._task_config import (
    _RoboTwinTaskConfigSnapshot,
    build_task_config,
    extract_serialized_task_config_snapshot,
    inject_serialized_task_config_snapshot,
    resolve_task_config_source,
    task_config_snapshot_restore_context,
)
from robo_orchard_lab.envs.robotwin.curobo_base_patch import (
    RoboTwinCuroboPatchUnsupportedError,
    prepare_robotwin_runtime_for_cfg,
    setup_robotwin_demo_with_runtime_guards,
)
from robo_orchard_lab.envs.robotwin.kinematics import (
    RoboTwinEEF,
    RoboTwinJointsToEEF,
)
from robo_orchard_lab.envs.robotwin.obs import (
    _build_mcap_observation_messages,
    _extract_video_frame,
    _format_observation,
    _pose_vector_to_tf,
)
from robo_orchard_lab.envs.robotwin.workspace import (
    config_robotwin_path,
    in_robotwin_workspace,
)
from robo_orchard_lab.envs.state import EnvStateScope, StatefulEnvMixin
from robo_orchard_lab.utils.state import (
    State,
    state2obj,
    validate_recovery_state,
)
from robo_orchard_lab.utils.video import (
    VideoBackendUnavailableError,
    VideoPixelFormat,
    VideoWriter,
    VideoWriterError,
)

if TYPE_CHECKING:
    from envs._base_task import (  # pyright: ignore[reportMissingImports]
        Base_Task,
    )

__all__ = [
    "RoboTwinEnvStepReturn",
    "RoboTwinEnv",
    "RoboTwinEnvCfg",
]

EVAL_SEED_BASE = 100000
EVAL_INSTRUCTION_NUM = 100
_logger_manager = LoggerManager()
_logger_manager_logger = _logger_manager.get_logger()
if _logger_manager_logger.handlers:
    _logger_manager_logger.propagate = False
logger = _logger_manager.get_child(__name__)

InstructionType: TypeAlias = Literal["seen", "unseen"]
RoboTwinObsType: TypeAlias = dict[str, Any] | None

ROBOTWIN_VIDEO_FPS = 10
ROBOTWIN_VIDEO_PIXEL_FORMAT = VideoPixelFormat.RGB24
ROBOTWIN_ENV_STATE_SCHEMA_VERSION = 1


@dataclass
class RoboTwinEnvStepReturn(EnvStepReturn[RoboTwinObsType, bool]):
    observations: RoboTwinObsType
    terminated: bool
    rewards: bool
    """The rewards is a boolean indicating whether the task was successful."""
    truncated: bool
    """Whether the episode was truncated due to reaching the step limit."""


class RoboTwinEpisodeInstructionsPayload(BaseModel):
    """Instruction candidates generated for one RoboTwin episode."""

    model_config = ConfigDict(extra="forbid")

    episode_index: int | None = None
    """Episode index used by RoboTwin instruction generation, if available."""
    seen: list[str] = Field(default_factory=list)
    """Instruction candidates sampled from RoboTwin's seen split."""
    unseen: list[str] = Field(default_factory=list)
    """Instruction candidates sampled from RoboTwin's unseen split."""


class RoboTwinObservationInstructionPayload(BaseModel):
    """Instruction metadata exported with one RoboTwin observation frame."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    """Payload schema version for MCAP compatibility checks."""
    instructions: RoboTwinEpisodeInstructionsPayload = Field(
        default_factory=RoboTwinEpisodeInstructionsPayload
    )
    """Instruction candidates attached to the current RoboTwin episode."""
    eval_chosen_instruction: str | None = None
    """Single instruction selected by eval-mode rollout logic, if any."""


class RoboTwinObservationMetaPayload(BaseModel):
    """Episode metadata exported with one RoboTwin observation frame."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    """Payload schema version for MCAP compatibility checks."""
    task_name: str
    """RoboTwin task name for the active episode."""
    action_type: Literal["qpos", "ee"]
    """RoboTwin action mode used by the active episode."""
    episode_id: int
    """Configured episode id for the rollout."""
    seed: int
    """Actual runtime seed after eval-mode resolution and retries."""
    start_seed: int
    """Caller-facing start seed stored on the env config."""
    resolved_start_seed: int
    """Eval-mode-normalized start seed before retry offset is applied."""
    offset_seed: int
    """Env-local retry offset added to ``resolved_start_seed``."""


# State.config owns task_name, start seed, and episode_id; this payload keeps
# only post-reset runtime values that cannot be derived from config.
class _RoboTwinPostResetStatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    scope: Literal[EnvStateScope.POST_RESET]
    offset_seed: int = Field(ge=0)
    task_config: dict[str, Any]
    instructions: RoboTwinEpisodeInstructionsPayload | None = None
    eval_chosen_instruction: str | None = None
    post_reset_state_available: bool | None = Field(
        default=None,
        exclude=True,
    )
    """Deprecated schema-v1 input; availability is derived on restore."""
    episode_finalized: bool = False


class RoboTwinEnv(
    EnvBase[RoboTwinObsType, bool],
    StatefulEnvMixin,
    EnvToMcapProtocol,
):
    """RoboTwin environment wrapped with the orchard env interface.

    This class adapts RoboTwin tasks to the ``robo_orchard_core`` env API.
    To use it, RoboTwin must be installed and the ``RoboTwin_PATH``
    environment variable must point to the RoboTwin package. The wrapper
    supports RoboTwin's official combined dual-arm articulation and separate
    left/right articulation layouts.

    Public interface:

    - ``reset(...)``: create or recreate the RoboTwin task and return the
      initial observation.
    - ``step(action)``: execute one RoboTwin action and return
      ``RoboTwinEnvStepReturn``.
    - ``close(...)``: close the current RoboTwin task.
    - ``finalize_episode()``: finalize episode-local artifacts without
      closing the reusable RoboTwin runtime.
    - ``unwrapped_env()``: return the underlying RoboTwin ``Base_Task``.
    - ``get_robot_urdf()``: return one compatibility ``"left"`` URDF for a
      combined articulation, or truthful ``"left"`` / ``"right"`` URDFs for
      split articulations.
    - ``get_obs_robots()``: return observation-facing robot metadata.
    - ``get_mcap_obs()``: export the latest reset/step observation as typed
      MCAP topic messages without re-sampling RoboTwin.
    - ``get_mcap_action_sidecars(action)``: export optional MCAP sidecars for
      the action that is about to be passed to ``step(action)``.
    - ``current_seed`` / ``instructions`` / ``num_envs``: runtime properties
      exposed by the wrapper.

    Typical usage::

        env = RoboTwinEnv(
            RoboTwinEnvCfg(
                task_name="place_object_basket",
                check_expert=False,
                check_task_init=False,
                action_type="qpos",
            )
        )
        obs, info = env.reset()
        action = np.zeros(14, dtype=np.float32)
        step_ret = env.step(action)
        env.close()

    The example above uses ``action_type="qpos"``. See ``step()`` for the
    exact action layout for both ``"qpos"`` and ``"ee"`` modes.
    """

    supported_state_scopes = frozenset({EnvStateScope.POST_RESET})

    def __init__(self, cfg: RoboTwinEnvCfg):
        self.cfg = cfg
        prepare_robotwin_runtime_for_cfg(self.cfg)
        self._task: Base_Task | None = None
        self._resolved_start_seed = self.cfg.resolve_start_seed(self.cfg.seed)
        self._offset_seed = 0
        self._instructions: dict[str, object] | None = None
        self._eval_chosen_instruction: str | None = None
        self._episode_finalized = True
        self._post_reset_state_available = False
        self._last_obs: dict[str, Any] | None = None
        self._last_obs_step_index: int | None = None
        self._joints_to_eef_transform: RoboTwinJointsToEEF | None = None
        self._cached_obs_robots: dict[str, Robot] | None = None
        self._runtime_layout: _RoboTwinRuntimeLayout | None = None
        self._active_task_config: dict[str, Any] | None = None
        self._pending_disposal_tasks: list[Base_Task] = []
        self._video_writer = VideoWriter(
            pixel_format=ROBOTWIN_VIDEO_PIXEL_FORMAT,
            fps=ROBOTWIN_VIDEO_FPS,
        )

    def _setup_task(
        self,
        *,
        cfg: RoboTwinEnvCfg,
        task: Base_Task,
        task_config: dict[str, Any],
    ) -> None:
        """Set up a task, fully disposing partial runtime state on failure."""

        try:
            setup_robotwin_demo_with_runtime_guards(cfg, task, task_config)
        except BaseException:
            self._dispose_task(task, clear_cache=True)
            raise

    def _dispose_task(
        self,
        task: Base_Task | None,
        *,
        clear_cache: bool,
    ) -> bool:
        """Best-effort full disposal for an owned RoboTwin task runtime.

        RoboTwin's upstream ``close_env`` does not stop heterogeneous planner
        workers. This Env boundary therefore owns the complete shutdown
        protocol for setup failures, retries, State candidates, reset, and
        close. Cleanup failures are logged and never replace the triggering
        exception.
        """

        if task is None:
            return True
        task_is_active = task is self._task
        workers_stopped = dispose_task_runtime(
            task,
            clear_cache=clear_cache,
        )

        if workers_stopped:
            self._forget_pending_disposal(task)
        else:
            self._remember_pending_disposal(task)

        if task_is_active:
            self._clear_active_task_runtime()
        return workers_stopped

    def _remember_pending_disposal(self, task: Base_Task) -> None:
        """Retain ownership when a worker could not be confirmed stopped."""

        if all(
            candidate is not task for candidate in self._pending_disposal_tasks
        ):
            self._pending_disposal_tasks.append(task)

    def _forget_pending_disposal(self, task: Base_Task) -> None:
        """Drop a retained task only after disposal is confirmed complete."""

        self._pending_disposal_tasks = [
            candidate
            for candidate in self._pending_disposal_tasks
            if candidate is not task
        ]

    def _drain_pending_disposals(self, *, clear_cache: bool) -> None:
        """Retry previously incomplete task disposal without losing handles."""

        for task in list(self._pending_disposal_tasks):
            self._dispose_task(task, clear_cache=clear_cache)

    def _clear_active_task_runtime(self) -> None:
        """Clear Env-owned handles and caches for the active task."""

        self._task = None
        self._runtime_layout = None
        self._active_task_config = None
        self._joints_to_eef_transform = None
        self._cached_obs_robots = None
        self._last_obs = None
        self._last_obs_step_index = None

    def _require_task(self) -> Base_Task:
        """Return the active RoboTwin task with optionality narrowed."""
        task = self._task
        if task is None:
            raise RuntimeError(
                "RoboTwinEnv has no active task. Call reset() or "
                "reset_from_state() first."
            )
        return task

    def _check_and_update_seed(
        self,
        task_config: dict[str, Any],
    ) -> tuple[
        Base_Task,
        dict[str, object] | None,
        dict[str, Any],
    ]:
        if not self.cfg.check_expert and not self.cfg.check_task_init:
            return create_task_from_name(self.cfg.task_name), None, task_config

        from description.utils.generate_episode_instructions import (  # pyright: ignore[reportMissingImports]
            generate_episode_descriptions,
        )

        if self.cfg.check_expert:
            logger.debug(
                "Checking RoboTwin expert trajectory: task=%s seed=%s",
                self.cfg.task_name,
                self.current_seed,
            )
            requested_seed = self.current_seed
            task, episode_info, success = self._check_expert_traj(task_config)
            retry_num = 0
            while not success:
                self._dispose_task(task, clear_cache=True)
                retry_num += 1
                if retry_num >= 50:
                    raise RuntimeError(
                        f"Failed to create task {self.cfg.task_name} "
                        f"with expert trajectory after {retry_num} retries. "
                        "Please check the task configuration!"
                    )

                failed_seed = self.current_seed
                self._drain_pending_disposals(clear_cache=True)
                if self._pending_disposal_tasks:
                    raise RuntimeError(
                        "RoboTwin expert retry cannot continue while a "
                        "planner worker from the failed attempt is still "
                        "alive."
                    )
                self._offset_seed += 1
                task_config = self.cfg.get_task_config_for_seed(
                    runtime_seed=self.current_seed
                )
                logger.debug(
                    "RoboTwin expert trajectory check failed: "
                    "task=%s seed=%s retry_seed=%s",
                    self.cfg.task_name,
                    failed_seed,
                    self.current_seed,
                )
                task, episode_info, success = self._check_expert_traj(
                    task_config
                )
            if retry_num > 0:
                logger.info(
                    "RoboTwin expert trajectory resolved after retry: "
                    "task=%s requested_seed=%s actual_seed=%s retries=%s",
                    self.cfg.task_name,
                    requested_seed,
                    self.current_seed,
                    retry_num,
                )

            if task is None or episode_info is None:
                self._dispose_task(task, clear_cache=True)
                raise RuntimeError(
                    "RoboTwin expert trajectory check reported success "
                    "without a reusable task."
                )
        else:
            logger.debug(
                "Checking RoboTwin task init: task=%s seed=%s",
                self.cfg.task_name,
                self.current_seed,
            )
            task, episode_info, _ = self._check_expert_traj(task_config)
            if task is None or episode_info is None:
                self._dispose_task(task, clear_cache=True)
                raise RuntimeError(
                    f"Failed to create task {self.cfg.task_name} "
                    f"with seed {self.cfg.seed}. Please try a different "
                    "seed or check the task configuration."
                )
        try:
            instructions = generate_episode_descriptions(
                self.cfg.task_name,
                [episode_info],
                max_descriptions=self.cfg.max_instruction_num,
            )[0]
        except BaseException:
            self._dispose_task(task, clear_cache=True)
            raise

        return task, instructions, task_config

    @property
    def current_seed(self) -> int:
        """The actual RoboTwin runtime seed for the current episode."""
        return self._resolved_start_seed + self._offset_seed

    @property
    def start_seed(self) -> int:
        """The caller-facing start seed configured on the env."""
        return self.cfg.seed

    @property
    def resolved_start_seed(self) -> int:
        """The eval-mode-normalized runtime start seed."""
        return self._resolved_start_seed

    @property
    def offset_seed(self) -> int:
        """The env-local retry offset from ``resolved_start_seed``."""
        return self._offset_seed

    @property
    def instructions(self) -> dict[str, object] | None | str:
        """The instructions for the environment.

        This property is only valid if the environment is initialized
        with `check_expert=True` or `check_task_init=True`.

        If in eval_mode, return the instruction from the task, usually
        a string, otherwise the returned instruction is a dictionary
        containing multiple instructions, with maximum number specified
        by `max_instruction_num`.

        """
        if self.cfg.eval_mode:
            if self._eval_chosen_instruction is None:
                assert self._instructions is not None
                # random pick one in unseen instructions
                eval_instruction_type: InstructionType = "unseen"
                self._eval_chosen_instruction = np.random.choice(
                    self._instructions[eval_instruction_type]
                )

            return self._eval_chosen_instruction

        else:
            return self._instructions

    def _get_state(self) -> State:
        if not self._post_reset_state_available:
            raise RuntimeError(
                "RoboTwinEnv state is only available after reset() and "
                "before the first step()."
            )
        if self._task is None:
            raise RuntimeError("RoboTwinEnv has no active task to capture.")
        if self._active_task_config is None:
            raise RuntimeError(
                "RoboTwinEnv has no active setup config to capture."
            )

        state_payload: dict[str, object] = _RoboTwinPostResetStatePayload(
            schema_version=ROBOTWIN_ENV_STATE_SCHEMA_VERSION,
            scope=EnvStateScope.POST_RESET,
            offset_seed=self._offset_seed,
            task_config=copy.deepcopy(self._active_task_config),
            instructions=copy.deepcopy(self._instructions),
            eval_chosen_instruction=copy.deepcopy(
                self._eval_chosen_instruction
            ),
            episode_finalized=self._episode_finalized,
        ).model_dump(mode="json")
        return State(
            class_type=type(self),
            config=copy.deepcopy(self.cfg),
            state=state_payload,
            hierarchical_save=None,
        )

    def _set_state(self, state: State) -> None:
        self._restore_post_reset_state(state)

    def reset_from_state(self, state: State) -> tuple[RoboTwinObsType, dict]:
        """Restore a post-reset State and return ``reset(...)`` output."""

        payload = self._restore_post_reset_state(state, activate=False)
        task = self._require_task()
        try:
            obs = self._format_obs(task.get_obs(), step_index=0)
            info = self._get_info()
        except BaseException:
            self._dispose_task(task, clear_cache=True)
            self._post_reset_state_available = False
            self._episode_finalized = True
            raise
        self._last_obs = obs
        self._last_obs_step_index = 0
        self._post_reset_state_available = True
        self._episode_finalized = payload.episode_finalized
        return obs, info

    @staticmethod
    def _validate_post_reset_state_config_consistency(
        cfg: RoboTwinEnvCfg,
        payload: _RoboTwinPostResetStatePayload,
    ) -> None:
        """Reject a task payload that conflicts with its reconstruction config.

        ``State.config`` owns the task identity and caller-facing reset
        inputs. The saved lowered task config must therefore preserve the
        corresponding task name, resolved runtime seed, and episode id before
        restore creates or replaces any live RoboTwin task.
        """

        expected_values = {
            "task_name": cfg.task_name,
            "seed": cfg.resolve_start_seed(cfg.seed) + payload.offset_seed,
            "now_ep_num": cfg.episode_id,
        }
        for key, expected_value in expected_values.items():
            actual_value = payload.task_config.get(key)
            if (
                type(actual_value) is not type(expected_value)
                or actual_value != expected_value
            ):
                raise ValueError(
                    "RoboTwin State payload task_config "
                    f"{key!r} must match State.config: expected "
                    f"{expected_value!r}, got {actual_value!r}."
                )

    def _restore_post_reset_state(
        self,
        state: State,
        *,
        activate: bool = True,
    ) -> _RoboTwinPostResetStatePayload:
        validate_recovery_state(
            state,
            require_class_type=True,
            require_config=True,
            context="RoboTwinEnv state",
        )
        state_class_type = state.class_type
        if state_class_type is not type(self):
            raise TypeError(
                "RoboTwinEnv state class_type must match the target env. "
                f"Got {state_class_type} for {type(self).__name__}."
            )
        if not isinstance(state.config, RoboTwinEnvCfg):
            raise TypeError(
                "RoboTwinEnv state config must be RoboTwinEnvCfg. "
                f"Got {type(state.config).__name__}."
            )

        payload = _RoboTwinPostResetStatePayload.model_validate(
            state2obj(state.state)
        ).model_copy(deep=True)
        cfg = copy.deepcopy(state.config)
        self._validate_post_reset_state_config_consistency(cfg, payload)
        prepare_robotwin_runtime_for_cfg(cfg)
        self._drain_pending_disposals(clear_cache=True)
        if self._pending_disposal_tasks:
            raise RuntimeError(
                "RoboTwinEnv cannot restore State while a planner worker "
                "from an earlier failed candidate is still alive. Call "
                "close() again after the worker exits."
            )

        staged_task: Base_Task | None = None
        try:
            with in_robotwin_workspace():
                if cfg.check_expert or cfg.check_task_init:
                    staged_task, episode_info, success = (
                        self._check_expert_traj(
                            payload.task_config,
                            cfg=cfg,
                        )
                    )
                    if staged_task is None or episode_info is None:
                        raise RuntimeError(
                            "RoboTwin State restore could not replay the "
                            "saved official play_once precheck."
                        )
                    if cfg.check_expert and not success:
                        raise RuntimeError(
                            "RoboTwin State restore replayed an expert "
                            "precheck that no longer succeeds for the saved "
                            "task config."
                        )
                else:
                    staged_task = create_task_from_name(cfg.task_name)
                try:
                    self._setup_task(
                        cfg=cfg,
                        task=staged_task,
                        task_config=payload.task_config,
                    )
                except BaseException:
                    # _setup_task already fully disposes its candidate.
                    staged_task = None
                    raise
            runtime_layout = derive_runtime_layout(staged_task)

            self.close(clear_cache=True)
            if self._pending_disposal_tasks:
                raise RuntimeError(
                    "RoboTwinEnv cannot restore State while a planner worker "
                    "from the previous runtime is still alive. Call close() "
                    "again after the worker exits."
                )
            self.cfg = cfg
            self._resolved_start_seed = self.cfg.resolve_start_seed(
                self.cfg.seed
            )
            self._offset_seed = payload.offset_seed
            self._instructions = (
                None
                if payload.instructions is None
                else payload.instructions.model_dump(mode="json")
            )
            self._eval_chosen_instruction = payload.eval_chosen_instruction
            self._task = staged_task
            self._runtime_layout = runtime_layout
            self._active_task_config = copy.deepcopy(payload.task_config)
            self._joints_to_eef_transform = None
            self._cached_obs_robots = None
            self._last_obs = None
            self._last_obs_step_index = 0
            if activate:
                self._post_reset_state_available = True
                self._episode_finalized = payload.episode_finalized
            else:
                self._post_reset_state_available = False
                self._episode_finalized = True
        except BaseException:
            self._dispose_task(staged_task, clear_cache=True)
            raise
        staged_task = None
        return payload

    def _check_expert_traj(
        self,
        task_config: dict[str, Any],
        *,
        cfg: RoboTwinEnvCfg | None = None,
    ) -> tuple[
        Base_Task | None,
        object | None,
        bool,
    ]:
        """Run RoboTwin's official ``play_once`` precheck lifecycle.

        Returns:
            tuple[Base_Task | None, object | None, bool]: The same task Python
                object initialized by ``play_once()``, a deep-copied
                episode-info payload, and whether the expert trajectory
                succeeded. Before a task is returned, its precheck scene is
                closed with upstream ``close_env()`` exactly as in RoboTwin's
                official evaluator. Planner workers remain attached so the
                caller can run the second ``setup_demo`` on that same object.
                Failed prechecks are fully disposed and return no task.

        """
        effective_cfg = self.cfg if cfg is None else cfg
        with in_robotwin_workspace():
            task = create_task_from_name(effective_cfg.task_name)
            config = copy.deepcopy(task_config)
            config["render_freq"] = 0
            setup_completed = False
            stage = "setup"
            try:
                self._setup_task(
                    cfg=effective_cfg,
                    task=task,
                    task_config=config,
                )
                setup_completed = True
                stage = "play"
                episode_info_payload = task.play_once()  # type: ignore
                stage = "precheck close"
                task.close_env(clear_cache=False)  # type: ignore
                stage = "success check"
                success: bool = bool(
                    task.plan_success and task.check_success()  # type: ignore
                )
                stage = "metadata capture"
                episode_info = copy.deepcopy(episode_info_payload["info"])
            except RoboTwinCuroboPatchUnsupportedError:
                if setup_completed:
                    self._dispose_task(task, clear_cache=True)
                raise
            except Exception as exc:
                logger.debug(
                    "RoboTwin expert trajectory check failed during %s: "
                    "task=%s seed=%s error=%s",
                    stage,
                    effective_cfg.task_name,
                    task_config.get("seed"),
                    exc,
                )
                if setup_completed:
                    self._dispose_task(task, clear_cache=True)
                return None, None, False
            except BaseException:
                if setup_completed:
                    self._dispose_task(task, clear_cache=True)
                raise

        return task, episode_info, success

    def step(self, action: list[float] | np.ndarray) -> RoboTwinEnvStepReturn:
        """Take a step in the environment.

        Args:
            action (list[float] | np.ndarray): The action to take in the
                environment. The exact semantics depend on
                `self.cfg.action_type`.

                - If `self.cfg.action_type == "qpos"`, the action must be a
                  1-D sequence in RoboTwin joint-control order:
                  `[left_arm_joint_targets..., left_gripper,
                  right_arm_joint_targets..., right_gripper]`.
                  The arm-joint counts come from the current RoboTwin robot
                  embodiment, so the expected total length is
                  `len(left_arm_joints_name) + 1 +
                  len(right_arm_joints_name) + 1`. The two gripper values use
                  RoboTwin's normalized gripper convention.

                - If `self.cfg.action_type == "ee"`, the action must be a
                  1-D sequence in RoboTwin end-effector-control order:
                  `[left_xyz(3), left_quat(4), left_gripper,
                  right_xyz(3), right_quat(4), right_gripper]`,
                  where each quaternion follows RoboTwin's
                  `[qw, qx, qy, qz]` convention. This representation always
                  contains 16 values for both combined and split layouts.

                The wrapper validates this expected width exactly before
                forwarding the action to RoboTwin.

        Returns:
            RoboTwinEnvStepReturn: The step result after taking the action.
                This function always returns a step result. Episode end is
                reported via `terminated` and `truncated` instead of
                returning None. `rewards` is a boolean indicating whether
                the task has succeeded.

        Raises:
            RuntimeError: If no active episode is available for stepping.
                This includes newly constructed, closed, and finalized env
                states. Call ``reset()`` or restore a non-finalized ``State``
                with ``reset_from_state()`` before stepping again.
        """
        if self._episode_finalized:
            raise RuntimeError(
                "RoboTwinEnv has no active episode. "
                "Call reset() or reset_from_state() before step()."
            )
        last_obs_step_index = self._last_obs_step_index
        if last_obs_step_index is None:
            raise RuntimeError(
                "RoboTwinEnv.step() requires a successful reset() or "
                "reset_from_state() before advancing the episode."
            )
        next_obs_step_index = last_obs_step_index + 1
        task = self._require_task()

        action_array = np.asarray(action)
        if action_array.ndim != 1:
            raise ValueError(
                "Action should be a 1-D array, "
                f"but got {action_array.ndim} dimensions."
            )

        if self.cfg.action_type == "qpos":
            expected_action_dim = len(
                task.robot.get_left_arm_jointState()
            ) + len(task.robot.get_right_arm_jointState())
        elif self.cfg.action_type == "ee":
            expected_action_dim = 16
        else:
            raise ValueError(
                f"Unsupported RoboTwin action_type: {self.cfg.action_type!r}."
            )

        # RoboTwin silently slices extra dimensions for qpos actions, so the
        # wrapper validates the exact width before forwarding the command.
        if action_array.shape[0] != expected_action_dim:
            raise ValueError(
                "Action width does not match RoboTwin action_type "
                f"{self.cfg.action_type!r}: expected {expected_action_dim}, "
                f"got {action_array.shape[0]}."
            )
        # the take_action method will do internal check if reach step limit
        # or task is successful. Either case, the task will not take further
        # actions.
        task.take_action(
            action_array,
            action_type=self.cfg.action_type,
        )
        self._post_reset_state_available = False

        # when reach step limit, truncated is True
        # Note that step_lim is None for default unlimited steps.
        # It will be set in evaluation mode.
        if task.step_lim is not None and task.take_action_cnt >= task.step_lim:
            truncated = True
        else:
            truncated = False

        # robotwin env does not have a concept of done.
        # when a task is evaluated as success, the task does not
        # take further actions anymore. We consider the episode
        # is done when the task is successful.
        if task.eval_success:
            terminated = True
        else:
            terminated = False

        raw_obs = task.get_obs()
        self._write_video_frame(raw_obs)
        obs = self._format_obs(raw_obs, step_index=next_obs_step_index)
        self._last_obs = obs
        self._last_obs_step_index = next_obs_step_index

        return RoboTwinEnvStepReturn(
            observations=obs,
            rewards=task.eval_success,
            terminated=terminated,
            truncated=truncated,
            info=self._get_info(),
        )

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        seed: int | None = None,
        offset_seed: int | None = None,
        task_name: str | None = None,
        clear_cache: bool = False,
        return_obs: bool = True,
        video_dir: str | None = None,
        episode_id: int | None = None,
    ) -> tuple[RoboTwinObsType, dict]:
        """Reset the environment.

        If the environment has not been reset before, or the seed is
        different from the previous one, or the task_name is different
        from the previous one, the environment will be re-created
        and check the seed. The config ``seed`` remains the caller-facing
        start seed while the env tracks runtime retries through
        ``offset_seed``.

        Warning:
            RoboTwin does not use local RandomGenerator, when the environment
            is re-created, the seed will be set to the one in the config
            for both numpy and torch. This may affect the randomness of other
            parts of the code!
            This is a BUG in RoboTwin!

        Args:
            env_ids (Sequence[int] | None, optional): Not supported.
                Defaults to None.
            seed (int | None, optional): The seed to reset the
                environment start point. If None, the seed in the config will
                be used. If an int is provided, it replaces the caller-facing
                start seed. Default is None.
            offset_seed (int | None, optional): Runtime offset from the
                resolved start seed. If None, the existing env offset is
                reused unless ``seed`` also changes, in which case the offset
                resets to 0. Default is None.
            task_name (str | None, optional): The task name to reset the
                environment. If None, the task name in the config will be used.
                Default is None.
            clear_cache (bool, optional): Whether to clear the cache
                when closing the environment. Default is False.
            return_obs (bool, optional): Whether to format and return the
                initial observation. Default is True.
            video_dir (str | None, optional): Directory where the env writes
                the episode video. The env controls the final file name using
                ``episode_{episode_id}_seed_{actual_seed}.mp4`` because the
                actual RoboTwin runtime seed is only known after reset.
                Default is None.
            episode_id (int | None, optional): Episode identifier forwarded to
                RoboTwin as ``now_ep_num``. When ``video_dir`` is set, this
                value is also used in the generated video file name. If None,
                the existing ``self.cfg.episode_id`` is reused. Default is
                None.

        Returns:
            tuple[RoboTwinObsType, dict]:
                A tuple containing the initial observation and
                environment info after reset.

        """
        if env_ids is not None:
            raise NotImplementedError(
                "RoboTwinEnv does not support env_ids in reset()."
            )

        if isinstance(seed, str):
            raise TypeError(
                "RoboTwinEnv.reset() seed must be an int or None. "
                f"Got {seed!r}."
            )
        if isinstance(offset_seed, str):
            raise TypeError(
                "RoboTwinEnv.reset() offset_seed must be an int or None. "
                f"Got {offset_seed!r}."
            )

        start_seed = (
            self.cfg.seed if seed is None else self.cfg.calculate_seed(seed)
        )
        seed_changes = start_seed != self.cfg.seed
        if offset_seed is not None:
            next_offset_seed = self._resolve_offset_seed(offset_seed)
        elif seed_changes:
            next_offset_seed = 0
        else:
            next_offset_seed = self._offset_seed
        next_task_name = self.cfg.task_name if task_name is None else task_name
        next_episode_id = (
            self.cfg.episode_id if episode_id is None else episode_id
        )
        next_resolved_start_seed = self.cfg.resolve_start_seed(start_seed)

        # Validate the prospective config before destructively releasing the
        # current simulator. The ordinary reset itself remains destructive
        # once runtime creation starts.
        prospective_cfg = copy.deepcopy(self.cfg)
        prospective_cfg.seed = start_seed
        prospective_cfg.task_name = next_task_name
        prospective_cfg.episode_id = next_episode_id
        prospective_runtime_seed = next_resolved_start_seed + next_offset_seed
        prospective_task_config = prospective_cfg.get_task_config_for_seed(
            prospective_runtime_seed
        )

        self.close(clear_cache=clear_cache)
        if self._pending_disposal_tasks:
            raise RuntimeError(
                "RoboTwinEnv cannot reset while a planner worker from the "
                "previous runtime is still alive. Call close() again after "
                "the worker exits."
            )
        self.cfg.seed = start_seed
        self.cfg.task_name = next_task_name
        self.cfg.episode_id = next_episode_id
        self._resolved_start_seed = next_resolved_start_seed
        self._offset_seed = next_offset_seed
        staged_task: Base_Task | None = None
        try:
            with in_robotwin_workspace():
                (
                    staged_task,
                    instructions,
                    task_config,
                ) = self._check_and_update_seed(prospective_task_config)
            self._task = staged_task
            task = staged_task
            staged_task = None
            self._instructions = instructions
            with in_robotwin_workspace():
                self._setup_task(
                    cfg=self.cfg,
                    task=task,
                    task_config=task_config,
                )
            self._runtime_layout = derive_runtime_layout(task)
            self._active_task_config = copy.deepcopy(task_config)
            # Reset episode-local metadata before formatting the first
            # observation so every cache reflects this runtime.
            self._joints_to_eef_transform = None
            self._cached_obs_robots = None
            self._eval_chosen_instruction = None

            episode_video_path = None
            if video_dir is not None:
                episode_video_path = os.path.join(
                    video_dir,
                    f"episode_{self.cfg.episode_id}_seed_"
                    f"{self.current_seed}.mp4",
                )

            self._stop_video_recording()
            raw_obs = task.get_obs()
            if episode_video_path is not None:
                frame = _extract_video_frame(raw_obs)
                if frame is None:
                    logger.warning(
                        "Skip RoboTwin episode video recording because the "
                        "head camera RGB frame is unavailable."
                    )
                else:
                    try:
                        writer = self._get_video_writer()
                        writer.open(episode_video_path)
                        writer.write_frame(frame)
                    except VideoBackendUnavailableError:
                        self._stop_video_recording()
                        logger.warning(
                            "Skip RoboTwin episode video recording because "
                            "ffmpeg is not available in PATH."
                        )
                    except VideoWriterError:
                        self._stop_video_recording()
                        logger.exception(
                            "Failed to start RoboTwin episode video recording "
                            "at %s.",
                            episode_video_path,
                        )
            obs = (
                self._format_obs(raw_obs, step_index=0) if return_obs else None
            )
            info = self._get_info()
        except BaseException:
            self._stop_video_recording()
            cleanup_task = (
                staged_task if staged_task is not None else self._task
            )
            self._dispose_task(cleanup_task, clear_cache=True)
            self._post_reset_state_available = False
            self._episode_finalized = True
            raise
        self._last_obs = obs
        self._last_obs_step_index = 0
        self._post_reset_state_available = True
        self._episode_finalized = False

        return obs, info

    def step_index_to_log_time_ns(self, step_index: int) -> int:
        """Map a RoboTwin rollout step index to MCAP log time.

        RoboTwin rollout MCAP uses the video frame rate as the logical env
        clock, so step 0 maps to 0 ns and later steps advance by
        ``1 / ROBOTWIN_VIDEO_FPS`` seconds.

        Args:
            step_index (int): Non-negative logical env step index.

        Returns:
            int: MCAP log time in nanoseconds.

        Raises:
            ValueError: If ``step_index`` is negative.
        """
        if step_index < 0:
            raise ValueError("step_index must be non-negative.")
        return int(round(step_index * 1_000_000_000 / ROBOTWIN_VIDEO_FPS))

    def get_mcap_obs(
        self,
        *,
        topic_prefix: str = "observation",
        anchor_log_time_ns: int | None = None,
    ) -> dict[str, list[StampedMessage[Any]]]:
        """Export the latest reset/step observation as MCAP messages.

        This method converts the observation cached by the last successful
        ``reset(...)``, ``reset_from_state(...)``, or ``step(...)`` call. It
        does not call RoboTwin ``get_obs()`` and therefore cannot create a
        simulator snapshot that differs from the policy-facing observation.
        Camera, joint, TF, instruction, episode metadata, and robot metadata
        messages are emitted under ``topic_prefix`` when those fields are
        present in the cached observation. TF messages preserve the
        policy-facing frame IDs and update only their timestamps to the
        selected MCAP log-time anchor.

        Args:
            topic_prefix (str, optional): Topic prefix for emitted observation
                topics. Default is ``"observation"``.
            anchor_log_time_ns (int | None, optional): Explicit MCAP log-time
                anchor in nanoseconds. When omitted, the env derives the
                anchor from the latest cached observation step. Default is
                None.

        Returns:
            dict[str, list[StampedMessage[Any]]]: Final-topic MCAP messages
            for the cached RoboTwin observation.

        Raises:
            RuntimeError: If no latest observation or logical step is
                available.
            ValueError: If ``topic_prefix`` is empty after normalization.
        """
        if self._last_obs is None:
            raise RuntimeError(
                "RoboTwinEnv has no latest observation. Call reset(), "
                "reset_from_state(), or step() before get_mcap_obs()."
            )
        obs = self._last_obs
        if anchor_log_time_ns is not None:
            log_time = int(anchor_log_time_ns)
        elif self._last_obs_step_index is not None:
            log_time = self.step_index_to_log_time_ns(
                self._last_obs_step_index
            )
        else:
            raise RuntimeError(
                "RoboTwinEnv has no logical step for latest observation. "
                "Pass anchor_log_time_ns explicitly."
            )
        prefix = topic_prefix.rstrip("/")
        if not prefix:
            raise ValueError("topic_prefix must not be empty.")
        instruction_payload = None
        if (
            self._instructions is not None
            or self._eval_chosen_instruction is not None
        ):
            instruction_payload = RoboTwinObservationInstructionPayload(
                instructions=(
                    RoboTwinEpisodeInstructionsPayload()
                    if self._instructions is None
                    else self._instructions
                ),
                eval_chosen_instruction=(
                    None
                    if self._eval_chosen_instruction is None
                    else str(self._eval_chosen_instruction)
                ),
            )
        return _build_mcap_observation_messages(
            obs=obs,
            topic_prefix=prefix,
            log_time=log_time,
            instruction_payload=instruction_payload,
            meta_payload=RoboTwinObservationMetaPayload(
                task_name=self.cfg.task_name,
                action_type=self.cfg.action_type,
                episode_id=self.cfg.episode_id,
                seed=self.current_seed,
                start_seed=self.start_seed,
                resolved_start_seed=self.resolved_start_seed,
                offset_seed=self.offset_seed,
            ),
        )

    def get_mcap_action_sidecars(
        self,
        action: Any,
        *,
        topic_prefix: str = "rollout/next_action",
        anchor_log_time_ns: int | None = None,
        frame_id_suffix: str | None = "next_action",
    ) -> dict[str, list[StampedMessage[Any]]]:
        """Export sidecars for the action about to be passed to step().

        For RoboTwin ``action_type="ee"``, the action is already a pair of
        world-frame end-effector targets with layout
        ``[left_xyz, left_quat_wxyz, left_gripper, right_xyz,
        right_quat_wxyz, right_gripper]``. This method records the left and
        right targets as a ``BatchFrameTransformGraph`` under
        ``{topic_prefix}/eef_tf``. Both transforms use ``"world"`` as
        ``parent_frame_id`` and child frame IDs
        ``left_eef_target_from_env_action_{frame_id_suffix}`` and
        ``right_eef_target_from_env_action_{frame_id_suffix}`` when a suffix
        is provided. Other action types currently rely on the raw action
        payload only and return an empty map.

        Args:
            action (Any): Action value about to be passed to ``step(action)``.
            topic_prefix (str, optional): Topic prefix for emitted action
                sidecar topics. Default is ``"rollout/next_action"``.
            anchor_log_time_ns (int | None, optional): Explicit MCAP log-time
                anchor in nanoseconds. When omitted, the env derives the
                anchor from the latest cached observation step. Default is
                None.
            frame_id_suffix (str | None, optional): Suffix appended to target
                child frame IDs. Default is ``"next_action"``.

        Returns:
            dict[str, list[StampedMessage[Any]]]: Final-topic MCAP sidecars.
            Returns an empty map for non-EE action mode or unsupported action
            shape.

        Raises:
            RuntimeError: If no latest observation or logical step is
                available.
            ValueError: If ``topic_prefix`` is empty after normalization.
        """
        if self._last_obs is None:
            raise RuntimeError(
                "RoboTwinEnv has no latest observation. Call reset(), "
                "reset_from_state(), or step() before "
                "get_mcap_action_sidecars()."
            )
        prefix = topic_prefix.rstrip("/")
        if not prefix:
            raise ValueError("topic_prefix must not be empty.")
        if self.cfg.action_type != "ee":
            return {}
        if anchor_log_time_ns is not None:
            log_time = int(anchor_log_time_ns)
        elif self._last_obs_step_index is not None:
            log_time = self.step_index_to_log_time_ns(
                self._last_obs_step_index
            )
        else:
            raise RuntimeError(
                "RoboTwinEnv has no logical step for latest observation. "
                "Pass anchor_log_time_ns explicitly."
            )

        action_array = np.asarray(action, dtype=np.float32)
        if action_array.ndim != 1 or action_array.shape[0] != 16:
            return {}
        left_child_frame_id = "left_eef_target_from_env_action"
        right_child_frame_id = "right_eef_target_from_env_action"
        if frame_id_suffix:
            left_child_frame_id = f"{left_child_frame_id}_{frame_id_suffix}"
            right_child_frame_id = f"{right_child_frame_id}_{frame_id_suffix}"

        left_tf = _pose_vector_to_tf(
            action_array[:7],
            child_frame_id=left_child_frame_id,
        )
        right_tf = _pose_vector_to_tf(
            action_array[8:15],
            child_frame_id=right_child_frame_id,
        )
        tf_graph = BatchFrameTransformGraph(
            [
                left_tf.model_copy(
                    update={"timestamps": [log_time] * left_tf.batch_size}
                ),
                right_tf.model_copy(
                    update={"timestamps": [log_time] * right_tf.batch_size}
                ),
            ]
        )
        return {
            f"{prefix}/eef_tf": [
                StampedMessage(
                    data=tf_graph,
                    log_time=log_time,
                    pub_time=log_time,
                )
            ]
        }

    def _joints2ee_pose(self, joints: np.ndarray) -> RoboTwinEEF:
        """Convert joint positions to world-frame end-effector transforms.

        Args:
            joints (np.ndarray): The joint positions of the robot.

        Returns:
            RoboTwinEEF: Left and right end-effector transforms in world
                frame. ``left_eef.parent_frame_id`` and
                ``right_eef.parent_frame_id`` are both ``"world"``.

        """
        layout = self._require_runtime_layout()
        return _transform_joint_vector_to_eef(
            self._get_joints_to_eef_transform(),
            joints,
            layout,
        )

    def _get_joints_to_eef_transform(self) -> RoboTwinJointsToEEF:
        """Get the cached RoboTwin joint-to-EEF forward-kinematics helper.

        The helper is built lazily from the current combined or split URDFs,
        runtime EEF names, and absolute robot-base transforms, then cached for
        the rest of the episode.
        """
        if self._joints_to_eef_transform is not None:
            return self._joints_to_eef_transform

        self._joints_to_eef_transform = build_joints_to_eef_transform(
            self._require_task(),
            self._require_runtime_layout(),
        )
        return self._joints_to_eef_transform

    def _get_info(self) -> dict[str, Any]:
        info = {
            "seed": self.current_seed,
            "start_seed": self.start_seed,
            "resolved_start_seed": self.resolved_start_seed,
            "offset_seed": self.offset_seed,
            "task": self.cfg.task_name,
        }
        info.update(self._require_task().info)
        return info

    def _resolve_offset_seed(self, offset_seed: int | None) -> int:
        if offset_seed is None:
            return self._offset_seed
        if isinstance(offset_seed, str):
            raise TypeError(
                "RoboTwinEnv.reset() offset_seed must be an int or None. "
                f"Got {offset_seed!r}."
            )
        if offset_seed < 0:
            raise ValueError(f"offset_seed must be >= 0, got {offset_seed}.")
        return offset_seed

    def _require_runtime_layout(self) -> _RoboTwinRuntimeLayout:
        layout = self._runtime_layout
        if layout is None:
            layout = derive_runtime_layout(self._require_task())
            self._runtime_layout = layout
        return layout

    def finalize_episode(self) -> None:
        """Finalize episode-local artifacts without closing the runtime.

        This method is idempotent and safe to call when no episode is active.
        It stops the current episode video writer but keeps the reusable
        RoboTwin task runtime open. After this call, ``step()`` rejects the
        finalized episode until ``reset()`` succeeds or ``reset_from_state()``
        restores a non-finalized episode state.
        """

        self._episode_finalized = True
        self._stop_video_recording()

    def close(self, clear_cache: bool = True):
        """Close the environment."""
        self._episode_finalized = True
        self._post_reset_state_available = False
        self._last_obs = None
        self._last_obs_step_index = None
        self._stop_video_recording()
        self._drain_pending_disposals(clear_cache=clear_cache)
        task = self._task
        if task is not None:
            self._dispose_task(task, clear_cache=clear_cache)
        else:
            self._clear_active_task_runtime()

    def _get_obs(self) -> dict[str, Any]:
        """Get the current observation from the environment.

        Note that in current RoboTwin implementation, the joints of the robot
        are provided in the "joint_action" key of the observation, and it
        actually represents the joint target positions! This is a design
        flaw in RoboTwin, and we leave it as is to be consistent with RoboTwin!

        """
        step_index = self._last_obs_step_index
        if step_index is None:
            raise RuntimeError(
                "RoboTwinEnv._get_obs() requires a successful reset() or "
                "reset_from_state() first."
            )
        ret = self._require_task().get_obs()
        return self._format_obs(ret, step_index=step_index)

    def _format_obs(
        self,
        ret: dict[str, Any],
        *,
        step_index: int,
    ) -> dict[str, Any]:
        """Format raw RoboTwin observations into orchard-compatible ones.

        The returned ``ret["tf"]`` graph includes ``world -> robot_base`` for
        combined layouts, or two absolute ``world -> *_robot_base`` edges for
        split layouts. When raw joint targets or RoboTwin end poses are
        available, it also includes world-frame end-effector edges:
        the joint-derived edges use ``*_eef_from_joint`` child frame IDs,
        while the raw RoboTwin end poses keep the runtime EE frame IDs
        reported by the RoboTwin robot object. The returned observation also
        includes ``ret["robots"]`` with one combined descriptor or two
        truthful split descriptors.
        ``ret["step_index"]`` is the episode-local observation step, and
        ``ret["step_timestamp"]`` is its env-owned logical time in seconds.
        """
        layout = self._require_runtime_layout()
        left_control_eef_frame_id = layout.left_control_eef_frame_id
        right_control_eef_frame_id = layout.right_control_eef_frame_id
        joint_names = (
            get_joint_state_names(self._require_task(), layout)
            if self.cfg.format_datatypes
            else None
        )
        joint_action = ret.get("joint_action")
        joint_eef = (
            self._joints2ee_pose(joint_action["vector"])
            if isinstance(joint_action, dict) and "vector" in joint_action
            else None
        )
        return _format_observation(
            ret,
            instructions=self.instructions,
            step_index=step_index,
            step_timestamp=(
                self.step_index_to_log_time_ns(step_index) / 1_000_000_000.0
            ),
            format_datatypes=self.cfg.format_datatypes,
            joint_names=joint_names,
            base_tf_graph=self._get_tf(),
            robots=self.get_obs_robots(),
            joint_eef=joint_eef,
            left_control_eef_frame_id=left_control_eef_frame_id,
            right_control_eef_frame_id=right_control_eef_frame_id,
        )

    def _write_video_frame(self, raw_obs: dict[str, Any]) -> None:
        writer = self._get_video_writer()
        if writer.is_closed:
            return

        frame = _extract_video_frame(raw_obs)
        if frame is None:
            return

        try:
            writer.write_frame(frame)
        except VideoBackendUnavailableError:
            self._stop_video_recording()
            logger.warning(
                "Skip RoboTwin episode video recording because ffmpeg is "
                "not available in PATH."
            )
        except VideoWriterError:
            self._stop_video_recording()
            logger.exception("Failed to write RoboTwin episode video frame.")

    def _stop_video_recording(self) -> None:
        writer = self._video_writer
        if writer is None or writer.is_closed:
            return
        try:
            writer.close()
        except VideoWriterError:
            logger.exception("Failed to finalize RoboTwin episode video.")

    def _get_video_writer(self) -> VideoWriter:
        writer = self._video_writer
        if writer is None:
            writer = VideoWriter(
                pixel_format=ROBOTWIN_VIDEO_PIXEL_FORMAT,
                fps=ROBOTWIN_VIDEO_FPS,
            )
            self._video_writer = writer
        return writer

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
        return self._require_task().action_space

    @property
    def observation_space(self) -> gym.Space:
        """The observation space of the environment.

        Actually RoboTwin does not implement the observation space!
        Call this method will raise an error!

        Returns:
            gym.Space: The observation space of the environment.
        """
        return self._require_task().observation_space

    def unwrapped_env(self) -> Base_Task:
        """Get the original RoboTwin environment."""
        return self._require_task()

    def _get_tf(self) -> BatchFrameTransformGraph:
        """Get the frame transforms in the environment.

        Combined layouts expose ``world -> robot_base``. Split layouts expose
        independent absolute ``world -> left_robot_base`` and
        ``world -> right_robot_base`` edges in the same graph.

        Returns:
            BatchFrameTransformGraph: The static robot base transform graph.
        """
        return get_robot_base_tf_graph(
            self._require_task(),
            self._require_runtime_layout(),
        )

    def get_robot_urdf(self) -> dict[str, bytes]:
        """Get URDF content for the initialized RoboTwin runtime layout.

        Returns:
            dict[str, bytes]: Combined layouts preserve RoboTwin's historical
                ``{"left": combined_urdf}`` mapping. Split layouts return
                independent ``"left"`` and ``"right"`` payloads.
        """
        return read_robot_urdfs(
            self._require_task(),
            self._require_runtime_layout(),
        )

    def get_obs_robots(self) -> dict[str, Robot]:
        """Return observation-facing robot metadata for the current layout.

        Combined layouts expose one descriptor under RoboTwin's historical
        ``"left"`` compatibility key. Split layouts expose truthful
        ``"left"`` and ``"right"`` descriptors with indices 0 and 1.
        """
        if self._cached_obs_robots is not None:
            return self._cached_obs_robots.copy()

        self._cached_obs_robots = build_obs_robots(
            self.get_robot_urdf(),
            self._require_runtime_layout(),
        )
        return self._cached_obs_robots.copy()


class RoboTwinEnvCfg(EnvBaseCfg[RoboTwinEnv]):
    """Configuration for the RoboTwin environment."""

    class_type: ClassType[RoboTwinEnv] = RoboTwinEnv
    _task_config_snapshot: _RoboTwinTaskConfigSnapshot = PrivateAttr()

    task_name: str
    """The name of the task to run, e.g., 'place_object_scale'."""

    seed: int = 0
    """The caller-facing start seed for the environment.

    In eval mode the env resolves this start seed into RoboTwin's reserved
    runtime seed range, but the config field itself remains unchanged.
    """

    episode_id: int = 0
    """Episode identifier forwarded to RoboTwin as ``now_ep_num``.

    The value may be updated per reset by passing ``episode_id`` to
    ``RoboTwinEnv.reset()``. When episode video recording is enabled, the env
    also uses this identifier in its output file-name convention together with
    the actual runtime seed selected during reset.
    """

    action_type: Literal["qpos", "ee"] = "qpos"
    """The RoboTwin action representation to use in the environment.

    `"qpos"` uses joint target positions. `"ee"` uses RoboTwin's
    end-effector action representation.
    """

    check_expert: bool = False
    """Whether to check the expert trajectory for the task.

    If true, the environment will attempt to run the task with the current
    runtime seed
    to check if the task can be completed successfully using the expert
    trajectory. If it fails, the env will increment its runtime
    ``offset_seed`` and retry until it finds a seed that can be completed by
    the expert trajectory.

    This mode is stronger than ``check_task_init``: it not only executes
    RoboTwin's ``play_once()`` initialization path, but also treats expert
    success as a requirement and may rewrite the env runtime offset to the
    first valid seed that passes the check.

    This field is used to make sure that the environment can be recorded
    successfully using the expert trajectory for imitation learning.

    ``check_expert`` and ``check_task_init`` are mutually exclusive. For
    evaluation, this is the recommended mode: use expert-verified seeds by
    setting ``check_expert=True`` and ``check_task_init=False``.

    """

    check_task_init: bool = True
    """Whether to check the task initialization.

    If true, the environment will call `play_once()` to execute the task
    with expert trajectory to check if the task can be initialized
    successfully.

    Compared with ``check_expert``, this mode is weaker and is meant as a
    RoboTwin warm-up path: it runs the same ``play_once()`` initialization
    flow once so task-specific runtime attributes are created, but it does
    not search for a new seed when the current one is unstable or cannot be
    solved by the expert trajectory.

    This field should be set to True because some task attributes that
    required for interaction may be initialized in the `play_once()` method,
    such as `place_object_scale` task.

    This should be a BUG in RoboTwin and will significantly affect the
    performance of the environment initialization.

    ``check_task_init`` and ``check_expert`` can not both be True. For
    evaluation, prefer ``check_expert`` instead of this flag because
    evaluation should use a seed that is known to be expert-solvable.

    """

    eval_mode: bool = False
    """Whether for evaluation.

    If true, the environment will use unseen texture_type.

    Evaluation also requires expert-verified seeds, so ``__post_init__``
    forces ``check_expert=True`` and ``check_task_init=False`` when
    ``eval_mode=True``. In other words, callers usually do not need to set
    those two flags manually for evaluation; enabling ``eval_mode`` is the
    recommended entrypoint.
    """

    max_instruction_num: int = 10
    """The maximum number of instructions to generate for the env."""

    format_datatypes: bool = False
    """whether to format obs as robo_orchard datatypes.

    If true, the observation will be formatted as:
        - "joints": dict of joint name to joint position. This key will
            replace the original "joint_action" key.
        - "cameras": dict of camera name to camera image. This key will
            replace the original "observation" key.
        - other keys in the original observation will be kept.

    The default is False for compatibility with original RoboTwin code.
    We highly recommend to set this field to True for better usability!
    """

    task_config_path: str | None = None
    """Path or preset name for the RoboTwin task configuration file.

    ``None`` defaults to the ``"demo_clean"`` preset. The preset names
    ``"demo_clean"`` and ``"demo_randomized"`` resolve to the corresponding
    file under ``<RoboTwin_PATH>/task_config/``. Any other string is treated
    as a file path.

    Note that we only support RoboTwin2.0 for now.
    """

    task_config_overrides: list[tuple[str, Any]] | None = None
    """Final overrides applied to the resolved task config.

    Each item is a `(path, value)` pair where `path` uses `/` to address
    nested dictionary keys, for example `("data_type/rgb", True)`.
    These overrides are applied after official task-config lowering finishes.
    """

    patch_curobo_base_transform: bool = False
    """Whether to guard RoboTwin Curobo target poses in base-link frame.

    When enabled for `action_type="ee"`, RoboOrchard patches RoboTwin's
    Curobo planner to derive the full fixed transform from the Curobo yml and
    URDF `base_link`, and bypasses RoboTwin's hard-coded `aloha-agilex`
    transform branch. The default is False so current RoboTwin deployments use
    their bundled planner behavior unless callers explicitly opt into the
    RoboOrchard compatibility patch. Once enabled in a process, comparing
    against the original RoboTwin behavior requires a fresh Python process.
    """

    @model_serializer(mode="wrap", return_type=dict, when_used="always")
    def _serialize_with_task_config_snapshot(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> dict[str, Any]:
        """Carry the private pinned input through config serialization."""

        serialized = super().wrapped_model_ser(handler, info)
        return inject_serialized_task_config_snapshot(
            serialized,
            self._task_config_snapshot,
        )

    @model_validator(mode="wrap")
    @classmethod
    def _restore_task_config_snapshot(
        cls,
        data: Any,
        handler: Any,
    ) -> RoboTwinEnvCfg:
        """Restore and verify the reserved serialized snapshot envelope."""

        data, snapshot = extract_serialized_task_config_snapshot(data)
        with task_config_snapshot_restore_context(snapshot):
            return handler(data)

    def __post_init__(self):
        self._refresh_task_config_snapshot()

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

        if self.eval_mode and self.max_instruction_num != EVAL_INSTRUCTION_NUM:
            logger.info(
                f"Set max_instruction_num from "
                f"{self.max_instruction_num} to "
                f"{EVAL_INSTRUCTION_NUM} for eval_mode."
            )
            self.max_instruction_num = EVAL_INSTRUCTION_NUM

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "task_config_path" and hasattr(
            self, "_task_config_snapshot"
        ):
            raise AttributeError(
                "RoboTwin task_config_path is pinned after construction; "
                "use replace(task_config_path=...) instead."
            )
        super().__setattr__(name, value)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> RoboTwinEnvCfg:
        """Copy this config while atomically replacing a task-config source.

        ``Config.replace()`` and ``ClassConfig.create_instance_by_cfg()``
        reach this Pydantic copy boundary. A changed task-config path must
        therefore refresh its canonical path and immutable snapshot together.
        """

        copied = super().model_copy(update=update, deep=deep)
        if (
            update is not None
            and update.get("task_config_path", self.task_config_path)
            != self.task_config_path
        ):
            copied._refresh_task_config_snapshot()
        return copied

    def _refresh_task_config_snapshot(self) -> None:
        """Resolve the public path and replace the immutable YAML snapshot."""

        task_config_path, task_config_snapshot = resolve_task_config_source(
            self.task_config_path,
            robotwin_root=config_robotwin_path(),
        )
        object.__setattr__(self, "task_config_path", task_config_path)
        object.__setattr__(self, "_task_config_snapshot", task_config_snapshot)

    @property
    def _task_config_content_sha256(self) -> str:
        """Return the pinned task-YAML digest for internal identity checks."""

        return self._task_config_snapshot.content_sha256

    def calculate_seed(self, seed: int) -> int:
        """Normalize the caller-facing start seed.

        This compatibility helper preserves the existing public method name
        while returning a caller-space start seed, not the actual runtime seed.
        """
        return seed

    def resolve_start_seed(self, seed: int) -> int:
        """Resolve a caller-facing start seed into a RoboTwin runtime seed.

        In eval mode, start seeds below ``EVAL_SEED_BASE`` are mapped into
        RoboTwin's reserved evaluation seed range.

        Args:
            seed (int): The caller-facing start seed.

        Returns:
            int: The resolved runtime start seed used in RoboTwin.
        """
        seed = self.calculate_seed(seed)

        if self.eval_mode and seed < EVAL_SEED_BASE:
            seed = EVAL_SEED_BASE * (1 + seed)

        if seed >= EVAL_SEED_BASE and self.eval_mode is False:
            raise ValueError(
                f"Seed {seed} is >= {EVAL_SEED_BASE} but eval_mode is "
                "False. This is reserved for RoboTwin evaluation mode."
            )

        return seed

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
        return self.get_task_config_for_seed(
            runtime_seed=self.resolve_start_seed(self.seed)
        )

    def get_task_config_for_seed(self, runtime_seed: int) -> dict[str, Any]:
        """Return the resolved task configuration for `setup_demo()`.

        The YAML bytes are pinned once when this config is constructed. The
        returned config parses a fresh mutable mapping from that immutable
        snapshot, combines official RoboTwin-derived fields, and finally
        applies `task_config_overrides`.
        """
        return build_task_config(
            snapshot=self._task_config_snapshot,
            source=self.task_config_path or "task config snapshot",
            task_name=self.task_name,
            runtime_seed=runtime_seed,
            episode_id=self.episode_id,
            eval_mode=self.eval_mode,
            embodiment_config_path=self.embodiment_config_path,
            camera_config_path=self.camera_config_path,
            robotwin_root=config_robotwin_path(),
            task_config_overrides=self.task_config_overrides,
        )


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
