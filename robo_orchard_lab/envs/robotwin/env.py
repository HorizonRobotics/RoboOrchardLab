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
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias

import gymnasium as gym
import numpy as np
import torch
import yaml
from robo_orchard_core.envs.env_base import EnvBase, EnvBaseCfg, EnvStepReturn
from robo_orchard_core.utils.logging import LoggerManager
from typing_extensions import Literal

from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchFrameTransformGraph,
)
from robo_orchard_lab.envs.robotwin.kinematics import (
    RoboTwinEEF,
    RoboTwinJointsToEEF,
)
from robo_orchard_lab.envs.robotwin.obs import (
    get_joints,
    get_observation_cams,
)
from robo_orchard_lab.envs.sapien import sapien_pose_to_orchard

if TYPE_CHECKING:
    from envs._base_task import Base_Task

EVAL_SEED_BASE = 100000
EVAL_INSTRUCTION_NUM = 100
logger = LoggerManager().get_child(__name__)

InstructionType: TypeAlias = Literal["seen", "unseen"]
RoboTwinObsType: TypeAlias = dict[str, Any] | None
__all__ = ["RoboTwinEnvStepReturn", "RoboTwinEnv", "RoboTwinEnvCfg"]


@dataclass
class RoboTwinEnvStepReturn(EnvStepReturn[RoboTwinObsType, bool]):
    observations: RoboTwinObsType
    terminated: bool
    rewards: bool
    """The rewards is a boolean indicating whether the task was successful."""
    truncated: bool
    """Whether the episode was truncated due to reaching the step limit."""


class RoboTwinEnv(EnvBase[RoboTwinObsType, bool]):
    """RoboTwin environment.

    This class provides RoboTwin environment with robo_orchard_core interface.
    To make it work, you need to install RoboTwin and set the `RoboTwin_PATH`
    environment variable to the path of the RoboTwin package.


    After initialization, you need to call `reset()` to create the environment.

    """

    def __init__(self, cfg: RoboTwinEnvCfg):
        self.cfg = cfg
        self._eval_chosen_instruction: str | None = None
        self._joints_to_eef_transform: RoboTwinJointsToEEF | None = None
        self._video_ffmpeg: subprocess.Popen[bytes] | None = None

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
            requested_seed = self.cfg.seed
            task, success = self._check_expert_traj()
            retry_num = 0
            while not success:
                retry_num += 1
                if retry_num >= 50:
                    raise RuntimeError(
                        f"Failed to create task {self.cfg.task_name} "
                        f"with expert trajectory after {retry_num} retries. "
                        "Please check the task configuration!"
                    )

                failed_seed = self.cfg.seed
                self.cfg.seed += 1
                logger.warning(
                    "Expert trajectory check failed for task "
                    f"{self.cfg.task_name} at seed {failed_seed}; "
                    f"retrying with seed {self.cfg.seed}."
                )
                task, success = self._check_expert_traj()
                if success:
                    logger.info(
                        f"Successfully created task {self.cfg.task_name} "
                        f"with seed {self.cfg.seed} using expert trajectory."
                    )
            if retry_num > 0:
                logger.info(
                    f"Requested seed {requested_seed} for task "
                    f"{self.cfg.task_name} resolved to actual seed "
                    f"{self.cfg.seed} after {retry_num} retries."
                )

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
                environment. The exact semantics depend on
                `self.cfg.action_type`: `"qpos"` expects RoboTwin joint target
                positions, while `"ee"` expects RoboTwin end-effector action
                values. The action should be a 1-D array with length matching
                the configured action type for the task.

        Returns:
            RoboTwinEnvStepReturn: The step result after taking the action.
                This function always returns a step result. Episode end is
                reported via `terminated` and `truncated` instead of
                returning None. `rewards` is a boolean indicating whether
                the task has succeeded.
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
        self._task.take_action(action, action_type=self.cfg.action_type)

        # when reach step limit, truncated is True
        # Note that step_lim is None for default unlimited steps.
        # It will be set in evaluation mode.
        if (
            self._task.step_lim is not None
            and self._task.take_action_cnt >= self._task.step_lim
        ):
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

        raw_obs = self._task.get_obs()
        self._write_video_frame(raw_obs)

        return RoboTwinEnvStepReturn(
            observations=self._format_obs(raw_obs),
            rewards=self._task.eval_success,
            terminated=terminated,
            truncated=truncated,
            info=self._get_info(),
        )

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        seed: int | str | None = None,
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
        and check the seed, and the seed will be updated in the config.

        Warning:
            RoboTwin does not use local RandomGenerator, when the environment
            is re-created, the seed will be set to the one in the config
            for both numpy and torch. This may affect the randomness of other
            parts of the code!
            This is a BUG in RoboTwin!

        Args:
            env_ids (Sequence[int] | None, optional): Not supported.
                Defaults to None.
            seed (int | str | None, optional): The seed to reset the
                environment. If None, the seed in the config will be used.
                If "next", the seed will be incremented by 1.
                Default is None.
            task_name (str | None, optional): The task name to reset the
                environment. If None, the task name in the config will be used.
                Default is None.
            clear_cache (bool, optional): Whether to clear the cache
                when closing the environment. Default is False.
            return_obs (bool, optional): Whether to format and return the
                initial observation. Default is True.
            video_dir (str | None, optional): Directory where the env writes
                the episode video using the fixed file-name convention
                ``episode_{episode_id}_seed_{seed}.mp4``. The env controls the
                file name and callers may only choose the output directory.
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

        self.close(clear_cache=clear_cache)
        # calculate the actual seed
        if seed is not None:
            seed = self.cfg.calculate_seed(seed)
        if episode_id is not None:
            self.cfg.episode_id = episode_id

        seed_changes = seed is not None and seed != self.cfg.seed
        task_name_changes = (
            task_name is not None and task_name != self.cfg.task_name
        )
        # check if task is not initialized or seed/task_name changes
        if not hasattr(self, "_task") or seed_changes or task_name_changes:
            # when need to create new env:
            # * when no existing env
            # * when seed changes
            if seed_changes:
                assert seed is not None
                self.cfg.seed = seed
            if task_name_changes:
                assert task_name is not None
                self.cfg.task_name = task_name
            with in_robotwin_workspace():
                task, instructions = self._check_and_update_seed()
            assert task is not None
            self._task = task
            self._instructions = instructions

        with in_robotwin_workspace():
            task_config = self.cfg.get_task_config()
            self._task.setup_demo(**task_config)  # type: ignore
        self._assert_supported_robot_layout()

        self._eval_chosen_instruction = None

        episode_video_path = None
        if video_dir is not None:
            episode_video_path = os.path.join(
                video_dir,
                f"episode_{self.cfg.episode_id}_seed_{self.cfg.seed}.mp4",
            )

        raw_obs = self._task.get_obs()
        self._start_video_recording(
            video_path=episode_video_path, raw_obs=raw_obs
        )
        obs = self._format_obs(raw_obs) if return_obs else None

        self._joints_to_eef_transform = None

        return obs, self._get_info()

    def _joints2ee_pose(self, joints: np.ndarray) -> RoboTwinEEF:
        """Convert joint positions to world-frame end-effector transforms.

        Args:
            joints (np.ndarray): The joint positions of the robot.

        Returns:
            RoboTwinEEF: Left and right end-effector transforms in world
                frame. ``left_eef.parent_frame_id`` and
                ``right_eef.parent_frame_id`` are both ``"world"``.

        """
        joints_np = np.asarray(joints, dtype=np.float32)
        if joints_np.ndim == 1:
            joints_np = joints_np[None, :]
        if joints_np.ndim != 2 or joints_np.shape[0] != 1:
            raise ValueError(
                "Expected joints to have shape (D,) or (1, D), got "
                f"{tuple(joints_np.shape)}."
            )

        left_joint_count = len(self._task.robot.left_arm_joints_name)
        right_joint_count = len(self._task.robot.right_arm_joints_name)
        total_arm_joint_count = left_joint_count + right_joint_count
        joint_dim = joints_np.shape[-1]
        if joint_dim == total_arm_joint_count + 2:
            left_arm_joints = joints_np[:, :left_joint_count]
            right_start = left_joint_count + 1
        elif joint_dim == total_arm_joint_count:
            left_arm_joints = joints_np[:, :left_joint_count]
            right_start = left_joint_count
        else:
            raise ValueError(
                "Expected RoboTwin joints to contain left/right arm joints "
                "with optional gripper values, got shape "
                f"{tuple(joints_np.shape)}."
            )
        right_arm_joints = joints_np[
            :,
            right_start : right_start + right_joint_count,
        ]

        return self._get_joints_to_eef_transform().transform(
            left_arm_joints=torch.from_numpy(left_arm_joints),
            right_arm_joints=torch.from_numpy(right_arm_joints),
        )

    def _get_joints_to_eef_transform(self) -> RoboTwinJointsToEEF:
        if self._joints_to_eef_transform is not None:
            return self._joints_to_eef_transform

        urdf_map = self.get_robot_urdf()
        urdf_content = urdf_map["left"]

        robot_base_tf = self._get_tf().get_tf("world", "robot_base")
        if not isinstance(robot_base_tf, BatchFrameTransform):
            raise RuntimeError(
                "Expected supported RoboTwin layouts to expose a single "
                "world->robot_base BatchFrameTransform."
            )

        self._joints_to_eef_transform = RoboTwinJointsToEEF(
            urdf_content=urdf_content,
            robot_base_xyz=robot_base_tf.xyz[0].tolist(),
            robot_base_quat=robot_base_tf.quat[0].tolist(),
        )
        return self._joints_to_eef_transform

    def _get_info(self) -> dict[str, Any]:
        info = {"seed": self.cfg.seed, "task": self.cfg.task_name}
        info.update(self._task.info)
        return info

    def _assert_supported_robot_layout(self) -> None:
        """Validate that the current RoboTwin robot layout is supported.

        RoboTwinEnv currently supports only the combined dual-arm layout with
        one shared robot base pose. Unsupported layouts are rejected at the
        env boundary during ``reset()`` and by robot-structure helper methods.
        """
        if self._task.robot.is_dual_arm is False:
            raise NotImplementedError(
                "RoboTwinEnv currently only supports a combined dual-arm "
                "robot layout. Separate left/right URDF layouts are not "
                "supported."
            )

        left_base_tf = sapien_pose_to_orchard(
            self._task.robot.left_entity_origion_pose
        )
        right_base_tf = sapien_pose_to_orchard(
            self._task.robot.right_entity_origion_pose
        )
        if left_base_tf != right_base_tf:
            raise NotImplementedError(
                "RoboTwinEnv currently only supports a combined dual-arm "
                "robot with a shared robot base. Separate left/right robot "
                "base poses are not supported."
            )

    def close(self, clear_cache: bool = True):
        """Close the environment."""
        self._stop_video_recording()
        if hasattr(self, "_task") and self._task is not None:
            self._task.close_env(clear_cache=clear_cache)
            if self._task.render_freq > 0:
                self._task.viewer.close()

    def _get_joint_state_names(self: RoboTwinEnv) -> list[str]:
        ret_names = []
        ret_names.extend(self._task.robot.left_arm_joints_name)
        ret_names.append(self._task.robot.left_gripper_name["base"])
        ret_names.extend(self._task.robot.right_arm_joints_name)
        ret_names.append(self._task.robot.right_gripper_name["base"])
        return ret_names

    def _get_obs(self) -> dict[str, Any]:
        """Get the current observation from the environment.

        Note that in current RoboTwin implementation, the joints of the robot
        are provided in the "joint_action" key of the observation, and it
        actually represents the joint target positions! This is a design
        flaw in RoboTwin, and we leave it as is to be consistent with RoboTwin!

        """
        ret = self._task.get_obs()
        return self._format_obs(ret)

    def _format_obs(self, ret: dict[str, Any]) -> dict[str, Any]:
        """Format raw RoboTwin observations into orchard-compatible ones."""
        ret["instructions"] = self.instructions
        if self.cfg.format_datatypes:
            ret["joints"] = get_joints(
                ret, joint_names=self._get_joint_state_names()
            )
            ret.pop("joint_action", None)
            ret["cameras"] = get_observation_cams(ret)
            ret.pop("observation")
        ret["tf"] = self._get_tf()
        return ret

    @staticmethod
    def _extract_video_frame(raw_obs: dict[str, Any]) -> np.ndarray | None:
        observation = raw_obs.get("observation")
        if not isinstance(observation, dict):
            return None
        head_camera = observation.get("head_camera")
        if not isinstance(head_camera, dict):
            return None
        frame = head_camera.get("rgb")
        if frame is None:
            return None

        frame_np = np.asarray(frame)
        if frame_np.ndim != 3 or frame_np.shape[2] != 3:
            return None
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)
        return np.ascontiguousarray(frame_np)

    def _start_video_recording(
        self,
        video_path: str | None,
        raw_obs: dict[str, Any],
    ) -> None:
        self._stop_video_recording()
        if video_path is None:
            return

        frame = self._extract_video_frame(raw_obs)
        if frame is None:
            logger.warning(
                "Skip RoboTwin episode video recording because the head "
                "camera RGB frame is unavailable."
            )
            return
        if shutil.which("ffmpeg") is None:
            logger.warning(
                "Skip RoboTwin episode video recording because ffmpeg is "
                "not available in PATH."
            )
            return

        output_dir = os.path.dirname(video_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        height, width, _ = frame.shape
        try:
            self._video_ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    video_path,
                ],
                stdin=subprocess.PIPE,
            )
        except Exception:
            self._video_ffmpeg = None
            logger.exception(
                "Failed to start RoboTwin episode video recording at %s.",
                video_path,
            )
            return

        self._write_video_frame(raw_obs)

    def _write_video_frame(self, raw_obs: dict[str, Any]) -> None:
        if self._video_ffmpeg is None or self._video_ffmpeg.stdin is None:
            return

        frame = self._extract_video_frame(raw_obs)
        if frame is None:
            return

        try:
            self._video_ffmpeg.stdin.write(frame.tobytes())
        except Exception:
            logger.exception("Failed to write RoboTwin episode video frame.")
            self._stop_video_recording()

    def _stop_video_recording(self) -> None:
        if self._video_ffmpeg is None:
            return
        try:
            if self._video_ffmpeg.stdin is not None:
                self._video_ffmpeg.stdin.close()
            self._video_ffmpeg.wait()
        except Exception:
            logger.exception("Failed to finalize RoboTwin episode video.")
        finally:
            self._video_ffmpeg = None

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

    def _get_tf(self) -> BatchFrameTransformGraph:
        """Get the frame transforms in the environment.

        For supported RoboTwin layouts this graph contains one static
        ``world -> robot_base`` edge. ``reset()`` rejects layouts with
        separate left/right robot bases before any observations are returned.

        Returns:
            BatchFrameTransformGraph: The static robot base transform graph.
        """
        self._assert_supported_robot_layout()
        left_base_tf = sapien_pose_to_orchard(
            self._task.robot.left_entity_origion_pose
        )
        return BatchFrameTransformGraph(
            tf_list=[
                BatchFrameTransform(
                    xyz=left_base_tf.xyz,
                    quat=left_base_tf.quat,
                    timestamps=left_base_tf.timestamps,
                    parent_frame_id="world",
                    child_frame_id="robot_base",
                )
            ],
            static_tf=[True],
        )

    def get_robot_urdf(self) -> dict[str, bytes]:
        """Get the supported combined dual-arm URDF content of the robot.

        Returns:
            dict[str, bytes]: A compatibility mapping containing the combined
                dual-arm URDF content under the ``"left"`` key.
        """
        self._assert_supported_robot_layout()

        assert self._task.robot.left_urdf_path is not None
        with in_robotwin_workspace():
            with open(self._task.robot.left_urdf_path, "rb") as f:
                urdf_content = f.read()

        return {"left": urdf_content}


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
    """Episode identifier forwarded to RoboTwin as ``now_ep_num``.

    The value may be updated per reset by passing ``episode_id`` to
    ``RoboTwinEnv.reset()``. When episode video recording is enabled, the env
    also uses this identifier in its fixed output file-name convention.
    """

    action_type: Literal["qpos", "ee"] = "qpos"
    """The RoboTwin action representation to use in the environment.

    `"qpos"` uses joint target positions. `"ee"` uses RoboTwin's
    end-effector action representation.
    """

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
    """Path to the task configuration file.

    If not provided, the path will be set to
    `<RoboTwin_PATH>/task_config/_config_template.yml` for RoboTwin2.0.

    Note that we only support RoboTwin2.0 for now.
    """

    task_config_overrides: list[tuple[str, Any]] | None = None
    """Final overrides applied to the resolved task config.

    Each item is a `(path, value)` pair where `path` uses `/` to address
    nested dictionary keys, for example `("data_type/rgb", True)`.
    These overrides are applied after `_update_task_config()` finishes.
    """

    def __post_init__(self):
        if self.task_config_path is None:
            robo_twin_root = config_robotwin_path()
            self.task_config_path = os.path.join(
                robo_twin_root, "task_config", "_config_template.yml"
            )

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

        if self.eval_mode and self.max_instruction_num != EVAL_INSTRUCTION_NUM:
            logger.info(
                f"Set max_instruction_num from "
                f"{self.max_instruction_num} to "
                f"{EVAL_INSTRUCTION_NUM} for eval_mode."
            )
            self.max_instruction_num = EVAL_INSTRUCTION_NUM

        self.seed = self.calculate_seed(self.seed)

    def calculate_seed(self, seed: int | str) -> int:
        """Calculate the actual seed used in RoboTwin.

        In eval_mode, the seed is calculated as `100000 * (1 + seed)`.

        Args:
            seed (int | str): The base seed. The string value `"next"`
                increments the current seed by 1.

        Returns:
            int: The actual seed used in RoboTwin.
        """
        if isinstance(seed, str):
            if seed == "next":
                seed = self.seed + 1
            else:
                raise ValueError(
                    f"Invalid seed string: {seed}. Only 'next' is supported."
                )

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
        """Return the resolved task configuration for `setup_demo()`.

        The returned config combines the YAML template, derived RoboTwin
        fields from `_update_task_config()`, and the final
        `task_config_overrides` patches.
        """
        assert self.task_config_path is not None
        with (
            open(self.task_config_path, "r", encoding="utf-8") as f,
        ):
            task_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            ret = self._update_task_config(task_config)
            self._apply_task_config_overrides(ret)

            ret["task_name"] = self.task_name
            return ret

    def _apply_task_config_overrides(
        self, task_config: dict[str, Any]
    ) -> None:
        """Apply final task-config overrides in place.

        This helper treats each item in `task_config_overrides` as a patch to
        the fully resolved task-config dictionary returned by
        `_update_task_config()`.

        Path rules:
            - Each override is a `(path, value)` pair.
            - `path` uses `/` as the separator for nested dictionary keys.
            - Only dictionary traversal is supported; list indices are not.
            - Every path segment must already exist in `task_config`.

        Guard rails:
            - Paths that correspond to env-managed fields or fields whose
              values are derived earlier in `get_task_config()` are rejected.
            - Invalid paths raise `ValueError`.
            - Missing intermediate or leaf keys raise `KeyError`.

        Args:
            task_config (dict[str, Any]): The resolved task config to patch.
        """
        reserved_paths = {
            "task_name",
            "seed",
            "now_ep_num",
            "eval_mode",
            "is_test",
            "camera/head_camera_type",
            "embodiment",
        }
        for path, value in self.task_config_overrides or []:
            if path in reserved_paths:
                raise ValueError(
                    f"Task config override path {path!r} is not supported "
                    "because it affects env-managed or derived fields."
                )

            keys = path.split("/")
            if not path or any(key == "" for key in keys):
                raise ValueError(
                    f"Invalid task config override path {path!r}."
                )

            target: Any = task_config
            for key in keys[:-1]:
                if not isinstance(target, dict):
                    raise KeyError(
                        f"Task config override path {path!r} does not "
                        f"resolve to a nested dict at {key!r}."
                    )
                if key not in target:
                    raise KeyError(
                        f"Task config override path {path!r} is missing "
                        f"segment {key!r}."
                    )
                target = target[key]

            if not isinstance(target, dict):
                raise KeyError(
                    f"Task config override path {path!r} does not resolve "
                    "to a dict parent."
                )

            leaf_key = keys[-1]
            if leaf_key not in target:
                raise KeyError(
                    f"Task config override path {path!r} is missing leaf "
                    f"key {leaf_key!r}."
                )
            target[leaf_key] = value

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
        task_args["is_test"] = self.eval_mode

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
