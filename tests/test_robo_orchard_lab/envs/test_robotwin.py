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

import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch
from pydantic import BaseModel, ValidationError
from test_robo_orchard_lab.dataset._mcap_pydantic_schema_helper import (
    assert_mcap_compatible_pydantic_schema,
)

import robo_orchard_lab.envs.robotwin.curobo_base_patch as curobo_base_patch
import robo_orchard_lab.envs.robotwin.env as robotwin_env
from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Robot,
    RobotDescriptionFormat,
)
from robo_orchard_lab.envs.robotwin._runtime import (
    derive_runtime_layout,
    get_joint_state_names,
)
from robo_orchard_lab.envs.robotwin.env import (
    ROBOTWIN_ENV_STATE_SCHEMA_VERSION,
    ROBOTWIN_VIDEO_FPS,
    ROBOTWIN_VIDEO_PIXEL_FORMAT,
    RoboTwinEnv,
    RoboTwinEnvCfg,
    RoboTwinEpisodeInstructionsPayload,
    RoboTwinObservationInstructionPayload,
    RoboTwinObservationMetaPayload,
)
from robo_orchard_lab.envs.robotwin.kinematics import (
    RoboTwinEEF,
    RoboTwinJointsToEEF,
)
from robo_orchard_lab.envs.robotwin.obs import (
    _LEFT_EEF_FROM_JOINT_FRAME_ID,
    _RIGHT_EEF_FROM_JOINT_FRAME_ID,
)
from robo_orchard_lab.envs.state import ENV_STATE_SCOPE_KEY, EnvStateScope
from robo_orchard_lab.utils.state import State
from robo_orchard_lab.utils.video import VideoWriter

pytestmark = pytest.mark.sim_env


def _install_fake_robotwin_instruction_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    description_module = ModuleType("description")
    utils_module = ModuleType("description.utils")
    generator_module = ModuleType(
        "description.utils.generate_episode_instructions"
    )
    description_module.__dict__["__path__"] = []
    utils_module.__dict__["__path__"] = []
    generator_module.__dict__["generate_episode_descriptions"] = (
        lambda task_name, infos, max_descriptions: [{"seen": [task_name]}]
    )
    monkeypatch.setitem(sys.modules, "description", description_module)
    monkeypatch.setitem(sys.modules, "description.utils", utils_module)
    monkeypatch.setitem(
        sys.modules,
        "description.utils.generate_episode_instructions",
        generator_module,
    )
    return generator_module


def _get_ffmpeg_binary(*, require_libx264: bool = False) -> str:
    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary is None:
        pytest.skip("ffmpeg is required for real RoboTwin video tests.")

    if not require_libx264:
        return ffmpeg_binary

    encoders = subprocess.run(
        [ffmpeg_binary, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    encoder_listing = f"{encoders.stdout}\n{encoders.stderr}"
    if encoders.returncode != 0 or "libx264" not in encoder_listing:
        pytest.skip(
            "ffmpeg with libx264 support is required for real RoboTwin "
            "video tests."
        )

    return ffmpeg_binary


def _make_fake_sapien_pose(
    xyz: tuple[float, float, float],
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> SimpleNamespace:
    return SimpleNamespace(
        get_p=lambda: np.asarray(xyz, dtype=np.float32),
        get_q=lambda: np.asarray(quat, dtype=np.float32),
    )


def _make_fake_joint_with_child_link(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        child_link=SimpleNamespace(get_name=lambda: name),
    )


def _make_uninitialized_env() -> RoboTwinEnv:
    """Build a typed Env shell for tests that intentionally skip __init__."""
    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env._task = None
    env._pending_disposal_tasks = []
    return env


class _RaiseOnExit:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        raise KeyboardInterrupt


def _make_reset_stub_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    robot: SimpleNamespace,
) -> RoboTwinEnv:
    if not hasattr(robot, "left_arm_joints_name"):
        robot.left_arm_joints_name = ["left_joint_0"]
    if not hasattr(robot, "right_arm_joints_name"):
        robot.right_arm_joints_name = ["right_joint_0"]
    if not hasattr(robot, "left_gripper_name"):
        robot.left_gripper_name = {"base": "left_gripper"}
    if not hasattr(robot, "right_gripper_name"):
        robot.right_gripper_name = {"base": "right_gripper"}
    if not hasattr(robot, "left_ee"):
        robot.left_ee = _make_fake_joint_with_child_link("left_eef")
    if not hasattr(robot, "right_ee"):
        robot.right_ee = _make_fake_joint_with_child_link("right_eef")
    if not hasattr(robot, "left_urdf_path"):
        robot.left_urdf_path = str(Path(__file__).resolve())
    if not hasattr(robot, "right_urdf_path"):
        robot.right_urdf_path = str(Path(__file__).resolve())
    if not hasattr(robot, "left_entity") or not hasattr(robot, "right_entity"):
        robot.left_entity = object()
        robot.right_entity = (
            robot.left_entity if robot.is_dual_arm else object()
        )

    env = _make_uninitialized_env()
    env.cfg = SimpleNamespace(
        seed=1,
        task_name="robotwin_dummy_task",
        action_type="qpos",
        patch_curobo_base_transform=False,
        episode_id=0,
        eval_mode=False,
        format_datatypes=False,
        get_task_config=lambda: {},
        get_task_config_for_seed=lambda runtime_seed: {"seed": runtime_seed},
        calculate_seed=lambda seed: seed,
        resolve_start_seed=lambda seed: seed,
    )
    env._resolved_start_seed = env.cfg.seed
    env._offset_seed = 0
    env._task = None
    env._instructions = None
    env._eval_chosen_instruction = None
    env._episode_finalized = True
    env._post_reset_state_available = False
    env._cached_obs_robots = None
    env._runtime_layout = None
    env._active_task_config = None
    env._video_writer = None
    env._check_and_update_seed = lambda task_config: (
        SimpleNamespace(
            robot=robot,
            setup_demo=lambda **kwargs: None,
            get_obs=lambda: {},
            info={},
        ),
        [],
        task_config,
    )
    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.env.in_robotwin_workspace",
        nullcontext,
    )
    monkeypatch.setattr(
        env,
        "get_robot_urdf",
        lambda: {"left": b"<robot/>"},
        raising=False,
    )
    return env


def _make_step_stub_env(
    *,
    action_type: str,
) -> tuple[RoboTwinEnv, MagicMock]:
    env = _make_uninitialized_env()
    env.cfg = SimpleNamespace(action_type=action_type)
    env._episode_finalized = False
    env._post_reset_state_available = False
    env._last_obs_step_index = 0
    take_action = MagicMock()
    env._task = SimpleNamespace(
        robot=SimpleNamespace(
            get_left_arm_jointState=lambda: [0.0] * 7,
            get_right_arm_jointState=lambda: [0.0] * 7,
        ),
        take_action=take_action,
        step_lim=None,
        take_action_cnt=0,
        eval_success=False,
        get_obs=lambda: {},
    )
    env._write_video_frame = lambda raw_obs: None
    env._format_obs = lambda raw_obs, *, step_index: {
        **raw_obs,
        "step_index": step_index,
        "step_timestamp": (
            env.step_index_to_log_time_ns(step_index) / 1_000_000_000.0
        ),
    }
    env._get_info = lambda: {}
    return env, take_action


class _StateFakeTask:
    def __init__(
        self,
        *,
        robot: SimpleNamespace,
        raw_obs: dict[str, object],
        info: dict[str, object] | None = None,
        close_calls: list[bool] | None = None,
        task_name: str = "robotwin_dummy_task",
    ) -> None:
        self.robot = robot
        self.raw_obs = raw_obs
        self.info = info or {}
        self.close_calls = [] if close_calls is None else close_calls
        self.task_name = task_name
        self.setup_calls: list[dict[str, object]] = []
        self.play_once_calls = 0
        self.plan_success = True
        self.render_freq = 0

    def setup_demo(self, **kwargs) -> None:
        self.setup_calls.append(kwargs)

    def play_once(self) -> dict[str, object]:
        self.play_once_calls += 1
        self.arm_tag = "left"
        self.info["info"] = {"arm": self.arm_tag}
        return self.info

    def get_obs(self) -> dict[str, object]:
        return self.raw_obs

    def check_success(self) -> bool:
        return getattr(self, "arm_tag", None) == "left"

    def close_env(self, clear_cache: bool) -> None:
        self.close_calls.append(clear_cache)


def _make_state_stub_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> tuple[RoboTwinEnv, list[bool], list[_StateFakeTask]]:
    robot = SimpleNamespace(
        is_dual_arm=True,
        left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        left_arm_joints_name=["left_joint"],
        right_arm_joints_name=["right_joint"],
        left_gripper_name={"base": "left_gripper"},
        right_gripper_name={"base": "right_gripper"},
        left_ee=_make_fake_joint_with_child_link("left_eef"),
        right_ee=_make_fake_joint_with_child_link("right_eef"),
        left_urdf_path=str(Path(__file__).resolve()),
        right_urdf_path=str(Path(__file__).resolve()),
    )
    robot.left_entity = object()
    robot.right_entity = robot.left_entity
    task_config_path = tmp_path / "task_config.yml"
    task_config_path.write_text("{}\n", encoding="utf-8")

    def _get_task_config_for_seed(
        cfg: RoboTwinEnvCfg,
        runtime_seed: int,
    ) -> dict[str, object]:
        return {
            "seed": runtime_seed,
            "now_ep_num": cfg.episode_id,
            "task_name": cfg.task_name,
        }

    monkeypatch.setattr(
        RoboTwinEnvCfg,
        "get_task_config_for_seed",
        _get_task_config_for_seed,
    )
    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.env.in_robotwin_workspace",
        nullcontext,
    )

    cfg = RoboTwinEnvCfg(
        task_name="robotwin_dummy_task",
        seed=1,
        episode_id=5,
        check_expert=False,
        check_task_init=False,
        task_config_path=str(task_config_path),
        patch_curobo_base_transform=False,
    )
    env = RoboTwinEnv(cfg)
    env._offset_seed = 2
    env._post_reset_state_available = True
    env._episode_finalized = False
    env._instructions = {"unseen": ["pick"]}
    env._eval_chosen_instruction = None
    close_calls: list[bool] = []
    env._task = _StateFakeTask(
        robot=robot,
        raw_obs={"initial": True},
        info={"source": "old"},
        close_calls=close_calls,
    )
    env._runtime_layout = derive_runtime_layout(env._require_task())
    env._active_task_config = {
        "seed": 3,
        "now_ep_num": 5,
        "task_name": "robotwin_dummy_task",
    }
    monkeypatch.setattr(
        env,
        "_format_obs",
        lambda raw_obs, *, step_index: {
            "formatted": raw_obs,
            "step_index": step_index,
            "step_timestamp": (
                env.step_index_to_log_time_ns(step_index) / 1_000_000_000.0
            ),
        },
        raising=False,
    )

    created_tasks: list[_StateFakeTask] = []

    def _create_task_from_name(task_name: str) -> _StateFakeTask:
        task = _StateFakeTask(
            robot=robot,
            raw_obs={"restored": task_name},
            info={"source": "restored"},
            task_name=task_name,
        )
        created_tasks.append(task)
        return task

    monkeypatch.setattr(
        "robo_orchard_lab.envs.robotwin.env.create_task_from_name",
        _create_task_from_name,
    )
    return env, close_calls, created_tasks


def _assert_pose_close(
    actual: BatchFrameTransform,
    expected: BatchFrameTransform,
    *,
    atol: float = 1e-4,
) -> None:
    assert torch.allclose(actual.xyz, expected.xyz, atol=atol)
    quat_alignment = torch.sum(actual.quat * expected.quat, dim=-1).abs()
    assert torch.allclose(
        quat_alignment,
        torch.ones_like(quat_alignment),
        atol=atol,
    )


def _decode_first_frame_rgb(
    video_path: str,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    ffmpeg_binary = _get_ffmpeg_binary()
    result = subprocess.run(
        [
            ffmpeg_binary,
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        height, width, 3
    )


class _FakeSerialChain:
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        end_frame_name: str,
        root_frame_name: str = "robot_base",
    ) -> None:
        self.dtype = dtype
        self.device = device
        self._end_frame_name = end_frame_name
        self._root_frame_name = root_frame_name
        self.recorded_joint_dtypes: list[torch.dtype] = []

    def forward_kinematics_tf(
        self,
        joint_positions: torch.Tensor,
    ) -> dict[str, BatchFrameTransform]:
        self.recorded_joint_dtypes.append(joint_positions.dtype)
        batch_size = joint_positions.shape[0]
        return {
            self._end_frame_name: BatchFrameTransform(
                xyz=torch.zeros(
                    batch_size,
                    3,
                    dtype=self.dtype,
                    device=self.device,
                ),
                quat=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]],
                    dtype=self.dtype,
                    device=self.device,
                ).repeat(batch_size, 1),
                parent_frame_id=self._root_frame_name,
                child_frame_id=self._end_frame_name,
            )
        }


@pytest.fixture()
def dummy_env_without_expert_check():
    env = RoboTwinEnv(
        RoboTwinEnvCfg(
            task_name="place_object_basket",
            check_expert=False,
            seed=1,
            check_task_init=False,  # for fast initialization
        )
    )
    yield env
    env.close()


class TestRoboTwinEnv:
    @pytest.mark.parametrize(
        "model_type",
        [
            RoboTwinEpisodeInstructionsPayload,
            RoboTwinObservationInstructionPayload,
            RoboTwinObservationMetaPayload,
        ],
    )
    def test_mcap_pydantic_schemas_are_compatible(
        self,
        model_type: type[BaseModel],
    ) -> None:
        assert_mcap_compatible_pydantic_schema(model_type)

    def test_logger_manager_does_not_duplicate_when_it_owns_handlers(self):
        manager_logger = robotwin_env._logger_manager.get_logger()

        if manager_logger.handlers:
            assert manager_logger.propagate is False

    def test_init_ensures_curobo_base_transform_patch(
        self,
        monkeypatch,
        tmp_path,
    ):
        task_config_path = tmp_path / "task_config.yml"
        task_config_path.write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(
            RoboTwinEnvCfg,
            "__post_init__",
            lambda self: None,
        )
        prepare_calls: list[RoboTwinEnvCfg] = []
        monkeypatch.setattr(
            robotwin_env,
            "prepare_robotwin_runtime_for_cfg",
            lambda cfg: prepare_calls.append(cfg),
            raising=False,
        )
        cfg = RoboTwinEnvCfg(
            task_name="robotwin_dummy_task",
            action_type="ee",
            check_expert=False,
            check_task_init=False,
            task_config_path=str(task_config_path),
        )

        env = RoboTwinEnv(cfg)

        assert prepare_calls == [cfg]
        env.close()

    def test_seed_check_without_precheck_only_creates_task(
        self,
        monkeypatch,
    ):
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            check_expert=False,
            check_task_init=False,
            task_name="robotwin_dummy_task",
        )
        task = SimpleNamespace()
        monkeypatch.setattr(
            robotwin_env,
            "create_task_from_name",
            lambda task_name: task,
        )

        task_config = {"seed": 1}
        created_task, instructions, accepted_config = (
            env._check_and_update_seed(task_config)
        )

        assert created_task is task
        assert instructions is None
        assert accepted_config is task_config

    def test_check_expert_logs_retry_summary_without_per_seed_warning(
        self,
        monkeypatch,
    ):
        _install_fake_robotwin_instruction_generator(monkeypatch)
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            check_expert=True,
            task_name="robotwin_dummy_task",
            max_instruction_num=1,
            get_task_config_for_seed=lambda runtime_seed: {
                "seed": runtime_seed
            },
        )
        env._resolved_start_seed = 100
        env._offset_seed = 0
        accepted_task = SimpleNamespace()
        outcomes = [
            (None, None, False),
            (accepted_task, {"seed": 101}, True),
        ]
        env._check_expert_traj = lambda task_config: outcomes.pop(0)
        env._dispose_task = MagicMock(return_value=True)
        logger = SimpleNamespace(
            debug=MagicMock(),
            info=MagicMock(),
            warning=MagicMock(),
            error=MagicMock(),
        )
        monkeypatch.setattr(robotwin_env, "logger", logger)
        ret_task, instructions, accepted_config = env._check_and_update_seed(
            {"seed": 100}
        )

        assert ret_task is accepted_task
        assert instructions == {"seen": ["robotwin_dummy_task"]}
        assert accepted_config == {"seed": 101}
        assert env.current_seed == 101
        env._dispose_task.assert_called_once_with(None, clear_cache=True)
        logger.info.assert_called_once_with(
            "RoboTwin expert trajectory resolved after retry: "
            "task=%s requested_seed=%s actual_seed=%s retries=%s",
            "robotwin_dummy_task",
            100,
            101,
            1,
        )
        logger.warning.assert_not_called()
        logger.error.assert_not_called()

    def test_checked_task_is_disposed_if_instruction_generation_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        description_module = _install_fake_robotwin_instruction_generator(
            monkeypatch
        )
        description_module.generate_episode_descriptions = MagicMock(
            side_effect=RuntimeError("instruction failed")
        )
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            check_expert=False,
            check_task_init=True,
            task_name="robotwin_dummy_task",
            max_instruction_num=1,
        )
        env._resolved_start_seed = 100
        env._offset_seed = 0
        accepted_task = SimpleNamespace()
        env._check_expert_traj = MagicMock(
            return_value=(accepted_task, {"seed": 100}, False)
        )
        env._dispose_task = MagicMock(return_value=True)

        with pytest.raises(RuntimeError, match="instruction failed"):
            env._check_and_update_seed({"seed": 100})

        env._dispose_task.assert_called_once_with(
            accepted_task,
            clear_cache=True,
        )

    def test_check_expert_does_not_retry_with_pending_worker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_robotwin_instruction_generator(monkeypatch)
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            check_expert=True,
            task_name="robotwin_dummy_task",
            max_instruction_num=1,
            get_task_config_for_seed=lambda runtime_seed: {
                "seed": runtime_seed
            },
        )
        env._resolved_start_seed = 100
        env._offset_seed = 0
        env._pending_disposal_tasks = [SimpleNamespace()]
        attempt_count = 0

        def _failed_attempt(task_config) -> tuple[None, None, bool]:
            del task_config
            nonlocal attempt_count
            attempt_count += 1
            return None, None, False

        env._check_expert_traj = _failed_attempt
        env._drain_pending_disposals = lambda *, clear_cache: None

        with pytest.raises(RuntimeError, match="expert retry cannot continue"):
            env._check_and_update_seed({"seed": 100})

        assert attempt_count == 1
        assert env.current_seed == 100

    def test_check_expert_traj_setup_failure_logs_concise_debug(
        self,
        monkeypatch,
    ):
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            task_name="robotwin_dummy_task",
            action_type="qpos",
            patch_curobo_base_transform=False,
        )
        env._resolved_start_seed = 100
        env._offset_seed = 2
        logger = SimpleNamespace(
            debug=MagicMock(),
            info=MagicMock(),
            warning=MagicMock(),
            error=MagicMock(),
        )
        failure = RuntimeError("unstable object")
        task = SimpleNamespace(
            setup_demo=MagicMock(side_effect=failure),
            play_once=MagicMock(),
            close_env=MagicMock(),
        )
        monkeypatch.setattr(robotwin_env, "logger", logger)
        monkeypatch.setattr(robotwin_env, "in_robotwin_workspace", nullcontext)
        monkeypatch.setattr(
            robotwin_env,
            "create_task_from_name",
            lambda task_name: task,
        )

        checked_task, episode_info, success = env._check_expert_traj(
            {"seed": 102, "large": "cfg"}
        )

        assert checked_task is None
        assert episode_info is None
        assert success is False
        task.close_env.assert_called_once_with(clear_cache=True)
        logger.debug.assert_called_once_with(
            "RoboTwin expert trajectory check failed during %s: "
            "task=%s seed=%s error=%s",
            "setup",
            "robotwin_dummy_task",
            102,
            failure,
        )
        logger.error.assert_not_called()

    def test_check_expert_traj_guard_failure_preserves_original_error(
        self,
        monkeypatch,
    ):
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            action_type="ee",
            patch_curobo_base_transform=True,
            task_name="robotwin_dummy_task",
        )
        env._resolved_start_seed = 100
        env._offset_seed = 2
        task = SimpleNamespace(
            robot=SimpleNamespace(communication_flag=True),
            setup_demo=MagicMock(),
            play_once=MagicMock(),
            close_env=MagicMock(side_effect=RuntimeError("close failed")),
        )
        monkeypatch.setattr(robotwin_env, "in_robotwin_workspace", nullcontext)
        monkeypatch.setattr(
            curobo_base_patch,
            "_PATCH_INSTALLED_IN_PROCESS",
            True,
        )
        monkeypatch.setattr(
            robotwin_env,
            "create_task_from_name",
            lambda task_name: task,
        )

        with pytest.raises(
            robotwin_env.RoboTwinCuroboPatchUnsupportedError,
            match="communication_flag",
        ):
            env._check_expert_traj({"seed": 102})

        task.setup_demo.assert_called_once_with(seed=102, render_freq=0)
        task.play_once.assert_not_called()
        task.close_env.assert_called_once_with(clear_cache=True)

    def test_check_expert_traj_disposes_after_success_check_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            task_name="robotwin_dummy_task",
            action_type="qpos",
            patch_curobo_base_transform=False,
        )
        env._resolved_start_seed = 100
        env._offset_seed = 2
        task = SimpleNamespace(
            setup_demo=MagicMock(),
            play_once=MagicMock(),
            plan_success=True,
            check_success=MagicMock(side_effect=RuntimeError("check failed")),
            close_env=MagicMock(),
        )
        monkeypatch.setattr(robotwin_env, "in_robotwin_workspace", nullcontext)
        monkeypatch.setattr(
            robotwin_env,
            "create_task_from_name",
            lambda task_name: task,
        )

        checked_task, episode_info, success = env._check_expert_traj(
            {"seed": 102}
        )

        assert checked_task is None
        assert episode_info is None
        assert success is False
        assert task.close_env.call_args_list == [
            call(clear_cache=False),
            call(clear_cache=True),
        ]

    def test_check_expert_traj_keeps_official_task_and_workers_for_reuse(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            task_name="robotwin_dummy_task",
            action_type="qpos",
            patch_curobo_base_transform=False,
        )
        env._resolved_start_seed = 100
        env._offset_seed = 2
        conn = SimpleNamespace(
            closed=False,
            send=MagicMock(),
            close=MagicMock(),
        )
        proc = SimpleNamespace(
            is_alive=MagicMock(side_effect=[True, False, False, False]),
            join=MagicMock(),
            terminate=MagicMock(),
            kill=MagicMock(),
            close=MagicMock(),
        )
        robot = SimpleNamespace(
            left_conn=conn,
            right_conn=None,
            left_proc=proc,
            right_proc=None,
        )
        task = SimpleNamespace(
            robot=robot,
            setup_demo=MagicMock(),
            play_once=MagicMock(return_value={"info": {"seed": 102}}),
            plan_success=True,
            check_success=MagicMock(return_value=True),
            close_env=MagicMock(),
        )
        monkeypatch.setattr(robotwin_env, "in_robotwin_workspace", nullcontext)
        monkeypatch.setattr(
            robotwin_env,
            "create_task_from_name",
            lambda task_name: task,
        )

        checked_task, episode_info, success = env._check_expert_traj(
            {"seed": 102}
        )

        assert checked_task is task
        assert episode_info == {"seed": 102}
        assert success is True
        task.close_env.assert_called_once_with(clear_cache=False)
        conn.send.assert_not_called()
        conn.close.assert_not_called()
        proc.join.assert_not_called()
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()
        proc.close.assert_not_called()
        assert robot.left_conn is conn
        assert robot.left_proc is proc

    def test_check_expert_traj_disposes_before_reraising_cancellation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            task_name="robotwin_dummy_task",
            action_type="qpos",
            patch_curobo_base_transform=False,
        )
        env._resolved_start_seed = 100
        env._offset_seed = 2
        cancellation = KeyboardInterrupt()
        task = SimpleNamespace(
            setup_demo=MagicMock(),
            play_once=MagicMock(side_effect=cancellation),
            close_env=MagicMock(),
        )
        monkeypatch.setattr(robotwin_env, "in_robotwin_workspace", nullcontext)
        monkeypatch.setattr(
            robotwin_env,
            "create_task_from_name",
            lambda task_name: task,
        )

        with pytest.raises(KeyboardInterrupt) as exc_info:
            env._check_expert_traj({"seed": 102})

        assert exc_info.value is cancellation
        task.close_env.assert_called_once_with(clear_cache=True)

    def test_joints2ee_pose_uses_arm_joints_only(self, monkeypatch):
        env = _make_uninitialized_env()
        env._task = SimpleNamespace(
            robot=SimpleNamespace(
                left_arm_joints_name=["left_joint_0", "left_joint_1"],
                right_arm_joints_name=["right_joint_0", "right_joint_1"],
            )
        )
        env._runtime_layout = SimpleNamespace(
            left_arm_joint_count=2,
            right_arm_joint_count=2,
        )
        captured: dict[str, torch.Tensor] = {}

        class _FakeJointsToEEF:
            def transform(
                self,
                left_arm_joints: torch.Tensor,
                right_arm_joints: torch.Tensor,
            ) -> RoboTwinEEF:
                captured["left"] = left_arm_joints.clone()
                captured["right"] = right_arm_joints.clone()
                return RoboTwinEEF(
                    left_eef=BatchFrameTransform(
                        xyz=torch.tensor([[10.0, 0.0, 0.0]]),
                        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                        parent_frame_id="world",
                        child_frame_id="left_eef",
                    ),
                    right_eef=BatchFrameTransform(
                        xyz=torch.tensor([[20.0, 0.0, 0.0]]),
                        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                        parent_frame_id="world",
                        child_frame_id="right_eef",
                    ),
                )

        monkeypatch.setattr(
            env,
            "_get_joints_to_eef_transform",
            lambda: _FakeJointsToEEF(),
            raising=False,
        )

        eef_tfs = env._joints2ee_pose(
            np.array([1.0, 2.0, 0.5, 3.0, 4.0, 0.75], dtype=np.float32)
        )

        assert torch.equal(
            captured["left"],
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        )
        assert torch.equal(
            captured["right"],
            torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        )
        assert eef_tfs.left_eef.parent_frame_id == "world"
        assert eef_tfs.right_eef.parent_frame_id == "world"
        assert eef_tfs.left_eef.child_frame_id == "left_eef"
        assert eef_tfs.right_eef.child_frame_id == "right_eef"
        assert torch.equal(
            eef_tfs.left_eef.xyz,
            torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
        )
        assert torch.equal(
            eef_tfs.right_eef.xyz,
            torch.tensor([[20.0, 0.0, 0.0]], dtype=torch.float32),
        )

    def test_default_runtime_observation_urdf_and_step(
        self,
        dummy_env_without_expert_check: RoboTwinEnv,
    ):
        """Reuse one default runtime for observation, URDF, and step checks."""
        env = dummy_env_without_expert_check
        obs, _ = env.reset()
        assert obs is not None
        assert "tf" in obs
        assert "robots" in obs

        urdf_dict = env.get_robot_urdf()
        assert urdf_dict is not None
        assert "left" in urdf_dict

        # Note that not all env can step because of robotwin BUG!
        step_return = env.step([1.0] * 14)
        assert step_return.observations is not None
        assert "tf" in step_return.observations

    def test_get_obs_robots_caches_robot_metadata(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        env.cfg.action_type = "qpos"
        env._task = SimpleNamespace(robot=robot)
        urdf_calls = {"count": 0}

        def _get_robot_urdf() -> dict[str, bytes]:
            urdf_calls["count"] += 1
            return {"left": b"<robot name='combined'/>"}

        monkeypatch.setattr(
            env,
            "get_robot_urdf",
            _get_robot_urdf,
            raising=False,
        )

        first = env.get_obs_robots()
        second = env.get_obs_robots()

        assert urdf_calls["count"] == 1
        assert set(first.keys()) == {"left"}
        first_robot = first["left"]
        assert isinstance(first_robot, Robot)
        assert first_robot.name == "left"
        assert first_robot.content == "<robot name='combined'/>"
        assert first_robot.content_format == RobotDescriptionFormat.URDF
        assert first_robot.md5
        assert second["left"] is first_robot

    def test_split_layout_exposes_two_robots_namespaced_tf_and_joints(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        left_urdf = tmp_path / "left.urdf"
        right_urdf = tmp_path / "right.urdf"
        left_urdf.write_text("<robot name='left'/>", encoding="utf-8")
        right_urdf.write_text("<robot name='right'/>", encoding="utf-8")
        robot = SimpleNamespace(
            is_dual_arm=False,
            left_entity_origion_pose=_make_fake_sapien_pose((-0.4, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.4, 0.0, 0.0)),
            left_arm_joints_name=["joint_1", "joint_2"],
            right_arm_joints_name=["joint_1", "joint_2", "joint_3"],
            left_gripper_name={"base": "gripper"},
            right_gripper_name={"base": "gripper"},
            left_ee=_make_fake_joint_with_child_link("link6"),
            right_ee=_make_fake_joint_with_child_link("link6"),
            left_urdf_path=str(left_urdf),
            right_urdf_path=str(right_urdf),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        env.cfg.action_type = "qpos"
        env._task = SimpleNamespace(robot=robot)
        env._runtime_layout = derive_runtime_layout(env._require_task())
        env.get_robot_urdf = RoboTwinEnv.get_robot_urdf.__get__(
            env, RoboTwinEnv
        )

        assert env.get_robot_urdf() == {
            "left": b"<robot name='left'/>",
            "right": b"<robot name='right'/>",
        }
        robots = env.get_obs_robots()
        assert list(robots) == ["left", "right"]
        assert [robots[side].index for side in robots] == [0, 1]
        assert [robots[side].name for side in robots] == ["left", "right"]
        joint_names = get_joint_state_names(
            env._require_task(), env._runtime_layout
        )
        assert joint_names == [
            "left/joint_1",
            "left/joint_2",
            "left/gripper",
            "right/joint_1",
            "right/joint_2",
            "right/joint_3",
            "right/gripper",
        ]
        assert (
            env._runtime_layout.left_control_eef_frame_id,
            env._runtime_layout.right_control_eef_frame_id,
        ) == (
            "left/link6_from_obs",
            "right/link6_from_obs",
        )

        obs = env._format_obs(
            {
                "endpose": {
                    "left_endpose": np.array(
                        [-0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    "right_endpose": np.array(
                        [0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                },
                "observation": {},
            },
            step_index=0,
        )
        tf_graph = obs["tf"]
        for child_frame_id in (
            "left_robot_base",
            "right_robot_base",
            "left/link6_from_obs",
            "right/link6_from_obs",
        ):
            assert tf_graph.get_tf("world", child_frame_id) is not None
        assert "embodiment_layout" not in obs

        env._last_obs = obs
        env._last_obs_step_index = 0
        messages = env.get_mcap_obs(anchor_log_time_ns=123)
        assert "observation/meta/robots/left" in messages
        assert "observation/meta/robots/right" in messages
        exported_tf = messages["observation/tf"][0].data
        assert exported_tf.get_tf("world", "left_robot_base") is not None
        assert exported_tf.get_tf("world", "right_robot_base") is not None

    def test_get_mcap_obs_exports_typed_json_payloads(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        env.cfg.action_type = "qpos"
        env._instructions = {
            "episode_index": 0,
            "seen": ["demo instruction"],
            "unseen": ["pick"],
        }
        env._eval_chosen_instruction = "pick"
        env._last_obs = {}
        env._last_obs_step_index = 0

        messages = env.get_mcap_obs(
            topic_prefix="rollout/observation",
            anchor_log_time_ns=123,
        )

        instruction = messages["rollout/observation/instruction"][0].data
        assert isinstance(instruction, RoboTwinObservationInstructionPayload)
        assert instruction.instructions is not None
        assert not isinstance(instruction.instructions, str)
        assert instruction.instructions.episode_index == 0
        assert instruction.instructions.seen == ["demo instruction"]
        assert instruction.instructions.unseen == ["pick"]
        assert instruction.eval_chosen_instruction == "pick"
        assert_mcap_compatible_pydantic_schema(
            RoboTwinObservationInstructionPayload
        )

        meta = messages["rollout/observation/meta"][0].data
        assert isinstance(meta, RoboTwinObservationMetaPayload)
        assert meta.task_name == "robotwin_dummy_task"
        assert meta.action_type == "qpos"
        assert meta.seed == 1

    def test_format_obs_adds_joint_and_endpose_tfs(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            left_ee=_make_fake_joint_with_child_link("left_real_eef"),
            right_ee=_make_fake_joint_with_child_link("right_real_eef"),
            left_arm_joints_name=["left_joint_0", "left_joint_1"],
            right_arm_joints_name=["right_joint_0", "right_joint_1"],
            left_gripper_name={"base": "left_gripper"},
            right_gripper_name={"base": "right_gripper"},
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        env._task = SimpleNamespace(robot=robot)
        joint_eef = RoboTwinEEF(
            left_eef=BatchFrameTransform(
                xyz=torch.tensor([[10.0, 1.0, 2.0]]),
                quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                parent_frame_id="world",
                child_frame_id="left_eef",
            ),
            right_eef=BatchFrameTransform(
                xyz=torch.tensor([[20.0, 3.0, 4.0]]),
                quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                parent_frame_id="world",
                child_frame_id="right_eef",
            ),
        )
        monkeypatch.setattr(
            env,
            "_joints2ee_pose",
            lambda joints: joint_eef,
            raising=False,
        )
        raw_urdf = env.get_robot_urdf()["left"]

        obs = env._format_obs(
            {
                "joint_action": {
                    "vector": np.array(
                        [1.0, 2.0, 0.5, 3.0, 4.0, 0.75],
                        dtype=np.float32,
                    )
                },
                "endpose": {
                    "left_endpose": np.array(
                        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    "left_gripper": 0.5,
                    "right_endpose": np.array(
                        [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    "right_gripper": 0.75,
                },
                "observation": {},
            },
            step_index=3,
        )

        assert obs["step_index"] == 3
        assert obs["step_timestamp"] == 0.3
        tf_graph = obs["tf"]
        layout = env._require_runtime_layout()
        left_endpose_frame_id, right_endpose_frame_id = (
            layout.left_control_eef_frame_id,
            layout.right_control_eef_frame_id,
        )
        assert tf_graph.get_tf(
            "world", _LEFT_EEF_FROM_JOINT_FRAME_ID
        ) == BatchFrameTransform(
            xyz=torch.tensor([[10.0, 1.0, 2.0]]),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            parent_frame_id="world",
            child_frame_id=_LEFT_EEF_FROM_JOINT_FRAME_ID,
        )
        assert tf_graph.get_tf(
            "world", _RIGHT_EEF_FROM_JOINT_FRAME_ID
        ) == BatchFrameTransform(
            xyz=torch.tensor([[20.0, 3.0, 4.0]]),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            parent_frame_id="world",
            child_frame_id=_RIGHT_EEF_FROM_JOINT_FRAME_ID,
        )
        assert tf_graph.get_tf(
            "world", left_endpose_frame_id
        ) == BatchFrameTransform(
            xyz=torch.tensor([[0.1, 0.2, 0.3]]),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            parent_frame_id="world",
            child_frame_id=left_endpose_frame_id,
        )
        assert tf_graph.get_tf(
            "world", right_endpose_frame_id
        ) == BatchFrameTransform(
            xyz=torch.tensor([[0.4, 0.5, 0.6]]),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            parent_frame_id="world",
            child_frame_id=right_endpose_frame_id,
        )
        assert set(obs["robots"].keys()) == {"left"}
        obs_robot = obs["robots"]["left"]
        assert isinstance(obs_robot, Robot)
        assert obs_robot.name == "left"
        assert obs_robot.content == raw_urdf.decode("utf-8")
        assert obs_robot.content_format == RobotDescriptionFormat.URDF
        assert obs_robot.md5

    def test_endpose_in_obs_when_enabled(self):
        """Current supported RoboTwin embodiments align raw endpose with FK.

        RoboTwin's raw ``endpose`` values come from its processed
        ``get_*_ee_pose()`` helpers rather than from an unmodified runtime
        link pose. For the currently supported combined dual-arm embodiment
        conventions, those values are still expected to numerically match the
        URDF joint-FK helper within a small tolerance.
        """
        env = RoboTwinEnv(
            RoboTwinEnvCfg(
                task_name="place_object_basket",
                check_expert=False,
                seed=1,
                check_task_init=False,  # for fast initialization
                task_config_overrides=[("data_type/endpose", True)],
            )
        )
        try:
            obs, info = env.reset()
            assert obs is not None
            assert "endpose" in obs
            endpose = obs["endpose"]
            assert isinstance(endpose, dict)
            assert np.asarray(endpose["left_endpose"]).size > 0
            assert np.asarray(endpose["right_endpose"]).size > 0
            assert endpose["left_gripper"] is not None
            assert endpose["right_gripper"] is not None
            tf_graph = obs["tf"]
            layout = env._require_runtime_layout()
            left_endpose_frame_id, right_endpose_frame_id = (
                layout.left_control_eef_frame_id,
                layout.right_control_eef_frame_id,
            )
            left_joint_tf = tf_graph.get_tf(
                "world", _LEFT_EEF_FROM_JOINT_FRAME_ID
            )
            right_joint_tf = tf_graph.get_tf(
                "world", _RIGHT_EEF_FROM_JOINT_FRAME_ID
            )
            left_endpose_tf = tf_graph.get_tf("world", left_endpose_frame_id)
            right_endpose_tf = tf_graph.get_tf("world", right_endpose_frame_id)
            assert isinstance(left_joint_tf, BatchFrameTransform)
            assert isinstance(right_joint_tf, BatchFrameTransform)
            assert isinstance(left_endpose_tf, BatchFrameTransform)
            assert isinstance(right_endpose_tf, BatchFrameTransform)
            _assert_pose_close(left_joint_tf, left_endpose_tf, atol=1e-3)
            _assert_pose_close(right_joint_tf, right_endpose_tf, atol=1e-3)
        finally:
            env.close()

    def test_step_rejects_qpos_action_width_mismatch(self):
        env, take_action = _make_step_stub_env(action_type="qpos")

        with pytest.raises(
            ValueError,
            match="expected 14, got 16",
        ):
            env.step([0.0] * 16)
        take_action.assert_not_called()

    @pytest.mark.parametrize(
        ("left_width", "right_width"), [(7, 7), (7, 8), (8, 8)]
    )
    def test_step_accepts_runtime_qpos_widths(
        self,
        left_width: int,
        right_width: int,
    ) -> None:
        env, take_action = _make_step_stub_env(action_type="qpos")
        env._task.robot.get_left_arm_jointState = lambda: [0.0] * left_width
        env._task.robot.get_right_arm_jointState = lambda: [0.0] * right_width

        env.step([0.0] * (left_width + right_width))

        take_action.assert_called_once()

    def test_step_advances_observation_step_metadata(self) -> None:
        env, take_action = _make_step_stub_env(action_type="qpos")

        first = env.step([0.0] * 14)
        second = env.step([0.0] * 14)

        assert first.observations is not None
        assert first.observations["step_index"] == 1
        assert first.observations["step_timestamp"] == 0.1
        assert second.observations is not None
        assert second.observations["step_index"] == 2
        assert second.observations["step_timestamp"] == 0.2
        assert env._last_obs_step_index == 2
        assert take_action.call_count == 2

    def test_step_requires_reset_step_clock_before_action(self) -> None:
        env, take_action = _make_step_stub_env(action_type="qpos")
        env._last_obs_step_index = None

        with pytest.raises(RuntimeError, match="successful reset"):
            env.step([0.0] * 14)

        take_action.assert_not_called()

    def test_step_format_failure_keeps_last_observation_step_index(
        self,
    ) -> None:
        env, take_action = _make_step_stub_env(action_type="qpos")

        def _raise_format_error(raw_obs, *, step_index):
            del raw_obs, step_index
            raise RuntimeError("format failed")

        env._format_obs = _raise_format_error

        with pytest.raises(RuntimeError, match="format failed"):
            env.step([0.0] * 14)

        take_action.assert_called_once()
        assert env._last_obs_step_index == 0

    def test_step_rejects_ee_action_width_mismatch(self):
        env, take_action = _make_step_stub_env(action_type="ee")

        with pytest.raises(
            ValueError,
            match="expected 16, got 14",
        ):
            env.step([0.0] * 14)

        take_action.assert_not_called()

    def test_step_rejects_unsupported_action_type(self):
        env, take_action = _make_step_stub_env(action_type="unsupported")

        with pytest.raises(
            ValueError,
            match="Unsupported RoboTwin action_type",
        ):
            env.step([0.0] * 16)

        take_action.assert_not_called()

    def test_get_mcap_action_sidecars_exports_ee_targets_without_simulator(
        self,
    ):
        env = object.__new__(RoboTwinEnv)
        env.cfg = SimpleNamespace(action_type="ee")
        env._last_obs = {"step": 3}
        env._last_obs_step_index = 3
        action = np.array(
            [
                1.0,
                2.0,
                3.0,
                1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                4.0,
                5.0,
                6.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        messages = RoboTwinEnv.get_mcap_action_sidecars(
            env,
            action,
            anchor_log_time_ns=123,
        )

        msg = messages["rollout/next_action/eef_tf"][0]
        assert msg.log_time == 123
        tf_list = msg.data.as_state().tf_list
        assert [tf.parent_frame_id for tf in tf_list] == ["world", "world"]
        assert [tf.child_frame_id for tf in tf_list] == [
            "left_eef_target_from_env_action_next_action",
            "right_eef_target_from_env_action_next_action",
        ]
        assert [tf.timestamps for tf in tf_list] == [[123], [123]]
        torch.testing.assert_close(
            tf_list[0].xyz,
            torch.tensor([[1.0, 2.0, 3.0]]),
        )
        torch.testing.assert_close(
            tf_list[1].xyz,
            torch.tensor([[4.0, 5.0, 6.0]]),
        )

    def test_get_mcap_action_sidecars_uses_explicit_frame_suffix(self):
        """Frame ids use caller-provided suffix, not topic-prefix parsing."""
        env = object.__new__(RoboTwinEnv)
        env.cfg = SimpleNamespace(action_type="ee")
        env._last_obs = {"step": 3}
        env._last_obs_step_index = 3
        action = np.array(
            [
                1.0,
                2.0,
                3.0,
                1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                4.0,
                5.0,
                6.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        next_msg = RoboTwinEnv.get_mcap_action_sidecars(
            env,
            action,
            topic_prefix="rollout/next_action",
            anchor_log_time_ns=123,
        )["rollout/next_action/eef_tf"][0]
        last_msg = RoboTwinEnv.get_mcap_action_sidecars(
            env,
            action,
            topic_prefix="rollout/custom_action",
            anchor_log_time_ns=123,
            frame_id_suffix="last_action",
        )["rollout/custom_action/eef_tf"][0]

        next_child_ids = [
            tf.child_frame_id for tf in next_msg.data.as_state().tf_list
        ]
        last_child_ids = [
            tf.child_frame_id for tf in last_msg.data.as_state().tf_list
        ]
        assert next_child_ids == [
            "left_eef_target_from_env_action_next_action",
            "right_eef_target_from_env_action_next_action",
        ]
        assert last_child_ids == [
            "left_eef_target_from_env_action_last_action",
            "right_eef_target_from_env_action_last_action",
        ]
        assert set(next_child_ids).isdisjoint(last_child_ids)

    def test_get_mcap_action_sidecars_returns_empty_for_qpos(self):
        env = object.__new__(RoboTwinEnv)
        env.cfg = SimpleNamespace(action_type="qpos")
        env._last_obs = {"step": 0}
        env._last_obs_step_index = 0

        assert RoboTwinEnv.get_mcap_action_sidecars(env, np.zeros(14)) == {}

    def test_step_rejects_after_finalize_until_reset(self) -> None:
        env, take_action = _make_step_stub_env(action_type="qpos")

        def _mark_finalized() -> None:
            env._episode_finalized = True

        env.finalize_episode = _mark_finalized
        env.finalize_episode()

        with pytest.raises(RuntimeError, match="reset"):
            env.step([0.0] * 14)
        take_action.assert_not_called()

        env._episode_finalized = False
        env.step([0.0] * 14)
        take_action.assert_called_once()

    def test_step_rejects_without_active_episode_message(self) -> None:
        env = _make_uninitialized_env()
        env._episode_finalized = True

        with pytest.raises(RuntimeError) as exc_info:
            env.step([0.0] * 14)

        error_message = str(exc_info.value)
        assert "no active episode" in error_message
        assert "reset()" in error_message
        assert "finalized" not in error_message

    def test_reset_accepts_split_arm_layout_with_two_absolute_bases(
        self, monkeypatch
    ):
        env = _make_reset_stub_env(
            monkeypatch,
            robot=SimpleNamespace(
                is_dual_arm=False,
                left_entity_origion_pose=_make_fake_sapien_pose(
                    (0.0, 0.0, 0.0)
                ),
                right_entity_origion_pose=_make_fake_sapien_pose(
                    (0.8, 0.0, 0.0)
                ),
            ),
        )

        obs, _ = env.reset(return_obs=False)

        assert obs is None
        assert env._runtime_layout.topology == "split_articulations"
        tf_graph = env._get_tf()
        assert tf_graph.get_tf("world", "left_robot_base") is not None
        assert tf_graph.get_tf("world", "right_robot_base") is not None

    def test_reset_rejects_combined_articulation_with_split_bases(
        self, monkeypatch
    ):
        env = _make_reset_stub_env(
            monkeypatch,
            robot=SimpleNamespace(
                is_dual_arm=True,
                left_entity_origion_pose=_make_fake_sapien_pose(
                    (0.0, 0.0, 0.0)
                ),
                right_entity_origion_pose=_make_fake_sapien_pose(
                    (1.0, 0.0, 0.0)
                ),
            ),
        )

        with pytest.raises(RuntimeError, match="shared"):
            env.reset(return_obs=False)

    def test_reset_rebuilds_joint_to_eef_cache_before_first_obs(
        self, monkeypatch
    ):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            left_arm_joints_name=["left_joint_0", "left_joint_1"],
            right_arm_joints_name=["right_joint_0", "right_joint_1"],
            left_gripper_name={"base": "left_gripper"},
            right_gripper_name={"base": "right_gripper"},
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        close_calls: list[bool] = []

        env._task = SimpleNamespace(
            close_env=lambda clear_cache: close_calls.append(clear_cache),
            render_freq=0,
            robot=robot,
            info={},
        )

        class _OldJointsToEEF:
            def transform(
                self,
                left_arm_joints: torch.Tensor,
                right_arm_joints: torch.Tensor,
            ) -> RoboTwinEEF:
                del left_arm_joints, right_arm_joints
                return RoboTwinEEF(
                    left_eef=BatchFrameTransform(
                        xyz=torch.tensor([[10.0, 0.0, 0.0]]),
                        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                        parent_frame_id="world",
                        child_frame_id="left_eef",
                    ),
                    right_eef=BatchFrameTransform(
                        xyz=torch.tensor([[20.0, 0.0, 0.0]]),
                        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                        parent_frame_id="world",
                        child_frame_id="right_eef",
                    ),
                )

        class _NewJointsToEEF:
            def transform(
                self,
                left_arm_joints: torch.Tensor,
                right_arm_joints: torch.Tensor,
            ) -> RoboTwinEEF:
                del left_arm_joints, right_arm_joints
                return RoboTwinEEF(
                    left_eef=BatchFrameTransform(
                        xyz=torch.tensor([[30.0, 0.0, 0.0]]),
                        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                        parent_frame_id="world",
                        child_frame_id="left_eef",
                    ),
                    right_eef=BatchFrameTransform(
                        xyz=torch.tensor([[40.0, 0.0, 0.0]]),
                        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                        parent_frame_id="world",
                        child_frame_id="right_eef",
                    ),
                )

        env._joints_to_eef_transform = _OldJointsToEEF()

        raw_obs = {
            "joint_action": {
                "vector": np.array(
                    [1.0, 2.0, 0.5, 3.0, 4.0, 0.75],
                    dtype=np.float32,
                )
            },
            "observation": {},
        }
        new_task = SimpleNamespace(
            robot=robot,
            setup_demo=lambda **kwargs: None,
            get_obs=lambda: raw_obs,
            info={},
            close_env=lambda clear_cache: None,
            render_freq=0,
        )
        env._check_and_update_seed = lambda task_config: (
            new_task,
            [],
            task_config,
        )

        built = {"count": 0}

        def _build_joints_to_eef(*args) -> _NewJointsToEEF:
            del args
            built["count"] += 1
            return _NewJointsToEEF()

        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.env.build_joints_to_eef_transform",
            _build_joints_to_eef,
        )

        obs, _ = env.reset(seed=2)

        assert len(close_calls) == 1
        assert built["count"] == 1
        assert obs is not None
        tf_graph = obs["tf"]
        left_joint_tf = tf_graph.get_tf("world", _LEFT_EEF_FROM_JOINT_FRAME_ID)
        right_joint_tf = tf_graph.get_tf(
            "world", _RIGHT_EEF_FROM_JOINT_FRAME_ID
        )
        assert isinstance(left_joint_tf, BatchFrameTransform)
        assert isinstance(right_joint_tf, BatchFrameTransform)
        assert torch.equal(
            left_joint_tf.xyz,
            torch.tensor([[30.0, 0.0, 0.0]], dtype=torch.float32),
        )
        assert torch.equal(
            right_joint_tf.xyz,
            torch.tensor([[40.0, 0.0, 0.0]], dtype=torch.float32),
        )

    def test_reset_success_clears_finalized_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)

        obs, _ = env.reset(return_obs=False)

        assert obs is None
        assert env._episode_finalized is False
        assert env._last_obs_step_index == 0

    def test_get_obs_reuses_current_step_index_without_advancing(self) -> None:
        env = _make_uninitialized_env()
        env._last_obs_step_index = 3
        get_obs = MagicMock(return_value={"raw": True})
        env._task = SimpleNamespace(get_obs=get_obs)
        env._format_obs = lambda raw_obs, *, step_index: {
            **raw_obs,
            "step_index": step_index,
        }

        obs = env._get_obs()

        assert obs == {"raw": True, "step_index": 3}
        assert env._last_obs_step_index == 3
        get_obs.assert_called_once()

    def test_get_obs_requires_reset_step_clock(self) -> None:
        env = _make_uninitialized_env()
        env._last_obs_step_index = None
        get_obs = MagicMock(return_value={})
        env._task = SimpleNamespace(get_obs=get_obs)

        with pytest.raises(RuntimeError, match="successful reset"):
            env._get_obs()

        get_obs.assert_not_called()

    def test_close_invalidates_observation_cache_and_step_clock(self) -> None:
        env = _make_uninitialized_env()
        env._episode_finalized = False
        env._post_reset_state_available = True
        env._last_obs = {"stale": True}
        env._last_obs_step_index = 7
        env._video_writer = None
        env._joints_to_eef_transform = object()
        env._cached_obs_robots = {"stale": object()}
        get_obs = MagicMock(return_value={})
        close_env = MagicMock()
        env._task = SimpleNamespace(
            get_obs=get_obs,
            close_env=close_env,
            render_freq=0,
        )

        env.close(clear_cache=False)

        assert env._last_obs is None
        assert env._last_obs_step_index is None
        close_env.assert_called_once_with(clear_cache=False)
        with pytest.raises(RuntimeError, match="successful reset"):
            env._get_obs()
        with pytest.raises(RuntimeError, match="no latest observation"):
            env.get_mcap_obs()
        get_obs.assert_not_called()

    def test_close_fully_disposes_heterogeneous_workers_idempotently(
        self,
    ) -> None:
        events: list[str] = []

        class _FakeConn:
            def __init__(self, side: str) -> None:
                self.side = side
                self.closed = False

            def send(self, message: object) -> None:
                events.append(f"send:{self.side}:{message}")

            def close(self) -> None:
                events.append(f"close_conn:{self.side}")
                self.closed = True

        class _FakeProc:
            def __init__(self, side: str, *, exits_on_join: bool) -> None:
                self.side = side
                self.exits_on_join = exits_on_join
                self.alive = True

            def join(self, timeout: float | None = None) -> None:
                events.append(f"join:{self.side}:{timeout}")
                if self.exits_on_join:
                    self.alive = False

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                events.append(f"terminate:{self.side}")
                self.alive = False

            def close(self) -> None:
                events.append(f"close_proc:{self.side}")

        robot = SimpleNamespace(
            left_conn=_FakeConn("left"),
            right_conn=_FakeConn("right"),
            left_proc=_FakeProc("left", exits_on_join=False),
            right_proc=_FakeProc("right", exits_on_join=True),
        )
        viewer = SimpleNamespace(
            is_closed=False,
            close=lambda: events.append("close_viewer"),
        )
        task = SimpleNamespace(
            robot=robot,
            close_env=lambda clear_cache: events.append(
                f"close_env:{clear_cache}"
            ),
            viewer=viewer,
        )
        env = _make_uninitialized_env()
        env._task = task
        env._runtime_layout = object()
        env._active_task_config = {"seed": 1}
        env._joints_to_eef_transform = object()
        env._cached_obs_robots = {"left": object()}
        env._last_obs = {"stale": True}
        env._last_obs_step_index = 3
        env._episode_finalized = False
        env._post_reset_state_available = True
        env._video_writer = None

        env.close(clear_cache=False)
        env.close(clear_cache=False)

        assert events == [
            "close_env:False",
            "send:left:{'cmd': 'exit'}",
            "close_conn:left",
            "send:right:{'cmd': 'exit'}",
            "close_conn:right",
            "join:left:5.0",
            "terminate:left",
            "join:left:5.0",
            "close_proc:left",
            "join:right:5.0",
            "close_proc:right",
            "close_viewer",
        ]
        assert env._task is None
        assert env._runtime_layout is None
        assert env._active_task_config is None
        assert env._joints_to_eef_transform is None
        assert env._cached_obs_robots is None

    def test_close_retains_stubborn_worker_handle_for_retry(self) -> None:
        class _StubbornProc:
            def __init__(self) -> None:
                self.alive = True
                self.kill_calls = 0
                self.close_calls = 0

            def join(self, timeout: float | None = None) -> None:
                assert timeout == 5.0

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                self.kill_calls += 1
                if self.kill_calls >= 2:
                    self.alive = False

            def close(self) -> None:
                self.close_calls += 1

        proc = _StubbornProc()
        robot = SimpleNamespace(left_proc=proc, right_proc=None)
        task = SimpleNamespace(
            robot=robot,
            close_env=MagicMock(),
        )
        env = _make_uninitialized_env()
        env._task = task
        env._pending_disposal_tasks = []
        env._runtime_layout = object()
        env._active_task_config = {"seed": 1}
        env._joints_to_eef_transform = object()
        env._cached_obs_robots = {"left": object()}
        env._last_obs = {"stale": True}
        env._last_obs_step_index = 3
        env._episode_finalized = False
        env._post_reset_state_available = True
        env._video_writer = None

        env.close()

        assert env._task is None
        assert env._pending_disposal_tasks == [task]
        assert robot.left_proc is proc

        env.close()

        assert env._pending_disposal_tasks == []
        assert robot.left_proc is None
        assert proc.kill_calls == 2
        assert proc.close_calls == 1

    def test_expert_retry_returns_accepted_official_task_with_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_robotwin_instruction_generator(monkeypatch)
        failed_tasks = [SimpleNamespace(), SimpleNamespace()]
        accepted_task = SimpleNamespace()
        outcomes = [
            (failed_tasks[0], {"name": "failed-1"}, False),
            (failed_tasks[1], {"name": "failed-2"}, False),
            (accepted_task, {"name": "successful"}, True),
        ]
        env = _make_uninitialized_env()
        env.cfg = SimpleNamespace(
            check_expert=True,
            task_name="open_laptop",
            max_instruction_num=1,
            get_task_config_for_seed=lambda runtime_seed: {
                "seed": runtime_seed
            },
        )
        env._task = None
        env._resolved_start_seed = 1
        env._offset_seed = 0
        env._check_expert_traj = lambda task_config: outcomes.pop(0)
        env._dispose_task = MagicMock(return_value=True)

        task, _, accepted_config = env._check_and_update_seed({"seed": 1})

        assert task is accepted_task
        assert accepted_config == {"seed": 3}
        assert env.current_seed == 3
        assert env._dispose_task.call_args_list == [
            call(failed_tasks[0], clear_cache=True),
            call(failed_tasks[1], clear_cache=True),
        ]

    def test_reset_failure_keeps_finalized_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        env._episode_finalized = False
        env._last_obs = {"stale": True}
        env._last_obs_step_index = 7
        env._check_and_update_seed = lambda task_config: (_ for _ in ()).throw(
            RuntimeError("reset failed")
        )

        with pytest.raises(RuntimeError, match="reset failed"):
            env.reset(return_obs=False)

        assert env._episode_finalized is True
        assert env._last_obs is None
        assert env._last_obs_step_index is None

    def test_reset_disposes_staged_task_if_workspace_exit_is_cancelled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        staged_task = SimpleNamespace(
            robot=robot,
            close_env=MagicMock(),
        )
        env._check_and_update_seed = lambda task_config: (
            staged_task,
            [],
            task_config,
        )
        monkeypatch.setattr(
            robotwin_env,
            "in_robotwin_workspace",
            _RaiseOnExit,
        )

        with pytest.raises(KeyboardInterrupt):
            env.reset(return_obs=False)

        staged_task.close_env.assert_called_once_with(clear_cache=True)
        assert env._task is None
        assert env._episode_finalized is True

    def test_reset_reuses_official_precheck_task_for_init_check(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_robotwin_instruction_generator(monkeypatch)
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        env.cfg.check_expert = False
        env.cfg.check_task_init = True
        env.cfg.max_instruction_num = 1
        env.cfg.task_name = "open_laptop"
        config_calls: list[int] = []
        setup_calls: list[dict[str, object]] = []
        env.cfg.get_task_config_for_seed = lambda runtime_seed: (
            config_calls.append(runtime_seed) or {"seed": runtime_seed}
        )
        task = SimpleNamespace(
            robot=robot,
            setup_demo=lambda **kwargs: setup_calls.append(kwargs),
            play_once=MagicMock(),
            plan_success=False,
            close_env=MagicMock(),
            get_obs=lambda: {},
            info={},
        )
        task.play_once.side_effect = lambda: (
            setattr(task, "arm_tag", "left") or {"info": {}}
        )
        create_task = MagicMock(return_value=task)
        monkeypatch.setattr(
            robotwin_env,
            "create_task_from_name",
            create_task,
        )
        env._check_and_update_seed = (
            RoboTwinEnv._check_and_update_seed.__get__(env, RoboTwinEnv)
        )

        env.reset(return_obs=False)

        assert config_calls == [1]
        assert setup_calls == [
            {"seed": 1, "render_freq": 0},
            {"seed": 1},
        ]
        task.close_env.assert_called_once_with(clear_cache=False)
        create_task.assert_called_once_with("open_laptop")
        assert env._task is task
        assert env._task.robot is robot
        assert env._task.arm_tag == "left"

    def test_reset_first_observation_failure_fully_disposes_new_task(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        close_calls: list[bool] = []
        task = SimpleNamespace(
            robot=robot,
            setup_demo=lambda **kwargs: None,
            get_obs=MagicMock(side_effect=RuntimeError("first obs failed")),
            close_env=lambda clear_cache: close_calls.append(clear_cache),
            info={},
        )
        env._check_and_update_seed = lambda task_config: (
            task,
            [],
            task_config,
        )

        with pytest.raises(RuntimeError, match="first obs failed"):
            env.reset()

        assert close_calls == [True]
        assert env._task is None
        assert env._runtime_layout is None
        assert env._active_task_config is None
        assert env._episode_finalized is True

    def test_reset_updates_episode_id_and_builds_video_path(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        setup_calls: list[dict[str, object]] = []
        task = SimpleNamespace(
            robot=robot,
            setup_demo=lambda **kwargs: setup_calls.append(kwargs),
            get_obs=lambda: {},
            info={},
        )
        env._check_and_update_seed = lambda task_config: (
            task,
            [],
            task_config,
        )

        def _get_task_config_for_seed(cfg, runtime_seed):
            return {
                "now_ep_num": cfg.episode_id,
                "seed": runtime_seed,
            }

        env.cfg.get_task_config_for_seed = MethodType(
            _get_task_config_for_seed, env.cfg
        )

        recorded: dict[str, object] = {}
        monkeypatch.setattr(
            robotwin_env,
            "_extract_video_frame",
            lambda raw_obs: np.zeros((16, 16, 3), dtype=np.uint8),
        )

        class FakeWriter:
            def __init__(self, **kwargs):
                recorded["writer_kwargs"] = kwargs
                recorded["is_open"] = False

            def open(self, output_path):
                recorded["video_path"] = output_path
                recorded["is_open"] = True

            def write_frame(self, frame):
                recorded["frame_shape"] = tuple(frame.shape)

            def close(self):
                recorded["closed"] = True
                recorded["is_open"] = False

            @property
            def is_closed(self):
                return not recorded["is_open"]

        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.env.VideoWriter",
            FakeWriter,
        )

        obs, _ = env.reset(
            return_obs=False,
            video_dir="/tmp/task/demo_clean",
            episode_id=7,
        )

        assert obs is None
        assert env.cfg.episode_id == 7
        assert setup_calls[-1]["now_ep_num"] == 7
        assert recorded["video_path"] == (
            "/tmp/task/demo_clean/episode_7_seed_1.mp4"
        )
        assert recorded["writer_kwargs"] == {
            "pixel_format": ROBOTWIN_VIDEO_PIXEL_FORMAT,
            "fps": ROBOTWIN_VIDEO_FPS,
        }
        assert recorded["frame_shape"] == (16, 16, 3)

    def test_reset_tracks_start_seed_and_offset_seed(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        task = SimpleNamespace(
            robot=robot,
            setup_demo=lambda **kwargs: None,
            get_obs=lambda: {},
            close_env=lambda clear_cache: None,
            render_freq=0,
            info={},
        )
        env._check_and_update_seed = lambda task_config: (
            task,
            [],
            task_config,
        )

        obs, info = env.reset(seed=2, offset_seed=3)

        assert obs is not None
        assert env.cfg.seed == 2
        assert env.start_seed == 2
        assert env.offset_seed == 3
        assert env.resolved_start_seed == 2
        assert env.current_seed == 5
        assert info["seed"] == 5
        assert info["start_seed"] == 2
        assert info["resolved_start_seed"] == 2
        assert info["offset_seed"] == 3

    def test_get_state_captures_post_reset_recreate_payload(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, _, _ = _make_state_stub_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            RoboTwinEnvCfg,
            "get_task_config_for_seed",
            MagicMock(
                side_effect=AssertionError(
                    "State capture must use active config"
                )
            ),
        )

        state = env.get_state()

        assert state.class_type is RoboTwinEnv
        assert isinstance(state.config, RoboTwinEnvCfg)
        assert state.config is not env.cfg
        assert state.config.episode_id == 5
        assert state.state["schema_version"] == (
            ROBOTWIN_ENV_STATE_SCHEMA_VERSION
        )
        assert state.state[ENV_STATE_SCOPE_KEY] == (
            EnvStateScope.POST_RESET.value
        )
        assert state.state["offset_seed"] == 2
        assert state.state["task_config"] == {
            "seed": 3,
            "now_ep_num": 5,
            "task_name": "robotwin_dummy_task",
        }
        assert "post_reset_state_available" not in state.state
        assert state.state["episode_finalized"] is False

    def test_state_restore_rejects_pending_worker_before_candidate_creation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        env, old_close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        state = env.get_state()
        env._pending_disposal_tasks = [SimpleNamespace()]
        env._drain_pending_disposals = MagicMock()

        with pytest.raises(
            RuntimeError,
            match="earlier failed candidate is still alive",
        ):
            env.load_state(state)

        env._drain_pending_disposals.assert_called_once_with(clear_cache=True)
        assert created_tasks == []
        assert old_close_calls == []
        assert env._task is not None

    def test_state_restore_disposes_candidate_if_workspace_exit_is_cancelled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        env, old_close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        old_task = env._require_task()
        state = env.get_state()
        monkeypatch.setattr(
            robotwin_env,
            "in_robotwin_workspace",
            _RaiseOnExit,
        )

        with pytest.raises(KeyboardInterrupt):
            env.load_state(state)

        assert len(created_tasks) == 1
        assert created_tasks[0].close_calls == [True]
        assert old_close_calls == []
        assert env._require_task() is old_task

    @pytest.mark.parametrize(
        "restore_method",
        ["load_state", "reset_from_state"],
    )
    def test_checked_state_restore_replays_official_precheck_on_same_task(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        restore_method: str,
    ) -> None:
        env, old_close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        env.cfg.check_task_init = True
        state = env.get_state()
        monkeypatch.setattr(
            RoboTwinEnvCfg,
            "get_task_config_for_seed",
            MagicMock(
                side_effect=AssertionError(
                    "State restore must use the saved exact task config"
                )
            ),
        )

        result = getattr(env, restore_method)(state)

        assert len(created_tasks) == 1
        restored_task = created_tasks[0]
        assert env._require_task() is restored_task
        assert restored_task.play_once_calls == 1
        assert restored_task.arm_tag == "left"
        assert restored_task.setup_calls == [
            {
                "seed": 3,
                "now_ep_num": 5,
                "task_name": "robotwin_dummy_task",
                "render_freq": 0,
            },
            {
                "seed": 3,
                "now_ep_num": 5,
                "task_name": "robotwin_dummy_task",
            },
        ]
        assert restored_task.close_calls == [False]
        assert old_close_calls == [True]
        if restore_method == "load_state":
            assert result is None
        else:
            obs, info = result
            assert obs["step_index"] == 0
            assert info["seed"] == 3

    def test_get_state_captures_episode_finalized_flag(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, _, _ = _make_state_stub_env(monkeypatch, tmp_path)
        env._episode_finalized = True

        state = env.get_state()

        assert state.state["episode_finalized"] is True

    def test_state_persistence_keeps_snapshot_after_source_deleted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        env, _, _ = _make_state_stub_env(monkeypatch, tmp_path)
        state = env.get_state()
        expected_digest = state.config._task_config_content_sha256
        save_path = tmp_path / "saved_state"

        state.save(str(save_path))
        Path(env.cfg.task_config_path).unlink()
        restored = State.load(str(save_path), protocol="cloudpickle")

        assert isinstance(restored.config, RoboTwinEnvCfg)
        assert restored.config._task_config_content_sha256 == expected_digest
        assert restored.state == state.state

    def test_get_state_rejects_before_reset_or_after_step(self):
        env, _ = _make_step_stub_env(action_type="qpos")

        with pytest.raises(RuntimeError, match="after reset"):
            env.get_state()

        env._post_reset_state_available = True
        env.step([0.0] * 14)

        with pytest.raises(RuntimeError, match="after reset"):
            env.get_state()

    def test_load_state_rejects_bad_payload_before_closing_current_task(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        state = env.get_state()
        bad_state = state.model_copy(deep=True)
        bad_state.state["offset_seed"] = -1

        with pytest.raises(ValidationError, match="offset_seed"):
            env.load_state(bad_state)

        assert close_calls == []
        assert created_tasks == []
        assert env._post_reset_state_available is True

    @pytest.mark.parametrize(
        ("task_config_key", "value"),
        [
            ("task_name", "other_robotwin_task"),
            ("seed", 4),
            ("now_ep_num", 6),
        ],
    )
    def test_load_state_rejects_task_config_mismatch_before_closing_task(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        task_config_key: str,
        value: int | str,
    ) -> None:
        env, close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        bad_state = env.get_state().model_copy(deep=True)
        bad_state.state["task_config"][task_config_key] = value

        with pytest.raises(
            ValueError,
            match=f"task_config {task_config_key!r} must match State.config",
        ):
            env.load_state(bad_state)

        assert close_calls == []
        assert created_tasks == []
        assert env._task is not None
        assert env._post_reset_state_available is True

    def test_load_state_rejects_unsupported_state_scope_before_closing_task(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        state = env.get_state()
        bad_state = state.model_copy(deep=True)
        bad_state.state[ENV_STATE_SCOPE_KEY] = EnvStateScope.MID_EPISODE.value

        with pytest.raises(ValidationError, match="POST_RESET"):
            env.load_state(bad_state)

        assert close_calls == []
        assert created_tasks == []

    def test_load_state_rejects_mismatched_class_type_before_closing_task(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        state = env.get_state()
        bad_state = state.model_copy(deep=True)
        bad_state.class_type = object

        with pytest.raises(TypeError, match="class_type"):
            env.load_state(bad_state)

        assert close_calls == []
        assert created_tasks == []

    def test_load_state_rejects_bad_recreated_task_before_closing_task(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, close_calls, _ = _make_state_stub_env(monkeypatch, tmp_path)
        state = env.get_state()
        old_task = env._task
        staged_close_calls: list[bool] = []
        bad_robot = SimpleNamespace(
            is_dual_arm=False,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        staged_task = _StateFakeTask(
            robot=bad_robot,
            raw_obs={"restored": "robotwin_dummy_task"},
            close_calls=staged_close_calls,
        )
        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.env.create_task_from_name",
            lambda task_name: staged_task,
        )

        with pytest.raises(RuntimeError, match="arm joints"):
            env.load_state(state)

        assert close_calls == []
        assert staged_close_calls == [True]
        assert env._task is old_task
        assert env._post_reset_state_available is True

    def test_load_state_disposes_candidate_when_old_worker_stays_pending(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        env, _, created_tasks = _make_state_stub_env(monkeypatch, tmp_path)
        state = env.get_state()
        pending_task = SimpleNamespace()
        disposed_tasks: list[object] = []
        original_dispose = env._dispose_task

        def _dispose_task(task, *, clear_cache):
            disposed_tasks.append(task)
            return original_dispose(task, clear_cache=clear_cache)

        def _close_with_pending_worker(clear_cache=True):
            del clear_cache
            env._pending_disposal_tasks = [pending_task]
            env._clear_active_task_runtime()

        monkeypatch.setattr(env, "_dispose_task", _dispose_task)
        monkeypatch.setattr(env, "close", _close_with_pending_worker)

        with pytest.raises(RuntimeError, match="cannot restore State"):
            env.load_state(state)

        assert len(created_tasks) == 1
        assert disposed_tasks == [created_tasks[0]]
        assert env._task is None
        assert env._pending_disposal_tasks == [pending_task]

    @pytest.mark.parametrize(
        ("last_obs", "last_obs_step_index"),
        [
            (None, None),
            ({"stale": True}, 7),
        ],
    )
    def test_load_state_resets_observation_cache_and_step_clock(
        self,
        monkeypatch,
        tmp_path,
        last_obs,
        last_obs_step_index,
    ):
        env, _, _ = _make_state_stub_env(monkeypatch, tmp_path)
        state = env.get_state()
        env._last_obs = last_obs
        env._last_obs_step_index = last_obs_step_index

        env.load_state(state)

        assert env._last_obs is None
        assert env._last_obs_step_index == 0
        current_obs = env._get_obs()
        assert current_obs["step_index"] == 0
        assert current_obs["step_timestamp"] == 0.0

        take_action = MagicMock()
        env._task.robot.get_left_arm_jointState = lambda: [0.0] * 7
        env._task.robot.get_right_arm_jointState = lambda: [0.0] * 7
        env._task.take_action = take_action
        env._task.step_lim = None
        env._task.take_action_cnt = 0
        env._task.eval_success = False
        env._write_video_frame = lambda raw_obs: None
        env._get_info = lambda: {}

        step_result = env.step([0.0] * 14)

        assert step_result.observations is not None
        assert step_result.observations["step_index"] == 1
        assert step_result.observations["step_timestamp"] == 0.1
        take_action.assert_called_once()

    def test_reset_from_state_restores_post_reset_state_and_returns_obs(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        state = env.get_state()
        env._offset_seed = 0
        env._post_reset_state_available = False
        env._episode_finalized = True

        obs, info = env.reset_from_state(state)

        assert close_calls == [True]
        assert len(created_tasks) == 1
        assert created_tasks[0].setup_calls == [
            {
                "seed": 3,
                "now_ep_num": 5,
                "task_name": "robotwin_dummy_task",
            }
        ]
        assert env.offset_seed == 2
        assert env.current_seed == 3
        assert env._post_reset_state_available is True
        assert env._episode_finalized is False
        assert obs == {
            "formatted": {"restored": "robotwin_dummy_task"},
            "step_index": 0,
            "step_timestamp": 0.0,
        }
        assert info["seed"] == 3
        assert info["offset_seed"] == 2
        assert info["source"] == "restored"

    def test_reset_from_state_restores_split_runtime_layout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        env, _, _ = _make_state_stub_env(monkeypatch, tmp_path)
        robot = env._task.robot
        robot.is_dual_arm = False
        robot.left_entity = object()
        robot.right_entity = object()
        robot.left_entity_origion_pose = _make_fake_sapien_pose(
            (-0.4, 0.0, 0.0)
        )
        robot.right_entity_origion_pose = _make_fake_sapien_pose(
            (0.4, 0.0, 0.0)
        )
        env._runtime_layout = derive_runtime_layout(env._require_task())
        env._active_task_config["embodiment"] = [
            "left_arm",
            "right_arm",
            0.8,
        ]
        state = env.get_state()

        env.reset_from_state(state)

        assert env._runtime_layout.topology == "split_articulations"
        assert env._active_task_config["embodiment"] == [
            "left_arm",
            "right_arm",
            0.8,
        ]

    def test_reset_from_state_uses_state_cfg_for_curobo_patch_guard(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, _, created_tasks = _make_state_stub_env(monkeypatch, tmp_path)
        env._instructions = None
        state = env.get_state()
        env.cfg.patch_curobo_base_transform = False
        state.config.action_type = "ee"
        state.config.patch_curobo_base_transform = True
        prepare_values: list[bool] = []
        setup_calls: list[tuple[bool, object]] = []
        monkeypatch.setattr(
            robotwin_env,
            "prepare_robotwin_runtime_for_cfg",
            lambda cfg: prepare_values.append(cfg.patch_curobo_base_transform),
            raising=False,
        )
        monkeypatch.setattr(
            robotwin_env,
            "setup_robotwin_demo_with_runtime_guards",
            lambda cfg, task, task_config: setup_calls.append(
                (cfg.patch_curobo_base_transform, task)
            ),
            raising=False,
        )

        env.reset_from_state(state)

        assert prepare_values == [True]
        assert setup_calls == [(True, created_tasks[0])]

    def test_reset_from_state_ignores_legacy_availability_and_enables_capture(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, _, _ = _make_state_stub_env(monkeypatch, tmp_path)
        state = env.get_state()
        state.state["post_reset_state_available"] = False
        state.state["episode_finalized"] = True
        env._post_reset_state_available = False
        env._episode_finalized = False

        env.reset_from_state(state)

        assert env._post_reset_state_available is True
        assert env._episode_finalized is True

    def test_reset_from_state_format_failure_leaves_episode_inactive(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, close_calls, created_tasks = _make_state_stub_env(
            monkeypatch,
            tmp_path,
        )
        state = env.get_state()

        def _raise_format_error(raw_obs, *, step_index):
            del raw_obs, step_index
            raise RuntimeError("format failed")

        monkeypatch.setattr(
            env,
            "_format_obs",
            _raise_format_error,
            raising=False,
        )

        with pytest.raises(RuntimeError, match="format failed"):
            env.reset_from_state(state)

        assert close_calls == [True]
        assert len(created_tasks) == 1
        assert env._task is None
        assert env._active_task_config is None
        assert env._runtime_layout is None
        assert env._post_reset_state_available is False
        assert env._episode_finalized is True
        with pytest.raises(RuntimeError, match="no active episode"):
            env.step([0.0] * 14)
        with pytest.raises(RuntimeError, match="after reset"):
            env.get_state()

    def test_reset_resets_offset_when_start_seed_changes(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
        task = SimpleNamespace(
            robot=robot,
            setup_demo=lambda **kwargs: None,
            get_obs=lambda: {},
            close_env=lambda clear_cache: None,
            info={},
        )
        env._check_and_update_seed = lambda task_config: (
            task,
            [],
            task_config,
        )
        env._offset_seed = 4

        _, info = env.reset(seed=2)

        assert env.start_seed == 2
        assert env.offset_seed == 0
        assert env.current_seed == 2
        assert info["offset_seed"] == 0

    def test_reset_rejects_string_seed(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)

        with pytest.raises(TypeError, match="seed must be an int or None"):
            env.reset(seed="next")

    def test_reset_rejects_string_offset_seed(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)

        with pytest.raises(
            TypeError,
            match="offset_seed must be an int or None",
        ):
            env.reset(offset_seed="next")

    def test_joints_to_eef_aligns_joint_and_base_tf_dtype(self, monkeypatch):
        fake_chain = SimpleNamespace(
            dtype=torch.float32,
            device=torch.device("cpu"),
            frame_names=["robot_base"],
        )

        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.kinematics.KinematicChain.from_content",
            lambda data, format: fake_chain,
        )
        left_chain = _FakeSerialChain(
            dtype=torch.float32,
            device=torch.device("cpu"),
            end_frame_name="fl_link6",
        )
        right_chain = _FakeSerialChain(
            dtype=torch.float32,
            device=torch.device("cpu"),
            end_frame_name="fr_link6",
        )
        created_chains = [left_chain, right_chain]
        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.kinematics.KinematicSerialChain",
            lambda chain, end_frame_name: created_chains.pop(0),
        )

        joints_to_eef = RoboTwinJointsToEEF(urdf_content="<robot/>")

        assert joints_to_eef._left_robot_base_tf.xyz.dtype == torch.float32
        assert joints_to_eef._right_robot_base_tf.quat.dtype == torch.float32

        ret = joints_to_eef.transform(
            left_arm_joints=torch.zeros(2, 6, dtype=torch.float64),
            right_arm_joints=torch.zeros(2, 6, dtype=torch.float64),
            robot_base_tf=BatchFrameTransform(
                xyz=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
                quat=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float64,
                ),
                parent_frame_id="world",
                child_frame_id="robot_base",
            ),
        )

        assert left_chain.recorded_joint_dtypes == [torch.float32]
        assert right_chain.recorded_joint_dtypes == [torch.float32]
        torch.testing.assert_close(
            ret.left_eef.xyz,
            torch.tensor(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                dtype=torch.float32,
            ),
        )
        assert ret.left_eef.xyz.dtype == torch.float32

    def test_joints_to_eef_supports_separate_urdfs_and_bases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parsed_contents: list[str] = []
        chain_models = [
            SimpleNamespace(
                dtype=torch.float32,
                device=torch.device("cpu"),
                frame_names=["left_base"],
            ),
            SimpleNamespace(
                dtype=torch.float32,
                device=torch.device("cpu"),
                frame_names=["right_base"],
            ),
        ]

        def _parse_chain(data: str, format: str) -> SimpleNamespace:
            assert format == "urdf"
            parsed_contents.append(data)
            return chain_models.pop(0)

        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.kinematics.KinematicChain.from_content",
            _parse_chain,
        )
        serial_chains = [
            _FakeSerialChain(
                dtype=torch.float32,
                device=torch.device("cpu"),
                end_frame_name="left_eef",
                root_frame_name="left_base",
            ),
            _FakeSerialChain(
                dtype=torch.float32,
                device=torch.device("cpu"),
                end_frame_name="right_eef",
                root_frame_name="right_base",
            ),
        ]
        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.kinematics.KinematicSerialChain",
            lambda chain, end_frame_name: serial_chains.pop(0),
        )

        joints_to_eef = RoboTwinJointsToEEF(
            urdf_content="<robot name='left'/>",
            right_urdf_content="<robot name='right'/>",
            left_eef_name="left_eef",
            right_eef_name="right_eef",
            robot_base_xyz=(-0.4, 0.0, 0.0),
            robot_base_quat=(1.0, 0.0, 0.0, 0.0),
            right_robot_base_xyz=(0.4, 0.0, 0.0),
            right_robot_base_quat=(1.0, 0.0, 0.0, 0.0),
        )

        ret = joints_to_eef.transform(
            left_arm_joints=torch.zeros(1, 6),
            right_arm_joints=torch.zeros(1, 7),
        )

        assert parsed_contents == [
            "<robot name='left'/>",
            "<robot name='right'/>",
        ]
        torch.testing.assert_close(
            ret.left_eef.xyz,
            torch.tensor([[-0.4, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            ret.right_eef.xyz,
            torch.tensor([[0.4, 0.0, 0.0]]),
        )

    def test_reset_reuses_cfg_episode_id_for_video_dir(self, monkeypatch):
        env = _make_reset_stub_env(
            monkeypatch,
            robot=SimpleNamespace(
                is_dual_arm=True,
                left_entity_origion_pose=_make_fake_sapien_pose(
                    (0.0, 0.0, 0.0)
                ),
                right_entity_origion_pose=_make_fake_sapien_pose(
                    (0.0, 0.0, 0.0)
                ),
            ),
        )
        env.cfg.episode_id = 3

        recorded: dict[str, object] = {}
        monkeypatch.setattr(
            robotwin_env,
            "_extract_video_frame",
            lambda raw_obs: np.zeros((16, 16, 3), dtype=np.uint8),
        )

        class FakeWriter:
            def __init__(self, **kwargs):
                recorded["writer_kwargs"] = kwargs
                recorded["is_open"] = False

            def open(self, output_path):
                recorded["video_path"] = output_path
                recorded["is_open"] = True

            def write_frame(self, frame):
                recorded["frame_shape"] = tuple(frame.shape)

            def close(self):
                recorded["closed"] = True
                recorded["is_open"] = False

            @property
            def is_closed(self):
                return not recorded["is_open"]

        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.env.VideoWriter",
            FakeWriter,
        )

        obs, _ = env.reset(
            return_obs=False,
            video_dir="/tmp/task/demo_clean",
        )

        assert obs is None
        assert env.cfg.episode_id == 3
        assert recorded["video_path"] == (
            "/tmp/task/demo_clean/episode_3_seed_1.mp4"
        )
        assert recorded["writer_kwargs"] == {
            "pixel_format": ROBOTWIN_VIDEO_PIXEL_FORMAT,
            "fps": ROBOTWIN_VIDEO_FPS,
        }
        assert recorded["frame_shape"] == (16, 16, 3)

    def test_video_recording_lifecycle(self, tmp_path):
        _get_ffmpeg_binary(require_libx264=True)
        env = _make_uninitialized_env()
        env._video_writer = VideoWriter(
            pixel_format=ROBOTWIN_VIDEO_PIXEL_FORMAT,
            fps=ROBOTWIN_VIDEO_FPS,
        )
        video_path = tmp_path / "episode.mp4"
        env._video_writer.open(video_path)

        raw_obs = {
            "observation": {
                "head_camera": {
                    "rgb": np.full((16, 16, 3), [0, 255, 0], dtype=np.uint8),
                }
            }
        }

        env._write_video_frame(raw_obs)
        env._write_video_frame(raw_obs)
        env._stop_video_recording()

        assert video_path.exists()
        assert video_path.stat().st_size > 0
        assert env._video_writer is not None
        assert env._video_writer.is_closed

        decoded = _decode_first_frame_rgb(
            str(video_path),
            width=16,
            height=16,
        )
        assert decoded[..., 1].mean() > 200
        assert decoded[..., 0].mean() < 40
        assert decoded[..., 2].mean() < 40

    def test_finalize_episode_stops_video_without_closing_task(self) -> None:
        env = _make_uninitialized_env()
        env._episode_finalized = False
        stop_calls: list[str] = []
        close_calls: list[bool] = []
        env._stop_video_recording = lambda: stop_calls.append("stop")
        env._task = SimpleNamespace(
            close_env=lambda clear_cache: close_calls.append(clear_cache),
            render_freq=0,
        )

        env.finalize_episode()

        assert env._episode_finalized is True
        assert stop_calls == ["stop"]
        assert close_calls == []

    def test_finalize_episode_marks_finalized_before_video_cleanup_failure(
        self,
    ) -> None:
        env = _make_uninitialized_env()
        env._episode_finalized = False

        def _fail_stop() -> None:
            raise RuntimeError("stop failed")

        env._stop_video_recording = _fail_stop

        with pytest.raises(RuntimeError, match="stop failed"):
            env.finalize_episode()

        assert env._episode_finalized is True
