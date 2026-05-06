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

import shutil
import subprocess
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from pydantic import ValidationError

import robo_orchard_lab.envs.robotwin.env as robotwin_env
from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchFrameTransformGraph,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Robot,
    RobotDescriptionFormat,
)
from robo_orchard_lab.envs.robotwin.env import (
    LEFT_EEF_FROM_JOINT_FRAME_ID,
    RIGHT_EEF_FROM_JOINT_FRAME_ID,
    ROBOTWIN_ENV_STATE_SCHEMA_VERSION,
    ROBOTWIN_VIDEO_FPS,
    ROBOTWIN_VIDEO_PIXEL_FORMAT,
    RoboTwinEnv,
    RoboTwinEnvCfg,
)
from robo_orchard_lab.envs.robotwin.kinematics import (
    RoboTwinEEF,
    RoboTwinJointsToEEF,
)
from robo_orchard_lab.envs.state import ENV_STATE_SCOPE_KEY, EnvStateScope
from robo_orchard_lab.utils.video import VideoWriter

pytestmark = pytest.mark.sim_env


def _install_fake_robotwin_instruction_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def _make_reset_stub_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    robot: SimpleNamespace,
) -> RoboTwinEnv:
    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.cfg = SimpleNamespace(
        seed=1,
        task_name="robotwin_dummy_task",
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
    env._video_writer = None
    env._check_and_update_seed = lambda: (
        SimpleNamespace(
            robot=robot,
            setup_demo=lambda **kwargs: None,
            get_obs=lambda: {},
            info={},
        ),
        [],
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
    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.cfg = SimpleNamespace(action_type=action_type)
    env._episode_finalized = False
    env._post_reset_state_available = False
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
    env._format_obs = lambda raw_obs: raw_obs
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
    ) -> None:
        self.robot = robot
        self.raw_obs = raw_obs
        self.info = info or {}
        self.close_calls = close_calls
        self.setup_calls: list[dict[str, object]] = []
        self.render_freq = 0

    def setup_demo(self, **kwargs) -> None:
        self.setup_calls.append(kwargs)

    def get_obs(self) -> dict[str, object]:
        return self.raw_obs

    def close_env(self, clear_cache: bool) -> None:
        if self.close_calls is not None:
            self.close_calls.append(clear_cache)


def _make_state_stub_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> tuple[RoboTwinEnv, list[bool], list[_StateFakeTask]]:
    robot = SimpleNamespace(
        is_dual_arm=True,
        left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
    )
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
    monkeypatch.setattr(
        env,
        "_format_obs",
        lambda raw_obs: {"formatted": raw_obs},
        raising=False,
    )

    created_tasks: list[_StateFakeTask] = []

    def _create_task_from_name(task_name: str) -> _StateFakeTask:
        task = _StateFakeTask(
            robot=robot,
            raw_obs={"restored": task_name},
            info={"source": "restored"},
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
    ) -> None:
        self.dtype = dtype
        self.device = device
        self._end_frame_name = end_frame_name
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
                parent_frame_id="robot_base",
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
    def test_logger_manager_does_not_duplicate_when_it_owns_handlers(self):
        manager_logger = robotwin_env._logger_manager.get_logger()

        if manager_logger.handlers:
            assert manager_logger.propagate is False

    def test_check_expert_logs_retry_summary_without_per_seed_warning(
        self,
        monkeypatch,
    ):
        _install_fake_robotwin_instruction_generator(monkeypatch)
        env = RoboTwinEnv.__new__(RoboTwinEnv)
        env.cfg = SimpleNamespace(
            check_expert=True,
            task_name="robotwin_dummy_task",
            max_instruction_num=1,
        )
        env._resolved_start_seed = 100
        env._offset_seed = 0
        task = SimpleNamespace(info={"info": {"seed": 101}})
        outcomes = [(None, False), (task, True)]
        env._check_expert_traj = lambda: outcomes.pop(0)
        logger = SimpleNamespace(
            debug=MagicMock(),
            info=MagicMock(),
            warning=MagicMock(),
            error=MagicMock(),
        )
        monkeypatch.setattr(robotwin_env, "logger", logger)

        ret_task, instructions = env._check_and_update_seed()

        assert ret_task is task
        assert instructions == {"seen": ["robotwin_dummy_task"]}
        assert env.current_seed == 101
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

    def test_check_expert_traj_setup_failure_logs_concise_debug(
        self,
        monkeypatch,
    ):
        env = RoboTwinEnv.__new__(RoboTwinEnv)
        env.cfg = SimpleNamespace(
            task_name="robotwin_dummy_task",
            get_task_config_for_seed=lambda runtime_seed: {
                "seed": runtime_seed,
                "large": "cfg",
            },
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

        ret_task, success = env._check_expert_traj()

        assert ret_task is task
        assert success is False
        logger.debug.assert_called_once_with(
            "RoboTwin expert trajectory check failed while playing "
            "task config: task=%s seed=%s error=%s",
            "robotwin_dummy_task",
            102,
            failure,
        )
        logger.error.assert_not_called()

    def test_joints2ee_pose_uses_arm_joints_only(self, monkeypatch):
        env = RoboTwinEnv.__new__(RoboTwinEnv)
        env._task = SimpleNamespace(
            robot=SimpleNamespace(
                left_arm_joints_name=["left_joint_0", "left_joint_1"],
                right_arm_joints_name=["right_joint_0", "right_joint_1"],
            )
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

    def test_tf_in_obs(self, dummy_env_without_expert_check: RoboTwinEnv):
        env = dummy_env_without_expert_check
        obs, info = env.reset()
        assert obs is not None
        assert "tf" in obs
        assert "robots" in obs

    def test_get_obs_robots_caches_robot_metadata(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)
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
            }
        )

        tf_graph = obs["tf"]
        left_endpose_frame_id, right_endpose_frame_id = (
            env._get_endpose_frame_ids()
        )
        assert tf_graph.get_tf(
            "world", LEFT_EEF_FROM_JOINT_FRAME_ID
        ) == BatchFrameTransform(
            xyz=torch.tensor([[10.0, 1.0, 2.0]]),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            parent_frame_id="world",
            child_frame_id=LEFT_EEF_FROM_JOINT_FRAME_ID,
        )
        assert tf_graph.get_tf(
            "world", RIGHT_EEF_FROM_JOINT_FRAME_ID
        ) == BatchFrameTransform(
            xyz=torch.tensor([[20.0, 3.0, 4.0]]),
            quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            parent_frame_id="world",
            child_frame_id=RIGHT_EEF_FROM_JOINT_FRAME_ID,
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
            left_endpose_frame_id, right_endpose_frame_id = (
                env._get_endpose_frame_ids()
            )
            left_joint_tf = tf_graph.get_tf(
                "world", LEFT_EEF_FROM_JOINT_FRAME_ID
            )
            right_joint_tf = tf_graph.get_tf(
                "world", RIGHT_EEF_FROM_JOINT_FRAME_ID
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

    def test_step(self, dummy_env_without_expert_check: RoboTwinEnv):
        # Note that not all env can step because of robotwin BUG!
        env = dummy_env_without_expert_check
        obs, info = env.reset()
        assert obs is not None

        action = [1.0] * 14
        step_return = env.step(action)
        assert step_return.observations is not None
        assert "tf" in step_return.observations

    def test_step_rejects_qpos_action_width_mismatch(self):
        env, take_action = _make_step_stub_env(action_type="qpos")

        with pytest.raises(
            ValueError,
            match="expected 14, got 16",
        ):
            env.step([0.0] * 16)

        take_action.assert_not_called()

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
        env = RoboTwinEnv.__new__(RoboTwinEnv)
        env._episode_finalized = True

        with pytest.raises(RuntimeError) as exc_info:
            env.step([0.0] * 14)

        error_message = str(exc_info.value)
        assert "no active episode" in error_message
        assert "reset()" in error_message
        assert "finalized" not in error_message

    def test_get_urdf(self, dummy_env_without_expert_check: RoboTwinEnv):
        env = dummy_env_without_expert_check
        env.reset()
        urdf_dict = env.get_robot_urdf()
        assert urdf_dict is not None
        assert "left" in urdf_dict

    def test_reset_rejects_separate_arm_layout(self, monkeypatch):
        env = _make_reset_stub_env(
            monkeypatch,
            robot=SimpleNamespace(
                is_dual_arm=False,
                left_entity_origion_pose=_make_fake_sapien_pose(
                    (0.0, 0.0, 0.0)
                ),
                right_entity_origion_pose=_make_fake_sapien_pose(
                    (0.0, 0.0, 0.0)
                ),
            ),
        )

        with pytest.raises(
            NotImplementedError,
            match="combined dual-arm robot layout",
        ):
            env.reset(return_obs=False)

    def test_reset_rejects_split_robot_bases(self, monkeypatch):
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

        with pytest.raises(
            NotImplementedError,
            match="shared robot base",
        ):
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
        env._check_and_update_seed = lambda: (new_task, [])

        built = {"count": 0}

        def _build_joints_to_eef(**kwargs) -> _NewJointsToEEF:
            del kwargs
            built["count"] += 1
            return _NewJointsToEEF()

        monkeypatch.setattr(
            "robo_orchard_lab.envs.robotwin.env.RoboTwinJointsToEEF",
            _build_joints_to_eef,
        )
        monkeypatch.setattr(
            env,
            "get_robot_urdf",
            lambda: {"left": b"<robot/>"},
            raising=False,
        )
        monkeypatch.setattr(
            env,
            "_get_tf",
            lambda: BatchFrameTransformGraph(
                tf_list=[
                    BatchFrameTransform(
                        xyz=torch.tensor([[0.0, 0.0, 0.0]]),
                        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                        parent_frame_id="world",
                        child_frame_id="robot_base",
                    )
                ],
                static_tf=[True],
            ),
            raising=False,
        )

        obs, _ = env.reset(seed=2)

        assert len(close_calls) == 1
        assert built["count"] == 1
        assert obs is not None
        tf_graph = obs["tf"]
        left_joint_tf = tf_graph.get_tf("world", LEFT_EEF_FROM_JOINT_FRAME_ID)
        right_joint_tf = tf_graph.get_tf(
            "world", RIGHT_EEF_FROM_JOINT_FRAME_ID
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
        env._check_and_update_seed = lambda: (_ for _ in ()).throw(
            RuntimeError("reset failed")
        )

        with pytest.raises(RuntimeError, match="reset failed"):
            env.reset(return_obs=False)

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
        env._check_and_update_seed = lambda: (task, [])
        env.cfg.get_task_config_for_seed = lambda runtime_seed: {
            "now_ep_num": env.cfg.episode_id,
            "seed": runtime_seed,
        }

        recorded: dict[str, object] = {}
        monkeypatch.setattr(
            env,
            "_extract_video_frame",
            lambda raw_obs: np.zeros((16, 16, 3), dtype=np.uint8),
            raising=False,
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
        env._check_and_update_seed = lambda: (task, [])

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
        assert state.state["post_reset_state_available"] is True
        assert state.state["episode_finalized"] is False

    def test_get_state_captures_episode_finalized_flag(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, _, _ = _make_state_stub_env(monkeypatch, tmp_path)
        env._episode_finalized = True

        state = env.get_state()

        assert state.state["episode_finalized"] is True

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

        with pytest.raises(NotImplementedError, match="combined dual-arm"):
            env.load_state(state)

        assert close_calls == []
        assert staged_close_calls == [True]
        assert env._task is old_task
        assert env._post_reset_state_available is True

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
        assert obs == {"formatted": {"restored": "robotwin_dummy_task"}}
        assert info["seed"] == 3
        assert info["offset_seed"] == 2
        assert info["source"] == "restored"

    def test_reset_from_state_restores_lifecycle_flags_from_state(
        self,
        monkeypatch,
        tmp_path,
    ):
        env, _, _ = _make_state_stub_env(monkeypatch, tmp_path)
        state = env.get_state()
        state.state["post_reset_state_available"] = False
        state.state["episode_finalized"] = True
        env._post_reset_state_available = True
        env._episode_finalized = False

        env.reset_from_state(state)

        assert env._post_reset_state_available is False
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

        def _raise_format_error(raw_obs):
            del raw_obs
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
        env._check_and_update_seed = lambda: (task, [])
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

        joints_to_eef.transform(
            left_arm_joints=torch.zeros(2, 6, dtype=torch.float64),
            right_arm_joints=torch.zeros(2, 6, dtype=torch.float64),
        )

        assert left_chain.recorded_joint_dtypes == [torch.float32]
        assert right_chain.recorded_joint_dtypes == [torch.float32]

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
            env,
            "_extract_video_frame",
            lambda raw_obs: np.zeros((16, 16, 3), dtype=np.uint8),
            raising=False,
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
        env = RoboTwinEnv.__new__(RoboTwinEnv)
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
        env = RoboTwinEnv.__new__(RoboTwinEnv)
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
        env = RoboTwinEnv.__new__(RoboTwinEnv)
        env._episode_finalized = False

        def _fail_stop() -> None:
            raise RuntimeError("stop failed")

        env._stop_video_recording = _fail_stop

        with pytest.raises(RuntimeError, match="stop failed"):
            env.finalize_episode()

        assert env._episode_finalized is True
