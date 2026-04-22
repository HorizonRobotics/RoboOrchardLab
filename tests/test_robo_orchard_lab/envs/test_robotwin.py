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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

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
    RoboTwinEnv,
    RoboTwinEnvCfg,
)
from robo_orchard_lab.envs.robotwin.kinematics import (
    RoboTwinEEF,
    RoboTwinJointsToEEF,
)

pytestmark = pytest.mark.sim_env


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
    env._instructions = None
    env._cached_obs_robots = None
    env._video_ffmpeg = None
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
            "_start_video_recording",
            lambda video_path, raw_obs: recorded.update(
                {"video_path": video_path, "raw_obs": raw_obs}
            ),
            raising=False,
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

    def test_reset_seed_next_advances_within_current_seed_family(
        self, monkeypatch
    ):
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
        check_calls = {"count": 0}

        def fake_check_and_update_seed():
            if check_calls["count"] == 0:
                env._offset_seed += 2
            check_calls["count"] += 1
            return task, []

        env._check_and_update_seed = fake_check_and_update_seed

        _, first_info = env.reset(seed=2)
        _, second_info = env.reset(seed="next")

        assert env.cfg.seed == 2
        assert env.start_seed == 2
        assert first_info["offset_seed"] == 2
        assert first_info["seed"] == 4
        assert second_info["offset_seed"] == 3
        assert second_info["seed"] == 5

    def test_reset_rejects_seed_next_with_explicit_offset(self, monkeypatch):
        robot = SimpleNamespace(
            is_dual_arm=True,
            left_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
            right_entity_origion_pose=_make_fake_sapien_pose((0.0, 0.0, 0.0)),
        )
        env = _make_reset_stub_env(monkeypatch, robot=robot)

        with pytest.raises(
            ValueError,
            match="seed='next' cannot be combined with offset_seed",
        ):
            env.reset(seed="next", offset_seed=3)

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
            "_start_video_recording",
            lambda video_path, raw_obs: recorded.update(
                {"video_path": video_path, "raw_obs": raw_obs}
            ),
            raising=False,
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

    def test_video_recording_lifecycle(self):
        env = RoboTwinEnv.__new__(RoboTwinEnv)
        env._video_ffmpeg = None

        raw_obs = {
            "observation": {
                "head_camera": {
                    "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                }
            }
        }
        ffmpeg = MagicMock()
        ffmpeg.stdin = MagicMock()

        with (
            patch(
                "robo_orchard_lab.envs.robotwin.env.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "robo_orchard_lab.envs.robotwin.env.subprocess.Popen",
                return_value=ffmpeg,
            ) as mock_popen,
        ):
            env._start_video_recording(
                video_path="/tmp/task/demo_clean/100000.mp4",
                raw_obs=raw_obs,
            )
            env._write_video_frame(raw_obs)
            env._stop_video_recording()

        mock_popen.assert_called_once()
        assert ffmpeg.stdin.write.call_count == 2
        ffmpeg.stdin.close.assert_called_once()
        ffmpeg.wait.assert_called_once()
        assert env._video_ffmpeg is None
