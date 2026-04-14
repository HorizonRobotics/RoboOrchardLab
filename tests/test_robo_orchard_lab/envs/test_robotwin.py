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
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from robo_orchard_lab.dataset.datatypes import BatchFrameTransform
from robo_orchard_lab.envs.robotwin.env import RoboTwinEnv, RoboTwinEnvCfg
from robo_orchard_lab.envs.robotwin.kinematics import RoboTwinEEF

pytestmark = pytest.mark.sim_env


def _make_fake_sapien_pose(
    xyz: tuple[float, float, float],
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> SimpleNamespace:
    return SimpleNamespace(
        get_p=lambda: np.asarray(xyz, dtype=np.float32),
        get_q=lambda: np.asarray(quat, dtype=np.float32),
    )


def _make_reset_stub_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    robot: SimpleNamespace,
) -> RoboTwinEnv:
    env = RoboTwinEnv.__new__(RoboTwinEnv)
    cast(Any, env).cfg = SimpleNamespace(
        seed=1,
        task_name="robotwin_dummy_task",
        episode_id=0,
        get_task_config=lambda: {},
        calculate_seed=lambda seed: seed,
    )
    cast(Any, env)._video_ffmpeg = None
    cast(Any, env)._check_and_update_seed = lambda: (
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
    return env


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
        cast(Any, env)._task = SimpleNamespace(
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
        cast(Any, env)._check_and_update_seed = lambda: (task, [])
        env.cfg.get_task_config = lambda: {"now_ep_num": env.cfg.episode_id}

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
