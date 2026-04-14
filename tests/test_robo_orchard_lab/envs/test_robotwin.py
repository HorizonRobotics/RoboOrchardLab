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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robo_orchard_lab.envs.robotwin.env import RoboTwinEnv, RoboTwinEnvCfg

pytestmark = pytest.mark.sim_env


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
