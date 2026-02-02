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

import pytest

from robo_orchard_lab.envs.robotwin.env import RoboTwinEnv, RoboTwinEnvCfg


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

    def test_step(self, dummy_env_without_expert_check: RoboTwinEnv):
        # Note that not all env can step because of robotwin BUG!
        env = dummy_env_without_expert_check
        obs, info = env.reset()
        assert obs is not None

        action = [1.0] * 14
        step_return = env.step(action)
        assert step_return.observations is not None
        assert "tf" in step_return.observations
