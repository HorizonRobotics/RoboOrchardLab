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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    import mani_skill  # noqa: F401
except ImportError:
    pytest.skip("mani_skill is not installed", allow_module_level=True)

from robo_orchard_lab.envs.maniskill import ManiSkillEnv, ManiSkillEnvCfg

pytestmark = pytest.mark.sim_env


class TestManiSkillEnv:
    def test_get_observations_reads_native_current_state(self):
        env = object.__new__(ManiSkillEnv)
        observation = {"agent": {"qpos": 1}}
        get_obs = MagicMock(return_value=observation)
        env.env = SimpleNamespace(get_obs=get_obs)

        assert env.get_observations() is observation
        get_obs.assert_called_once_with()

    def test_env_create(self):
        env = ManiSkillEnv(
            ManiSkillEnvCfg(env_id="PickCube-v1", obs_mode="rgbd")
        )
        assert env is not None

    def test_env_step(self):
        env = ManiSkillEnv(
            ManiSkillEnvCfg(env_id="PickCube-v1", obs_mode="rgbd")
        )
        assert env is not None
        env.reset()
        ret = env.step(env.action_space.sample())
        assert ret is not None
