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

from robo_orchard_lab.envs.robotwin.env import RoboTwinEnv, RoboTwinEnvCfg


class TestRoboTwinEnv:
    def test_init_without_expert_check(self):
        env = RoboTwinEnv(
            RoboTwinEnvCfg(
                task_name="place_object_scale",
                check_expert=False,
                seed=1,
                check_task_init=False,  # for fast initialization
            )
        )
        env.reset()
        assert env is not None
        assert env._task is not None
