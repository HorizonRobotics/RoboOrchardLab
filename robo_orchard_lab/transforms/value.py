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


class ValueSampling:
    def __init__(self, norm_mode="episode", task_max_step=None):
        assert norm_mode in ["episode", "task"]
        self.norm_mode = norm_mode
        self.task_max_step = task_max_step

    def __call__(self, data):

        is_rollout = data.get("metas", {}).get("rollout", [False])[0]
        is_success = data.get("metas", {}).get("success", [True])[0]

        if is_rollout:
            raise NotImplementedError(
                "Rollout data is not used during training in this version."
            )

        if not is_success:
            raise NotImplementedError(
                "Failure data is not used during training in this version."
            )

        assert (
            "joint_state" in data.keys()
        )  # some transforms may remove this key. put this transform earlier.
        num_episode_step = data["joint_state"].shape[0]
        step_index = data["step_index"]

        if self.norm_mode == "episode":
            max_step = num_episode_step
        elif self.norm_mode == "task":
            max_step = self.task_max_step
            raise NotImplementedError(
                "'task' norm_mode is not implemented yet. "
            )
        else:
            raise NotImplementedError(
                f"Only 'episode' and 'task' norm_mode are supported, "
                f"got '{self.norm_mode}'."
            )

        value = (step_index - num_episode_step) / max_step
        data["eps_steps"] = num_episode_step
        data["value"] = value

        return data
