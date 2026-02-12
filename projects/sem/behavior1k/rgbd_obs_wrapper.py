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


from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.learning.utils.eval_utils import (
    HEAD_RESOLUTION,
    ROBOT_CAMERA_NAMES,
    WRIST_RESOLUTION,
)
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
logger = create_module_logger("RGBDObservationWrapper")


class RGBDObservationWrapper(EnvironmentWrapper):
    """RGBDObservationWrapper.

    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
        super().__init__(env=env)
        # Note that from eval.py we already set the robot to include
        # rgb + depth + seg_instance_id modalities
        robot = env.robots[0]

        # Here, we change the camera resolution and head camera aperture
        # to match the one we used in data collection
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0
                robot.sensors[sensor_name].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = HEAD_RESOLUTION[1]
            else:
                robot.sensors[sensor_name].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = WRIST_RESOLUTION[1]

            robot.sensors[sensor_name].add_modality("depth_linear")

        # reload observation space
        env.load_observation_space()

        # we also set task to include obs
        env.task._include_obs = True
        logger.info("Reloaded observation space!")

    def step(self, action, n_render_iterations=1):
        """Step function.

        By default, run the normal environment step() function

        Args:
            action (th.tensor): action to take in environment
            n_render_iterations (int): Number of rendering iterations
                to use before returning observations

        Returns:
            4-tuple:
                - (dict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is terminated
                - (bool) whether the current episode is truncated
                - (dict) misc information
        """
        obs, reward, terminated, truncated, info = self.env.step(
            action, n_render_iterations=n_render_iterations
        )

        # Now, query for some additional privileged task info
        obs["task"] = self.env.task.get_obs(self.env)
        return obs, reward, terminated, truncated, info

    def reset(self):
        # Note that we need to also add additional observations in reset()
        # because the returned observation will be passed into policy
        ret = self.env.reset()
        ret[0]["task"] = self.env.task.get_obs(self.env)
        return ret
