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
from __future__ import annotations

import gymnasium as gym
from robo_orchard_core.policy.base import (
    ACTType,
    OBSType,
    PolicyConfig,
    PolicyMixin,
)
from robo_orchard_core.utils.config import ClassType_co, ConfigInstanceOf

from robo_orchard_lab.inference.mixin import (
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
)

__all__ = ["InferencePipelinePolicy", "InferencePipelinePolicyCfg"]


class InferencePipelinePolicy(PolicyMixin[OBSType, ACTType]):
    cfg: InferencePipelinePolicyCfg

    pipeline: InferencePipelineMixin[OBSType, ACTType]

    def __init__(
        self,
        pipeline: InferencePipelinePolicyCfg
        | InferencePipelineMixin[OBSType, ACTType],
        observation_space: gym.Space[OBSType] | None = None,
        action_space: gym.Space[ACTType] | None = None,
    ):
        if isinstance(pipeline, InferencePipelinePolicyCfg):
            super().__init__(
                pipeline,
                observation_space=observation_space,
                action_space=action_space,
            )
        else:
            super().__init__(
                InferencePipelinePolicyCfg(
                    pipeline_cfg=pipeline.cfg,
                ),
                observation_space=observation_space,
                action_space=action_space,
            )
        if isinstance(pipeline, InferencePipelinePolicyCfg):
            pipeline = pipeline.pipeline_cfg()
        assert isinstance(pipeline, InferencePipelineMixin)
        self.pipeline = pipeline

    def act(self, obs: OBSType) -> ACTType:
        """Generate an action based on the observation.

        Args:
            obs (OBSType): The observation from the environment.

        Returns:
            ACTType: The action to be taken in the environment.
        """

        action = self.pipeline(obs)
        return action

    def reset(self) -> None:
        self.pipeline.reset()


class InferencePipelinePolicyCfg(PolicyConfig[InferencePipelinePolicy]):
    class_type: ClassType_co[InferencePipelinePolicy] = InferencePipelinePolicy

    pipeline_cfg: ConfigInstanceOf[InferencePipelineMixinCfg]
