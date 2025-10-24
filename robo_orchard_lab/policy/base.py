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
        cfg: InferencePipelinePolicyCfg,
        observation_space: gym.Space[OBSType] | None = None,
        action_space: gym.Space[ACTType] | None = None,
    ):
        super().__init__(
            cfg, observation_space=observation_space, action_space=action_space
        )
        self.pipeline = cfg.pipeline()

    @classmethod
    def from_infer_pipeline(
        cls,
        pipeline: InferencePipelineMixin,
        observation_space: gym.Space[OBSType] | None = None,
        action_space: gym.Space[ACTType] | None = None,
    ):
        ret = cls.__new__(cls)
        ret.cfg = InferencePipelinePolicyCfg.from_inference_pipeline(
            pipeline.cfg
        )
        ret.pipeline = pipeline
        ret.observation_space = observation_space
        ret.action_space = action_space
        return ret

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
        if hasattr(self.pipeline, "reset"):
            self.pipeline.reset()  # type: ignore


class InferencePipelinePolicyCfg(PolicyConfig[InferencePipelinePolicy]):
    class_type: ClassType_co[InferencePipelinePolicy] = InferencePipelinePolicy

    pipeline: ConfigInstanceOf[InferencePipelineMixinCfg]

    @classmethod
    def from_inference_pipeline(
        cls,
        pipeline: InferencePipelineMixinCfg,
    ) -> InferencePipelinePolicyCfg:
        """Create an from an existing InferencePipeline Config.

        Args:
            pipeline (InferencePipelineMixinCfg): An instance of pipeline
                config.

        Returns:
            InferencePipelinePolicyCfg: A configuration object for
                InferencePipelinePolicy.
        """
        cfg = cls(pipeline=pipeline)
        return cfg
