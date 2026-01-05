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
from typing import Any, TypeVar

import gymnasium as gym
import torch
from robo_orchard_core.policy.base import (
    ACTType,
    OBSType,
    PolicyConfig as _PolicyConfig,
    PolicyMixin as _PolicyMixin,
)
from robo_orchard_core.utils.config import ClassType_co, ConfigInstanceOf

from robo_orchard_lab.inference.mixin import (
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
)
from robo_orchard_lab.utils.state import State, StateSaveLoadMixin

__all__ = [
    "InferencePipelinePolicy",
    "InferencePipelinePolicyCfg",
    "PolicyConfig",
    "PolicyMixin",
]


class PolicyMixin(StateSaveLoadMixin, _PolicyMixin[OBSType, ACTType]):
    """A base class for policies with state save/load capability."""

    def _get_state(self) -> State:
        """Get the state of the object for saving."""
        # pull out cfg from state for better clarity
        ret = super()._get_state()
        ret.config = ret.state.pop("cfg", None)
        return ret

    def _set_state(self, state: State) -> None:
        """Set the state of the object from the unpickled state."""
        # push cfg back to state for consistency
        state.state["cfg"] = state.config
        state.config = None
        super()._set_state(state)

    def to(self, device: torch.device | str):
        """Moves the pipeline to the specified device.

        Args:
            device (str): The target device to move the model to.
        """
        pass


PolicyType = TypeVar("PolicyType", bound=PolicyMixin, covariant=True)


class PolicyConfig(_PolicyConfig[PolicyType]):
    """Configuration for PolicyMixin."""

    def __call__(self, *args, **kwargs) -> PolicyType:
        return self.create_instance_by_cfg(*args, **kwargs)


class InferencePipelinePolicy(PolicyMixin[OBSType, ACTType]):
    """A policy that uses an inference pipeline to generate actions.

    Args:
        cfg (InferencePipelinePolicyCfg): The configuration for the policy.
        observation_space (gym.Space | None, optional): The observation space
            of the environment. Defaults to None.
        action_space (gym.Space | None, optional): The action space of
            the environment. Defaults to None.
        pipeline (InferencePipelineMixin| None, optional): The inference
            pipeline to use. If None, it will be created from the
            configuration. If provided, its configuration must match
            the policy's pipeline_cfg. Defaults to None.
    """

    cfg: InferencePipelinePolicyCfg

    pipeline: InferencePipelineMixin[OBSType, ACTType]

    def __init__(
        self,
        cfg: InferencePipelinePolicyCfg,
        observation_space: gym.Space[OBSType] | None = None,
        action_space: gym.Space[ACTType] | None = None,
        pipeline: InferencePipelineMixin[OBSType, ACTType] | None = None,
    ):
        super().__init__(
            cfg,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._setup(
            cfg,
            observation_space=observation_space,
            action_space=action_space,
            pipeline=pipeline or cfg.pipeline_cfg(),
        )

    def _setup(
        self,
        cfg: InferencePipelinePolicyCfg,
        observation_space: gym.Space[OBSType] | None,
        action_space: gym.Space[ACTType] | None,
        pipeline: InferencePipelineMixin[OBSType, ACTType],
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        if pipeline.cfg != cfg.pipeline_cfg:
            raise ValueError(
                "The pipeline's cfg does not match the policy's pipeline_cfg. "
                f"Got pipeline.cfg: {pipeline.cfg}, "
                f"policy.pipeline_cfg: {cfg.pipeline_cfg}",
            )
        self.cfg = cfg
        self.pipeline = pipeline

    def _set_state(self, state: State) -> None:
        super()._set_state(state)
        self._setup(
            cfg=self.cfg,
            observation_space=self.observation_space,
            action_space=self.action_space,
            pipeline=self.pipeline,
        )

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

    def to(self, device: torch.device | str):
        """Moves the pipeline to the specified device.

        Args:
            device (str): The target device to move the model to.
        """
        self.pipeline.to(device)

    @property
    def device(self) -> torch.device:
        """The device where the pipeline's parameters are located."""
        return self.pipeline.device


class InferencePipelinePolicyCfg(PolicyConfig[InferencePipelinePolicy]):
    class_type: ClassType_co[InferencePipelinePolicy] = InferencePipelinePolicy

    pipeline_cfg: ConfigInstanceOf[InferencePipelineMixinCfg[Any]]
