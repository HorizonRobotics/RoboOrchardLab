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
import concurrent
import concurrent.futures
from typing import Any, Generator

import ray
from robo_orchard_core.envs.env_base import EnvBaseCfg
from robo_orchard_core.policy.base import PolicyConfig
from robo_orchard_core.utils.config import (
    ClassType,
    ConfigInstanceOf,
)
from robo_orchard_core.utils.ray import (
    RayRemoteInstance,
    RayRemoteInstanceConfig,
)

from robo_orchard_lab.metrics.base import MetricDict
from robo_orchard_lab.policy.evaluator import (
    PolicyEvaluator,
    PolicyEvaluatorConfig,
)

__all__ = ["PolicyEvaluatorRemote", "PolicyEvaluatorRemoteConfig"]


class PolicyEvaluatorRemote(RayRemoteInstance[PolicyEvaluator]):
    cfg: PolicyEvaluatorRemoteConfig
    InitFromConfig: bool = True

    def __init__(
        self,
        cfg: PolicyEvaluatorRemoteConfig,
    ) -> None:
        super().__init__(cfg)

    def reconfigure_metrics(self, metrics: MetricDict) -> None:
        """Reconfigure the metrics.

        Args:
            metrics (MetricDict): The new metrics.
        """
        ray.get(self._remote.reconfigure_metrics.remote(metrics))

    def reconfigure_env(self, env_cfg: EnvBaseCfg) -> None:
        """Reconfigure the environment.

        Args:
            env_cfg (EnvBaseCfg): The new environment configuration.
        """
        ray.get(self._remote.reconfigure_env.remote(env_cfg))

    def reconfigure_policy(self, policy_cfg: PolicyConfig) -> None:
        """Reconfigure the policy.

        Args:
            policy_cfg (PolicyConfig): The new policy configuration.
        """
        ray.get(self._remote.reconfigure_policy.remote(policy_cfg))

    def evaluate_episode(
        self,
        max_steps: int,
        env_reset_kwargs: dict[str, Any] | None = None,
        policy_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        return self.async_evaluate_episode(
            max_steps=max_steps,
            env_reset_kwargs=env_reset_kwargs,
            policy_reset_kwargs=policy_reset_kwargs,
        ).result()

    def async_evaluate_episode(
        self,
        max_steps: int,
        env_reset_kwargs: dict[str, Any] | None = None,
        policy_reset_kwargs: dict[str, Any] | None = None,
    ) -> concurrent.futures.Future:
        return self._remote.evaluate_episode.remote(
            max_steps=max_steps,
            env_reset_kwargs=env_reset_kwargs,
            policy_reset_kwargs=policy_reset_kwargs,
        ).future()

    def make_episode_evaluation(
        self,
        max_steps: int,
        env_reset_kwargs: dict[str, Any] | None = None,
        policy_reset_kwargs: dict[str, Any] | None = None,
        rollout_steps: int = 5,
    ) -> Generator[int, None, None]:
        """Make an episode evaluation.

        Args:
            max_steps (int): The maximum number of steps to run in the episode.
            env_reset_kwargs (dict[str, Any] | None): Additional arguments to
                pass to the environment's reset method.
            policy_reset_kwargs (dict[str, Any] | None): Additional arguments
                to pass to the policy's reset method.
            rollout_steps (int, optional): The number of steps to rollout
                at each iteration. Defaults to 5.

        Returns:
            Generator[int, None, None]: A generator that yields the number
                of steps taken in each rollout batch.
        """
        gen = self._remote.make_episode_evaluation.remote(
            max_steps=max_steps,
            env_reset_kwargs=env_reset_kwargs,
            policy_reset_kwargs=policy_reset_kwargs,
            rollout_steps=rollout_steps,
        )
        for ref in gen:
            yield ray.get(ref)

        del gen

    def reset_metrics(self) -> None:
        """Reset the metrics."""
        ray.get(self._remote.reset_metrics.remote())

    def reset_policy(self) -> None:
        """Reset the policy."""
        ray.get(self._remote.reset_policy.remote())

    def compute_metrics(self) -> dict[str, Any]:
        """Compute the metrics and return the results.

        Returns:
            dict[str, Any]: The computed metrics.
        """
        return ray.get(self._remote.compute_metrics.remote())


class PolicyEvaluatorRemoteConfig(
    RayRemoteInstanceConfig[PolicyEvaluatorRemote, PolicyEvaluatorConfig]
):
    """Configuration for PolicyEvaluatorRemote."""

    class_type: ClassType[PolicyEvaluatorRemote] = PolicyEvaluatorRemote
    instance_config: ConfigInstanceOf[PolicyEvaluatorConfig]
