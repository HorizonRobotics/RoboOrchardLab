# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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
from typing import TYPE_CHECKING, Any, Generator

from robo_orchard_core.envs.env_base import (
    EnvBase,
    EnvBaseCfg,
    EnvStepReturn,
)
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import (
    ClassConfig,
    ConfigInstanceOf,
)
from robo_orchard_core.utils.ray import RayRemoteClassConfig

from robo_orchard_lab.metrics.base import (
    MetricDict,
    MetricDictConfig,
)

if TYPE_CHECKING:
    from robo_orchard_lab.policy.remote import PolicyEvaluatorRemoteConfig

__all__ = ["PolicyEvaluator", "PolicyEvaluatorConfig"]


def evaluate_rollout_stop_condition(
    step_ret: EnvStepReturn | tuple[Any, Any, bool, bool, dict[str, Any]],
) -> bool:
    """Determine whether to stop the rollout based on terminal conditions.

    Returns:
        bool: True if the rollout should stop, False otherwise.
    """
    # case for gym.Env that return tuple
    if isinstance(step_ret, tuple):
        done = step_ret[2]
        truncated = step_ret[3]
        return done or truncated
    elif isinstance(step_ret, EnvStepReturn):
        if not isinstance(step_ret.terminated, bool):
            raise ValueError(
                "The `terminated` field in `EnvStepReturn` must be a boolean."
            )
        if not isinstance(step_ret.truncated, bool):
            raise ValueError(
                "The `truncated` field in `EnvStepReturn` must be a boolean."
            )
        return step_ret.terminated or step_ret.truncated
    else:
        raise NotImplementedError(
            "The `step_ret` must be either `EnvStepReturn` or tuple."
        )
    return False


class PolicyEvaluator:
    """Evaluate a policy using a set of metrics on an environment.

    Args:
        cfg (PolicyEvaluatorConfig): Configuration for the
            PolicyEvaluator instance.

    """

    InitFromConfig: bool = True

    metrics: MetricDict
    policy: PolicyMixin
    env: EnvBase

    cfg: PolicyEvaluatorConfig

    def __init__(
        self,
        cfg: PolicyEvaluatorConfig,
    ) -> None:
        self.cfg = cfg

        self.policy = cfg.policy_cfg()
        self.env = cfg.env_cfg()
        self.metrics = cfg.metrics()

    def reconfigure_metrics(self, metrics: MetricDict) -> None:
        """Reconfigure the metrics with a new set of metrics.

        Args:
            metrics (MetricDict): A dictionary where keys are
                metric names and values are callable metric functions that
                follow the MetricProtocol.
        """
        self.metrics = metrics

    def reconfigure_env(self, env_cfg: EnvBaseCfg) -> None:
        """Reconfigure the environment with a new configuration.

        Args:
            env_cfg (EnvBaseCfg): The new configuration for the environment.
        """
        self.env = env_cfg()

    def reconfigure_policy(self, policy_cfg: PolicyConfig) -> None:
        """Reconfigure the policy with a new configuration.

        Args:
            policy_cfg (PolicyConfig): The new configuration for the policy.
        """
        self.policy = policy_cfg()

    def evaluate_episode(
        self,
        max_steps: int,
        env_reset_kwargs: dict[str, Any] | None = None,
        policy_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Non-generator version of make_episode_evaluation."""
        for _ in self.make_episode_evaluation(
            max_steps=max_steps,
            env_reset_kwargs=env_reset_kwargs,
            policy_reset_kwargs=policy_reset_kwargs,
            rollout_steps=max_steps,
        ):
            pass

    def make_episode_evaluation(
        self,
        max_steps: int,
        env_reset_kwargs: dict[str, Any] | None = None,
        policy_reset_kwargs: dict[str, Any] | None = None,
        rollout_steps: int = 5,
    ) -> Generator[int, None, None]:
        """Make a generator to evaluate the policy on episodes.

        This method yields the number of steps taken in each rollout
        until the episode ends or the maximum number of steps is reached.

        At the beginning of each episode, the environment and policy
        are reset using the provided keyword arguments. And after the
        episode ends, the metrics are updated with the last action
        and step result.

        Args:
            max_steps (int): The maximum number of steps to evaluate
                the policy for.
            env_reset_kwargs (dict, optional): Keyword arguments to
                pass to the environment's reset method. Defaults to None.
            policy_reset_kwargs (dict, optional): Keyword arguments to
                pass to the policy's reset method. Defaults to None.
            rollout_steps (int, optional): The number of steps to roll
                out in each iteration. Defaults to 5.

        """
        env_reset_kwargs = env_reset_kwargs or {}
        env_reset_ret: tuple[Any, dict[str, Any]] = self.env.reset(
            **env_reset_kwargs
        )
        policy_reset_kwargs = policy_reset_kwargs or {}
        self.policy.reset(**policy_reset_kwargs)
        init_obs = env_reset_ret[0]

        last_action = None
        last_step_ret = None

        for i in range(0, max_steps, rollout_steps):
            rollout_ret = self.env.rollout(
                init_obs=init_obs,
                max_steps=min(rollout_steps, max_steps - i),
                policy=self.policy,
                terminal_condition=evaluate_rollout_stop_condition,
                keep_last_results=1,
            )
            if isinstance(rollout_ret.step_results[-1], EnvStepReturn):
                init_obs = rollout_ret.step_results[-1].observations
            else:
                init_obs = rollout_ret.step_results[-1][0]
            last_action = rollout_ret.actions[-1]
            last_step_ret = rollout_ret.step_results[-1]
            yield rollout_ret.rollout_actual_steps
            if rollout_ret.terminal_condition_triggered:
                break

        assert last_action is not None
        assert last_step_ret is not None

        self.metrics.update(last_action, last_step_ret)

    def reset_metrics(
        self, metrics_reset_kwargs: dict[str, Any] | None = None
    ):
        """Reset all metrics.

        Args:
            metrics_reset_kwargs (dict[str, Any] | None): Additional
                arguments to pass to the reset method of each metric.
                Defaults to None.
        """
        if metrics_reset_kwargs is None:
            metrics_reset_kwargs = {}
        for _, metric in self.metrics.items():
            metric.reset(**metrics_reset_kwargs)

    def reset_policy(self, policy_reset_kwargs: dict[str, Any] | None = None):
        """Reset the policy.

        Args:
            policy_reset_kwargs (dict[str, Any] | None): Additional
                arguments to pass to the reset method of the policy.
                Defaults to None.
        """
        if policy_reset_kwargs is None:
            policy_reset_kwargs = {}
        self.policy.reset(**policy_reset_kwargs)

    def compute_metrics(self) -> dict[str, Any]:
        """Compute all metrics and return the results as a dictionary.

        Returns:
            dict: A dictionary where keys are metric names and values are
                the computed metric values.
        """
        return self.metrics.compute()


class PolicyEvaluatorConfig(ClassConfig[PolicyEvaluator]):
    """Configuration class for PolicyEvaluator.

    This class is used to configure and instantiate a PolicyEvaluator
    object with the specified environment, policy, and metrics.

    Args:
        env_cfg (EnvBaseCfg): The configuration for the environment.
        policy_cfg (PolicyConfig): The configuration for the policy.
        metrics (dict | MetricDict): A dictionary where keys are
            metric names and values are callable metric functions that
            follow the MetricProtocol.

    """

    class_type: type[PolicyEvaluator] = PolicyEvaluator

    env_cfg: ConfigInstanceOf[EnvBaseCfg]
    policy_cfg: ConfigInstanceOf[PolicyConfig]
    metrics: MetricDictConfig

    def as_remote(
        self,
        remote_class_config: RayRemoteClassConfig | None = None,
        ray_init_config: dict[str, Any] | None = None,
        check_init_timeout: int = 60,
    ) -> PolicyEvaluatorRemoteConfig:
        from robo_orchard_lab.policy.remote import PolicyEvaluatorRemoteConfig

        if remote_class_config is None:
            remote_class_config = RayRemoteClassConfig()
        return PolicyEvaluatorRemoteConfig(
            instance_config=self,
            remote_class_config=remote_class_config,
            ray_init_config=ray_init_config,
            check_init_timeout=check_init_timeout,
        )
