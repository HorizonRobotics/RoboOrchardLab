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
from typing import TYPE_CHECKING, Any, Callable, Generator

import torch
from robo_orchard_core.envs.env_base import (
    EnvBase,
    EnvBaseCfg,
    EnvStepReturn,
)
from robo_orchard_core.utils.config import (
    ClassConfig,
)
from robo_orchard_core.utils.ray import RayRemoteClassConfig

from robo_orchard_lab.metrics.base import (
    MetricDict,
    MetricProtocol,
)
from robo_orchard_lab.policy.base import PolicyConfig, PolicyMixin

if TYPE_CHECKING:
    from robo_orchard_lab.policy.evaluator.remote import (
        PolicyEvaluatorRemoteConfig,
    )

__all__ = [
    "PolicyEvaluator",
    "PolicyEvaluatorConfig",
    "RollOutStopCondition",
    "evaluate_rollout_stop_condition",
]

RollOutStopCondition = Callable[
    [EnvStepReturn | tuple[Any, Any, bool, bool, dict[str, Any]]], bool
]


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

    The evaluator setup process may be time-consuming, so it is separated
    from the initialization process. Please call the `setup()` method after
    initialization to prepare the evaluator for use.

    Args:
        cfg (PolicyEvaluatorConfig): Configuration for the
            PolicyEvaluator instance.

    """

    InitFromConfig: bool = True

    metrics: MetricDict | MetricProtocol
    policy: PolicyMixin
    env: EnvBase

    cfg: PolicyEvaluatorConfig

    def __init__(
        self,
        cfg: PolicyEvaluatorConfig,
    ) -> None:
        self.cfg = cfg

    def setup(
        self,
        env_cfg: EnvBaseCfg,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        metrics: MetricDict,
        device: str | torch.device | None = None,
    ):
        """Setup the evaluator with the current configuration."""
        self.reconfigure_env(env_cfg)
        self.reconfigure_policy(policy_or_cfg, device=device)
        self.reconfigure_metrics(metrics)

    def reconfigure_metrics(
        self, metrics: MetricDict | MetricProtocol
    ) -> None:
        """Reconfigure the metrics with a new set of metrics.

        Args:
            metrics (MetricDict|MetricProtocol): A dictionary where keys are
                metric names and values are callable metric functions that
                follow the MetricProtocol.
        """
        self.metrics = metrics

    def reconfigure_env(self, env_cfg: EnvBaseCfg) -> None:
        """Reconfigure the environment with a new configuration.

        We only provide reconfiguration via configuration here because for
        most cases, the environment does not support pickling/unpickling.

        Args:
            env_cfg (EnvBaseCfg): The new configuration for the environment.
        """
        self.env = env_cfg()

    def reconfigure_policy(
        self,
        policy_or_cfg: PolicyConfig | PolicyMixin,
        device: str | torch.device | None = None,
    ) -> None:
        """Reconfigure the policy with a new configuration.

        Args:
            policy_or_cfg (PolicyConfig | PolicyMixin): The new configuration
                for the policy or a policy instance.
        """
        if isinstance(policy_or_cfg, PolicyMixin):
            self.policy = policy_or_cfg
        else:
            self.policy = policy_or_cfg()
        if device is not None:
            self.policy.to(device=device)

    def evaluate_episode(
        self,
        max_steps: int,
        env_reset_kwargs: dict[str, Any] | None = None,
        policy_reset_kwargs: dict[str, Any] | None = None,
        rollout_stop_condition: RollOutStopCondition = evaluate_rollout_stop_condition,  # noqa: E501
    ) -> dict[str, Any]:
        """Evaluate the policy on a single episode and return metrics."""
        for _ in self.make_episode_evaluation(
            max_steps=max_steps,
            env_reset_kwargs=env_reset_kwargs,
            policy_reset_kwargs=policy_reset_kwargs,
            rollout_steps=max_steps,
            rollout_stop_condition=rollout_stop_condition,
        ):
            pass
        return self.compute_metrics()

    def make_episode_evaluation(
        self,
        max_steps: int,
        env_reset_kwargs: dict[str, Any] | None = None,
        policy_reset_kwargs: dict[str, Any] | None = None,
        rollout_steps: int = 5,
        rollout_stop_condition: RollOutStopCondition = evaluate_rollout_stop_condition,  # noqa: E501
    ) -> Generator[int, None, None]:
        """Make a generator to evaluate the policy on episodes.

        This method yields the number of steps taken in each rollout
        until the episode ends or the maximum number of steps is reached.

        At the beginning of each episode, the environment and policy
        are reset using the provided keyword arguments. And after the
        final step, the metrics are updated with the last action
        and step result.

        Warning:
            The final action and step result are used to update the metrics,
            regardless of whether the episode ended due to reaching the
            maximum number of steps or due to the terminal condition. This
            may lead to unexpected behavior if the last action and step result
            are not representative of the entire episode.

        Args:
            max_steps (int): The maximum number of steps to evaluate
                the policy for.
            env_reset_kwargs (dict, optional): Keyword arguments to
                pass to the environment's reset method. Defaults to None.
            policy_reset_kwargs (dict, optional): Keyword arguments to
                pass to the policy's reset method. Defaults to None.
            rollout_steps (int, optional): The number of steps to roll
                out in each iteration. Defaults to 5.
        yields:
            int: The number of steps taken in each rollout.

        """
        ready = (
            hasattr(self, "policy")
            and hasattr(self, "env")
            and hasattr(self, "metrics")
        )
        if not ready:
            raise RuntimeError(
                "PolicyEvaluator is not ready. Please call setup() first "
                "or reconfigure the policy, environment, and metrics."
            )

        env_reset_ret: tuple[Any, dict[str, Any]] = self.reset_env(
            **(env_reset_kwargs or {})
        )
        self.reset_policy(**(policy_reset_kwargs or {}))
        init_obs = env_reset_ret[0]

        last_action = None
        last_step_ret = None

        for i in range(0, max_steps, rollout_steps):
            rollout_ret = self.env.rollout(
                init_obs=init_obs,
                max_steps=min(rollout_steps, max_steps - i),
                policy=self.policy,
                terminal_condition=rollout_stop_condition,
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

    def reset_metrics(self, **kwargs) -> None:
        """Reset all metrics.

        Args:
            kwargs: Additional arguments to pass to the
                metrics' reset method.
        """
        if not hasattr(self, "metrics"):
            raise RuntimeError(
                "Metrics are not configured. Cannot reset metrics."
            )

        return self.metrics.reset(**kwargs)

    def reset_env(self, **kwargs) -> Any:
        """Reset the environment.

        Args:
            kwargs: Additional arguments to pass to the
                environment's reset method.

        """

        if not hasattr(self, "env"):
            raise RuntimeError(
                "Environment is not configured. Cannot reset environment."
            )

        return self.env.reset(**kwargs)

    def reset_policy(self, **kwargs) -> None:
        """Reset the policy.

        Args:
            kwargs: Additional arguments to pass to the policy's
                reset method.

        """

        if not hasattr(self, "policy"):
            raise RuntimeError(
                "Policy is not configured. Cannot reset policy."
            )

        return self.policy.reset(**kwargs)

    def get_metrics(self) -> MetricDict | MetricProtocol | None:
        """Get the current metrics.

        Returns:
            MetricDict: A dictionary where keys are metric names
                and values are the current metric values.
        """
        if not hasattr(self, "metrics"):
            return None

        return self.metrics

    def compute_metrics(self) -> dict[str, Any]:
        """Compute all metrics and return the results as a dictionary.

        Returns:
            dict: A dictionary where keys are metric names and values are
                the computed metric values.
        """
        if not hasattr(self, "metrics"):
            raise RuntimeError(
                "Metrics are not configured. Cannot compute metrics."
            )

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

    def as_remote(
        self,
        remote_class_config: RayRemoteClassConfig | None = None,
        ray_init_config: dict[str, Any] | None = None,
        check_init_timeout: int = 60,
    ) -> PolicyEvaluatorRemoteConfig:
        from robo_orchard_lab.policy.evaluator.remote import (
            PolicyEvaluatorRemoteConfig,
        )

        if remote_class_config is None:
            remote_class_config = RayRemoteClassConfig()
        return PolicyEvaluatorRemoteConfig(
            instance_config=self,
            remote_class_config=remote_class_config,
            ray_init_config=ray_init_config,
            check_init_timeout=check_init_timeout,
        )
