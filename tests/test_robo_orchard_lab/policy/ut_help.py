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
from typing import Any

import torch
from robo_orchard_core.envs.env_base import EnvBase, EnvBaseCfg
from robo_orchard_core.utils.config import ClassType

from robo_orchard_lab.metrics.base import MetricConfig
from robo_orchard_lab.policy.base import PolicyConfig, PolicyMixin


class DummyEnv(EnvBase):
    cfg: DummyEnvConfig

    def __init__(self, cfg: DummyEnvConfig) -> None:
        self.cfg = cfg
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return {"obs": 0}, {"step": self.step_count}

    def step(self, action):
        self.step_count += 1
        obs = self.step_count
        reward = 1.0 if self.step_count >= self.cfg.success_step_count else 0.0
        terminated = self.step_count >= self.cfg.success_step_count
        truncated = False
        info = {"step": self.step_count}
        return (
            {"obs": obs, "act": action},
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass

    def num_envs(self) -> int:
        return 1


class DummyEnvConfig(EnvBaseCfg[DummyEnv]):
    class_type: ClassType[DummyEnv] = DummyEnv

    success_step_count: int = 5


class DummyPolicy(PolicyMixin):
    cfg: DummyPolicyConfig

    data: torch.Tensor | None

    def __init__(self, cfg: DummyPolicyConfig, *args, **kwargs) -> None:
        self.cfg = cfg
        self.data = None

    def reset(self, **kwargs) -> None:
        pass

    def act(self, obs):
        # return obs with policy data
        ret = {"data": self.data}
        ret.update(obs)
        return ret

    def close(self) -> None:
        pass


class DummyPolicyConfig(PolicyConfig[DummyPolicy]):
    class_type: ClassType[DummyPolicy] = DummyPolicy


class DummySuccessRateMetric:
    cfg: DummySuccessRateMetricConfig

    def __init__(self, cfg: DummySuccessRateMetricConfig) -> None:
        self.cfg = cfg
        self.success_count = 0
        self.episode_count = 0

    def reset(self, **kwargs) -> None:
        self.success_count = 0
        self.episode_count = 0

    def update(self, action, step_ret) -> None:
        success = step_ret[1]  # reward indicates success in DummyEnv
        if success > 0:
            self.success_count += 1
        self.episode_count += 1

    def compute(self) -> float:
        if self.episode_count == 0:
            return 0.0
        return float(self.success_count) / self.episode_count

    def __call__(self, *args, **kwargs) -> None:
        self.update(*args, **kwargs)

    def to(self, *args, **kwargs):
        pass


class DummyPolicyDataMetric:
    cfg: DummySuccessRateMetricConfig
    data: Any

    def __init__(self, cfg: DummySuccessRateMetricConfig) -> None:
        self.cfg = cfg
        self.data = None

    def reset(self) -> None:
        self.data = None

    def update(self, action, step_ret) -> None:
        self.data = {"action": action, "step_ret": step_ret}

    def compute(self) -> Any:
        return self.data

    def __call__(self, *args, **kwargs) -> None:
        self.update(*args, **kwargs)

    def to(self, *args, **kwargs):
        pass


class DummySuccessRateMetricConfig(MetricConfig[Any]):
    class_type: ClassType[Any] = DummySuccessRateMetric
