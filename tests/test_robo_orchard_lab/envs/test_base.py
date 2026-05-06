# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

from robo_orchard_lab.envs.base import (
    EnvBase,
    EnvBaseCfg,
    EnvStepReturn,
    EpisodeFinalizableEnvProtocol,
    finalize_env_episode,
)


class _FinalizableEnv:
    def __init__(self) -> None:
        self.finalize_calls = 0

    def finalize_episode(self) -> None:
        self.finalize_calls += 1


class _FailingFinalizableEnv:
    def __init__(self) -> None:
        self.finalize_calls = 0

    def finalize_episode(self) -> None:
        self.finalize_calls += 1
        raise RuntimeError("finalize failed")


class _PlainEnv:
    pass


def test_env_base_reexports_core_env_types() -> None:
    assert EnvBase.__name__ == "EnvBase"
    assert EnvBaseCfg.__name__ == "EnvBaseCfg"
    assert EnvStepReturn.__name__ == "EnvStepReturn"


def test_episode_finalizable_protocol_accepts_structural_implementation() -> (
    None
):
    assert isinstance(_FinalizableEnv(), EpisodeFinalizableEnvProtocol)


def test_finalize_env_episode_calls_supported_env() -> None:
    env = _FinalizableEnv()

    finalize_env_episode(env)

    assert env.finalize_calls == 1


def test_finalize_env_episode_noops_for_plain_env() -> None:
    finalize_env_episode(_PlainEnv())


def test_finalize_env_episode_swallows_finalize_failure() -> None:
    env = _FailingFinalizableEnv()

    finalize_env_episode(env)

    assert env.finalize_calls == 1
