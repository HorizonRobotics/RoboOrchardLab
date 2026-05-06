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

import pytest

from robo_orchard_lab.envs.state import (
    ENV_STATE_SCOPE_KEY,
    EnvStateScope,
    StatefulEnvMixin,
    require_env_supports_state_scope,
)
from robo_orchard_lab.utils.state import State


class _StatefulEnv(StatefulEnvMixin):
    supported_state_scopes = frozenset({EnvStateScope.POST_RESET})

    def _get_state(self) -> State:
        return State(
            state={ENV_STATE_SCOPE_KEY: EnvStateScope.POST_RESET.value}
        )

    def _set_state(self, state: State) -> None:
        self.loaded_state = state

    def reset_from_state(self, state: State):
        self.load_state(state)
        return {"obs": state.state}, {}


class _NoStateResetEnv(StatefulEnvMixin):
    supported_state_scopes = frozenset({EnvStateScope.POST_RESET})

    def _get_state(self) -> State:
        return State(
            state={ENV_STATE_SCOPE_KEY: EnvStateScope.POST_RESET.value}
        )

    def _set_state(self, state: State) -> None:
        self.loaded_state = state


class _UnsupportedScopeEnv(_StatefulEnv):
    supported_state_scopes = frozenset({EnvStateScope.MID_EPISODE})


def test_get_env_state_scope_accepts_post_reset() -> None:
    state = State(state={ENV_STATE_SCOPE_KEY: "post_reset"})

    assert (
        StatefulEnvMixin.get_env_state_scope(state) is EnvStateScope.POST_RESET
    )


def test_get_env_state_scope_rejects_missing_scope() -> None:
    with pytest.raises(ValueError, match="requires `scope`"):
        StatefulEnvMixin.get_env_state_scope(State(state={}))


def test_get_env_state_scope_rejects_invalid_scope() -> None:
    with pytest.raises(ValueError, match="Unsupported env state scope"):
        StatefulEnvMixin.get_env_state_scope(
            State(state={ENV_STATE_SCOPE_KEY: "bad"})
        )


def test_get_env_state_scope_rejects_non_mapping_state_payload() -> None:
    state = State.model_construct(state=[])

    with pytest.raises(TypeError, match="State.state to be a dict"):
        StatefulEnvMixin.get_env_state_scope(state)


def test_stateful_env_mixin_default_state_hooks_require_override() -> None:
    env = StatefulEnvMixin()

    with pytest.raises(NotImplementedError, match="explicit env State schema"):
        env.get_state()
    with pytest.raises(
        NotImplementedError,
        match="explicit env State restore",
    ):
        env.load_state(State(state={}))


def test_require_env_supports_state_scope_accepts_capability() -> None:
    env = _StatefulEnv()

    assert (
        require_env_supports_state_scope(env, EnvStateScope.POST_RESET) is env
    )


def test_require_env_supports_state_scope_rejects_missing_reset_from_state():
    with pytest.raises(TypeError, match="StatefulEnvProtocol"):
        require_env_supports_state_scope(
            _NoStateResetEnv(),
            EnvStateScope.POST_RESET,
        )


def test_require_env_supports_state_scope_rejects_unsupported_scope() -> None:
    with pytest.raises(ValueError, match="does not support state scope"):
        require_env_supports_state_scope(
            _UnsupportedScopeEnv(),
            EnvStateScope.POST_RESET,
        )
