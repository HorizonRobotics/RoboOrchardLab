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

"""Lab env base contracts and lightweight core env re-exports."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from robo_orchard_core.envs.env_base import EnvBase, EnvBaseCfg, EnvStepReturn
from robo_orchard_core.utils.logging import LoggerManager

if TYPE_CHECKING:
    from robo_orchard_lab.dataset.experimental.mcap.messages import (
        StampedMessage,
    )

logger = LoggerManager().get_child(__name__)

__all__ = [
    "EnvBase",
    "EnvBaseCfg",
    "EnvStepReturn",
    "EnvToMcapProtocol",
    "EpisodeFinalizableEnvProtocol",
    "finalize_env_episode",
]


@runtime_checkable
class EpisodeFinalizableEnvProtocol(Protocol):
    """Env capability for finalizing episode-local resources.

    Implementations must make this method idempotent. Calling it when no
    episode is active, after reset failure, or after a previous finalization
    should be safe. Finalization must not close the environment runtime.
    """

    def finalize_episode(self) -> None:
        """Finalize episode-local resources without closing the environment."""
        ...


@runtime_checkable
class EnvToMcapProtocol(Protocol):
    """Optional env capability for exporting rollout MCAP messages.

    Observation messages must be derived from the latest successful
    ``reset(...)`` or ``step(...)`` observation saved by the env.
    Action sidecars describe the action that is about to be passed to
    ``env.step(action)`` and may read the latest cached observation or
    simulator state. These methods must not step, reset, rebuild, or re-sample
    the simulator.
    """

    def step_index_to_log_time_ns(self, step_index: int) -> int:
        """Map an env rollout step index to an MCAP log-time anchor.

        Args:
            step_index (int): Non-negative logical env step index, where the
                observation returned by ``reset(...)`` is step 0.

        Returns:
            int: MCAP log time in nanoseconds for messages aligned to that
            logical env step.

        Raises:
            ValueError: If ``step_index`` is outside the env-supported range.
        """
        ...

    def get_mcap_obs(
        self,
        *,
        topic_prefix: str = "observation",
        anchor_log_time_ns: int | None = None,
    ) -> dict[str, list["StampedMessage[Any]"]]:
        """Export the latest cached reset/step observation as MCAP messages.

        Implementations return a final-topic map where each key is a complete
        MCAP topic under ``topic_prefix`` and each value is already stamped.
        If ``anchor_log_time_ns`` is provided, all observation messages are
        aligned to that anchor. Otherwise the env derives the anchor from its
        latest logical step state.

        Args:
            topic_prefix (str, optional): Topic prefix for emitted observation
                topics. Default is ``"observation"``.
            anchor_log_time_ns (int | None, optional): Explicit MCAP log-time
                anchor in nanoseconds. Default is None.

        Returns:
            dict[str, list[StampedMessage[Any]]]: Final-topic MCAP messages
            for the latest env observation.

        Raises:
            RuntimeError: If no latest reset/step observation is available.
            ValueError: If ``topic_prefix`` is invalid.
        """
        ...

    def get_mcap_action_sidecars(
        self,
        action: Any,
        *,
        topic_prefix: str = "rollout/next_action",
        anchor_log_time_ns: int | None = None,
        frame_id_suffix: str | None = "next_action",
    ) -> dict[str, list["StampedMessage[Any]"]]:
        """Export sidecars for the action about to be passed to env.step.

        Implementations may use the latest cached observation or simulator
        state to derive sidecar messages for ``action``. They must interpret
        ``action`` exactly as the value that will be passed to
        ``env.step(action)``. ``frame_id_suffix`` lets callers distinguish TF
        child frames for the same env action written on different rollout
        axes, such as next action versus last action at the same log time.

        Args:
            action (Any): Action that the caller is about to pass to
                ``env.step(action)``.
            topic_prefix (str, optional): Topic prefix for emitted action
                sidecar topics. Default is ``"rollout/next_action"``.
            anchor_log_time_ns (int | None, optional): Explicit MCAP log-time
                anchor in nanoseconds. Default is None.
            frame_id_suffix (str | None, optional): Suffix appended to
                frame-bearing child IDs when an env emits TF sidecars. Default
                is ``"next_action"``.

        Returns:
            dict[str, list[StampedMessage[Any]]]: Final-topic MCAP sidecars
            for ``action``. An empty map means this env has no sidecars for
            the action shape or configured action mode.

        Raises:
            RuntimeError: If the env state required to interpret ``action`` is
                unavailable.
            ValueError: If ``topic_prefix`` is invalid.
        """
        ...


def finalize_env_episode(env: object) -> None:
    """Best-effort finalize one env episode when the env supports it."""

    if not isinstance(env, EpisodeFinalizableEnvProtocol):
        return
    try:
        env.finalize_episode()
    except Exception:
        logger.exception("Failed to finalize env episode.")
