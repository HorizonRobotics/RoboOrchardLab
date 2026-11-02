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
from dataclasses import dataclass, field, replace
from typing import Any

from robo_orchard_lab.pipeline.hooks.mixin import MicroStepProgressState
from robo_orchard_lab.utils.huggingface import AcceleratorState

__all__ = ["TrainerProgressState"]


@dataclass
class TrainerProgressState(AcceleratorState):
    """Checkpointable training progress owned by ``HookBasedTrainer``.

    ``step_id`` and ``global_step_id`` track completed accumulation windows,
    including windows whose optimizer update was skipped by mixed precision.
    ``micro_step`` tracks dataloader-batch progress and the active gradient
    accumulation window.
    """

    schema_version: int = 3
    """Persisted progress schema version."""
    epoch_id: int = 0
    """The current epoch. Starts from 0."""
    step_id: int = 0
    """The number of completed accumulation windows in the current epoch."""
    global_step_id: int = 0
    """The total number of completed accumulation windows across epochs."""
    micro_step: MicroStepProgressState = field(
        default_factory=MicroStepProgressState
    )
    """The dataloader-batch micro-step progress."""
    rank_states: list[dict[str, Any]] = field(default_factory=list)
    """Ordered per-rank data and accelerator state captured at checkpoint."""

    def update_step(self) -> None:
        """Increments the completed-window counters by 1."""
        self.commit_optimizer_step(micro_steps=1)

    def preview_next_micro_step(self) -> MicroStepProgressState:
        """Returns the next micro-step snapshot without mutating state."""
        return MicroStepProgressState(
            epoch_step_id=self.micro_step.epoch_step_id,
            global_step_id=self.micro_step.global_step_id,
            index_in_optimizer_step=(
                self.micro_step.index_in_optimizer_step + 1
            ),
            last_optimizer_step_size=(
                self.micro_step.last_optimizer_step_size
            ),
        )

    def commit_micro_step(
        self,
        current_micro_step: MicroStepProgressState,
    ) -> None:
        """Commits a successfully processed micro step."""
        self.rank_states = []
        self.micro_step.epoch_step_id = current_micro_step.epoch_step_id + 1
        self.micro_step.global_step_id = current_micro_step.global_step_id + 1
        self.micro_step.index_in_optimizer_step = (
            current_micro_step.index_in_optimizer_step
        )

    def commit_optimizer_step(self, *, micro_steps: int) -> None:
        """Commit one completed accumulation window, including an AMP skip."""
        if micro_steps < 1:
            raise ValueError("micro_steps must be greater than 0.")
        self.rank_states = []
        self.step_id += 1
        self.global_step_id += 1
        self.micro_step.last_optimizer_step_size = micro_steps
        self.micro_step.index_in_optimizer_step = 0

    def reset_optimizer_step_window(self) -> None:
        """Resets the active accumulation window without committing a step."""
        self.rank_states = []
        self.micro_step.index_in_optimizer_step = 0

    def update_epoch(self) -> None:
        """Increments the epoch and resets epoch-local progress."""
        self.rank_states = []
        self.epoch_id += 1
        self.step_id = 0
        self.micro_step.epoch_step_id = 0
        # With sync_with_dataloader=False, a window can span epochs. Preserve
        # that position; trainer/data checkpoints reject an unfinished window.

    def is_training_end(
        self, max_step: int | None, max_epoch: int | None
    ) -> bool:
        """Checks if the training loop should end based on current state.

        Args:
            max_step (int | None): The maximum completed windows allowed.
            max_epoch (int | None): The maximum epochs allowed.

        Returns:
            bool: True if the training loop should end, False otherwise.
        """
        if max_step is not None and self.global_step_id >= max_step:
            return True
        if max_epoch is not None and self.epoch_id >= max_epoch:
            return True
        return False

    def sync_pipeline_hook_arg(self, hook_args: Any) -> None:
        """Synchronizes committed progress into hook arguments.

        Args:
            hook_args (Any): The hook arguments to update.
        """
        hook_args.epoch_id = self.epoch_id
        hook_args.step_id = self.step_id
        hook_args.global_step_id = self.global_step_id
        hook_args.micro_step = replace(self.micro_step)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Loads progress state and restores nested micro-step state.

        Earlier checkpoints counted successful updates or dataloader steps,
        not completed windows, and cannot be loaded as current progress.
        """
        state = dict(state)
        schema_version = state.get("schema_version")
        if schema_version is None:
            raise RuntimeError(
                "Legacy TrainerProgressState checkpoint detected: older "
                "progress counters cannot be interpreted as completed "
                "accumulation windows. Restart from weights only."
            )
        elif schema_version != self.schema_version:
            raise RuntimeError(
                "Unsupported TrainerProgressState schema_version="
                f"{schema_version!r}; expected {self.schema_version}."
            )
        micro_step = state.get("micro_step")
        if isinstance(micro_step, dict):
            state["micro_step"] = MicroStepProgressState(**micro_step)
        super().load_state_dict(state)
