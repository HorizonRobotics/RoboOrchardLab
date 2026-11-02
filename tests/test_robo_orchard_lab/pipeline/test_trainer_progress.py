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

"""Test Trainer micro-step and optimizer-step progress state."""

import pytest

import robo_orchard_lab.pipeline.training.hook_based_trainer as trainer_module
from robo_orchard_lab.pipeline.hooks.mixin import (
    MicroStepProgressState,
)


def test_trainer_progress_state_tracks_micro_and_optimizer_steps():
    """Progress state separates micro steps from optimizer steps."""
    progress = trainer_module.TrainerProgressState()

    first_micro_step = progress.preview_next_micro_step()
    assert first_micro_step == MicroStepProgressState(
        epoch_step_id=0,
        global_step_id=0,
        index_in_optimizer_step=1,
        last_optimizer_step_size=0,
    )

    progress.commit_micro_step(first_micro_step)
    assert progress.step_id == 0
    assert progress.global_step_id == 0
    assert progress.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=1,
        last_optimizer_step_size=0,
    )

    progress.commit_optimizer_step(micro_steps=1)
    assert progress.step_id == 1
    assert progress.global_step_id == 1
    assert progress.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=0,
        last_optimizer_step_size=1,
    )

    state_dict = progress.state_dict()
    assert state_dict["schema_version"] == 3
    restored = trainer_module.TrainerProgressState()
    restored.load_state_dict(state_dict)

    assert restored == progress
    assert isinstance(restored.micro_step, MicroStepProgressState)

    progress.update_epoch()
    assert progress.epoch_id == 1
    assert progress.step_id == 0
    assert progress.global_step_id == 1
    assert progress.micro_step.epoch_step_id == 0
    assert progress.micro_step.global_step_id == 1


def test_trainer_progress_state_rejects_legacy_checkpoint_payload():
    """Legacy dataloader-step checkpoints should fail with a clear error."""

    progress = trainer_module.TrainerProgressState()

    with pytest.raises(RuntimeError, match="Legacy TrainerProgressState"):
        progress.load_state_dict(
            {
                "epoch_id": 0,
                "step_id": 3,
                "global_step_id": 3,
            }
        )


def test_trainer_progress_state_rejects_unsupported_schema_version():
    """Unknown progress schemas should fail before partial state mutation."""

    progress = trainer_module.TrainerProgressState()

    with pytest.raises(RuntimeError, match="Unsupported.*schema_version"):
        progress.load_state_dict(
            {
                "schema_version": 999,
                "epoch_id": 0,
                "step_id": 0,
                "global_step_id": 0,
                "micro_step": {
                    "epoch_step_id": 0,
                    "global_step_id": 0,
                    "index_in_optimizer_step": 0,
                    "last_optimizer_step_size": 0,
                },
            }
        )

    assert progress == trainer_module.TrainerProgressState()


def test_trainer_progress_rejects_v2_and_preserves_cross_epoch_window():
    """Reject old counters and preserve an unfinished window."""
    progress = trainer_module.TrainerProgressState()
    with pytest.raises(RuntimeError, match="Unsupported.*schema_version"):
        progress.load_state_dict({"schema_version": 2})

    progress.commit_micro_step(progress.preview_next_micro_step())
    progress.update_epoch()
    assert progress.epoch_id == 1
    assert progress.micro_step.epoch_step_id == 0
    assert progress.micro_step.index_in_optimizer_step == 1


def test_trainer_progress_state_resets_optimizer_step_window():
    """Progress state can discard an uncommitted accumulation window."""
    progress = trainer_module.TrainerProgressState()
    current_micro_step = progress.preview_next_micro_step()
    progress.commit_micro_step(current_micro_step)

    progress.reset_optimizer_step_window()

    assert progress.global_step_id == 0
    assert progress.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=0,
        last_optimizer_step_size=0,
    )
