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

from unittest.mock import MagicMock

import pytest
from accelerate import Accelerator

from robo_orchard_lab.pipeline.hooks.checkpoint import SaveCheckpointConfig
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs


@pytest.fixture(scope="function")
def mock_accelerator():
    """Fixture to create a mock Accelerator instance."""
    # Create the mock for Accelerator
    accelerator = MagicMock(spec=Accelerator)

    # Mock the project_configuration attribute
    accelerator.project_configuration = MagicMock()
    accelerator.project_configuration.automatic_checkpoint_naming = True

    # Mock the save_state method
    accelerator.save_state = MagicMock(return_value="mock_checkpoint_path")

    # Mock wait_for_everyone method (used for synchronization)
    accelerator.wait_for_everyone = MagicMock()
    accelerator._models = []  # Ensure _models attribute exists

    return accelerator


def test_checkpoint_init_valid():
    """Checkpoint test.

    Test that the DoCheckpoint hook initializes correctly with
    valid parameters.
    """
    hook = SaveCheckpointConfig(
        save_root="checkpoints", save_epoch_freq=1, save_step_freq=10
    )()
    assert hook.save_root == "checkpoints"
    assert hook.save_epoch_freq == 1
    assert hook.save_step_freq == 10


def test_checkpoint_init_invalid():
    """Test that invalid initialization parameters raise ValueError."""
    with pytest.raises(ValueError):
        SaveCheckpointConfig(
            save_root=None, save_epoch_freq=None, save_step_freq=None
        )
    with pytest.raises(ValueError):
        SaveCheckpointConfig(save_root="checkpoints", save_epoch_freq=0)
    with pytest.raises(ValueError):
        SaveCheckpointConfig(save_root="checkpoints", save_step_freq=0)


def test_on_step_end(mocker, mock_accelerator):
    """Test that checkpoint is saved at the correct step interval."""
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.checkpoint.logger"
    )

    hook = SaveCheckpointConfig(save_root="checkpoints", save_step_freq=2)()

    args = PipelineHookArgs(
        accelerator=mock_accelerator,
        global_step_id=1,  # This step should trigger a checkpoint save
        epoch_id=0,
        step_id=0,
    )

    with hook.begin("on_step", args):
        pass

    mock_accelerator.save_state.assert_called_once_with("checkpoints")
    mock_logger.info.assert_called_once_with(
        "Save checkpoint at the end of step 1 to mock_checkpoint_path"
    )


def test_on_epoch_end(mocker, mock_accelerator):
    """Test that checkpoint is saved at the correct epoch interval."""
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.checkpoint.logger"
    )

    hook = SaveCheckpointConfig(save_root="checkpoints", save_epoch_freq=2)()

    args = PipelineHookArgs(
        accelerator=mock_accelerator,
        global_step_id=0,
        epoch_id=1,  # This epoch should trigger a checkpoint save
        step_id=0,
    )

    with hook.begin("on_epoch", args):
        pass
    mock_accelerator.save_state.assert_called_once_with("checkpoints")
    mock_logger.info.assert_called_once_with(
        "Save checkpoint at the end of epoch 1 to mock_checkpoint_path"
    )


def test_skip_checkpoint_on_step(mocker, mock_accelerator):
    """Checkpoint test.

    Test that checkpoint is skipped when the step does not match the interval.
    """
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.checkpoint.logger"
    )

    hook = SaveCheckpointConfig(save_root="checkpoints", save_step_freq=5)()

    args = PipelineHookArgs(
        accelerator=mock_accelerator,
        global_step_id=3,  # This step should not trigger a checkpoint save
        epoch_id=0,
        step_id=0,
    )

    with hook.begin("on_step", args):
        pass
    mock_accelerator.save_state.assert_not_called()
    mock_logger.info.assert_not_called()


def test_skip_checkpoint_on_epoch(mocker, mock_accelerator):
    """CheckPoint test.

    Test that checkpoint is skipped when the epoch does not match the interval.
    """
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.checkpoint.logger"
    )

    hook = SaveCheckpointConfig(save_root="checkpoints", save_epoch_freq=3)()

    args = PipelineHookArgs(
        accelerator=mock_accelerator,
        global_step_id=0,
        epoch_id=1,  # This epoch should not trigger a checkpoint save
        step_id=0,
    )

    with hook.begin("on_epoch", args):
        pass
    mock_accelerator.save_state.assert_not_called()
    mock_logger.info.assert_not_called()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
