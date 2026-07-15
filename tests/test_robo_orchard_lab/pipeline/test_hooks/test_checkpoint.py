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

import os
from unittest.mock import MagicMock

import pytest
from accelerate import Accelerator

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)
from robo_orchard_lab.pipeline.hooks.checkpoint import (
    SaveCheckpointConfig,
    SaveModelConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs


class CheckpointDummyModel(ModelMixin):
    def __init__(self, cfg: "CheckpointDummyModelCfg"):
        super().__init__(cfg)


class CheckpointDummyModelCfg(TorchModuleCfg[CheckpointDummyModel]):
    class_type: ClassType_co[CheckpointDummyModel] = CheckpointDummyModel


@pytest.fixture(scope="function")
def mock_accelerator():
    """Fixture to create a mock Accelerator instance."""
    # Create the mock for Accelerator
    accelerator = MagicMock(spec=Accelerator)
    accelerator.is_main_process = True
    accelerator.is_local_main_process = True

    # Mock the project_configuration attribute
    accelerator.project_configuration = MagicMock()
    accelerator.project_configuration.automatic_checkpoint_naming = True
    accelerator.project_configuration.save_on_each_node = False
    accelerator.project_dir = "mock_project"

    # Mock the save_state method
    accelerator.save_state = MagicMock(return_value="mock_checkpoint_path")

    # Mock wait_for_everyone method (used for synchronization)
    accelerator.wait_for_everyone = MagicMock()
    accelerator._models = []  # Ensure _models attribute exists

    return accelerator


def _make_mock_accelerator(
    tmp_path,
    *,
    is_main_process: bool = True,
    is_local_main_process: bool = True,
    save_on_each_node: bool = False,
) -> MagicMock:
    accelerator = MagicMock(spec=Accelerator)
    accelerator.is_main_process = is_main_process
    accelerator.is_local_main_process = is_local_main_process
    accelerator.project_configuration = MagicMock()
    accelerator.project_configuration.automatic_checkpoint_naming = False
    accelerator.project_configuration.save_on_each_node = save_on_each_node
    accelerator.project_dir = str(tmp_path)
    accelerator.wait_for_everyone = MagicMock()
    accelerator.save_model = MagicMock()
    accelerator._models = []
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
        global_step_id=2,  # This committed step should trigger a save
        epoch_id=0,
        step_id=2,
        is_optimizer_step_committed=True,
    )

    with hook.begin("on_optimizer_step", args):
        pass

    mock_accelerator.save_state.assert_called_once_with("checkpoints")
    mock_logger.info.assert_called_once_with(
        "Save checkpoint at the end of step 2 to mock_checkpoint_path"
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
        step_id=3,
        is_optimizer_step_committed=True,
    )

    with hook.begin("on_optimizer_step", args):
        pass
    mock_accelerator.save_state.assert_not_called()
    mock_logger.info.assert_not_called()


def test_checkpoint_skips_uncommitted_optimizer_boundary(
    mocker,
    mock_accelerator,
):
    """Skipped optimizer boundaries should not save step checkpoints."""
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.checkpoint.logger"
    )
    hook = SaveCheckpointConfig(save_root="checkpoints", save_step_freq=1)()
    args = PipelineHookArgs(
        accelerator=mock_accelerator,
        global_step_id=0,
        epoch_id=0,
        step_id=0,
        is_optimizer_step_committed=False,
    )

    with hook.begin("on_optimizer_step", args):
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


def test_checkpoint_skips_loop_end_save_when_body_raises(
    mock_accelerator,
):
    hook = SaveCheckpointConfig(save_root="checkpoints")()
    args = PipelineHookArgs(accelerator=mock_accelerator)

    with pytest.raises(RuntimeError, match="body failed") as exc_info:
        with hook.begin("on_loop", args):
            raise RuntimeError("body failed")

    assert args.exception is exc_info.value
    mock_accelerator.wait_for_everyone.assert_not_called()
    mock_accelerator.save_state.assert_not_called()


def test_save_checkpoint_save_model_runs_on_non_main(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    accelerator = _make_mock_accelerator(
        tmp_path,
        is_main_process=False,
        is_local_main_process=False,
    )
    accelerator.save_state = MagicMock(return_value=str(checkpoint_dir))
    accelerator._models = [object()]

    hook = SaveCheckpointConfig(save_root="checkpoints", save_step_freq=1)()

    hook._save_state(accelerator)

    accelerator.save_model.assert_called_once()


def test_save_checkpoint_waits_after_model_export(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    accelerator = _make_mock_accelerator(tmp_path)
    accelerator.save_state = MagicMock(return_value=str(checkpoint_dir))
    accelerator._models = [object()]

    def save_model_artifact(_model, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "model.safetensors"), "wb"):
            pass

    accelerator.save_model.side_effect = save_model_artifact
    hook = SaveCheckpointConfig(save_root="checkpoints", save_step_freq=1)()

    hook._save_state(accelerator)

    assert accelerator.wait_for_everyone.call_count == 2


def test_save_checkpoint_skips_config_copy_without_model_artifact(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.config.json").write_text("{}", encoding="utf-8")
    accelerator = _make_mock_accelerator(tmp_path)
    accelerator.save_state = MagicMock(return_value=str(checkpoint_dir))
    accelerator._models = [object()]

    def save_empty_model(_model, save_directory):
        os.makedirs(save_directory, exist_ok=True)

    accelerator.save_model.side_effect = save_empty_model
    hook = SaveCheckpointConfig(save_root="checkpoints", save_step_freq=1)()

    hook._save_state(accelerator)

    copied_config_path = checkpoint_dir / "model" / "model.config.json"
    assert not copied_config_path.exists()


def test_save_checkpoint_check_warning_only_on_global_main(mocker, tmp_path):
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.checkpoint.logger"
    )
    accelerator = _make_mock_accelerator(
        tmp_path,
        is_main_process=False,
        is_local_main_process=False,
    )
    accelerator.project_configuration.automatic_checkpoint_naming = True
    hook = SaveCheckpointConfig(save_root="checkpoints", save_step_freq=1)()

    hook._check(accelerator)

    mock_logger.warning.assert_not_called()


def test_save_model_hook_calls_save_model_on_non_main(tmp_path):
    accelerator = _make_mock_accelerator(
        tmp_path,
        is_main_process=False,
        is_local_main_process=False,
    )
    accelerator._models = [object()]
    hook = SaveModelConfig(save_root=str(tmp_path), save_step_freq=1)()
    args = PipelineHookArgs(
        accelerator=accelerator,
        model=object(),
        global_step_id=0,
        epoch_id=0,
        step_id=0,
    )

    hook._save_model(args, step_id=0, epoch_id=0)

    accelerator.save_model.assert_called_once()


def test_save_model_hook_on_step_runs_save_on_non_main(mocker, tmp_path):
    accelerator = _make_mock_accelerator(
        tmp_path,
        is_main_process=False,
        is_local_main_process=False,
    )
    hook = SaveModelConfig(save_root=str(tmp_path), save_step_freq=1)()
    save_model = mocker.patch.object(hook, "_save_model", return_value="path")
    args = PipelineHookArgs(
        accelerator=accelerator,
        model=object(),
        global_step_id=1,
        epoch_id=0,
        step_id=1,
        is_optimizer_step_committed=True,
    )

    with hook.begin("on_optimizer_step", args):
        pass

    save_model.assert_called_once_with(args, step_id=1, epoch_id=0)


def test_save_model_hook_skips_config_without_model_artifact(tmp_path):
    model = CheckpointDummyModel(CheckpointDummyModelCfg())
    accelerator = _make_mock_accelerator(tmp_path)
    accelerator._models = [model]

    def save_empty_model(_model, save_directory):
        os.makedirs(save_directory, exist_ok=True)

    accelerator.save_model.side_effect = save_empty_model
    hook = SaveModelConfig(save_root=str(tmp_path), save_step_freq=1)()
    args = PipelineHookArgs(
        accelerator=accelerator,
        model=model,
        global_step_id=0,
        epoch_id=0,
        step_id=0,
    )

    hook._save_model(args, step_id=0, epoch_id=0)

    config_path = tmp_path / "model_epoch_0_step_0" / "0" / "model.config.json"
    assert not config_path.exists()


def test_save_model_config_warning_only_on_global_main(mocker, tmp_path):
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.checkpoint.logger"
    )
    accelerator = _make_mock_accelerator(
        tmp_path,
        is_main_process=False,
        is_local_main_process=False,
    )
    accelerator.project_configuration.automatic_checkpoint_naming = True
    cfg = SaveModelConfig(save_root="models", save_step_freq=1)

    cfg.get_save_root(accelerator)

    mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
