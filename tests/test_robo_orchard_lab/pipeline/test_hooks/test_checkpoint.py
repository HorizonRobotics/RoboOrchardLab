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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from accelerate.utils import DistributedType

import robo_orchard_lab.pipeline.hooks.checkpoint as checkpoint_module
from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)
from robo_orchard_lab.pipeline.hooks.checkpoint import (
    SaveCheckpointConfig,
    SaveModelConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    MicroStepProgressState,
    PipelineHookArgs,
)
from robo_orchard_lab.pipeline.training._progress import TrainerProgressState


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


@pytest.fixture
def real_checkpoint(tmp_path, monkeypatch):
    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator.prepare(model, optimizer)
    progress = TrainerProgressState()
    accelerator.register_for_checkpointing(progress)
    root = tmp_path / "checkpoints"
    hook = SaveCheckpointConfig(save_root=str(root), save_model=False)()
    return accelerator, hook, root


def test_native_save_retains_rank_progress(real_checkpoint):
    accelerator, hook, root = real_checkpoint
    destination = Path(hook._save_state(accelerator))
    assert destination == root
    state = torch.load(root / "custom_checkpoint_0.pkl", weights_only=False)
    assert state["schema_version"] == 3
    assert "checkpoint_complete" not in state
    assert state["rank_states"] == []
    assert (root / "random_states_0.pkl").is_file()


@pytest.mark.parametrize("boundary", ["step", "epoch", "deferred", "loop"])
def test_checkpoint_records_default_sampler_epoch_origin(
    real_checkpoint, boundary
):
    from robo_orchard_lab.utils.accelerate import (
        _bind_managed_map_generators,
        _configure_prepared_loader_recovery,
    )

    accelerator, hook, root = real_checkpoint
    source = torch.utils.data.DataLoader(range(4), batch_size=1, shuffle=True)
    _bind_managed_map_generators(source)
    expected = source.sampler.generator.get_state().clone()
    prepared = accelerator.prepare(source)
    _configure_prepared_loader_recovery(accelerator, prepared, source)
    args = PipelineHookArgs(
        accelerator=accelerator,
        dataloader=prepared,
        global_step_id=1,
    )
    if boundary == "step":
        hook.save_step_freq = 1
        hook._on_step_end(args)
    elif boundary == "loop":
        hook._on_loop_end(args)
    else:
        pending = []
        if boundary == "deferred":
            args._defer_epoch_checkpoint = pending.append
        hook._on_epoch_end(args)
        if boundary == "deferred":
            assert not root.exists()
            assert len(pending) == 1
            pending[0]()

    state = torch.load(root / "custom_checkpoint_0.pkl", weights_only=False)
    assert torch.equal(state["rank_states"][0]["sampler_state"], expected)
    assert not hasattr(accelerator._custom_objects[0], "_data_loader")


def test_standalone_hook_accepts_unprepared_dataloader(real_checkpoint):
    accelerator, hook, root = real_checkpoint
    args = PipelineHookArgs(
        accelerator=accelerator,
        dataloader=torch.utils.data.DataLoader(range(4), batch_size=1),
    )
    hook._on_epoch_end(args)
    state = torch.load(root / "custom_checkpoint_0.pkl", weights_only=False)
    assert state["rank_states"] == []


def test_failed_save_leaves_partial_directory(real_checkpoint, monkeypatch):
    accelerator, hook, root = real_checkpoint
    error = OSError("injected save failure")

    def fail_save(directory):
        Path(directory).mkdir(parents=True)
        (Path(directory) / "partial.bin").write_bytes(b"partial")
        raise error

    monkeypatch.setattr(accelerator, "save_state", fail_save)
    with pytest.raises(OSError) as caught:
        hook._save_state(accelerator)
    assert caught.value is error
    assert (root / "partial.bin").read_bytes() == b"partial"


def test_unfinished_window_rejected_before_native_save(
    real_checkpoint, monkeypatch
):
    accelerator, hook, _ = real_checkpoint
    progress = accelerator._custom_objects[0]
    progress.micro_step.index_in_optimizer_step = 1
    save = MagicMock()
    monkeypatch.setattr(accelerator, "save_state", save)
    with pytest.raises(RuntimeError, match="unfinished window"):
        hook._save_state(accelerator)
    save.assert_not_called()


def test_custom_save_hook_preserves_trainer_progress(real_checkpoint):
    accelerator, hook, root = real_checkpoint
    seen = []
    accelerator.register_save_state_pre_hook(
        lambda models, weights, output_dir: seen.append(output_dir)
    )
    destination = hook._save_state(accelerator)
    assert Path(destination) == root
    assert seen == [str(root)]
    state = torch.load(root / "custom_checkpoint_0.pkl", weights_only=False)
    assert state["rank_states"] == []


def test_native_model_export_failure_preserves_partial_save(
    real_checkpoint, monkeypatch
):
    accelerator, _, root = real_checkpoint
    hook = SaveCheckpointConfig(save_root=str(root), save_model=True)()
    error = OSError("injected model export failure")

    def fail_export(*args, **kwargs):
        raise error

    monkeypatch.setattr(accelerator, "save_model", fail_export)
    with pytest.raises(OSError) as caught:
        hook._save_state(accelerator)
    assert caught.value is error
    assert root.is_dir()
    assert (root / "custom_checkpoint_0.pkl").is_file()


@pytest.mark.parametrize("writer", [False, True], ids=["nonmain", "main"])
@pytest.mark.parametrize("model_count", [1, 2])
def test_native_model_export_preserves_config_policy(
    tmp_path, monkeypatch, writer, model_count
):
    accelerator = _make_mock_accelerator(tmp_path, is_main_process=writer)
    accelerator._models = [object() for _ in range(model_count)]
    monkeypatch.setattr(
        checkpoint_module,
        "_should_write_like_accelerate_save",
        lambda accelerator: writer,
    )
    monkeypatch.setattr(
        checkpoint_module, "_has_model_save_artifact", lambda path: True
    )
    accelerator.save_model.side_effect = lambda model, directory: Path(
        directory
    ).mkdir()
    prefixes = (
        ["model"] if model_count == 1 else [f"model_{i}" for i in range(2)]
    )
    for prefix in prefixes:
        (tmp_path / f"{prefix}.config.json").write_text("{}", encoding="utf-8")
    checkpoint_module._export_checkpoint_models(accelerator, str(tmp_path))
    assert accelerator.save_model.call_count == model_count
    for prefix in prefixes:
        assert (
            tmp_path / prefix / f"{prefix}.config.json"
        ).is_file() is writer


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


@pytest.mark.parametrize(
    "backend", [DistributedType.DEEPSPEED, DistributedType.FSDP]
)
def test_native_only_backend_preserves_legacy_save(mock_accelerator, backend):
    progress = TrainerProgressState()
    mock_accelerator._custom_objects = [progress]
    mock_accelerator.distributed_type = backend
    hook = SaveCheckpointConfig(save_root="checkpoints", save_model=False)()

    assert hook._save_state(mock_accelerator) == "mock_checkpoint_path"
    mock_accelerator.save_state.assert_called_once_with("checkpoints")
    assert progress.rank_states == []


def test_unverified_device_preserves_native_save(mock_accelerator):
    progress = TrainerProgressState()
    mock_accelerator._custom_objects = [progress]
    mock_accelerator.distributed_type = DistributedType.NO
    mock_accelerator.device = SimpleNamespace(type="mps")
    hook = SaveCheckpointConfig(save_root="checkpoints", save_model=False)()

    assert hook._save_state(mock_accelerator) == "mock_checkpoint_path"
    assert progress.rank_states == []


@pytest.mark.parametrize(
    "backend",
    [DistributedType.XLA, DistributedType.DEEPSPEED, DistributedType.FSDP],
)
@pytest.mark.parametrize("outcome", ["save", "pre_error", "save_error"])
def test_unsupported_backend_saves_without_object_collective(
    mock_accelerator, monkeypatch, backend, outcome
):
    from accelerate.utils import operations

    # Exercise Accelerate's real XLA dispatch, which rejects object gathering.
    monkeypatch.setattr(
        operations,
        "PartialState",
        lambda: SimpleNamespace(distributed_type=DistributedType.XLA),
    )
    with pytest.raises(NotImplementedError, match="TPU"):
        operations.gather_object([{"probe": True}])
    progress = TrainerProgressState(rank_states=[{"rank": 0}])
    mock_accelerator._custom_objects = [progress]
    mock_accelerator.distributed_type = backend
    hook = SaveCheckpointConfig(save_root="checkpoints", save_model=False)()
    error = OSError("native failure")
    if outcome == "save_error":
        mock_accelerator.save_state.side_effect = error
    if outcome == "pre_error":
        monkeypatch.setattr(hook, "_check", MagicMock(side_effect=error))
        with pytest.raises(OSError) as caught:
            hook._save_state(mock_accelerator)
        assert caught.value is error
        mock_accelerator.save_state.assert_not_called()
    elif outcome == "save_error":
        with pytest.raises(OSError) as caught:
            hook._save_state(mock_accelerator)
        assert caught.value is error
    else:
        assert hook._save_state(mock_accelerator) == "mock_checkpoint_path"
        mock_accelerator.save_state.assert_called_once_with("checkpoints")
        assert progress.rank_states == []


def test_rank_local_loader_config_uses_save_state_gather(
    real_checkpoint, monkeypatch
):
    accelerator, hook, _ = real_checkpoint
    accelerator.project_configuration.save_on_each_node = True
    original_gather = checkpoint_module.gather_object
    gather = MagicMock(wraps=original_gather)
    monkeypatch.setattr(checkpoint_module, "gather_object", gather)
    hook._save_state(accelerator)
    gather.assert_called_once()
    assert "node-local" in gather.call_args.args[0][0]["reason"]


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


@pytest.mark.parametrize("channel", ["on_epoch", "on_loop"])
def test_unfinished_window_skips_epoch_and_loop_saves(
    mock_accelerator, caplog, channel
):
    hook = SaveCheckpointConfig(
        save_root="checkpoints", save_epoch_freq=1, save_when_loop_end=True
    )()
    args = PipelineHookArgs(
        accelerator=mock_accelerator,
        micro_step=MicroStepProgressState(index_in_optimizer_step=1),
    )
    with hook.begin(channel, args):
        pass
    mock_accelerator.save_state.assert_not_called()
    assert "gradient accumulation is unfinished" in caplog.text


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


def test_checkpoint_saves_after_skipped_optimizer_update(
    mocker,
    mock_accelerator,
):
    """An AMP skip remains a completed window for checkpoint cadence."""
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.checkpoint.logger"
    )
    hook = SaveCheckpointConfig(save_root="checkpoints", save_step_freq=1)()
    args = PipelineHookArgs(
        accelerator=mock_accelerator,
        global_step_id=1,
        epoch_id=0,
        step_id=1,
        is_optimizer_step_committed=False,
    )

    with hook.begin("on_optimizer_step", args):
        pass

    mock_accelerator.save_state.assert_called_once()
    mock_logger.info.assert_called_once()


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


def test_save_checkpoint_delegates_synchronization_to_accelerate(tmp_path):
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

    accelerator.wait_for_everyone.assert_not_called()


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


def test_checkpoint_recaptures_data_when_current_capability_recovers(
    real_checkpoint,
):
    from robo_orchard_lab.utils.accelerate import prepare_data_loader

    accelerator, hook, root = real_checkpoint
    source = torch.utils.data.DataLoader(range(4), batch_size=1, shuffle=True)
    loader = prepare_data_loader(accelerator, source)
    progress = accelerator._custom_objects[0]
    for node_local in (True, False, True, False):
        accelerator.project_configuration.save_on_each_node = node_local
        hook._save_state(accelerator, loader)
        saved = torch.load(
            root / "custom_checkpoint_0.pkl", weights_only=False
        )
        if node_local:
            assert saved["rank_states"] == progress.rank_states == []
        else:
            assert len(saved["rank_states"]) == 1
            state = saved["rank_states"][0]
            assert state["rank"] == 0
            assert state["loader_state"] is None
            assert state["seed"] == 42
            assert isinstance(state["sampler_state"], torch.Tensor)


@pytest.mark.parametrize("supported", [False, True])
def test_checkpoint_multiple_progress_owners_keep_backend_boundary(
    mock_accelerator, monkeypatch, supported
):
    owners = [
        TrainerProgressState(rank_states=[{"rank": rank}]) for rank in range(2)
    ]
    mock_accelerator._custom_objects = owners
    mock_accelerator.distributed_type = (
        DistributedType.NO if supported else DistributedType.DEEPSPEED
    )
    gather = MagicMock(side_effect=AssertionError("unexpected gather"))
    monkeypatch.setattr(checkpoint_module, "gather_object", gather)
    hook = SaveCheckpointConfig(save_root="checkpoints", save_model=False)()
    if supported:
        with pytest.raises(RuntimeError, match="one trainer progress owner"):
            hook._save_state(mock_accelerator)
        mock_accelerator.save_state.assert_not_called()
    else:
        assert hook._save_state(mock_accelerator) == "mock_checkpoint_path"
        assert owners[0].rank_states == owners[1].rank_states == []
        assert owners[0].rank_states is not owners[1].rank_states
        mock_accelerator.save_state.assert_called_once_with("checkpoints")
    gather.assert_not_called()
