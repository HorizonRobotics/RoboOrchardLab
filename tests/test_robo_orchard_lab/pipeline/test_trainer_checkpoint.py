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

"""Test checkpoint boundaries and epoch checkpoint lifecycle."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader as RODataLoader,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHooks,
)
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
)
from test_robo_orchard_lab.pipeline._trainer_test_helpers import (
    DummyBatchProcessor,
    SimpleModel,
    _epoch_checkpoint_trainer,
    dummy_trainer as dummy_trainer,
)


@pytest.mark.parametrize("failure", ["processor", "after_hook"])
def test_data_checkpoint_commits_only_at_complete_optimizer_boundary(
    dummy_trainer, failure
):
    """Neither processing nor step-hook failure commits data recovery."""
    from robo_orchard_lab.utils.accelerate import (
        _capture_prepared_loader_recovery,
        _configure_prepared_loader_recovery,
    )

    trainer = dummy_trainer
    source = RODataLoader(trainer.dataloader.dataset, batch_size=2)
    _configure_prepared_loader_recovery(
        trainer.accelerator, trainer.dataloader, source
    )
    before = _capture_prepared_loader_recovery(trainer.dataloader)
    processor = MagicMock()
    if failure == "processor":
        processor.side_effect = RuntimeError("processor failed")
    trainer.batch_processor = processor
    if failure == "after_hook":
        hook = PipelineHooks()

        def fail_after(args):
            raise RuntimeError("after hook failed")

        hook.register_hook(
            "on_step", HookContext.from_callable(after=fail_after)
        )
        trainer.hooks += hook
    with pytest.raises(RuntimeError, match="failed"):
        trainer()
    assert _capture_prepared_loader_recovery(trainer.dataloader) == before
    assert trainer.trainer_progress_state.micro_step.global_step_id == (
        failure == "after_hook"
    )


@pytest.mark.parametrize("custom_sampler", [False, True])
def test_checkpoint_accepts_unowned_generator_attributes(
    tmp_path, monkeypatch, custom_sampler
):
    """Checkpointing must not claim arbitrary dataset or sampler RNGs."""
    import random

    from accelerate.state import AcceleratorState, GradientState, PartialState

    from robo_orchard_lab.pipeline.hooks.checkpoint import SaveCheckpointConfig

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})

    class Data(torch.utils.data.Dataset):
        generator = object()

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return torch.tensor([0.5, 0.1 * index])

    class Sampler(torch.utils.data.Sampler):
        def __init__(self):
            self.generator = random.Random(3)

        def __iter__(self):
            return iter(self.generator.sample(range(4), k=4))

        def __len__(self):
            return 4

    dataset = Data()
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=Sampler() if custom_sampler else None,
    )
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    trainer = HookBasedTrainer(
        accelerator=Accelerator(cpu=True),
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=StepLR(optimizer, step_size=1),
        batch_processor=DummyBatchProcessor(),
        max_step=1,
        hooks=[
            SaveCheckpointConfig(
                save_root=str(tmp_path),
                save_epoch_freq=None,
                save_step_freq=1,
                save_when_loop_end=False,
                save_model=False,
            )
        ],
    )
    trainer()
    assert trainer.trainer_progress_state.global_step_id == 1
    checkpoint = tmp_path / "custom_checkpoint_0.pkl"
    progress = torch.load(checkpoint, weights_only=False)
    assert progress["schema_version"] == 3
    assert progress["rank_states"][0]["sampler_state"] is None


def test_accumulation_short_epoch_tail_is_committed_and_resumable(
    tmp_path, monkeypatch
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    root = tmp_path / "checkpoints"
    batches = []
    windows = []
    trainer = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=batches,
        max_epoch=1,
        accumulation_steps=4,
        dataset_size=5,
    )
    observer = PipelineHooks()
    observer.register_hook(
        "on_optimizer_step",
        HookContext.from_callable(
            after=lambda args: windows.append(
                args.micro_step.last_optimizer_step_size
            )
        ),
    )
    trainer.hooks += observer
    trainer()
    assert batches == [0, 1, 2, 3, 4]
    assert windows == [4, 1]
    assert trainer.trainer_progress_state.global_step_id == 2

    resumed_batches = []
    resumed = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=resumed_batches,
        resume=root / "checkpoint_0",
        max_epoch=2,
        accumulation_steps=4,
        dataset_size=5,
        save_epoch_freq=None,
    )
    resumed()
    assert resumed_batches == [0, 1, 2, 3, 4]
    assert resumed.trainer_progress_state.global_step_id == 4


@pytest.mark.parametrize("lab_loader", [False, True], ids=["map", "lab"])
def test_epoch_checkpoint_resumes_after_completed_epoch(
    tmp_path, monkeypatch, caplog, lab_loader
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    root = tmp_path / "checkpoints"
    events, first_batches = [], []
    first = _epoch_checkpoint_trainer(
        root,
        lab_loader=lab_loader,
        events=events,
        batches=first_batches,
        max_epoch=1,
    )
    first()
    assert events == [("eval", None), ("after", 0)]
    assert first_batches == [0, 1, 2, 3]
    assert "aggregation state is not restored" not in caplog.text
    checkpoint = root / "checkpoint_0"
    saved = torch.load(
        checkpoint / "custom_checkpoint_0.pkl", weights_only=False
    )
    assert saved["schema_version"] == 3
    assert saved["epoch_id"] == 1
    assert saved["step_id"] == 0
    assert saved["micro_step"]["epoch_step_id"] == 0
    if lab_loader:
        loader_state = saved["rank_states"][0]["loader_state"]
        assert loader_state["epoch"] == 1
        assert loader_state["delivered_batches"] == 0

    resumed_events, resumed_batches = [], []
    resumed = _epoch_checkpoint_trainer(
        root,
        lab_loader=lab_loader,
        events=resumed_events,
        batches=resumed_batches,
        resume=checkpoint,
        max_epoch=2,
    )
    assert caplog.text.count("aggregation state is not restored") == 1
    resumed()
    assert resumed_events == [("eval", None), ("after", 1)]
    assert resumed_batches == [0, 1, 2, 3]
    assert resumed.trainer_progress_state.epoch_id == 2


@pytest.mark.parametrize("boundary", ["step", "epoch"])
def test_checkpoint_uses_trainer_loader_when_hook_rebinds_public_loader(
    tmp_path, monkeypatch, boundary
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    root = tmp_path / "checkpoints"
    original_batches, resumed_batches = [], []
    trainer = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=original_batches,
        max_step=1 if boundary == "step" else None,
        max_epoch=1 if boundary == "epoch" else None,
        save_epoch_freq=1 if boundary == "epoch" else None,
        save_step_freq=1 if boundary == "step" else None,
    )
    observer = PipelineHooks()

    def replace_loader(args):
        args.dataloader = None

    observer.register_hook(
        "on_step" if boundary == "step" else "on_epoch",
        HookContext.from_callable(
            before=replace_loader if boundary == "step" else None,
            after=replace_loader if boundary == "epoch" else None,
        ),
    )
    trainer.hooks += observer
    trainer()

    checkpoint = root / "checkpoint_0"
    state = torch.load(
        checkpoint / "custom_checkpoint_0.pkl", weights_only=False
    )
    loader_state = state["rank_states"][0]["loader_state"]
    assert loader_state is not None
    assert loader_state["epoch"] == (0 if boundary == "step" else 1)
    assert loader_state["delivered_batches"] == (
        1 if boundary == "step" else 0
    )
    resumed = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=resumed_batches,
        resume=checkpoint,
        max_epoch=1 if boundary == "step" else 2,
        save_epoch_freq=None,
    )
    resumed()
    assert original_batches == ([0] if boundary == "step" else [0, 1, 2, 3])
    assert resumed_batches == (
        [1, 2, 3] if boundary == "step" else [0, 1, 2, 3]
    )


def test_epoch_checkpoint_on_max_step_keeps_partial_epoch(
    tmp_path, monkeypatch
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    root = tmp_path / "checkpoints"
    first_batches, resumed_batches = [], []
    first = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=first_batches,
        max_step=2,
    )
    first()
    saved = torch.load(
        root / "checkpoint_0" / "custom_checkpoint_0.pkl", weights_only=False
    )
    assert saved["epoch_id"] == 0
    assert saved["micro_step"]["epoch_step_id"] == 2
    resumed = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=resumed_batches,
        resume=root / "checkpoint_0",
        max_epoch=1,
    )
    resumed()
    assert first_batches == [0, 1]
    assert resumed_batches == [2, 3]


@pytest.mark.parametrize("remove", [False, True], ids=["scheduled", "removed"])
def test_nested_epoch_checkpoint_retains_frequency_and_removal(
    dummy_trainer, monkeypatch, remove
):
    from robo_orchard_lab.pipeline.hooks.checkpoint import SaveCheckpointConfig

    checkpoint = SaveCheckpointConfig(
        save_epoch_freq=2, save_when_loop_end=False, save_model=False
    )()
    monkeypatch.setattr(checkpoint, "_check", lambda accelerator: None)
    save = MagicMock(return_value="checkpoint")
    monkeypatch.setattr(checkpoint, "_save_state", save)
    nested = PipelineHooks()
    nested += checkpoint
    handle = dummy_trainer.hooks.register_pipeline_hooks(nested)
    if remove:
        handle()
    dummy_trainer.max_epoch = 2
    dummy_trainer()
    if remove:
        save.assert_not_called()
    else:
        save.assert_called_once()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_epoch_hook_can_copy_args_before_and_after(
    dummy_trainer, tmp_path, checkpoint, monkeypatch
):
    if checkpoint:
        from accelerate.state import (
            AcceleratorState,
            GradientState,
            PartialState,
        )

        for state in (AcceleratorState, PartialState, GradientState):
            monkeypatch.setattr(state, "_shared_state", {})
        trainer = _epoch_checkpoint_trainer(
            tmp_path / "checkpoints",
            lab_loader=True,
            events=[],
            batches=[],
            max_epoch=1,
        )
    else:
        trainer = dummy_trainer
    calls = []
    observer = PipelineHooks()

    def record(args):
        calls.append(args.epoch_id)
        for copied in (
            args.copy_with_updates(global_step_id=123),
            replace(args, global_step_id=123),
        ):
            assert copied.global_step_id == 123
            assert (
                copied._defer_epoch_checkpoint is args._defer_epoch_checkpoint
            )
            assert copied._checkpoint_dataloader is args._checkpoint_dataloader
        assert args.global_step_id != 123

    observer.register_hook(
        "on_epoch", HookContext.from_callable(before=record, after=record)
    )
    trainer.hooks += observer
    trainer()
    assert calls == [0, 0]
    if checkpoint:
        assert (tmp_path / "checkpoints" / "checkpoint_0").is_dir()
