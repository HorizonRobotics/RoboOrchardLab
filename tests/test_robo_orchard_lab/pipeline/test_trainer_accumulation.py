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

"""Test accumulation windows and optimizer-step hook boundaries."""

import pytest
import torch
from accelerate import Accelerator
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    MicroStepProgressState,
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
)
from test_robo_orchard_lab.pipeline._trainer_test_helpers import (
    DummyBatchProcessor,
    SimpleModel,
    TensorDataset,
)


def test_hook_based_trainer_supports_manual_scheduler_with_accumulation():
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                [[0.5, 0.5], [0.25, 0.25], [0.1, 0.2], [0.3, 0.4]],
                dtype=torch.float32,
            )
        ),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )

    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        max_step=2,
    )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)


def test_hook_based_trainer_counts_optimizer_steps_with_accumulation():
    """With accumulation, optimizer progress advances only on boundaries."""
    records: list[tuple[int, int, MicroStepProgressState, bool]] = []

    class RecordingStepHook(PipelineHooks):
        def __init__(self):
            super().__init__()
            self.register_hook(
                "on_step",
                HookContext.from_callable(after=self.on_step_end),
            )

        def on_step_end(self, args: PipelineHookArgs):
            assert args.micro_step is not None
            records.append(
                (
                    args.step_id,
                    args.global_step_id,
                    args.micro_step,
                    args.is_optimizer_step_boundary,
                )
            )

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor(
            [[0.5, 0.5], [0.25, 0.25], [0.1, 0.2], [0.3, 0.4]],
            dtype=torch.float32,
        ),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[RecordingStepHook()],
        max_step=2,
    )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 2
    assert trainer.trainer_progress_state.step_id == 2
    assert trainer.trainer_progress_state.micro_step.global_step_id == 4
    assert (
        trainer.trainer_progress_state.micro_step.last_optimizer_step_size == 2
    )
    assert records == [
        (
            0,
            0,
            MicroStepProgressState(
                epoch_step_id=1,
                global_step_id=1,
                index_in_optimizer_step=1,
                last_optimizer_step_size=0,
            ),
            False,
        ),
        (
            0,
            0,
            MicroStepProgressState(
                epoch_step_id=2,
                global_step_id=2,
                index_in_optimizer_step=2,
                last_optimizer_step_size=0,
            ),
            True,
        ),
        (
            1,
            1,
            MicroStepProgressState(
                epoch_step_id=3,
                global_step_id=3,
                index_in_optimizer_step=1,
                last_optimizer_step_size=2,
            ),
            False,
        ),
        (
            1,
            1,
            MicroStepProgressState(
                epoch_step_id=4,
                global_step_id=4,
                index_in_optimizer_step=2,
                last_optimizer_step_size=2,
            ),
            True,
        ),
    ]


def test_hook_based_trainer_runs_optimizer_step_hook_on_boundaries():
    """on_optimizer_step wraps only optimizer-step boundaries."""
    records: list[
        tuple[
            str,
            int,
            int,
            MicroStepProgressState,
            bool,
            bool,
        ]
    ] = []

    class RecordingOptimizerStepHook(PipelineHooks):
        def __init__(self):
            super().__init__()
            self.register_hook(
                "on_optimizer_step",
                HookContext.from_callable(
                    before=self.on_optimizer_step_begin,
                    after=self.on_optimizer_step_end,
                ),
            )

        def _record(self, phase: str, args: PipelineHookArgs) -> None:
            assert args.micro_step is not None
            records.append(
                (
                    phase,
                    args.step_id,
                    args.global_step_id,
                    args.micro_step,
                    args.is_optimizer_step_boundary,
                    args.is_optimizer_step_committed,
                )
            )

        def on_optimizer_step_begin(self, args: PipelineHookArgs) -> None:
            self._record("before", args)

        def on_optimizer_step_end(self, args: PipelineHookArgs) -> None:
            self._record("after", args)

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor(
            [[0.5, 0.5], [0.25, 0.25], [0.1, 0.2], [0.3, 0.4]],
            dtype=torch.float32,
        ),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[RecordingOptimizerStepHook()],
        max_step=2,
    )

    trainer()

    assert records == [
        (
            "before",
            0,
            0,
            MicroStepProgressState(
                epoch_step_id=2,
                global_step_id=2,
                index_in_optimizer_step=2,
                last_optimizer_step_size=0,
            ),
            True,
            False,
        ),
        (
            "after",
            1,
            1,
            MicroStepProgressState(
                epoch_step_id=2,
                global_step_id=2,
                index_in_optimizer_step=0,
                last_optimizer_step_size=2,
            ),
            True,
            True,
        ),
        (
            "before",
            1,
            1,
            MicroStepProgressState(
                epoch_step_id=4,
                global_step_id=4,
                index_in_optimizer_step=2,
                last_optimizer_step_size=2,
            ),
            True,
            False,
        ),
        (
            "after",
            2,
            2,
            MicroStepProgressState(
                epoch_step_id=4,
                global_step_id=4,
                index_in_optimizer_step=0,
                last_optimizer_step_size=2,
            ),
            True,
            True,
        ),
    ]


def test_hook_based_trainer_ignores_user_boundary_enable_mutation():
    """User hook mutation cannot turn a non-boundary into a commit."""

    observed_step_events: list[tuple[int, bool]] = []

    class BoundaryEnablingHook(PipelineHooks):
        def __init__(self):
            super().__init__()
            self.register_hook(
                "on_step",
                HookContext.from_callable(
                    before=None,
                    after=self.on_step_end,
                ),
            )

        def on_step_end(self, args: PipelineHookArgs) -> None:
            assert args.micro_step is not None
            observed_step_events.append(
                (
                    args.micro_step.global_step_id,
                    args.is_optimizer_step_boundary,
                )
            )
            if args.micro_step.global_step_id == 1:
                args.is_optimizer_step_boundary = True

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                [[0.5, 0.5], [0.25, 0.25]],
                dtype=torch.float32,
            )
        ),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[BoundaryEnablingHook()],
        max_step=1,
    )

    trainer()

    assert observed_step_events == [(1, False), (2, True)]
    assert trainer.trainer_progress_state.global_step_id == 1
    assert trainer.trainer_progress_state.micro_step.global_step_id == 2
    assert (
        trainer.trainer_progress_state.micro_step.last_optimizer_step_size == 2
    )


def test_hook_based_trainer_restores_boundary_snapshot_for_optimizer_hooks():
    """User hook mutation cannot corrupt optimizer boundary snapshots."""

    optimizer_events: list[tuple[str, bool, int, bool]] = []

    class BoundaryCorruptingHook(PipelineHooks):
        def __init__(self):
            super().__init__()
            self.register_hook(
                "on_step",
                HookContext.from_callable(
                    before=None,
                    after=self.on_step_end,
                ),
            )
            self.register_hook(
                "on_optimizer_step",
                HookContext.from_callable(
                    before=self.on_optimizer_step_begin,
                    after=self.on_optimizer_step_end,
                ),
            )

        def on_step_end(self, args: PipelineHookArgs) -> None:
            assert args.micro_step is not None
            if args.micro_step.global_step_id == 2:
                args.is_optimizer_step_boundary = False
                args.micro_step.index_in_optimizer_step = 99

        def _record(self, phase: str, args: PipelineHookArgs) -> None:
            assert args.micro_step is not None
            optimizer_events.append(
                (
                    phase,
                    args.is_optimizer_step_boundary,
                    args.micro_step.index_in_optimizer_step,
                    args.is_optimizer_step_committed,
                )
            )

        def on_optimizer_step_begin(self, args: PipelineHookArgs) -> None:
            self._record("before", args)

        def on_optimizer_step_end(self, args: PipelineHookArgs) -> None:
            self._record("after", args)

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                [[0.5, 0.5], [0.25, 0.25]],
                dtype=torch.float32,
            )
        ),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[BoundaryCorruptingHook()],
        max_epoch=1,
    )

    trainer()

    assert optimizer_events == [
        ("before", True, 2, False),
        ("after", True, 0, True),
    ]
    assert trainer.trainer_progress_state.global_step_id == 1
    assert trainer.trainer_progress_state.micro_step.global_step_id == 2
    assert (
        trainer.trainer_progress_state.micro_step.last_optimizer_step_size == 2
    )


@pytest.mark.parametrize(
    ("epochs", "expected_windows"), [(1, []), (2, [(1, 4)])]
)
def test_accumulation_window_can_span_epochs(
    monkeypatch, epochs, expected_windows
):
    from accelerate.state import AcceleratorState, GradientState, PartialState
    from accelerate.utils import GradientAccumulationPlugin

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=4, sync_with_dataloader=False
        ),
    )
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    windows = []
    observer = PipelineHooks()
    observer.register_hook(
        "on_optimizer_step",
        HookContext.from_callable(
            after=lambda args: windows.append(
                (args.epoch_id, args.micro_step.last_optimizer_step_size)
            )
        ),
    )
    trainer = HookBasedTrainer(
        accelerator=accelerator,
        model=model,
        dataloader=DataLoader(torch.ones(2, 2), batch_size=1),
        optimizer=optimizer,
        lr_scheduler=StepLR(optimizer, step_size=1),
        batch_processor=DummyBatchProcessor(),
        max_epoch=epochs,
        hooks=[observer],
    )
    trainer()
    assert windows == expected_windows
    assert trainer.trainer_progress_state.global_step_id == len(
        expected_windows
    )
