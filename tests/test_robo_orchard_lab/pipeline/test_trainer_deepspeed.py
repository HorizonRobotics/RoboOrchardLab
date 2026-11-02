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

"""Test DeepSpeed preparation ownership and observed step progress."""

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedType,
)
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import robo_orchard_lab.pipeline.training.hook_based_trainer as trainer_module
from robo_orchard_lab.pipeline.hooks.mixin import (
    MicroStepProgressState,
    PipelineHookArgs,
)
from robo_orchard_lab.pipeline.training import _deepspeed as deepspeed_training
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
)
from test_robo_orchard_lab.pipeline._trainer_test_helpers import (
    DummyBatchProcessor,
    PrepareOnlyDeepSpeedAccelerator,
    SimpleModel,
    TensorDataset,
    _adamw_deepspeed_config,
    _deepspeed_topology_config,
    build_micro_step_hook_args,
)


class FakeDeepSpeedEngine:
    def __init__(self) -> None:
        self.global_steps = 0
        self.skipped_steps = 0


class FakeDeepSpeedEngineWrapper:
    def __init__(self, engine: FakeDeepSpeedEngine) -> None:
        self.engine = engine


class FakeDeepSpeedAccelerator:
    distributed_type = DistributedType.DEEPSPEED

    def __init__(self, engine: FakeDeepSpeedEngine) -> None:
        self.deepspeed_engine_wrapped = FakeDeepSpeedEngineWrapper(engine)


def build_deepspeed_observer_trainer(
    engine: FakeDeepSpeedEngine,
) -> HookBasedTrainer:
    trainer = HookBasedTrainer.__new__(HookBasedTrainer)
    trainer.accelerator = cast(Accelerator, FakeDeepSpeedAccelerator(engine))
    trainer.trainer_progress_state = trainer_module.TrainerProgressState()
    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = MagicMock()
    return trainer


def test_hook_based_trainer_observes_deepspeed_accumulation_boundary():
    """DeepSpeed progress is committed from engine counters only."""
    engine = FakeDeepSpeedEngine()
    trainer = build_deepspeed_observer_trainer(engine)

    first_step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    first_hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=False,
    )

    committed = trainer._run_optimizer_step(
        first_hook_args,
        is_optimizer_step_boundary=False,
        micro_steps=first_hook_args.micro_step.index_in_optimizer_step,
        deepspeed_step_before=first_step_before,
    )

    assert committed is False
    assert trainer.trainer_progress_state.global_step_id == 0
    assert (
        trainer.trainer_progress_state.micro_step.index_in_optimizer_step == 1
    )

    second_step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    second_hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=True,
    )
    engine.global_steps = 1

    committed = trainer._run_optimizer_step(
        second_hook_args,
        is_optimizer_step_boundary=True,
        micro_steps=second_hook_args.micro_step.index_in_optimizer_step,
        deepspeed_step_before=second_step_before,
    )

    assert committed is True
    assert trainer.trainer_progress_state.global_step_id == 1
    assert trainer.trainer_progress_state.micro_step == MicroStepProgressState(
        epoch_step_id=2,
        global_step_id=2,
        index_in_optimizer_step=0,
        last_optimizer_step_size=2,
    )
    trainer.optimizer.step.assert_not_called()
    trainer.lr_scheduler.step.assert_not_called()


def test_hook_based_trainer_observes_deepspeed_skipped_step():
    """A DeepSpeed overflow still completes the accumulation window."""
    engine = FakeDeepSpeedEngine()
    trainer = build_deepspeed_observer_trainer(engine)
    step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=True,
    )
    engine.global_steps = 1
    engine.skipped_steps = 1

    committed = trainer._run_optimizer_step(
        hook_args,
        is_optimizer_step_boundary=True,
        micro_steps=hook_args.micro_step.index_in_optimizer_step,
        deepspeed_step_before=step_before,
    )

    assert committed is False
    assert trainer.trainer_progress_state.global_step_id == 1
    assert trainer.trainer_progress_state.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=0,
        last_optimizer_step_size=1,
    )


def test_hook_based_trainer_requires_deepspeed_boundary_step():
    """DeepSpeed boundary without engine progress fails fast."""
    engine = FakeDeepSpeedEngine()
    trainer = build_deepspeed_observer_trainer(engine)
    step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=True,
    )

    with pytest.raises(RuntimeError, match="did not advance"):
        trainer._run_optimizer_step(
            hook_args,
            is_optimizer_step_boundary=True,
            micro_steps=hook_args.micro_step.index_in_optimizer_step,
            deepspeed_step_before=step_before,
        )


def test_deepspeed_step_requires_micro_step():
    """The trainer fails fast without the step's micro-step state."""
    engine = FakeDeepSpeedEngine()
    trainer = build_deepspeed_observer_trainer(engine)
    step_before = deepspeed_training.read_deepspeed_step_snapshot(
        trainer.accelerator
    )
    assert step_before is not None

    with pytest.raises(RuntimeError, match="PipelineHookArgs.micro_step"):
        trainer._run_optimizer_step(
            PipelineHookArgs(
                accelerator=trainer.accelerator,
                is_optimizer_step_boundary=True,
            ),
            is_optimizer_step_boundary=True,
            micro_steps=1,
            deepspeed_step_before=step_before,
        )


def test_hook_based_trainer_converts_adamw_factory_before_deepspeed_prepare():
    """Trainer routes a real AdamW and factory to one DeepSpeed Dummy pair."""

    deepspeed_config = _deepspeed_topology_config()
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)
    model = SimpleModel()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [model.linear.weight],
                "lr": 0.01,
                "weight_decay": 0.001,
            },
            {
                "params": [model.linear.bias],
                "lr": 0.02,
                "betas": (0.8, 0.95),
                "eps": 1e-6,
            },
        ],
        lr=0.1,
        weight_decay=0.0005,
    )
    factory_calls: list[torch.optim.Optimizer] = []

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        factory_calls.append(actual_optimizer)
        return StepLR(actual_optimizer, step_size=1, gamma=0.1)

    HookBasedTrainer(
        model=model,
        dataloader=DataLoader(
            TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
        ),
        optimizer=optimizer,
        lr_scheduler=build_scheduler,
        accelerator=cast(Accelerator, accelerator),
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )

    assert accelerator.prepare_calls == 1
    prepared_optimizer = accelerator.prepared_args[2]
    prepared_scheduler = accelerator.prepared_args[3]
    assert isinstance(prepared_optimizer, DummyOptim)
    assert isinstance(prepared_scheduler, DummyScheduler)
    assert prepared_scheduler.optimizer is prepared_optimizer
    assert factory_calls == []
    assert [group["lr"] for group in prepared_optimizer.params] == [0.01, 0.02]
    assert prepared_optimizer.params[0]["weight_decay"] == 0.001
    assert prepared_optimizer.params[1]["betas"] == (0.8, 0.95)
    assert prepared_optimizer.params[1]["eps"] == 1e-6
    assert all(
        set(group) == {"params", "lr", "weight_decay", "betas", "eps"}
        for group in prepared_optimizer.params
    )
    assert deepspeed_config["optimizer"] == {
        "type": "AdamW",
        "params": {
            "lr": 0.1,
            "weight_decay": 0.0005,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    }

    actual_optimizer = torch.optim.AdamW(
        prepared_optimizer.params,
        lr=0.1,
        weight_decay=0.0005,
    )
    actual_scheduler = prepared_scheduler.lr_scheduler_callable(
        actual_optimizer
    )
    assert actual_scheduler.optimizer is actual_optimizer
    assert factory_calls == [actual_optimizer]


@pytest.mark.parametrize("dummy_kind", ["optimizer", "scheduler"])
def test_hook_based_trainer_rejects_dummy_inputs_without_deepspeed(
    dummy_kind: str,
):
    """Dummy prepare inputs have no runtime owner on regular backends."""

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = SGD([parameter], lr=0.01)
    if dummy_kind == "optimizer":
        dummy_optimizer = DummyOptim(params=[parameter], lr=0.01)
        optimizer_input = dummy_optimizer
        scheduler_input = DummyScheduler(
            optimizer=dummy_optimizer,
            lr_scheduler_callable=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )
    else:
        optimizer_input = optimizer
        scheduler_input = DummyScheduler(optimizer=optimizer)

    with pytest.raises(ValueError, match="only with a DeepSpeed accelerator"):
        trainer_module._resolve_optimizer_scheduler_before_prepare(
            accelerator=Accelerator(),
            optimizer=optimizer_input,
            lr_scheduler=scheduler_input,
        )


def test_deepspeed_prepare_accepts_existing_dummy_pair_and_factory():
    """Advanced callers can pass a valid dummy pair or a factory."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_adamw_deepspeed_config())
    parameter = torch.nn.Parameter(torch.ones(1))
    dummy_optimizer = DummyOptim(params=[parameter], lr=0.01)

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return StepLR(actual_optimizer, step_size=1)

    resolved_optimizer, resolved_scheduler = (
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=dummy_optimizer,
            lr_scheduler=build_scheduler,
        )
    )
    assert resolved_optimizer is dummy_optimizer
    assert isinstance(resolved_scheduler, DummyScheduler)
    assert resolved_scheduler.optimizer is dummy_optimizer
    assert resolved_scheduler.lr_scheduler_callable is build_scheduler

    assert deepspeed_training.prepare_deepspeed_optimizer_scheduler(
        accelerator=cast(Accelerator, accelerator),
        optimizer=dummy_optimizer,
        lr_scheduler=resolved_scheduler,
    ) == (dummy_optimizer, resolved_scheduler)


def test_deepspeed_prepare_accepts_fully_config_owned_dummy_pair():
    """A config-owned optimizer and scheduler use explicit placeholders."""

    config = _adamw_deepspeed_config()
    config["scheduler"] = {"type": "WarmupLR"}
    accelerator = PrepareOnlyDeepSpeedAccelerator(config)
    parameter = torch.nn.Parameter(torch.ones(1))
    dummy_optimizer = DummyOptim(params=[parameter], lr=0.01)
    dummy_scheduler = DummyScheduler(optimizer=dummy_optimizer)

    assert deepspeed_training.prepare_deepspeed_optimizer_scheduler(
        accelerator=cast(Accelerator, accelerator),
        optimizer=dummy_optimizer,
        lr_scheduler=dummy_scheduler,
    ) == (dummy_optimizer, dummy_scheduler)


def test_deepspeed_prepare_rejects_dummy_optimizer_without_config_owner():
    """DeepSpeed must have a config optimizer before accepting DummyOptim."""

    accelerator = PrepareOnlyDeepSpeedAccelerator({})
    dummy_optimizer = DummyOptim(
        params=[torch.nn.Parameter(torch.ones(1))],
        lr=0.01,
    )
    dummy_scheduler = DummyScheduler(
        optimizer=dummy_optimizer,
        lr_scheduler_callable=lambda actual_optimizer: StepLR(
            actual_optimizer,
            step_size=1,
        ),
    )

    with pytest.raises(ValueError, match="explicit DummyOptim requires"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=dummy_optimizer,
            lr_scheduler=dummy_scheduler,
        )


def test_deepspeed_prepare_rejects_concrete_scheduler_for_real_optimizer():
    """A concrete scheduler cannot bind the later DeepSpeed optimizer."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_deepspeed_topology_config())
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )

    with pytest.raises(ValueError, match="requires an LRSchedulerFactory"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=StepLR(optimizer, step_size=1),
        )


@pytest.mark.parametrize(
    ("optimizer", "match"),
    [
        (
            torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-4),
            "only exact torch.optim.AdamW",
        ),
        (
            torch.optim.AdamW(
                [torch.nn.Parameter(torch.ones(1))],
                lr=1e-4,
                amsgrad=True,
            ),
            "amsgrad",
        ),
        (
            torch.optim.AdamW(
                [
                    {
                        "params": [torch.nn.Parameter(torch.ones(1))],
                        "amsgrad": True,
                    }
                ],
                lr=1e-4,
            ),
            "param group 0.*amsgrad",
        ),
    ],
)
def test_deepspeed_prepare_rejects_unsupported_optimizer_recipe(
    optimizer: torch.optim.Optimizer,
    match: str,
):
    """Runtime lowering accepts only a supported exact AdamW recipe."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_deepspeed_topology_config())

    with pytest.raises((TypeError, ValueError), match=match):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )


def test_deepspeed_prepare_accepts_adamw_without_newer_defaults():
    """Older exact AdamW shapes have implicit decoupled weight decay."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_deepspeed_topology_config())
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )
    optimizer.defaults.pop("decoupled_weight_decay", None)
    for param_group in optimizer.param_groups:
        param_group.pop("decoupled_weight_decay", None)

    resolved_optimizer, _ = (
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )
    )

    assert isinstance(resolved_optimizer, DummyOptim)


def test_deepspeed_prepare_rejects_adamw_with_existing_state():
    """Lowering must not silently discard already-restored Adam moments."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_deepspeed_topology_config())
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    optimizer.state[parameter]["step"] = torch.tensor(1.0)

    with pytest.raises(ValueError, match="pristine AdamW"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )


def test_deepspeed_prepare_rejects_config_and_factory_scheduler_owners():
    """Python factory and config scheduler cannot both own the schedule."""

    config = {
        **_deepspeed_topology_config(),
        "scheduler": {"type": "WarmupLR"},
    }
    accelerator = PrepareOnlyDeepSpeedAccelerator(config)
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )

    with pytest.raises(ValueError, match="fully config-owned pair"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )


def test_deepspeed_prepare_rejects_duplicate_real_optimizer_source():
    """Real optimizer lowering cannot coexist with a config optimizer."""

    accelerator = PrepareOnlyDeepSpeedAccelerator(_adamw_deepspeed_config())
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )

    with pytest.raises(ValueError, match="must not also define optimizer"):
        deepspeed_training.prepare_deepspeed_optimizer_scheduler(
            accelerator=cast(Accelerator, accelerator),
            optimizer=optimizer,
            lr_scheduler=lambda actual_optimizer: StepLR(
                actual_optimizer,
                step_size=1,
            ),
        )
