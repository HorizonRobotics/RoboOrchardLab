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

"""Test Trainer gradient clipping ownership and hook configuration."""

import warnings
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedType,
)
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import robo_orchard_lab.pipeline.training.hook_based_trainer as trainer_module
from robo_orchard_lab.pipeline.hooks.grad_clip import (
    GradientClippingHookConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    MicroStepProgressState,
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.pipeline.training import _deepspeed as deepspeed_training
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
)
from robo_orchard_lab.pipeline.training.trainer import SimpleTrainer
from test_robo_orchard_lab.pipeline._trainer_test_helpers import (
    DummyBatchProcessor,
    PrepareOnlyDeepSpeedAccelerator,
    SimpleModel,
    TensorDataset,
    _adamw_deepspeed_config,
    _deepspeed_topology_config,
    build_micro_step_hook_args,
)


def test_hook_based_trainer_runs_internal_gradient_clipping(
    monkeypatch: pytest.MonkeyPatch,
):
    """HookBasedTrainer applies configured grad clipping internally."""
    clip_calls: list[tuple[int, float, float]] = []
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    def record_clip_grad_norm_(params, max_norm, norm_type):
        clip_calls.append((len(list(params)), max_norm, norm_type))

    monkeypatch.setattr(
        accelerator,
        "clip_grad_norm_",
        record_clip_grad_norm_,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
            norm_type=2.0,
        ),
        max_step=1,
    )

    trainer()

    assert clip_calls == [(2, 1.0, 2.0)]


def test_hook_based_trainer_clips_only_accumulation_boundaries(
    monkeypatch: pytest.MonkeyPatch,
):
    """Trainer-owned grad clipping waits for optimizer-step boundaries."""

    events: list[str] = []
    clip_calls: list[tuple[int, float, float]] = []

    class RecordingSGD(SGD):
        def step(self, *args, **kwargs):
            events.append("optimizer_step")
            return super().step(*args, **kwargs)

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor(
            [[0.5, 0.5], [0.25, 0.25]],
            dtype=torch.float32,
        ),
        batch_size=1,
    )
    optimizer = RecordingSGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )

    def record_clip_grad_norm_(params, max_norm, norm_type):
        events.append("clip")
        clip_calls.append((len(list(params)), max_norm, norm_type))

    monkeypatch.setattr(
        accelerator,
        "clip_grad_norm_",
        record_clip_grad_norm_,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
            norm_type=2.0,
        ),
        max_step=1,
    )

    trainer()

    assert events == ["clip", "optimizer_step"]
    assert clip_calls == [(2, 1.0, 2.0)]
    assert (
        trainer.trainer_progress_state.micro_step.last_optimizer_step_size == 2
    )


def test_hook_based_trainer_clips_skipped_boundary_without_commit():
    """An AMP-skipped update still advances completed-window progress."""

    events: list[str] = []

    class SkippedStepAccelerator:
        distributed_type = DistributedType.NO
        optimizer_step_was_skipped = True

        def clip_grad_norm_(self, params, max_norm, norm_type):
            del max_norm, norm_type
            events.append(f"clip:{len(list(params))}")

    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([2.0])
    optimizer = MagicMock()
    optimizer.param_groups = [{"params": [param]}]
    trainer = HookBasedTrainer.__new__(HookBasedTrainer)
    trainer.accelerator = cast(Accelerator, SkippedStepAccelerator())
    trainer.trainer_progress_state = trainer_module.TrainerProgressState()
    trainer.optimizer = optimizer
    trainer.lr_scheduler = MagicMock()
    trainer._grad_clip = GradientClippingHookConfig(
        clip_mode="norm",
        max_norm=1.0,
    )
    hook_args = build_micro_step_hook_args(
        trainer,
        is_optimizer_step_boundary=True,
    )

    committed = trainer._run_optimizer_step(
        hook_args,
        is_optimizer_step_boundary=True,
        micro_steps=hook_args.micro_step.index_in_optimizer_step,
    )

    assert committed is False
    assert events == ["clip:1"]
    optimizer.step.assert_called_once()
    trainer.lr_scheduler.step.assert_not_called()
    optimizer.zero_grad.assert_called_once()
    assert trainer.trainer_progress_state.global_step_id == 1
    assert trainer.trainer_progress_state.micro_step == MicroStepProgressState(
        epoch_step_id=1,
        global_step_id=1,
        index_in_optimizer_step=0,
        last_optimizer_step_size=1,
    )


def test_hook_based_trainer_rejects_duplicate_gradient_clipping_config():
    """Logical grad clipping has a single trainer-owned source of truth."""

    with pytest.raises(ValueError, match="Only one gradient clipping"):
        trainer_module._split_gradient_clipping_config(
            hooks=[
                GradientClippingHookConfig(
                    clip_mode="norm",
                    max_norm=1.0,
                )
            ],
            grad_clip=GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=1.0,
            ),
            is_deepspeed=False,
        )


def test_hook_based_trainer_rejects_direct_hook_with_logical_grad_clip():
    """A directly constructed legacy hook cannot share trainer ownership."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy_hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    with pytest.raises(ValueError, match="already constructed"):
        trainer_module._split_gradient_clipping_config(
            hooks=[legacy_hook],
            grad_clip=GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=1.0,
            ),
            is_deepspeed=False,
        )


def test_hook_based_trainer_rejects_direct_grad_clip_hook_for_deepspeed():
    """DeepSpeed cannot use a late GradientClippingHook instance."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy_hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    with pytest.raises(ValueError, match="DeepSpeed gradient clipping"):
        trainer_module._split_gradient_clipping_config(
            hooks=[legacy_hook],
            grad_clip=None,
            is_deepspeed=True,
        )


def test_hook_based_trainer_allows_opaque_deepspeed_hooks():
    """Opaque PipelineHooks are not rejected without a grad-clip marker."""

    opaque_hooks = PipelineHooks()

    remaining_hooks, grad_clip = (
        trainer_module._split_gradient_clipping_config(
            hooks=[opaque_hooks],
            grad_clip=None,
            is_deepspeed=True,
        )
    )

    assert remaining_hooks == [opaque_hooks]
    assert grad_clip is None


def test_hook_based_trainer_rejects_direct_hook_before_grad_clip_config():
    """Direct hook before logical config should not double clip."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy_hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    with pytest.raises(ValueError, match="already constructed"):
        trainer_module._split_gradient_clipping_config(
            hooks=[
                legacy_hook,
                GradientClippingHookConfig(
                    clip_mode="norm",
                    max_norm=0.5,
                ),
            ],
            grad_clip=None,
            is_deepspeed=False,
        )


def test_hook_based_trainer_rejects_multiple_direct_grad_clip_hooks():
    """Multiple direct legacy hooks would clip the same gradients twice."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy_hooks = [
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=1.0,
            )(),
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=0.5,
            )(),
        ]

    with pytest.raises(ValueError, match="Only one gradient clipping"):
        trainer_module._split_gradient_clipping_config(
            hooks=legacy_hooks,
            grad_clip=None,
            is_deepspeed=False,
        )


def test_hook_based_trainer_injects_missing_deepspeed_gradient_clipping():
    """DeepSpeed config receives trainer max_norm before prepare."""

    deepspeed_config: dict[str, object] = {}
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)

    deepspeed_training.configure_deepspeed_gradient_clipping(
        accelerator=cast(Accelerator, accelerator),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=0.25,
        ),
    )

    assert deepspeed_config["gradient_clipping"] == 0.25
    assert accelerator.deepspeed_plugin.gradient_clipping == 0.25


def test_hook_based_trainer_overwrites_deepspeed_auto_with_warning(
    monkeypatch: pytest.MonkeyPatch,
):
    """Trainer grad_clip owns DeepSpeed 'auto' when both are provided."""

    warnings_: list[str] = []

    def record_warning(message: str, *args, **kwargs) -> None:
        del kwargs
        warnings_.append(message % args if args else message)

    monkeypatch.setattr(deepspeed_training.logger, "warning", record_warning)
    deepspeed_config = _adamw_deepspeed_config()
    deepspeed_config["gradient_clipping"] = "auto"
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)

    deepspeed_training.configure_deepspeed_gradient_clipping(
        accelerator=cast(Accelerator, accelerator),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=0.25,
        ),
    )

    assert deepspeed_config["gradient_clipping"] == 0.25
    assert accelerator.deepspeed_plugin.gradient_clipping == 0.25
    assert len(warnings_) == 1
    assert "overriding it with grad_clip.max_norm=0.25" in warnings_[0]


def test_hook_based_trainer_accepts_matching_deepspeed_gradient_clipping():
    """Matching config and trainer grad_clip values share one numeric value."""

    deepspeed_config: dict[str, object] = {"gradient_clipping": "0.25"}
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)

    deepspeed_training.configure_deepspeed_gradient_clipping(
        accelerator=cast(Accelerator, accelerator),
        grad_clip=GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=0.25,
        ),
    )

    assert deepspeed_config["gradient_clipping"] == 0.25
    assert accelerator.deepspeed_plugin.gradient_clipping == 0.25


def test_hook_based_trainer_rejects_conflicting_deepspeed_gradient_clipping():
    """Conflicting DeepSpeed config and trainer grad_clip fail fast."""

    deepspeed_config: dict[str, object] = {"gradient_clipping": 0.5}
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)

    with pytest.raises(ValueError, match="Conflicting gradient clipping"):
        deepspeed_training.configure_deepspeed_gradient_clipping(
            accelerator=cast(Accelerator, accelerator),
            grad_clip=GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=0.25,
            ),
        )


@pytest.mark.parametrize(
    ("grad_clip", "match"),
    [
        (
            GradientClippingHookConfig(
                clip_mode="value",
                clip_value=0.1,
            ),
            "supports only norm clipping",
        ),
        (
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=0.25,
                norm_type=1.0,
            ),
            "norm_type must be 2.0",
        ),
    ],
)
def test_hook_based_trainer_rejects_non_equivalent_deepspeed_grad_clip(
    grad_clip: GradientClippingHookConfig,
    match: str,
):
    """DeepSpeed config injection accepts only equivalent L2 norm clipping."""

    accelerator = PrepareOnlyDeepSpeedAccelerator({})

    with pytest.raises(ValueError, match=match):
        deepspeed_training.configure_deepspeed_gradient_clipping(
            accelerator=cast(Accelerator, accelerator),
            grad_clip=grad_clip,
        )


def test_hook_based_trainer_rejects_deepspeed_norm_without_max_norm():
    """DeepSpeed clipping cannot be configured without a max norm."""

    grad_clip = GradientClippingHookConfig(
        clip_mode="norm",
        max_norm=0.25,
    )
    grad_clip.max_norm = None
    accelerator = PrepareOnlyDeepSpeedAccelerator({})

    with pytest.raises(ValueError, match="requires max_norm"):
        deepspeed_training.configure_deepspeed_gradient_clipping(
            accelerator=cast(Accelerator, accelerator),
            grad_clip=grad_clip,
        )


def test_hook_based_trainer_leaves_deepspeed_auto_without_grad_clip():
    """Without trainer grad_clip, DeepSpeed auto stays owned by Accelerate."""

    deepspeed_config = _deepspeed_topology_config()
    deepspeed_config["gradient_clipping"] = "auto"
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01)

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return StepLR(actual_optimizer, step_size=1, gamma=0.1)

    HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=build_scheduler,
        accelerator=cast(Accelerator, accelerator),
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )

    assert deepspeed_config["gradient_clipping"] == "auto"
    assert accelerator.prepare_calls == 1


def test_hook_based_trainer_injects_deepspeed_grad_clip_from_raw_hooks():
    """Raw GradientClippingHookConfig is routed before DeepSpeed prepare."""

    deepspeed_config = _deepspeed_topology_config()
    accelerator = PrepareOnlyDeepSpeedAccelerator(deepspeed_config)
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01)

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return StepLR(actual_optimizer, step_size=1, gamma=0.1)

    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=build_scheduler,
        accelerator=cast(Accelerator, accelerator),
        batch_processor=DummyBatchProcessor(),
        hooks=[
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=0.25,
            )
        ],
        max_step=1,
    )

    assert deepspeed_config["gradient_clipping"] == 0.25
    assert accelerator.deepspeed_plugin.gradient_clipping == 0.25
    assert trainer._grad_clip is None
    assert accelerator.prepare_calls == 1


def test_hook_based_trainer_preserves_user_hook_order_before_optimizer(
    monkeypatch: pytest.MonkeyPatch,
):
    """User on_step after-hooks observe state before optimizer finalization.

    Post-update user hooks should use ``on_optimizer_step`` instead.
    """
    events: list[str] = []

    class RecordingSGD(SGD):
        def step(self, *args, **kwargs):
            events.append("optimizer_step")
            return super().step(*args, **kwargs)

    class RecordingStepHook(PipelineHooks):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.register_hook(
                "on_step",
                HookContext.from_callable(after=self.on_step_end),
            )

        def on_step_end(self, args: PipelineHookArgs):
            events.append(self.name)

    class RecordingOptimizerStepBeforeHook(PipelineHooks):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.register_hook(
                "on_optimizer_step",
                HookContext.from_callable(before=self.on_optimizer_step_begin),
            )

        def on_optimizer_step_begin(self, args: PipelineHookArgs):
            events.append(self.name)

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = RecordingSGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    def record_clip_grad_norm_(params, max_norm, norm_type):
        events.append("clip")

    monkeypatch.setattr(
        accelerator,
        "clip_grad_norm_",
        record_clip_grad_norm_,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        hooks=[
            RecordingStepHook("first_user_hook"),
            GradientClippingHookConfig(
                clip_mode="norm",
                max_norm=1.0,
                norm_type=2.0,
            ),
            RecordingOptimizerStepBeforeHook("user_optimizer_before"),
            RecordingStepHook("last_user_hook"),
        ],
        max_step=1,
    )

    trainer()

    assert events == [
        "first_user_hook",
        "last_user_hook",
        "user_optimizer_before",
        "clip",
        "optimizer_step",
    ]


def test_simple_trainer_passes_raw_gradient_clipping_hook_config(
    monkeypatch: pytest.MonkeyPatch,
):
    """SimpleTrainer must not instantiate grad clip configs before parent."""

    events: list[str] = []

    class RecordingSGD(SGD):
        def step(self, *args, **kwargs):
            events.append("optimizer_step")
            return super().step(*args, **kwargs)

    class RecordingStepHook(PipelineHooks):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.register_hook(
                "on_step",
                HookContext.from_callable(after=self.on_step_end),
            )

        def on_step_end(self, args: PipelineHookArgs):
            events.append(self.name)

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = RecordingSGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    def record_clip_grad_norm_(params, max_norm, norm_type):
        events.append("clip")

    monkeypatch.setattr(
        accelerator,
        "clip_grad_norm_",
        record_clip_grad_norm_,
    )
    with pytest.warns(DeprecationWarning):
        trainer = SimpleTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            batch_processor=DummyBatchProcessor(),
            hooks=[
                GradientClippingHookConfig(
                    clip_mode="norm",
                    max_norm=1.0,
                    norm_type=2.0,
                ),
                RecordingStepHook("user_hook"),
            ],
            max_step=1,
        )

    trainer()

    assert events == [
        "user_hook",
        "clip",
        "optimizer_step",
    ]


def test_simple_trainer_rejects_duplicate_legacy_and_raw_grad_clip():
    """SimpleTrainer forwards duplicate logical grad clip sources.

    The parent trainer should then reject the duplicate logical ownership.
    """

    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="Only one gradient clipping"):
            SimpleTrainer(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                accelerator=accelerator,
                batch_processor=DummyBatchProcessor(),
                grad_clip_mode="norm",
                grad_max_norm=1.0,
                hooks=[
                    GradientClippingHookConfig(
                        clip_mode="norm",
                        max_norm=1.0,
                    )
                ],
                max_step=1,
            )
