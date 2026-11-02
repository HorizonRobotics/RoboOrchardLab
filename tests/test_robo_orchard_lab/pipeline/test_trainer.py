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

"""Test Trainer initialization, lifecycle, and core training loops."""

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderShard
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedType,
)
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import robo_orchard_lab.pipeline as pipeline_module
import robo_orchard_lab.pipeline.training as training_module
import robo_orchard_lab.pipeline.training.hook_based_trainer as trainer_module
import robo_orchard_lab.utils.accelerate as accelerate_utils
from robo_orchard_lab.dataset.robot import (
    DictIterableDataset,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader as RODataLoader,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.pipeline.hooks.optimizer import OptimizerHookConfig
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
)
from test_robo_orchard_lab.pipeline._trainer_test_helpers import (
    ArrayDatasetItem,
    DummyBatchProcessor,
    SimpleModel,
    TensorDataset,
    dummy_trainer as dummy_trainer,
)


@dataclass
class TrainerState:
    epoch: int = 0
    step: int = 0
    global_step: int = 0


class DummyPipelineHook(PipelineHooks):
    def __init__(self):
        super().__init__()

        self._on_loop_begin_cnt = 0
        self._on_loop_end_cnt = 0
        self._on_epoch_begin_cnt = 0
        self._on_epoch_end_cnt = 0
        self._on_step_begin_cnt = 0
        self._on_step_end_cnt = 0

        self.register_hook(
            "on_loop",
            HookContext.from_callable(
                before=self.on_loop_begin, after=self.on_loop_end
            ),
        )
        self.register_hook(
            "on_epoch",
            HookContext.from_callable(
                before=self.on_epoch_begin, after=self.on_epoch_end
            ),
        )
        self.register_hook(
            "on_step",
            HookContext.from_callable(
                before=self.on_step_begin, after=self.on_step_end
            ),
        )
        self._on_epoch_begin_state: Optional[TrainerState] = None
        self._on_epoch_end_state: Optional[TrainerState] = None

        self._on_step_begin_state: Optional[TrainerState] = None
        self._on_step_end_state: Optional[TrainerState] = None

    def on_loop_begin(self, args: PipelineHookArgs):
        self._on_loop_begin_cnt += 1

    def on_loop_end(self, args: PipelineHookArgs):
        self._on_loop_end_cnt += 1

    def on_epoch_begin(self, args: PipelineHookArgs):
        self._on_epoch_begin_cnt += 1
        self._on_epoch_begin_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )

    def on_epoch_end(self, args: PipelineHookArgs):
        self._on_epoch_end_cnt += 1
        self._on_epoch_end_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )
        assert self._on_epoch_begin_state is not None
        assert (
            self._on_epoch_end_state.epoch == self._on_epoch_begin_state.epoch
        )
        assert (
            self._on_epoch_end_state.global_step
            != self._on_epoch_begin_state.global_step
        )
        assert self._on_epoch_end_state.step != self._on_epoch_begin_state.step

    def on_step_begin(self, args: PipelineHookArgs):
        self._on_step_begin_cnt += 1
        self._on_step_begin_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )

    def on_step_end(self, args: PipelineHookArgs):
        self._on_step_end_cnt += 1
        self._on_step_end_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )
        assert self._on_step_begin_state is not None
        assert self._on_step_end_state.step == self._on_step_begin_state.step
        assert (
            self._on_step_end_state.global_step
            == self._on_step_begin_state.global_step
        )
        assert self._on_step_end_state.epoch == self._on_step_begin_state.epoch


def test_legacy_trainer_facades_export_runtime_types():
    from robo_orchard_lab.pipeline.training.trainer import SimpleTrainer

    with pytest.warns(DeprecationWarning):
        legacy_hook_based_trainer = importlib.import_module(
            "robo_orchard_lab.pipeline.hook_based_trainer"
        )
        legacy_hook_based_trainer = importlib.reload(legacy_hook_based_trainer)

    with pytest.warns(DeprecationWarning):
        legacy_trainer_module = importlib.import_module(
            "robo_orchard_lab.pipeline.trainer"
        )
        legacy_trainer_module = importlib.reload(legacy_trainer_module)

    assert legacy_hook_based_trainer.HookBasedTrainer is HookBasedTrainer
    assert legacy_trainer_module.SimpleTrainer is SimpleTrainer
    assert pipeline_module.HookBasedTrainer is HookBasedTrainer
    assert pipeline_module.SimpleTrainer is SimpleTrainer
    assert (
        training_module.LRSchedulerFactory is trainer_module.LRSchedulerFactory
    )


def test_trainer_initialization(dummy_trainer):
    assert dummy_trainer.max_epoch == 1


@pytest.mark.parametrize("use_factory", [False, True])
def test_trainer_rejects_reduce_on_plateau_before_training(use_factory):
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau
        if use_factory
        else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    )

    with pytest.raises(
        ValueError, match="ReduceLROnPlateau.*validation metric"
    ):
        HookBasedTrainer(
            model=model,
            dataloader=DataLoader(TensorDataset(torch.zeros(1, 2))),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            accelerator=Accelerator(),
            batch_processor=DummyBatchProcessor(),
            max_epoch=1,
        )


def test_trainer_does_not_iterate_when_loaded_limits_are_reached(
    dummy_trainer,
):
    """Restored completed progress must be checked before fetching a batch."""
    dummy_trainer.max_epoch = None
    dummy_trainer.max_step = 1
    dummy_trainer.trainer_progress_state.global_step_id = 1
    processor = MagicMock()
    dummy_trainer.batch_processor = processor
    dummy_trainer()
    processor.assert_not_called()
    assert dummy_trainer.trainer_progress_state.global_step_id == 1


def test_trainer_notifies_unmanaged_dataset_once_per_epoch():
    """Unmanaged dataset epoch hooks retain Accelerate's original cadence."""

    class CountingDataset(torch.utils.data.IterableDataset):
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)
            torch.rand(1)

        def __iter__(self):
            yield torch.tensor([0.5, 0.5])

    dataset = CountingDataset()
    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    trainer = HookBasedTrainer(
        model=model,
        dataloader=DataLoader(dataset, batch_size=1),
        optimizer=optimizer,
        lr_scheduler=StepLR(optimizer, step_size=1, gamma=0.1),
        accelerator=Accelerator(
            step_scheduler_with_optimizer=False,
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        ),
        batch_processor=DummyBatchProcessor(),
        max_epoch=1,
    )
    trainer()
    assert dataset.epochs == [0]


def test_hook_based_trainer_supports_scheduler_coupled_accelerator():
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(device_placement=True)

    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_hook_based_trainer_ignores_deprecated_user_optimizer_hook():
    """Deprecated user OptimizerHook config is a no-op compatibility shim."""
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

    with pytest.warns(DeprecationWarning, match="OptimizerHook is deprecated"):
        trainer = HookBasedTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            batch_processor=DummyBatchProcessor(),
            hooks=[OptimizerHookConfig()],
            max_epoch=1,
        )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_hook_based_trainer_materializes_scheduler_factory_before_prepare():
    """A regular backend binds the factory to its code-owned optimizer."""

    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    factory_calls: list[torch.optim.Optimizer] = []

    def build_scheduler(
        actual_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        factory_calls.append(actual_optimizer)
        return StepLR(actual_optimizer, step_size=1, gamma=0.1)

    trainer = HookBasedTrainer(
        model=model,
        dataloader=DataLoader(
            TensorDataset(torch.tensor([[0.5, 0.5]], dtype=torch.float32)),
        ),
        optimizer=optimizer,
        lr_scheduler=build_scheduler,
        accelerator=Accelerator(step_scheduler_with_optimizer=False),
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )

    assert factory_calls == [optimizer]
    assert (
        trainer.lr_scheduler.scheduler.optimizer is trainer.optimizer.optimizer
    )


def test_hook_based_trainer_prepares_iterable_dataloader_once(
    monkeypatch: pytest.MonkeyPatch,
):
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            dispatch_batches=False,
            split_batches=False,
            even_batches=False,
        ),
    )
    monkeypatch.setattr(accelerator.state, "num_processes", 4)
    monkeypatch.setattr(accelerator.state, "process_index", 0)
    monkeypatch.setattr(accelerator.state, "local_process_index", 0)

    dataset = DictIterableDataset(
        [ArrayDatasetItem(data=list(range(32)))],
    )
    dataloader = RODataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
    )
    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = HookBasedTrainer(
        model=model,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_epoch=1,
    )

    assert isinstance(trainer.dataloader, DataLoaderShard)
    base_dataset = trainer.dataloader.base_dataloader.dataset
    assert isinstance(base_dataset, DictIterableDataset)
    assert base_dataset.shard_kwargs.shard_strategy == "pad_last"
    assert base_dataset.dataset_items[0].num_shards == 4
    assert base_dataset.dataset_items[0].shard_id == 0
    assert base_dataset.total_iterator_length == 8


def test_hook_based_trainer_early_break_cleans_accelerate_dataloader_state(
    monkeypatch: pytest.MonkeyPatch,
):
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            dispatch_batches=False,
            split_batches=False,
            even_batches=False,
        ),
    )
    monkeypatch.setattr(accelerator.state, "num_processes", 4)
    monkeypatch.setattr(accelerator.state, "process_index", 0)
    monkeypatch.setattr(accelerator.state, "local_process_index", 0)

    dataset = DictIterableDataset(
        [ArrayDatasetItem(data=list(range(32)))],
    )
    dataloader = RODataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=0,
    )
    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = HookBasedTrainer(
        model=model,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_step=1,
    )

    trainer()

    assert not accelerator.gradient_state.in_dataloader
    assert (
        sum(
            ref is not None
            for ref in accelerator.gradient_state.dataloader_references
        )
        == 0
    )


def test_training_loop(dummy_trainer: HookBasedTrainer):
    hook = DummyPipelineHook()
    dummy_trainer.hooks += hook
    dummy_trainer()

    assert dummy_trainer.dataloader is not None

    assert hook._on_loop_begin_cnt == 1
    assert hook._on_epoch_begin_cnt == 1
    assert hook._on_step_begin_cnt == len(dummy_trainer.dataloader)
    assert hook._on_step_end_cnt == len(dummy_trainer.dataloader)
    assert hook._on_epoch_end_cnt == 1
    assert hook._on_loop_end_cnt == 1


def test_hook_based_trainer_honors_peer_epoch_stop_before_step(
    monkeypatch: pytest.MonkeyPatch,
):
    event_order: list[str] = []
    close_reasons: list[accelerate_utils.DataloaderCloseReason] = []

    class RecordingBatchProcessor(DummyBatchProcessor):
        def forward(self, model: torch.nn.Module, batch: torch.Tensor):
            event_order.append("step")
            return super().forward(model=model, batch=batch)

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=RecordingBatchProcessor(),
        max_epoch=1,
    )
    check_results = iter([False, True, False, True])

    def record_sync() -> None:
        event_order.append("sync")

    def record_set_trigger() -> None:
        event_order.append("set")

    def record_check_trigger() -> bool:
        event_order.append("check")
        return next(check_results, False)

    monkeypatch.setattr(accelerator, "wait_for_everyone", record_sync)
    monkeypatch.setattr(accelerator, "set_trigger", record_set_trigger)
    monkeypatch.setattr(accelerator, "check_trigger", record_check_trigger)
    original_close = accelerate_utils.close_dataloader_resources

    def record_close(*args, **kwargs):
        close_reasons.append(kwargs["reason"])
        return original_close(*args, **kwargs)

    monkeypatch.setattr(
        accelerate_utils,
        "close_dataloader_resources",
        record_close,
    )

    trainer()

    assert event_order == ["check", "check", "set", "check"]
    assert trainer.trainer_progress_state.global_step_id == 0
    assert close_reasons == [
        accelerate_utils.DataloaderCloseReason.COORDINATED_EPOCH_END
    ]


def test_hook_based_trainer_sets_trigger_when_dataloader_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
):
    event_order: list[str] = []
    close_reasons: list[accelerate_utils.DataloaderCloseReason] = []

    class RecordingBatchProcessor(DummyBatchProcessor):
        def forward(self, model: torch.nn.Module, batch: torch.Tensor):
            event_order.append("step")
            return super().forward(model=model, batch=batch)

    model = SimpleModel()
    dataloader = DataLoader(
        torch.empty((0, 2), dtype=torch.float32),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=RecordingBatchProcessor(),
        max_epoch=1,
    )
    check_results = iter([False, True, False, True])

    def record_sync() -> None:
        event_order.append("sync")

    def record_set_trigger() -> None:
        event_order.append("set")

    def record_check_trigger() -> bool:
        event_order.append("check")
        return next(check_results, False)

    monkeypatch.setattr(accelerator, "wait_for_everyone", record_sync)
    monkeypatch.setattr(accelerator, "set_trigger", record_set_trigger)
    monkeypatch.setattr(accelerator, "check_trigger", record_check_trigger)
    original_close = accelerate_utils.close_dataloader_resources

    def record_close(*args, **kwargs):
        close_reasons.append(kwargs["reason"])
        return original_close(*args, **kwargs)

    monkeypatch.setattr(
        accelerate_utils,
        "close_dataloader_resources",
        record_close,
    )

    trainer()

    assert event_order == ["check", "set", "check", "set", "check"]
    assert trainer.trainer_progress_state.global_step_id == 0
    assert close_reasons == [
        accelerate_utils.DataloaderCloseReason.EPOCH_EXHAUSTED
    ]


@pytest.mark.parametrize("failure", [False, True], ids=["eof", "error"])
@pytest.mark.parametrize("checkpoint_hook", ["none", "step_only", "removed"])
def test_xla_epoch_termination_avoids_object_collective(
    dummy_trainer, monkeypatch, failure, checkpoint_hook
):
    from accelerate.utils import operations

    trainer = dummy_trainer
    accelerator = trainer.accelerator
    monkeypatch.setattr(
        operations,
        "PartialState",
        lambda: SimpleNamespace(distributed_type=DistributedType.XLA),
    )
    with pytest.raises(NotImplementedError, match="TPU"):
        operations.gather_object([{"probe": True}])
    monkeypatch.setattr(
        accelerator.state, "distributed_type", DistributedType.XLA
    )
    active = [False]

    def set_trigger():
        active[0] = True

    def check_trigger():
        value = active[0]
        active[0] = False
        return value

    monkeypatch.setattr(accelerator, "set_trigger", set_trigger)
    monkeypatch.setattr(accelerator, "check_trigger", check_trigger)
    error = OSError("XLA source failure")
    if checkpoint_hook != "none":
        from robo_orchard_lab.pipeline.hooks.checkpoint import (
            SaveCheckpointConfig,
        )

        handle = trainer.hooks.register_pipeline_hooks(
            SaveCheckpointConfig(
                save_epoch_freq=None,
                save_step_freq=100,
                save_when_loop_end=False,
                save_model=False,
            )()
        )
        if checkpoint_hook == "removed":
            handle()

    class Source:
        def __iter__(self):
            if failure:
                raise error
            return iter(())

    replacement = Source()
    replacement._lab_recovery = trainer.dataloader._lab_recovery
    trainer.dataloader = replacement
    if failure:
        with pytest.raises(OSError) as caught:
            trainer()
        assert caught.value is error
    else:
        trainer()
        assert trainer.trainer_progress_state.epoch_id == 1
    assert trainer.trainer_progress_state.global_step_id == 0


def test_hook_based_trainer_closes_max_step_with_max_step_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    """A local max-step stop must tear down the active iterator."""
    close_reasons: list[accelerate_utils.DataloaderCloseReason] = []
    teardown_primary_excs: list[BaseException | None] = []

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor([[0.5, 0.5], [0.25, 0.25]], dtype=torch.float32),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=DummyBatchProcessor(),
        max_step=1,
    )
    original_close = accelerate_utils.close_dataloader_resources
    original_teardown = trainer_module._close_dataloader_owner_resources

    def record_close(*args, **kwargs):
        close_reasons.append(kwargs["reason"])
        return original_close(*args, **kwargs)

    def record_teardown(*args, **kwargs):
        teardown_primary_excs.append(kwargs.get("primary_exc"))
        return original_teardown(*args, **kwargs)

    monkeypatch.setattr(
        accelerate_utils,
        "close_dataloader_resources",
        record_close,
    )
    monkeypatch.setattr(
        trainer_module,
        "_close_dataloader_owner_resources",
        record_teardown,
    )

    trainer()

    assert trainer.trainer_progress_state.global_step_id == 1
    assert close_reasons == [
        accelerate_utils.DataloaderCloseReason.MAX_STEP_END
    ]
    assert teardown_primary_excs == [None]


def test_hook_based_trainer_closes_exception_with_exception_abort_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    """An exception in the step body must preserve the primary failure."""
    close_reasons: list[accelerate_utils.DataloaderCloseReason] = []
    teardown_primary_excs: list[BaseException | None] = []

    class FailingBatchProcessor(DummyBatchProcessor):
        def forward(self, model: torch.nn.Module, batch: torch.Tensor):
            raise RuntimeError("step failed")

    model = SimpleModel()
    dataloader = DataLoader(
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        batch_size=1,
    )
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=FailingBatchProcessor(),
        max_epoch=1,
    )
    original_close = accelerate_utils.close_dataloader_resources
    original_teardown = trainer_module._close_dataloader_owner_resources

    def record_close(*args, **kwargs):
        close_reasons.append(kwargs["reason"])
        return original_close(*args, **kwargs)

    def record_teardown(*args, **kwargs):
        teardown_primary_excs.append(kwargs.get("primary_exc"))
        return original_teardown(*args, **kwargs)

    monkeypatch.setattr(
        accelerate_utils,
        "close_dataloader_resources",
        record_close,
    )
    monkeypatch.setattr(
        trainer_module,
        "_close_dataloader_owner_resources",
        record_teardown,
    )

    with pytest.raises(RuntimeError, match="step failed"):
        trainer()

    assert close_reasons == [
        accelerate_utils.DataloaderCloseReason.EXCEPTION_ABORT
    ]
    assert len(teardown_primary_excs) == 1
    assert isinstance(teardown_primary_excs[0], RuntimeError)
    assert trainer.trainer_progress_state.global_step_id == 0
    assert trainer.trainer_progress_state.micro_step.global_step_id == 0


def test_optimizer_and_scheduler(dummy_trainer):
    initial_lr = dummy_trainer.optimizer.param_groups[0]["lr"]
    dummy_trainer()
    final_lr = dummy_trainer.optimizer.param_groups[0]["lr"]

    assert final_lr < initial_lr


if __name__ == "__main__":
    pytest.main(["-s", __file__])
