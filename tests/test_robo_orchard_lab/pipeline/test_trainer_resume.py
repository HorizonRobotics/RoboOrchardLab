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

"""Test Trainer data-position recovery and resume policy."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedType,
)
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import robo_orchard_lab.utils.accelerate as accelerate_utils
from robo_orchard_lab.dataset.robot import (
    DictIterableDataset,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader as RODataLoader,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHooks,
)
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
    ResumeCheckpointConfig,
)
from test_robo_orchard_lab.pipeline._trainer_test_helpers import (
    ArrayDatasetItem,
    DummyBatchProcessor,
    SimpleModel,
    _epoch_checkpoint_trainer,
)


@pytest.mark.parametrize("stop", ["eof", "early", "error"])
def test_map_index_skip_closes_temporary_persistent_workers(monkeypatch, stop):
    from accelerate.data_loader import SkipBatchSampler
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})

    processed = []

    class RecordingProcessor(DummyBatchProcessor):
        def forward(self, model, batch):
            processed.extend(int(value) for value in batch[:, 0].tolist())
            return super().forward(model, batch)

    loader = DataLoader(
        torch.stack((torch.arange(8), torch.ones(8)), dim=1),
        batch_size=2,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="fork",
    )
    accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    trainer = HookBasedTrainer(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=StepLR(optimizer, step_size=1),
        accelerator=accelerator,
        batch_processor=RecordingProcessor(),
        max_epoch=2 if stop == "eof" else 1,
        max_step=1 if stop == "early" else None,
    )
    trainer.dataloader._lab_recovery["replay"] = 2
    original_base = trainer.dataloader.base_dataloader
    wrapper = trainer.dataloader
    pools = []
    cleanup = accelerate_utils.close_dataloader_resources

    def record_cleanup(owner, iterator, **kwargs):
        base = owner.base_dataloader
        if isinstance(getattr(base, "batch_sampler", None), SkipBatchSampler):
            native = getattr(base, "_iterator", None)
            if native is not None:
                pools.append((base, native, list(native._workers)))
        return cleanup(owner, iterator, **kwargs)

    monkeypatch.setattr(
        accelerate_utils, "close_dataloader_resources", record_cleanup
    )
    if stop == "error":
        trainer.batch_processor = MagicMock(
            side_effect=RuntimeError("injected processor failure")
        )
        with pytest.raises(RuntimeError, match="injected processor failure"):
            trainer()
    else:
        trainer()

    assert trainer.dataloader is wrapper
    assert trainer.accelerator._dataloaders[0] is wrapper
    assert wrapper.base_dataloader is original_base
    assert len(pools) == 1
    temporary_base, iterator, workers = pools[0]
    assert temporary_base._iterator is None
    assert iterator._shutdown
    assert all(not worker.is_alive() for worker in workers)
    if stop == "eof":
        assert processed == [4, 5, 6, 7, *range(8)]
        assert trainer.trainer_progress_state.epoch_id == 2
    elif stop == "early":
        assert processed == [4, 5]
    else:
        assert processed == []


def test_map_custom_sampler_skips_batches_through_prepared_loader(monkeypatch):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})

    class CountingDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.reads = []

        def __len__(self):
            return 4

        def __getitem__(self, index):
            self.reads.append(index)
            return torch.tensor([float(index), 1.0])

    class CustomSampler(torch.utils.data.SequentialSampler):
        pass

    dataset = CountingDataset()
    loader = DataLoader(dataset, batch_size=2, sampler=CustomSampler(dataset))
    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    trainer = HookBasedTrainer(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=StepLR(optimizer, step_size=1),
        accelerator=Accelerator(cpu=True, step_scheduler_with_optimizer=False),
        batch_processor=DummyBatchProcessor(),
        max_epoch=1,
    )
    trainer.dataloader._lab_recovery["replay"] = 1
    original_base = trainer.dataloader.base_dataloader
    trainer()

    assert trainer.dataloader.base_dataloader is original_base
    assert dataset.reads == [2, 3]
    assert trainer.trainer_progress_state.global_step_id == 1


def test_trainer_aligns_managed_loader_to_nonzero_epoch():
    """Trainer epoch wins over a stale prepared-wrapper iteration."""
    dataset = DictIterableDataset(
        [ArrayDatasetItem(data=list(range(8)))], shuffle=True, seed=42
    )
    model = SimpleModel()
    optimizer = SGD(params=model.parameters(), lr=0.01)
    trainer = HookBasedTrainer(
        model=model,
        dataloader=RODataLoader(dataset, batch_size=2),
        optimizer=optimizer,
        lr_scheduler=StepLR(optimizer, step_size=1, gamma=0.1),
        accelerator=Accelerator(
            step_scheduler_with_optimizer=False,
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        ),
        batch_processor=DummyBatchProcessor(),
        max_epoch=4,
    )
    trainer.trainer_progress_state.epoch_id = 3
    trainer.dataloader.iteration = 77
    trainer()
    assert trainer.trainer_progress_state.epoch_id == 4
    assert trainer.dataloader.iteration == 4


@pytest.mark.parametrize("position", [0, 2])
def test_random_sampler_resume_nonzero_epoch(tmp_path, monkeypatch, position):
    """Rebuild the same epoch permutation before skipping consumed batches."""
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})

    root = tmp_path / "checkpoints"
    baseline, prefix, suffix = [], [], []
    _epoch_checkpoint_trainer(
        root,
        lab_loader=False,
        events=[],
        batches=baseline,
        max_epoch=2,
        shuffle=True,
        save_epoch_freq=None,
    )()
    _epoch_checkpoint_trainer(
        root,
        lab_loader=False,
        events=[],
        batches=prefix,
        max_epoch=1 if position == 0 else None,
        max_step=4 + position if position else None,
        shuffle=True,
        save_epoch_freq=1 if position == 0 else None,
        save_step_freq=4 + position if position else None,
    )()
    _epoch_checkpoint_trainer(
        root,
        lab_loader=False,
        events=[],
        batches=suffix,
        resume=root / "checkpoint_0",
        max_epoch=2,
        shuffle=True,
        save_epoch_freq=None,
    )()
    assert prefix == baseline[: 4 + position]
    assert suffix == baseline[4 + position :]


def test_resume_keeps_explicit_loader_generator(tmp_path, monkeypatch):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})

    root = tmp_path / "checkpoints"
    _epoch_checkpoint_trainer(
        root,
        lab_loader=False,
        events=[],
        batches=[],
        max_step=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(103),
        save_epoch_freq=None,
        save_step_freq=2,
    )()
    generator = torch.Generator().manual_seed(731)
    torch.rand(5, generator=generator)
    before = generator.get_state().clone()
    batches = []
    trainer = _epoch_checkpoint_trainer(
        root,
        lab_loader=False,
        events=[],
        batches=batches,
        max_epoch=1,
        resume=root / "checkpoint_0",
        shuffle=True,
        generator=generator,
        save_epoch_freq=None,
    )
    assert torch.equal(generator.get_state(), before)
    trainer()
    assert len(batches) == 2


@pytest.mark.parametrize("lab_loader", [False, True], ids=["map", "lab"])
def test_resume_with_changed_accumulation_starts_a_new_window(
    tmp_path, monkeypatch, lab_loader
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    root = tmp_path / "checkpoints"
    first_batches, resumed_batches = [], []
    first = _epoch_checkpoint_trainer(
        root,
        lab_loader=lab_loader,
        events=[],
        batches=first_batches,
        max_step=1,
        accumulation_steps=2,
        dataset_size=10,
        save_epoch_freq=None,
        save_step_freq=1,
    )
    first()

    resumed = _epoch_checkpoint_trainer(
        root,
        lab_loader=lab_loader,
        events=[],
        batches=resumed_batches,
        resume=root / "checkpoint_0",
        max_step=2,
        accumulation_steps=4,
        dataset_size=10,
        save_epoch_freq=None,
        save_step_freq=100,
    )
    resumed()
    assert first_batches == [0, 1]
    assert resumed_batches == [2, 3, 4, 5]
    assert resumed.trainer_progress_state.global_step_id == 2


@pytest.mark.parametrize("stale_snapshot", [False, True])
def test_native_checkpoint_after_progress_replays_with_warning(
    tmp_path, monkeypatch, stale_snapshot
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    root = tmp_path / "checkpoints"
    first_batches, second_batches = [], []
    resumed_batches, final_batches = [], []
    _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=first_batches,
        max_step=1,
        save_epoch_freq=None,
        save_step_freq=1,
    )()

    raw_paths = []
    second = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=second_batches,
        resume=root / "checkpoint_0",
        max_step=2,
        save_epoch_freq=None,
    )
    observer = PipelineHooks()
    observer.register_hook(
        "on_optimizer_step",
        HookContext.from_callable(
            after=lambda args: raw_paths.append(args.accelerator.save_state())
        ),
    )
    second.hooks += observer
    second()
    assert len(raw_paths) == 1
    saved = torch.load(
        Path(raw_paths[0]) / "custom_checkpoint_0.pkl", weights_only=False
    )
    assert saved["rank_states"] == []
    if stale_snapshot:
        earlier = torch.load(
            root / "checkpoint_0" / "custom_checkpoint_0.pkl",
            weights_only=False,
        )
        saved["rank_states"] = earlier["rank_states"]
        torch.save(saved, Path(raw_paths[0]) / "custom_checkpoint_0.pkl")

    with pytest.warns(
        UserWarning, match="no current DataLoader recovery state"
    ):
        resumed = _epoch_checkpoint_trainer(
            root,
            lab_loader=True,
            events=[],
            batches=resumed_batches,
            resume=raw_paths[0],
            max_step=3,
            save_epoch_freq=None,
            save_step_freq=1,
        )
    resumed()
    resaved = root / "checkpoint_2"
    resaved_state = torch.load(
        resaved / "custom_checkpoint_0.pkl", weights_only=False
    )
    assert (
        resaved_state["rank_states"][0]["loader_state"]["delivered_batches"]
        == 3
    )
    final = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=final_batches,
        resume=resaved,
        max_epoch=1,
        save_epoch_freq=None,
    )
    final()
    assert first_batches == [0]
    assert second_batches == [1]
    assert resumed_batches == [2]
    assert final_batches == [3]


def test_native_map_resume_can_save_before_any_new_batch(
    tmp_path, monkeypatch
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    root = tmp_path / "checkpoints"
    raw_paths = []
    first = _epoch_checkpoint_trainer(
        root,
        lab_loader=False,
        events=[],
        batches=[],
        max_step=1,
        save_epoch_freq=None,
        shuffle=True,
    )
    observer = PipelineHooks()
    observer.register_hook(
        "on_optimizer_step",
        HookContext.from_callable(
            after=lambda args: raw_paths.append(args.accelerator.save_state())
        ),
    )
    first.hooks += observer
    first()

    no_batches = []
    with pytest.warns(
        UserWarning, match="no current DataLoader recovery state"
    ):
        stopped = _epoch_checkpoint_trainer(
            root,
            lab_loader=False,
            events=[],
            batches=no_batches,
            resume=raw_paths[0],
            max_step=1,
            save_epoch_freq=None,
            save_step_freq=100,
            save_when_loop_end=True,
            shuffle=True,
        )
    stopped()
    assert no_batches == []
    resaved = root / "checkpoint_1"
    state = torch.load(resaved / "custom_checkpoint_0.pkl", weights_only=False)
    assert state["rank_states"][0]["sampler_state"] is not None

    next_batches = []
    continued = _epoch_checkpoint_trainer(
        root,
        lab_loader=False,
        events=[],
        batches=next_batches,
        resume=resaved,
        max_step=2,
        save_epoch_freq=None,
        shuffle=True,
    )
    continued()
    assert len(next_batches) == 1


def test_native_checkpoint_in_accumulation_starts_new_window_with_warning(
    tmp_path, monkeypatch
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    root = tmp_path / "checkpoints"
    native_paths = []
    first = _epoch_checkpoint_trainer(
        root,
        lab_loader=True,
        events=[],
        batches=[],
        max_epoch=1,
        accumulation_steps=4,
        dataset_size=5,
        save_epoch_freq=None,
    )
    observer = PipelineHooks()

    def save_first_micro_step(args):
        if args.micro_step.epoch_step_id == 1:
            native_paths.append(args.accelerator.save_state())

    observer.register_hook(
        "on_step", HookContext.from_callable(after=save_first_micro_step)
    )
    first.hooks += observer
    first()
    assert len(native_paths) == 1

    batches = []
    with pytest.warns(UserWarning, match="unfinished accumulation window"):
        resumed = _epoch_checkpoint_trainer(
            root,
            lab_loader=True,
            events=[],
            batches=batches,
            resume=native_paths[0],
            max_epoch=1,
            accumulation_steps=4,
            dataset_size=5,
            save_epoch_freq=None,
            save_step_freq=1,
        )
    assert (
        resumed.trainer_progress_state.micro_step.index_in_optimizer_step == 0
    )
    resumed()
    assert batches == [1, 2, 3, 4]
    assert resumed.trainer_progress_state.global_step_id == 1


def test_xla_trainer_resume_rejects_before_object_collective(
    tmp_path, monkeypatch
):
    from accelerate.utils import operations

    from robo_orchard_lab.pipeline.training._progress import (
        TrainerProgressState,
    )

    monkeypatch.setattr(
        operations,
        "PartialState",
        lambda: SimpleNamespace(distributed_type=DistributedType.XLA),
    )
    with pytest.raises(NotImplementedError, match="TPU"):
        operations.gather_object([{"probe": True}])
    accelerator = SimpleNamespace(
        _custom_objects=[TrainerProgressState()],
        distributed_type=DistributedType.XLA,
        load_state=MagicMock(),
    )
    with pytest.raises(RuntimeError, match="(?i)trainer/data.*unsupported"):
        ResumeCheckpointConfig(resume_from=str(tmp_path)).load_state(
            accelerator
        )
    accelerator.load_state.assert_not_called()


@pytest.mark.parametrize("load_fails", [False, True])
def test_resume_config_resets_phase_only_after_full_load(
    monkeypatch, tmp_path, load_fails
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    from robo_orchard_lab.pipeline.training._progress import (
        TrainerProgressState,
    )

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    accelerator.prepare(torch.nn.Linear(1, 1))
    accelerator.register_for_checkpointing(TrainerProgressState())
    checkpoint = tmp_path / "training-phase"
    accelerator.step = 7
    accelerator.save_state(str(checkpoint))
    accelerator.step = 99
    native_load = accelerator.load_state
    error = OSError("native load failed")

    def observed_load(*args, **kwargs):
        assert accelerator.step == 99
        if load_fails:
            raise error
        native_load(*args, **kwargs)
        # Native loading restores phase 7; training policy runs afterward.
        assert accelerator.step == 7

    monkeypatch.setattr(accelerator, "load_state", observed_load)
    config = ResumeCheckpointConfig(resume_from=str(checkpoint))
    if load_fails:
        with pytest.raises(OSError) as caught:
            config.load_state(accelerator)
        assert caught.value is error
        assert accelerator.step == 99
    else:
        config.load_state(accelerator)
        assert accelerator.step == 0
