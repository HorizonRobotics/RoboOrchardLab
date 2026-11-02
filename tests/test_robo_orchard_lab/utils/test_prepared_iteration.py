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

"""Prepared DataLoader iteration and logical-position recovery tests."""

import random

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.utils.dataclasses import DataLoaderConfiguration
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
)

import robo_orchard_lab.utils.accelerate as accelerate_utils
from robo_orchard_lab.dataset.robot import (
    DictIterableDataset,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader as LabDataLoader,
)
from test_robo_orchard_lab.utils._accelerate_test_helpers import (
    _ManagedRngItem,
)

prepare_data_loader = accelerate_utils.prepare_data_loader


@pytest.fixture
def preserve_torch_rng():
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def test_loader_restore_rejects_changed_rank_count_before_selecting_rank(
    monkeypatch,
):
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    source = LabDataLoader(torch.arange(8), batch_size=1)
    prepared = accelerator.prepare(source)
    accelerate_utils._configure_prepared_loader_recovery(
        accelerator, prepared, source
    )
    saved = accelerate_utils._capture_prepared_loader_recovery(prepared)
    with pytest.raises(RuntimeError, match="rank topology"):
        accelerate_utils._load_prepared_loader_recovery(
            accelerator, prepared, [saved, saved], epoch=0, processed_batches=0
        )


def test_unknown_source_seed_does_not_reseed_destination(
    monkeypatch, preserve_torch_rng
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    class CustomLoader(DataLoader):
        pass

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    prepared = prepare_data_loader(
        accelerator, CustomLoader(torch.arange(8), batch_size=2, shuffle=True)
    )
    assert prepared.get_sampler().generator is None
    assert not hasattr(prepared.get_sampler(), "_lab_default_generator")
    saved = accelerate_utils._capture_prepared_loader_recovery(
        prepared, processed_batches=0
    )
    assert saved["seed"] is None
    assert saved["sampler_state"] is None

    destination_accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True, data_seed=17
        ),
    )
    destination = prepare_data_loader(
        destination_accelerator,
        DataLoader(torch.arange(8), batch_size=2, shuffle=True),
    )
    assert destination.get_sampler().initial_seed == 17
    accelerate_utils._load_prepared_loader_recovery(
        destination_accelerator,
        destination,
        [saved],
        epoch=0,
        processed_batches=0,
    )
    assert destination.get_sampler().initial_seed == 17


def test_stale_lab_snapshot_replays_with_a_different_loader(monkeypatch):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    source = DataLoader(torch.arange(5), batch_size=1)
    prepared = accelerator.prepare(source)
    accelerate_utils._configure_prepared_loader_recovery(
        accelerator, prepared, source
    )
    saved = {"loader_state": {"epoch": 0, "delivered_batches": 1}}
    with pytest.warns(
        UserWarning, match="no current DataLoader recovery state"
    ):
        accelerate_utils._load_prepared_loader_recovery(
            accelerator, prepared, [saved], epoch=0, processed_batches=2
        )
    with accelerate_utils.iterate_prepared_loader(
        accelerator, prepared, epoch=0
    ) as session:
        assert [batch.item() for batch in session] == [2, 3, 4]

    saved["loader_state"]["delivered_batches"] = 2
    with pytest.raises(RuntimeError, match="compatible Lab DataLoader"):
        accelerate_utils._load_prepared_loader_recovery(
            accelerator, prepared, [saved], epoch=0, processed_batches=2
        )


def test_managed_iteration_preserves_init_error_when_cleanup_fails(
    monkeypatch,
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    init_error = RuntimeError("iterator initialization failed")
    cleanup_events = []

    class FailingLoader(DataLoader):
        def __iter__(self):
            raise init_error

        def end(self):
            cleanup_events.append("end")
            raise RuntimeError("cleanup failed")

    loader = FailingLoader([1])
    loader._lab_recovery = {
        "sampler_state": None,
        "replay": 0,
        "resume_pending": False,
    }
    with pytest.raises(
        RuntimeError, match="iterator initialization failed"
    ) as e:
        with accelerate_utils.iterate_prepared_loader(
            accelerator, loader, epoch=0
        ) as session:
            next(session)
    assert e.value is init_error
    assert cleanup_events == ["end"]


@pytest.mark.parametrize("lab", [False, True])
def test_prepared_nonzero_epoch_notifies_dataset_and_sampler_once(
    monkeypatch, lab
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})

    class Source(Dataset):
        def __init__(self):
            self.epochs = []

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return index

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    class Sampler(torch.utils.data.SequentialSampler):
        def __init__(self, source):
            super().__init__(source)
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    accelerator = Accelerator(cpu=True)
    source = Source()
    sampler = Sampler(source)
    loader_type = LabDataLoader if lab else DataLoader
    loader = loader_type(source, sampler=sampler, batch_size=2)
    prepared = accelerator.prepare(loader)
    accelerate_utils._configure_prepared_loader_recovery(
        accelerator, prepared, loader
    )
    with accelerate_utils.iterate_prepared_loader(
        accelerator, prepared, epoch=3
    ) as session:
        assert [batch.tolist() for batch in session] == [
            [0, 1],
            [2, 3],
        ]
    assert source.epochs == [3]
    assert sampler.epochs == [3]


@pytest.mark.parametrize("lab", [False, True])
@pytest.mark.parametrize("sampler_kind", ["sequential", "random", "seedable"])
def test_plain_map_private_sampler_skips_prefix_without_old_reads(
    monkeypatch, lab, sampler_kind
):
    from accelerate.data_loader import SeedableRandomSampler
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)

    class Source(Dataset):
        def __init__(self):
            self.reads = []

        def __len__(self):
            return 16

        def __getitem__(self, index):
            self.reads.append(index)
            return index

    def build():
        data = Source()
        sampler = (
            torch.utils.data.SequentialSampler(data)
            if sampler_kind == "sequential"
            else torch.utils.data.RandomSampler(data)
            if sampler_kind == "random"
            else SeedableRandomSampler(data, generator=None, data_seed=17)
        )
        source = (LabDataLoader if lab else DataLoader)(
            data, sampler=sampler, batch_size=2
        )
        accelerate_utils.configure_data_loader_for_accelerate(
            accelerator, source
        )
        prepared = accelerator.prepare(source)
        accelerate_utils._configure_prepared_loader_recovery(
            accelerator, prepared, source
        )
        return data, prepared

    _, original = build()
    with accelerate_utils.iterate_prepared_loader(
        accelerator, original, epoch=0
    ) as session:
        iterator = session
        next(iterator)
        next(iterator)
        accelerate_utils._commit_prepared_loader_position(original)
        saved = accelerate_utils._capture_prepared_loader_recovery(original)
        remaining = [batch.tolist() for batch in iterator]
    data, restored = build()
    accelerate_utils._load_prepared_loader_recovery(
        accelerator, restored, [saved], epoch=0, processed_batches=2
    )
    with accelerate_utils.iterate_prepared_loader(
        accelerator, restored, epoch=0
    ) as session:
        assert [batch.tolist() for batch in session] == remaining
    assert data.reads == [value for batch in remaining for value in batch]


@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
@pytest.mark.parametrize("sampler_kind", ["random", "seedable", "sequential"])
@pytest.mark.parametrize("processed", [0, 2])
def test_hf_skip_restores_epoch_one_and_keeps_next_epoch(
    monkeypatch, loader_type, sampler_kind, processed
):
    from accelerate.data_loader import SeedableRandomSampler
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)

    def build():
        data = torch.arange(20)
        sampler = (
            torch.utils.data.RandomSampler(data)
            if sampler_kind == "random"
            else SeedableRandomSampler(data, data_seed=17)
            if sampler_kind == "seedable"
            else torch.utils.data.SequentialSampler(data)
        )
        source = loader_type(data, batch_size=2, sampler=sampler)
        accelerate_utils.configure_data_loader_for_accelerate(
            accelerator, source
        )
        prepared = accelerator.prepare(source)
        accelerate_utils._configure_prepared_loader_recovery(
            accelerator, prepared, source
        )
        assert not isinstance(prepared.base_dataloader, LabDataLoader)
        return prepared

    def consume(loader, epoch, snapshot=False):
        with accelerate_utils.iterate_prepared_loader(
            accelerator, loader, epoch=epoch
        ) as session:
            batches = []
            saved = (
                accelerate_utils._capture_prepared_loader_recovery(
                    loader, processed_batches=0
                )
                if snapshot and processed == 0
                else None
            )
            for batch in session:
                batches.append(batch.tolist())
                if snapshot and len(batches) == processed:
                    saved = accelerate_utils._capture_prepared_loader_recovery(
                        loader, processed_batches=processed
                    )
            return batches, saved

    original = build()
    consume(original, 0)
    accelerate_utils._set_prepared_loader_epoch(original, 1)
    expected, saved = consume(original, 1, snapshot=True)
    accelerate_utils._set_prepared_loader_epoch(original, 2)
    following, _ = consume(original, 2)

    restored = build()
    accelerate_utils._load_prepared_loader_recovery(
        accelerator, restored, [saved], epoch=1, processed_batches=processed
    )
    actual, _ = consume(restored, 1)
    assert actual == expected[processed:]
    accelerate_utils._set_prepared_loader_epoch(restored, 2)
    actual_following, _ = consume(restored, 2)
    assert actual_following == following


@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
def test_hf_skip_iterable_replays_and_restores_original_next_epoch(
    monkeypatch, loader_type
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)

    class Source(IterableDataset):
        def __init__(self):
            self.reads = []

        def __iter__(self):
            for index in range(8):
                self.reads.append(index)
                yield index

    def build():
        dataset = Source()
        source = loader_type(dataset, batch_size=2)
        accelerate_utils.configure_data_loader_for_accelerate(
            accelerator, source
        )
        prepared = accelerator.prepare(source)
        accelerate_utils._configure_prepared_loader_recovery(
            accelerator, prepared, source
        )
        assert not isinstance(prepared.base_dataloader, LabDataLoader)
        return dataset, prepared

    dataset, loader = build()
    saved = accelerate_utils._capture_prepared_loader_recovery(
        loader, processed_batches=2
    )
    restored_dataset, restored = build()
    accelerate_utils._load_prepared_loader_recovery(
        accelerator, restored, [saved], epoch=0, processed_batches=2
    )
    for epoch, expected in (
        (0, [[4, 5], [6, 7]]),
        (1, [[0, 1], [2, 3], [4, 5], [6, 7]]),
    ):
        with accelerate_utils.iterate_prepared_loader(
            accelerator, restored, epoch=epoch
        ) as session:
            assert [batch.tolist() for batch in session] == expected
    assert restored_dataset.reads == list(range(8)) * 2


@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
def test_explicit_generator_is_not_rewound_on_recovery(
    monkeypatch, loader_type
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    generator = torch.Generator().manual_seed(22)
    source = loader_type(
        torch.arange(8), batch_size=2, shuffle=True, generator=generator
    )
    accelerate_utils.configure_data_loader_for_accelerate(accelerator, source)
    prepared = accelerator.prepare(source)
    accelerate_utils._configure_prepared_loader_recovery(
        accelerator, prepared, source
    )
    saved = accelerate_utils._capture_prepared_loader_recovery(
        prepared, processed_batches=0
    )
    assert saved["sampler_state"] is None
    torch.rand(3, generator=generator)
    expected_state = generator.get_state().clone()
    accelerate_utils._load_prepared_loader_recovery(
        accelerator, prepared, [saved], epoch=0, processed_batches=1
    )
    with accelerate_utils.iterate_prepared_loader(
        accelerator, prepared, epoch=0
    ) as session:
        assert torch.equal(generator.get_state(), expected_state)
        list(session)


@pytest.mark.parametrize("stop_early", [False, True])
def test_handwritten_loop_uses_prepared_iteration(monkeypatch, stop_early):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )
    dataset = DictIterableDataset(
        [_ManagedRngItem(data=list(range(8)))], shuffle=False
    )
    source = LabDataLoader(dataset, batch_size=2)
    loader = prepare_data_loader(accelerator, source)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    before = model.weight.detach().clone()
    received = []
    for epoch in range(2):
        with accelerate_utils.iterate_prepared_loader(
            accelerator, loader, epoch=epoch
        ) as batches:
            for batch in batches:
                received.extend(batch.tolist())
                with accelerator.accumulate(model):
                    loss = model(batch.float().reshape(-1, 1)).square().mean()
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    if accelerator.sync_gradients:
                        batches.commit()
                if batches.should_stop(local_stop=stop_early):
                    break
        saved = accelerate_utils._capture_prepared_loader_recovery(loader)
        assert saved["loader_state"]["epoch"] == epoch
        assert saved["loader_state"]["delivered_batches"] == (
            1 if stop_early else 4
        )
        assert not accelerator.gradient_state.in_dataloader
        assert not loader.base_dataloader._resume_runtime._live_iterators()
    assert received == (list(range(2)) if stop_early else list(range(8))) * 2
    assert not torch.equal(model.weight.detach(), before)
    assert next(batches, "finished") == "finished"


@pytest.mark.parametrize("outcome", ["batch", "eof", "peer_stop", "error"])
def test_prepared_iteration_first_fetch_rng_and_coordination(
    monkeypatch, preserve_torch_rng, outcome
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    events = []
    failure = OSError("source failed")

    class Source(IterableDataset):
        def __iter__(self):
            events.append("read")
            random.random()
            np.random.random()
            torch.rand(())
            if outcome == "error":
                raise failure
            if outcome != "eof":
                yield torch.tensor(3)

    loader = prepare_data_loader(
        accelerator, DataLoader(Source(), batch_size=1)
    )
    # Stage recovery through the same entrypoint as the trainer. It captures
    # process RNG without adding it to the checkpoint payload.
    saved = accelerate_utils._capture_prepared_loader_recovery(loader)
    accelerate_utils._load_prepared_loader_recovery(
        accelerator, loader, [saved], epoch=0, processed_batches=1
    )
    loader._lab_recovery["replay"] = 0
    expected = loader._lab_recovery["training_rng"]
    assert "training_rng" not in saved

    def set_trigger():
        events.append("set")

    def check_trigger():
        events.append("check")
        if outcome != "error":
            assert random.getstate() == expected[0]
            np.testing.assert_equal(np.random.get_state(), expected[1])
            assert torch.equal(torch.get_rng_state(), expected[2])
        return outcome in {"eof", "peer_stop", "error"}

    monkeypatch.setattr(accelerator, "set_trigger", set_trigger)
    monkeypatch.setattr(accelerator, "check_trigger", check_trigger)
    try:
        with accelerate_utils.iterate_prepared_loader(
            accelerator, loader, epoch=0
        ) as batches:
            if outcome == "error":
                with pytest.raises(OSError) as caught:
                    next(batches)
                assert caught.value is failure
                assert loader._lab_recovery["training_rng"] is expected
                # Propagate the source error through the owning context too.
                raise failure
            elif outcome in {"eof", "peer_stop"}:
                assert list(batches) == []
            else:
                assert next(batches).item() == 3
            assert loader._lab_recovery["training_rng"] is None
    except OSError as exc:
        assert outcome == "error" and exc is failure
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
    assert events[0] == "read"
    assert events[-2:] == (
        ["set", "check"] if outcome in {"eof", "error"} else ["read", "check"]
    )
    assert not accelerator.gradient_state.in_dataloader


@pytest.mark.parametrize(("epoch", "processed_batches"), [(1, 0), (0, 1)])
def test_capture_recovery_validates_expected_lab_position(
    monkeypatch, epoch, processed_batches
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )
    source = LabDataLoader(
        DictIterableDataset([_ManagedRngItem(data=[0, 1])], shuffle=False),
        batch_size=1,
    )
    loader = prepare_data_loader(accelerator, source)
    capture = accelerate_utils._capture_prepared_loader_recovery
    expected = capture(loader, epoch=0, processed_batches=0)
    assert expected["loader_state"]["epoch"] == 0
    assert expected["loader_state"]["delivered_batches"] == 0
    with pytest.raises(RuntimeError, match="data and progress do not match"):
        capture(loader, epoch=epoch, processed_batches=processed_batches)
    assert capture(loader, epoch=0, processed_batches=0) == expected
    # A missing snapshot retains the original empty-payload contract.
    loader.base_dataloader._resume_runtime._prepared_checkpoint = None
    assert capture(
        loader, epoch=epoch, processed_batches=processed_batches
    ) == {"loader_state": None}


@pytest.mark.parametrize("mode", ["owner_change", "restore_without_seed"])
def test_capture_reads_current_sampling_seed_owner(monkeypatch, mode):
    from accelerate.data_loader import SeedableRandomSampler
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    data = torch.arange(8)
    source = DataLoader(
        data, sampler=SeedableRandomSampler(data, data_seed=17), batch_size=2
    )
    loader = prepare_data_loader(accelerator, source)
    sampler = loader.get_sampler()
    generator = sampler.generator
    generator_state = generator.get_state().clone()
    if mode == "owner_change":
        sampler.initial_seed = 29
        expected_seed = 29
    else:
        # An absent saved seed preserves the destination's sampling mode.
        saved = {"loader_state": None, "sampler_state": None, "seed": None}
        accelerate_utils._load_prepared_loader_recovery(
            accelerator, loader, [saved], epoch=0, processed_batches=0
        )
        expected_seed = 17
    state = accelerate_utils._capture_prepared_loader_recovery(
        loader, processed_batches=0
    )
    assert state["seed"] == expected_seed
    assert sampler.generator is generator
    assert torch.equal(generator.get_state(), generator_state)
