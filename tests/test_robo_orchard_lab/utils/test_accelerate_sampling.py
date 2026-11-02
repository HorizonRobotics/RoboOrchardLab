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

"""Accelerate sampling ownership and random-generator isolation tests."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.utils.dataclasses import DataLoaderConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
)

import robo_orchard_lab.utils.accelerate as accelerate_utils
from robo_orchard_lab.dataset.robot import (
    DatasetItem,
    DictIterableDataset,
    _resume,
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


class _RandomWorkerMap(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return (
            index,
            torch.rand(()).item(),
            np.random.random(),
            random.random(),
        )


class _RandomWorkerItem(DatasetItem[_RandomWorkerMap]):
    class_type: ClassType[_RandomWorkerMap] = _RandomWorkerMap

    def get_dataset_row_num(self):
        return 8

    def _create_dataset(self):
        return _RandomWorkerMap()


class _RandomWorkerIterable(IterableDataset):
    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        for index in range(worker_id, 8, worker_count):
            yield (
                index,
                torch.rand(()).item(),
                np.random.random(),
                random.random(),
            )


@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
@pytest.mark.parametrize("kind", ["map", "iterable"])
def test_default_worker_seed_follows_run_seed_after_prepare(
    loader_type, kind, monkeypatch, preserve_torch_rng
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)

    def collect(seed):
        torch.manual_seed(seed)
        dataset = (
            _RandomWorkerMap() if kind == "map" else _RandomWorkerIterable()
        )
        options = {"shuffle": True} if kind == "map" else {}
        loader = loader_type(dataset, batch_size=2, num_workers=1, **options)
        before = torch.get_rng_state().clone()
        prepared = prepare_data_loader(accelerator, loader)
        batches = list(prepared)
        assert torch.equal(torch.get_rng_state(), before)
        return [
            tuple(tensor.tolist() for tensor in batch) for batch in batches
        ]

    first = collect(11)
    repeated = collect(11)
    changed = collect(99)
    assert first == repeated
    assert [batch[0] for batch in first] == [batch[0] for batch in changed]
    for field in (1, 2, 3):
        assert [batch[field] for batch in first] != [
            batch[field] for batch in changed
        ]


@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
def test_default_worker_generator_rank_binding_and_explicit_alias(
    loader_type, preserve_torch_rng
):
    torch.manual_seed(77)
    before = torch.get_rng_state().clone()
    seeds = []
    batches = []
    for rank in (0, 1):
        loader = loader_type(_RandomWorkerMap(), shuffle=True, num_workers=1)
        accelerator = SimpleNamespace(num_processes=1, process_index=rank)
        accelerate_utils.configure_data_loader_for_accelerate(
            accelerator, loader
        )
        seeds.append(loader.generator.initial_seed())
        assert loader.sampler.generator.initial_seed() == 42
        assert loader.sampler.generator is not loader.generator
        batches.append(list(loader))
    assert seeds[0] != seeds[1]
    assert [batch[0].item() for batch in batches[0]] == [
        batch[0].item() for batch in batches[1]
    ]
    assert [batch[1].item() for batch in batches[0]] != [
        batch[1].item() for batch in batches[1]
    ]
    assert torch.equal(torch.get_rng_state(), before)

    shared = torch.Generator().manual_seed(8)
    loader = loader_type(range(8), shuffle=True, generator=shared)
    accelerator = SimpleNamespace(num_processes=1, process_index=1)
    accelerate_utils.configure_data_loader_for_accelerate(accelerator, loader)
    assert loader.generator is shared
    assert loader.sampler.generator is shared
    assert shared.initial_seed() == 8


@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
def test_advanced_worker_generator_stays_bound_to_its_rank(
    loader_type, preserve_torch_rng
):
    loader = loader_type(range(4), shuffle=True)
    rank_one = SimpleNamespace(num_processes=1, process_index=1)
    accelerate_utils.configure_data_loader_for_accelerate(rank_one, loader)
    torch.rand((), generator=loader.generator)
    advanced = loader.generator.get_state().clone()

    accelerate_utils.configure_data_loader_for_accelerate(rank_one, loader)
    assert torch.equal(loader.generator.get_state(), advanced)

    rank_zero = SimpleNamespace(num_processes=1, process_index=0)
    with pytest.raises(RuntimeError, match="advanced worker RNG"):
        accelerate_utils.configure_data_loader_for_accelerate(
            rank_zero, loader
        )


def test_dict_sampling_stays_fixed_when_worker_run_seed_changes(
    monkeypatch, preserve_torch_rng
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    results = []
    for seed in (11, 99):
        torch.manual_seed(seed)
        source = LabDataLoader(
            DictIterableDataset([_RandomWorkerItem()], shuffle=True),
            batch_size=2,
            num_workers=1,
        )
        assert source.dataset._sampling_seed == 42
        before = torch.get_rng_state().clone()
        batches = list(prepare_data_loader(accelerator, source))
        assert torch.equal(torch.get_rng_state(), before)
        results.append(
            [tuple(value.tolist() for value in batch) for batch in batches]
        )
    assert [batch[0] for batch in results[0]] == [
        batch[0] for batch in results[1]
    ]
    assert [batch[1:] for batch in results[0]] != [
        batch[1:] for batch in results[1]
    ]


@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
@pytest.mark.parametrize("use_wrapper", [False, True])
def test_dispatcher_retains_explicit_worker_generator_and_recovery(
    loader_type, use_wrapper, monkeypatch
):
    from accelerate.data_loader import DataLoaderDispatcher
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    generator = torch.Generator().manual_seed(109)
    source = loader_type(
        _RandomWorkerIterable(),
        batch_size=2,
        num_workers=1,
        generator=generator,
    )
    before = generator.get_state().clone()
    if use_wrapper:
        prepared = prepare_data_loader(accelerator, source)
    else:
        accelerate_utils.configure_data_loader_for_accelerate(
            accelerator, source
        )
        prepared = accelerator.prepare(source)
        accelerate_utils._attach_prepared_lab_loader(prepared, source)
    assert isinstance(prepared, DataLoaderDispatcher)
    assert prepared.base_dataloader.generator is generator
    list(prepared)
    assert not torch.equal(generator.get_state(), before)


def test_dispatcher_preserves_explicit_dataset_loader_generator_alias(
    monkeypatch,
):
    from accelerate.data_loader import DataLoaderDispatcher
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    shared = torch.Generator().manual_seed(17)
    dataset = DictIterableDataset([_RandomWorkerItem()], generator=shared)
    source = LabDataLoader(dataset, generator=shared, num_workers=1)
    prepared = prepare_data_loader(accelerator, source)
    assert isinstance(prepared, DataLoaderDispatcher)
    assert prepared.base_dataloader.generator is shared
    assert prepared.dataset._generator is shared
    before = shared.get_state().clone()
    list(prepared)
    assert not torch.equal(shared.get_state(), before)
    assert prepared.dataset._generator is prepared.base_dataloader.generator


@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
def test_managed_map_generators_are_isolated_and_keep_explicit_aliases(
    loader_type, monkeypatch
):
    """Preparing missing data streams must not consume model RNG state."""
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    loader = loader_type(list(range(6)), batch_size=2, shuffle=True)
    before = torch.get_rng_state().clone()
    accelerate_utils._bind_managed_map_generators(loader)
    prepared = accelerator.prepare(loader)
    assert torch.equal(torch.get_rng_state(), before)
    assert isinstance(loader.generator, torch.Generator)
    assert isinstance(loader.sampler.generator, torch.Generator)
    assert loader.sampler.generator.initial_seed() == 42
    assert loader.generator is not loader.sampler.generator
    assert prepared.base_dataloader.generator is loader.generator
    assert prepared.get_sampler().generator is loader.sampler.generator
    assert not accelerate_utils._attach_prepared_lab_loader(prepared, loader)
    assert accelerate_utils._identified_data_seed(prepared) == 42

    shared = torch.Generator().manual_seed(7)
    explicit = loader_type(
        list(range(6)), batch_size=2, shuffle=True, generator=shared
    )
    before = torch.get_rng_state().clone()
    accelerate_utils._bind_managed_map_generators(explicit)
    prepared_explicit = accelerator.prepare(explicit)
    assert torch.equal(torch.get_rng_state(), before)
    assert explicit.generator is shared
    assert explicit.sampler.generator is shared
    assert prepared_explicit.base_dataloader.generator is shared
    assert prepared_explicit.get_sampler().generator is shared
    assert not accelerate_utils._attach_prepared_lab_loader(
        prepared_explicit, explicit
    )
    assert accelerate_utils._identified_data_seed(prepared_explicit) is None


def test_managed_map_does_not_change_unknown_loader_or_sampler():
    class CustomLoader(DataLoader):
        pass

    class CustomSampler(torch.utils.data.RandomSampler):
        pass

    custom = CustomLoader(list(range(6)), shuffle=True)
    accelerate_utils._bind_managed_map_generators(custom)
    assert custom.generator is None
    assert custom.sampler.generator is None
    assert accelerate_utils._identified_data_seed(custom) is None

    data = list(range(6))
    sampler = CustomSampler(data)
    loader = DataLoader(data, sampler=sampler)
    accelerate_utils._bind_managed_map_generators(loader)
    assert loader.generator.initial_seed() == _resume._worker_seed(
        torch.initial_seed(), 0
    )
    assert sampler.generator is None

    explicit = torch.Generator().manual_seed(17)
    sampler = torch.utils.data.RandomSampler(data, generator=explicit)
    loader = DataLoader(data, sampler=sampler)
    accelerate_utils._bind_managed_map_generators(loader)
    assert sampler.generator is explicit


def test_capture_reads_current_owned_sampler_seed(
    monkeypatch, preserve_torch_rng
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    prepared = prepare_data_loader(
        accelerator, DataLoader(torch.arange(8), batch_size=2, shuffle=True)
    )
    sampler = prepared.get_sampler()
    assert sampler._lab_default_generator is sampler.generator
    sampler.generator.manual_seed(17)
    saved = accelerate_utils._capture_prepared_loader_recovery(
        prepared, processed_batches=0
    )
    assert saved["seed"] == 17
    origin = torch.Generator().set_state(saved["sampler_state"])
    assert origin.initial_seed() == saved["seed"]


@pytest.mark.parametrize("already_common", [False, True])
def test_bind_fallback_preserves_caller_epoch_and_generator(already_common):
    from accelerate.data_loader import DataLoaderShard, IterableDatasetShard

    source = DictIterableDataset(
        [_ManagedRngItem(data=list(range(24)))], shuffle=True, seed=42
    )
    source._sampling_rank = None if already_common else 0
    expected = list(source)
    wrapper = IterableDatasetShard(
        source, batch_size=1, num_processes=2, process_index=0
    )
    prepared = DataLoaderShard(wrapper, batch_size=1)
    accelerator = SimpleNamespace(process_index=0)
    assert accelerate_utils.bind_prepared_data_loader_sampling(
        accelerator, prepared
    )
    owned = prepared.dataset.dataset
    assert owned is not source
    assert owned._sampling_rank is None
    prepared.set_epoch(3)
    assert list(source) == expected
    assert source._sampling_epoch.value == 0
    assert owned._sampling_epoch.value == 3
    rows = list(owned)
    assert accelerate_utils.bind_prepared_data_loader_sampling(
        accelerator, prepared
    )
    assert prepared.dataset.dataset is not owned
    assert list(prepared.dataset.dataset) == rows
    assert prepared.dataset.dataset._sampling_epoch.value == 3


def test_dispatcher_binding_leaves_shared_source_untouched():
    from accelerate import Accelerator
    from accelerate.data_loader import DataLoaderDispatcher

    source = DictIterableDataset(
        [_ManagedRngItem(data=list(range(24)))], shuffle=True, seed=42
    )
    loader = DataLoader(source, batch_size=2)
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(dispatch_batches=True),
    )
    prepared = accelerator.prepare_data_loader(loader)
    assert isinstance(prepared, DataLoaderDispatcher)
    assert prepared.dataset is source
    expected = list(source)
    state = source._generator.get_state().clone()
    assert (
        accelerate_utils.bind_prepared_data_loader_sampling(
            accelerator, prepared
        )
        is False
    )
    assert prepared.dataset is source
    assert source._sampling_seed == 42
    assert source._generator.get_state().equal(state)
    assert list(source) == expected


def test_fallback_owner_can_be_safely_reused_in_native_loader():
    from accelerate.data_loader import DataLoaderShard, IterableDatasetShard

    from robo_orchard_lab.dataset.robot.dataset_ex import (
        DataLoader as RODataLoader,
    )

    source = DictIterableDataset(
        [_ManagedRngItem(data=list(range(24)))], shuffle=True, seed=42
    )
    fallback = DataLoaderShard(
        IterableDatasetShard(
            source, batch_size=1, num_processes=2, process_index=0
        ),
        batch_size=1,
    )
    accelerator = SimpleNamespace(process_index=0)
    assert accelerate_utils.bind_prepared_data_loader_sampling(
        accelerator, fallback
    )
    common_owner = fallback.dataset.dataset
    assert common_owner._sampling_rank is None

    loader = RODataLoader(common_owner, batch_size=1)
    native = DataLoaderShard(loader.dataset, batch_size=1)
    assert accelerate_utils.bind_prepared_data_loader_sampling(
        accelerator, native
    )
    fresh = DictIterableDataset(
        [_ManagedRngItem(data=list(range(24)))], shuffle=True, seed=42
    )
    assert list(native.dataset) == list(fresh)
    assert common_owner._sampling_rank is None

    shared_torch_loader = DataLoader(common_owner, batch_size=1)
    shared_native = DataLoaderShard(shared_torch_loader.dataset, batch_size=1)
    with pytest.raises(ValueError, match="without a new owner"):
        accelerate_utils.bind_prepared_data_loader_sampling(
            accelerator, shared_native
        )
    assert common_owner._sampling_rank is None


def test_fallback_rebinds_a_preexisting_static_shard_across_ranks():
    """Static dataset sharding cannot be mistaken for final native mode."""
    from collections import Counter

    from accelerate import Accelerator
    from accelerate.data_loader import DataLoaderShard, IterableDatasetShard

    Accelerator()
    sources = [
        DictIterableDataset(
            [_ManagedRngItem(data=list(range(24)))],
            shuffle=True,
            seed=42,
        ).shard(num_shards=2, index=0)
        for rank in (0, 1)
    ]
    expected = list(sources[0])
    prepared = [
        DataLoaderShard(
            IterableDatasetShard(
                sources[rank],
                batch_size=1,
                num_processes=2,
                process_index=rank,
            ),
            batch_size=1,
        )
        for rank in (0, 1)
    ]
    for rank, loader in enumerate(prepared):
        assert accelerate_utils.bind_prepared_data_loader_sampling(
            SimpleNamespace(process_index=rank), loader
        )
    rows = [
        int(value)
        for loader in prepared
        for batch in loader
        for value in batch
    ]
    assert Counter(rows) == Counter(expected)
    assert all(source._sampling_rank == 0 for source in sources)


@pytest.mark.parametrize("explicit_generator", [False, True])
@pytest.mark.parametrize("loader_type", [DataLoader, LabDataLoader])
def test_prepared_sampler_keeps_custom_batch_generator_separate(
    monkeypatch, explicit_generator, loader_type
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(cpu=True)
    data = list(range(8))
    generator = (
        torch.Generator().manual_seed(17) if explicit_generator else None
    )
    sampler = torch.utils.data.RandomSampler(data, generator=generator)

    class CustomBatchSampler(torch.utils.data.BatchSampler):
        def __init__(self):
            super().__init__(sampler, batch_size=2, drop_last=False)
            self.generator = torch.Generator().manual_seed(99)

    batch_sampler = CustomBatchSampler()
    batch_generator_state = batch_sampler.generator.get_state().clone()
    source = loader_type(data, batch_sampler=batch_sampler)
    loader = prepare_data_loader(accelerator, source)
    actual_generator = sampler.generator
    expected_generator = torch.Generator().set_state(
        actual_generator.get_state()
    )
    expected = torch.randperm(len(data), generator=expected_generator).tolist()
    assert accelerate_utils._private_random_sampler_generator(loader) is (
        None if explicit_generator else actual_generator
    )
    assert accelerate_utils._identified_data_seed(loader) == (
        None if explicit_generator else 42
    )
    assert [value for batch in loader for value in batch.tolist()] == expected
    assert torch.equal(
        batch_sampler.generator.get_state(), batch_generator_state
    )
    if explicit_generator:
        assert actual_generator is generator
