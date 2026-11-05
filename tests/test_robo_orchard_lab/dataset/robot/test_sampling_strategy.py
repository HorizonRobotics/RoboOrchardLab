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

"""Configurable mixture policies preserve quotas and default RNG ordering."""

import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from accelerate.data_loader import IterableDatasetShard
from robo_orchard_core.utils.config import ClassType
from test_robo_orchard_lab.dataset.robot._dataset_ex_helpers import (
    ArrayDatasetItem,
)

from robo_orchard_lab.dataset.robot._prefetch import (
    _close_dataloader_owner_resources,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    BatchLoaderConfig,
    DataLoader,
    DatasetItemGroup,
    DictIterableDataset,
)
from robo_orchard_lab.dataset.robot.sampling_strategy import (
    DefaultMixtureSamplingStrategyConfig,
    MixtureSamplingStrategy,
    MixtureSamplingStrategyConfig,
    MixtureSamplingView,
)
from robo_orchard_lab.utils.accelerate import (
    bind_prepared_data_loader_sampling,
)


class ArrayGroup(DatasetItemGroup):
    members: list[ArrayDatasetItem]

    def iter_dataset_items(self):
        yield from self.members


class ReverseStrategy(MixtureSamplingStrategy):
    """Exercise both selection hooks with a visibly different source order."""

    instances: ClassVar[list] = []

    def __init__(self, cfg, *, generator, view):
        super().__init__(cfg, generator=generator, view=view)
        self.initial_view = view
        self.admission_views = []
        self.completions = []
        self.active_calls = 0
        type(self).instances.append(self)
        self._record("created", 0)

    def _record(self, event, value):
        if self.cfg.event_path is None:
            return
        descriptor = os.open(
            self.cfg.event_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
        )
        try:
            os.write(
                descriptor,
                (json.dumps([os.getpid(), event, value]) + "\n").encode(),
            )
        finally:
            os.close(descriptor)

    def choose_admission(self, view):
        self.admission_views.append(view)
        if self.cfg.invalid_phase == "admission":
            return 999
        return max(view.pending_ids)

    def choose_active(self, view):
        self.active_calls += 1
        self._record("active", self.active_calls)
        if self.cfg.invalid_phase == "active":
            return 999
        return max(view.active_ids)

    def residency_completed(self, item_id, *, requeue):
        self.completions.append((item_id, requeue))


class ReverseStrategyConfig(MixtureSamplingStrategyConfig):
    class_type: ClassType[ReverseStrategy] = ReverseStrategy
    event_path: str | None = None
    invalid_phase: str | None = None


def make_items():
    return [
        ArrayDatasetItem(name="single", data=list(range(5))),
        ArrayGroup(
            name="group",
            members=[
                ArrayDatasetItem(name="left", data=list(range(100, 107))),
                ArrayDatasetItem(name="right", data=list(range(200, 204))),
            ],
        ),
    ]


def rng_options(kind):
    return {
        "seed": {"seed": 17},
        "torch": {"generator": torch.Generator().manual_seed(17)},
        "numpy": {"generator": np.random.default_rng(17)},
    }[kind]


def generator_hash(generator):
    if isinstance(generator, torch.Generator):
        state = bytes(generator.get_state().tolist())
    else:
        state = json.dumps(
            generator.bit_generator.state,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    return hashlib.sha256(state).hexdigest()


def collect(dataset):
    batches = [
        item.tolist() if isinstance(item, torch.Tensor) else [item]
        for item in dataset
    ]
    return [value for batch in batches for value in batch], [
        len(batch) for batch in batches
    ]


class TestDefaultStrategyCompatibility:
    @pytest.mark.parametrize("kind", ["torch", "numpy"])
    @pytest.mark.parametrize("slots", [0, 1, 4, 8])
    def test_initial_budget_uses_one_parent_prefetch_and_no_hook_draws(
        self, kind, slots
    ):
        generator = rng_options(kind)["generator"]
        reference = pickle.loads(pickle.dumps(generator))
        uniforms = []
        if slots:
            uniforms = (
                torch.rand(
                    slots, dtype=torch.float64, generator=reference
                ).tolist()
                if kind == "torch"
                else reference.random(slots).tolist()
            )
        view = MixtureSamplingView(
            pending_ids=tuple(range(slots + 1)),
            active_ids=(),
            targets=(20,) * (slots + 1),
            logical_indices=(0,) * (slots + 1),
            remaining_targets=dict.fromkeys(range(slots + 1), 20),
            residence_rows=dict.fromkeys(range(slots + 1), 8),
            segment_remaining={},
            batch_size=1,
            logical_names=("source",),
            initial_residency_slots=slots,
        )
        strategy = DefaultMixtureSamplingStrategyConfig()(
            generator=generator, view=view
        )
        state = generator_hash(reference)
        assert generator_hash(generator) == state
        for idx, uniform in enumerate(uniforms):
            assert strategy.initial_residency_rows(
                idx, rows=8, batch_size=1
            ) == 1 + int(uniform * 8)
        if not slots:
            assert (
                strategy.initial_residency_rows(0, rows=11, batch_size=4) == 11
            )
        assert generator_hash(generator) == state

    @pytest.mark.parametrize("kind", ["seed", "torch", "numpy"])
    @pytest.mark.parametrize("offload", [False, True])
    @pytest.mark.parametrize("mode", ["sample", "short", "drop"])
    def test_default_contract_replays_from_independent_rng_owners(
        self, kind, offload, mode, tmp_path
    ):
        def mixture(config):
            return DictIterableDataset(
                make_items(),
                shuffle=True,
                resample_ratios=[1.4, 0.75],
                max_dataset_concurrency=2,
                offload_num_batches=2 if offload else None,
                offload_dir=str(tmp_path),
                batch_loader_kwargs=(
                    BatchLoaderConfig(batch_size=2, drop_last=mode == "drop")
                    if mode != "sample"
                    else None
                ),
                sampling_strategy=config,
                **rng_options(kind),
            )

        default = DefaultMixtureSamplingStrategyConfig()
        restored = DefaultMixtureSamplingStrategyConfig.model_validate_json(
            default.model_dump_json()
        )
        assert restored.class_type is default.class_type
        assert restored.supports_epoch_replay
        numpy_before = pickle.dumps(np.random.get_state())
        with torch.random.fork_rng(devices=[]):
            global_before = torch.get_rng_state().clone()
            first = mixture(None)
            initial_rng = generator_hash(first._generator)
            rows, sizes = collect(first)
            final_rng = generator_hash(first._generator)
            assert final_rng != initial_rng
            assert torch.equal(torch.get_rng_state(), global_before)
            # Independent owners reproduce the complete stream even after
            # unrelated caller draws change Torch's default RNG.
            torch.rand(7)
            global_before = torch.get_rng_state().clone()
            for config in (restored, None):
                peer = mixture(config)
                assert peer._generator is not first._generator
                assert collect(peer) == (rows, sizes)
                assert generator_hash(peer._generator) == final_rng
                assert generator_hash(first._generator) == final_rng
                assert torch.equal(torch.get_rng_state(), global_before)
        assert pickle.dumps(np.random.get_state()) == numpy_before

        assert Counter(value // 100 for value in rows) == (
            {0: 6, 1: 4, 2: 2} if mode == "drop" else {0: 7, 1: 5, 2: 3}
        )
        assert set(rows) <= (
            set(range(5)) | set(range(100, 107)) | set(range(200, 204))
        )
        assert (
            Counter(sizes)
            == {
                "sample": {1: 15},
                "short": {2: 6, 1: 3},
                "drop": {2: 6},
            }[mode]
        )
        offset = 0
        short_sources = Counter()
        for size in sizes:
            batch = rows[offset : offset + size]
            source = batch[0] // 100
            assert all(value // 100 == source for value in batch)
            if mode == "short" and size == 1:
                short_sources[source] += 1
            offset += size
        if mode == "short":
            assert short_sources == {0: 1, 1: 1, 2: 1}

    @pytest.mark.parametrize("kind", ["torch", "numpy"])
    def test_ordered_offload_keeps_source_order_without_rng_draws(
        self, kind, tmp_path
    ):
        dataset = DictIterableDataset(
            make_items(),
            shuffle=False,
            batch_loader_kwargs=BatchLoaderConfig(batch_size=2),
            offload_num_batches=2,
            offload_dir=str(tmp_path),
            **rng_options(kind),
        )
        before = generator_hash(dataset._generator)
        rows, sizes = collect(dataset)
        assert rows == list(range(5)) + list(range(100, 107)) + list(
            range(200, 204)
        )
        assert sizes == [2, 2, 1, 2, 2, 2, 1, 2, 2]
        assert generator_hash(dataset._generator) == before


class TestCustomStrategy:
    def test_inherited_budget_hook_keeps_natural_nonmultiple_tail(
        self, tmp_path
    ):
        ReverseStrategy.instances.clear()
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(11)))],
            shuffle=True,
            seed=17,
            sampling_strategy=ReverseStrategyConfig(),
            offload_num_batches=2,
            offload_dir=str(tmp_path),
            batch_loader_kwargs=BatchLoaderConfig(batch_size=4),
        )
        rows, sizes = collect(dataset)
        assert sorted(rows) == list(range(11))
        assert sizes == [4, 4, 3]
        assert ReverseStrategy.instances[-1].completions == [(0, False)]

    def test_shortened_custom_budget_requires_complete_yields(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            ReverseStrategy,
            "initial_residency_rows",
            lambda self, item_id, *, rows, batch_size: 3,
        )
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(11)))],
            shuffle=True,
            sampling_strategy=ReverseStrategyConfig(),
            offload_num_batches=2,
            offload_dir=str(tmp_path),
            batch_loader_kwargs=BatchLoaderConfig(batch_size=4),
        )
        with pytest.raises(ValueError, match="initial_residency_rows"):
            list(dataset)

    def test_config_roundtrip_reverse_order_quotas_and_fresh_runtime(self):
        config = ReverseStrategyConfig.model_validate(
            json.loads(ReverseStrategyConfig().model_dump_json())
        )
        assert config.class_type is ReverseStrategy
        assert not config.supports_epoch_replay
        assert (
            ReverseStrategyConfig.model_validate_json(config.model_dump_json())
            == config
        )
        dataset = DictIterableDataset(
            make_items(),
            shuffle=True,
            seed=17,
            sampling_strategy=config,
            resample_ratios=[1.4, 0.75],
            max_dataset_concurrency=2,
        )
        ReverseStrategy.instances.clear()
        first = list(dataset)
        second = list(dataset)
        assert first == second
        assert [row // 100 for row in first] == [2] * 3 + [1] * 5 + [0] * 7
        default = DictIterableDataset(
            make_items(),
            shuffle=True,
            seed=17,
            resample_ratios=[1.4, 0.75],
            max_dataset_concurrency=2,
        )
        assert first != list(default)
        assert Counter(value // 100 for value in first) == {0: 7, 1: 5, 2: 3}
        runtime_a, runtime_b = ReverseStrategy.instances
        assert runtime_a is not runtime_b
        assert runtime_a.active_calls == runtime_b.active_calls == 15
        assert runtime_a.completions == [(2, False), (1, False), (0, False)]

        view = runtime_a.initial_view
        assert view.logical_indices == (0, 1, 1)
        assert view.logical_names == ("single", "group")
        assert view.targets == (7, 5, 3)
        assert dict(view.remaining_targets) == {0: 0, 1: 0, 2: 0}
        for mapping in (
            view.remaining_targets,
            view.segment_remaining,
            runtime_a.admission_views[0].residence_rows,
        ):
            with pytest.raises(TypeError):
                mapping[0] = 999
        with pytest.raises(FrozenInstanceError):
            view.pending_ids = ()

    @pytest.mark.parametrize("offload", [False, True])
    def test_clone_shard_pickle_keep_configuration(self, offload, tmp_path):
        config = ReverseStrategyConfig()
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(start, start + 8)))
                for start in (0, 100, 200)
            ],
            shuffle=True,
            sampling_strategy=config,
            offload_num_batches=2 if offload else None,
            offload_dir=str(tmp_path),
        )
        shard = dataset.shard(num_shards=2, index=1)
        restored = pickle.loads(pickle.dumps(dataset))
        loader = DataLoader(dataset, batch_size=2)
        try:
            assert loader.dataset is not dataset
            for owner in (shard, restored, loader.dataset):
                assert owner.sampling_strategy == config
                assert owner.sampling_strategy.class_type is ReverseStrategy
            assert list(restored) == list(dataset)
            rows, _ = collect(loader)
            assert rows == list(dataset)
            assert Counter(value // 100 for value in shard) == {
                0: 4,
                1: 4,
                2: 4,
            }
        finally:
            _close_dataloader_owner_resources(loader)

    def test_only_successful_residencies_emit_completion(self, tmp_path):
        def mixture():
            return DictIterableDataset(
                [ArrayDatasetItem(data=list(range(9)))],
                shuffle=True,
                sampling_strategy=ReverseStrategyConfig(),
                offload_num_batches=2,
                offload_dir=str(tmp_path),
            )

        ReverseStrategy.instances.clear()
        iterator = iter(mixture())
        next(iterator)
        next(iterator)
        runtime = ReverseStrategy.instances[-1]
        iterator.close()
        assert runtime.completions == []

        assert len(list(mixture())) == 9
        assert ReverseStrategy.instances[-1].completions == [
            (0, True),
            (0, True),
            (0, True),
            (0, False),
        ]

    @pytest.mark.parametrize("phase", ["admission", "active"])
    def test_invalid_selection_fails_at_framework_boundary(self, phase):
        dataset = DictIterableDataset(
            make_items(),
            shuffle=True,
            sampling_strategy=ReverseStrategyConfig(invalid_phase=phase),
        )
        with pytest.raises(ValueError, match=f"choose_{phase}"):
            list(dataset)

    @pytest.mark.parametrize("kind", ["seed", "torch", "numpy"])
    def test_fallback_keeps_common_stream_and_positional_coverage(self, kind):
        def mixture():
            return DictIterableDataset(
                [
                    ArrayDatasetItem(data=list(range(start, start + 8)))
                    for start in (0, 100, 200)
                ],
                shuffle=True,
                sampling_strategy=ReverseStrategyConfig(),
                **rng_options(kind),
            )

        source = mixture()
        common = source._bind_sampling(None)
        expected = list(common)
        ranks = []
        for rank in (0, 1):
            dataset = mixture()
            dataset._bind_sampling(rank)
            before = generator_hash(dataset._generator)
            wrapper = IterableDatasetShard(
                dataset,
                batch_size=2,
                drop_last=True,
                num_processes=2,
                process_index=rank,
            )
            loader = torch.utils.data.DataLoader(wrapper, batch_size=None)
            assert bind_prepared_data_loader_sampling(
                SimpleNamespace(process_index=rank), loader
            )
            assert wrapper.dataset.sampling_strategy.class_type is (
                ReverseStrategy
            )
            ranks.append(list(loader))
            assert generator_hash(dataset._generator) == before
            assert dataset._sampling_rank == rank
        reconstructed = [
            value
            for offset in range(0, len(ranks[0]), 2)
            for rank in ranks
            for value in rank[offset : offset + 2]
        ]
        assert reconstructed == expected
        assert sorted(reconstructed) == sorted(
            value
            for start in (0, 100, 200)
            for value in range(start, start + 8)
        )

    def test_spawn_workers_rebuild_independent_custom_runtime(self, tmp_path):
        event_path = tmp_path / "strategy-events.jsonl"
        config = ReverseStrategyConfig.model_validate_json(
            ReverseStrategyConfig(event_path=str(event_path)).model_dump_json()
        )
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(start, start + 24)))
                for start in (0, 100, 200)
            ],
            shuffle=True,
            seed=17,
            sampling_strategy=config,
            offload_num_batches=2,
            offload_dir=str(tmp_path / "offload"),
            max_dataset_concurrency=2,
        )
        loader = DataLoader(
            dataset,
            batch_size=3,
            num_workers=2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        epochs = []
        try:
            for epoch in (0, 1):
                loader.dataset.set_epoch(epoch)
                batches = [batch.tolist() for batch in loader]
                for worker in (0, 1):
                    values = [v for batch in batches[worker::2] for v in batch]
                    assert [v // 100 for v in values] == (
                        [2] * 12 + [1] * 12 + [0] * 12
                    )
                epochs.append([v for batch in batches for v in batch])
        finally:
            _close_dataloader_owner_resources(loader)
        assert epochs[0] != epochs[1]
        assert (
            sorted(epochs[0])
            == sorted(epochs[1])
            == sorted(
                value
                for start in (0, 100, 200)
                for value in range(start, start + 24)
            )
        )
        by_process = defaultdict(list)
        for line in event_path.read_text().splitlines():
            pid, event, counter = json.loads(line)
            by_process[pid].append((event, counter))
        assert len(by_process) == 2
        assert os.getpid() not in by_process
        expected_runtime = [("created", 0)] + [
            ("active", index) for index in range(1, 37)
        ]
        assert all(
            events == expected_runtime * 2 for events in by_process.values()
        )
