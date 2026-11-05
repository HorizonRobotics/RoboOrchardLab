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

"""Tests for native DataLoader checkpoint state and iterator recovery."""

import gc
import multiprocessing as mp
import pickle
import random
import warnings
import weakref

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from robo_orchard_core.utils.config import ClassType
from test_robo_orchard_lab.dataset.robot._dataset_ex_helpers import (
    ArrayDataset,
    ArrayDatasetItem,
    BatchedArrayDataset,
    isolated_accelerate_state as isolated_accelerate_state,
)
from torch.utils.data import (
    Dataset,
)

from robo_orchard_lab.dataset.robot import (
    BatchLoaderConfig,
    DataLoader,
    DatasetWithIndices,
    IterableWithLenDataset,
)
from robo_orchard_lab.dataset.robot._prefetch import (
    _close_dataloader_owner_resources,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DictIterableDataset,
    ShuffleConfig,
)
from robo_orchard_lab.dataset.robot.sampling_strategy import (
    DefaultMixtureSamplingStrategyConfig,
    MixtureSamplingStrategy,
    MixtureSamplingStrategyConfig,
)


class _RecoveryMixtureStrategy(MixtureSamplingStrategy):
    def choose_admission(self, view):
        choose = max if self.cfg.reverse else min
        return choose(view.pending_ids)

    def choose_active(self, view):
        choose = max if self.cfg.reverse else min
        return choose(view.active_ids)

    def residency_completed(self, item_id, *, requeue):
        pass


class _RecoveryMixtureConfig(MixtureSamplingStrategyConfig):
    class_type: ClassType[_RecoveryMixtureStrategy] = _RecoveryMixtureStrategy
    reverse: bool = False


class TestMixtureStrategyRecovery:
    @staticmethod
    def build(strategy=None, *, offload=False, shuffle=True):
        return DataLoader(
            DictIterableDataset(
                [
                    ArrayDatasetItem(data=list(range(12))),
                    ArrayDatasetItem(data=list(range(100, 112))),
                ],
                shuffle=shuffle,
                seed=42,
                sampling_strategy=strategy,
                offload_num_batches=2 if offload else None,
            ),
            batch_size=2,
        )

    @pytest.mark.parametrize("offload", [False, True])
    def test_default_identity_restores_current_sample_order(self, offload):
        expected = [batch.tolist() for batch in self.build(offload=offload)]
        loader = self.build(offload=offload)
        iterator = iter(loader)
        try:
            next(iterator)
            next(iterator)
            state = loader.state_dict()
        finally:
            iterator.close()
        assert state["layout"]["sampling_strategy"] == (
            DefaultMixtureSamplingStrategyConfig().model_dump(mode="json")
        )
        assert state["layout"]["sampling_policy_version"] == (
            "remaining-quota-initial-jitter-v1"
        )
        assert state["recovery"]["accuracy"] == (
            "unknown" if offload else "worker_local"
        )
        restored = self.build(offload=offload)
        restored.load_state_dict(state)
        if offload:
            with pytest.warns(UserWarning, match="replaying"):
                actual = [batch.tolist() for batch in restored]
        else:
            actual = [batch.tolist() for batch in restored]
        assert actual == expected[2:]

    @pytest.mark.parametrize(
        "field", ["sampling_strategy", "sampling_policy_version"]
    )
    def test_old_default_layout_cannot_replay_new_policy(self, field):
        state = self.build().state_dict()
        state["layout"].pop(field)
        with pytest.raises(ValueError, match="sampling strategy differs"):
            self.build().load_state_dict(state)

    def test_ordered_layout_keeps_existing_replay_format(self):
        expected = [batch.tolist() for batch in self.build(shuffle=False)]
        loader = self.build(shuffle=False)
        iterator = iter(loader)
        try:
            next(iterator)
            state = loader.state_dict()
        finally:
            iterator.close()
        assert "sampling_strategy" not in state["layout"]
        assert "sampling_policy_version" not in state["layout"]
        restored = self.build(shuffle=False)
        restored.load_state_dict(state)
        assert [batch.tolist() for batch in restored] == expected[1:]

    @pytest.mark.parametrize("supports_replay", [False, True])
    def test_custom_capability_and_reconstruction(self, supports_replay):
        cfg = _RecoveryMixtureConfig(supports_epoch_replay=supports_replay)
        expected = [batch.tolist() for batch in self.build(cfg)]
        loader = self.build(cfg)
        iterator = iter(loader)
        try:
            next(iterator)
            state = loader.state_dict()
        finally:
            iterator.close()
        assert "sampling_strategy" in state["layout"]
        assert state["recovery"]["accuracy"] == (
            "worker_local" if supports_replay else "unknown"
        )
        restored = self.build(cfg)
        restored.load_state_dict(state)
        if supports_replay:
            actual = [batch.tolist() for batch in restored]
        else:
            with pytest.warns(UserWarning, match="replaying"):
                actual = [batch.tolist() for batch in restored]
        assert actual == expected[1:]

    def test_changed_custom_config_rejects_replay(self):
        loader = self.build(_RecoveryMixtureConfig(supports_epoch_replay=True))
        iterator = iter(loader)
        try:
            next(iterator)
            state = loader.state_dict()
        finally:
            iterator.close()
        restored = self.build(
            _RecoveryMixtureConfig(supports_epoch_replay=True, reverse=True)
        )
        with pytest.raises(ValueError, match="sampling strategy differs"):
            restored.load_state_dict(state)


def _resume_worker_init(worker_id: int) -> None:
    dataset = torch.utils.data.get_worker_info().dataset
    assert isinstance(dataset, DictIterableDataset)
    dataset._generator.manual_seed(1000 + worker_id)


def _seeded_resume_worker_init(worker_id: int) -> None:
    dataset = torch.utils.data.get_worker_info().dataset
    assert isinstance(dataset, DictIterableDataset)
    dataset._sampling_seed += worker_id + 1


def _plus_hundred(batch):
    return torch.utils.data.default_collate(batch) + 100


def _consume_loader_in_spawn(loader, result_queue):
    batches = [batch.tolist() for batch in loader]
    dataset = loader.dataset
    if isinstance(dataset, DictIterableDataset):
        dataset_generator = dataset._generator
    elif isinstance(dataset, IterableWithLenDataset):
        dataset_generator = dataset.indice_sampler.generator
    else:
        dataset_generator = None
    same_generator = dataset_generator is loader.generator
    result_queue.put((batches, same_generator))


class TestResumeIteratorFactory:
    @pytest.mark.parametrize("kind", ["mixture", "indexed"])
    def test_batched_sample_prefetch_skip_replays(self, kind):
        kwargs = {
            "shuffle": ShuffleConfig(
                shuffle=True, chunk_size=2, prefetch_factor=4
            ),
            "seed": 42,
            "batch_loader_kwargs": BatchLoaderConfig(batch_size=3),
        }
        if kind == "mixture":
            dataset = DictIterableDataset(
                [ArrayDatasetItem(data=list(range(60)))], **kwargs
            )
        else:
            dataset = IterableWithLenDataset(list(range(60)), **kwargs)
        iterator = dataset.create_iterator(skip_n_items=1)
        try:
            batches = list(iterator)
        finally:
            iterator.close()
        values = [value for batch in batches for value in batch.tolist()]
        assert len(batches) == 19
        assert len(values) == len(set(values)) == 57
        assert set(values) <= set(range(60))

    def test_virtual_getitems_skip_avoids_reading_old_rows(self):
        def build():
            source = BatchedArrayDataset(list(range(80)))
            dataset = DatasetWithIndices(source).to_iterable_dataset(
                shuffle=True, generator=torch.Generator().manual_seed(6)
            )
            return source, dataset

        _, expected_dataset = build()
        expected = list(expected_dataset)
        source, dataset = build()
        actual = list(dataset.create_iterator(skip_n_items=21))
        assert actual == expected[21:]
        assert sorted(
            x for batch in source.getitems_calls for x in batch
        ) == sorted(actual)
        assert source.getitem_calls == []

    @pytest.mark.parametrize("batching", [False, True])
    def test_mixture_skip_replays_scheduler_without_old_sample_reads(
        self, monkeypatch, batching
    ):
        def build():
            return DictIterableDataset(
                [
                    ArrayDatasetItem(data=list(range(start, start + 24)))
                    for start in (0, 100, 200)
                ],
                shuffle=True,
                seed=0,
                max_dataset_concurrency=2,
                batch_loader_kwargs=(
                    BatchLoaderConfig(batch_size=3) if batching else None
                ),
            )

        def normalize(items):
            return [
                item.tolist() if isinstance(item, torch.Tensor) else item
                for item in items
            ]

        expected = normalize(list(build()))
        reads: list[int] = []
        original = ArrayDataset.__getitem__

        def track_read(dataset, index):
            value = original(dataset, index)
            reads.append(value)
            return value

        monkeypatch.setattr(ArrayDataset, "__getitem__", track_read)
        actual = normalize(list(build().create_iterator(skip_n_items=7)))
        assert actual == expected[7:]
        expected_reads = (
            [value for batch in expected[7:] for value in batch]
            if batching
            else expected[7:]
        )
        assert sorted(reads) == sorted(expected_reads)
        assert list(build().create_iterator(skip_n_items=10_000)) == []

    def test_default_seed_is_42_without_consuming_default_rng(self):
        before = torch.get_rng_state().clone()
        implicit = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(12)))], shuffle=True
        )
        explicit = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(12)))], shuffle=True, seed=42
        )
        assert list(implicit) == list(explicit)
        assert torch.equal(before, torch.get_rng_state())

    @pytest.mark.parametrize("bad", [-1, True, 1.5])
    def test_factory_rejects_invalid_skip(self, bad):
        with pytest.raises(ValueError):
            DictIterableDataset([]).create_iterator(skip_n_items=bad)


class TestLoaderResume:
    @pytest.mark.parametrize("num_workers", [0, 1])
    @pytest.mark.parametrize("prefetch_factor", [1, 4])
    def test_batched_sample_prefetch_resume(
        self, num_workers, prefetch_factor
    ):
        def build():
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(60)))],
                    shuffle=ShuffleConfig(
                        shuffle=True,
                        chunk_size=1 if prefetch_factor == 1 else 2,
                        prefetch_factor=prefetch_factor,
                    ),
                    seed=42,
                    batch_loader_kwargs=BatchLoaderConfig(batch_size=3),
                ),
                batch_size=3,
                num_workers=num_workers,
            )

        original = build()
        restored = build()
        try:
            iterator = iter(original)
            try:
                assert next(iterator).numel() == 3
                state = original.state_dict()
            finally:
                iterator.close()
            fast = prefetch_factor == 1
            assert state["recovery"]["strategy"] == (
                "worker_prefix" if fast else "batch_replay"
            )
            assert state["recovery"]["accuracy"] == (
                "worker_local" if fast else "unknown"
            )
            restored.load_state_dict(state)
            if fast:
                suffix = list(restored)
            else:
                with pytest.warns(UserWarning, match="replaying"):
                    suffix = list(restored)
            values = [value for batch in suffix for value in batch.tolist()]
            assert len(suffix) == 19
            assert len(values) == len(set(values)) == 57
            assert set(values) <= set(range(60))
            assert restored.state_dict()["delivered_batches"] == 20
        finally:
            _close_dataloader_owner_resources(original)
            _close_dataloader_owner_resources(restored)

    def test_state_schema_rejects_unsupported_version(self):
        loader = DataLoader(
            DictIterableDataset([ArrayDatasetItem(data=list(range(8)))]),
            batch_size=2,
        )
        state = loader.state_dict()
        assert state["schema"] == 1
        assert "rng" not in state
        unsupported = dict(state, schema=999)
        with pytest.raises(
            ValueError, match="unsupported DataLoader state schema"
        ):
            loader.load_state_dict(unsupported)
        without_seed = dict(state, seed=None)
        with pytest.raises(ValueError, match="managed sampling seed"):
            loader.load_state_dict(without_seed)

    def test_restored_worker_prefix_skips_old_sample_reads(self, monkeypatch):
        def build():
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(24)))],
                    shuffle=True,
                    seed=42,
                ),
                batch_size=2,
            )

        reference = [batch.tolist() for batch in build()]
        first = build()
        iterator = iter(first)
        prefix = [next(iterator).tolist() for _ in range(3)]
        state = first.state_dict()
        iterator.close()
        reads = []
        original = ArrayDataset.__getitem__

        def track(dataset, index):
            value = original(dataset, index)
            reads.append(value)
            return value

        monkeypatch.setattr(ArrayDataset, "__getitem__", track)
        resumed = build()
        resumed.load_state_dict(state)
        suffix = [batch.tolist() for batch in resumed]
        assert prefix + suffix == reference
        assert sorted(reads) == sorted(
            value for batch in suffix for value in batch
        )

    def test_iterator_start_failure_clears_worker_offsets(self, monkeypatch):
        from robo_orchard_lab.dataset.robot import _resume

        loader = DataLoader(
            DictIterableDataset([ArrayDatasetItem(data=list(range(8)))]),
            batch_size=2,
        )
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()
        iterator.close()
        loader.load_state_dict(state)

        def fail(_self):
            raise RuntimeError("injected iterator startup failure")

        monkeypatch.setattr(_resume.TorchDataLoader, "__iter__", fail)
        with pytest.raises(
            RuntimeError, match="injected iterator startup failure"
        ):
            next(iter(loader))
        assert loader.dataset._resume_worker_offsets is None

    @pytest.mark.parametrize("close_early", [False, True])
    def test_restored_pool_does_not_reapply_offsets_next_epoch(
        self, close_early
    ):
        def build():
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(24)))],
                    shuffle=True,
                    seed=42,
                ),
                batch_size=2,
                num_workers=2,
                persistent_workers=True,
            )

        first = build()
        restored = build()
        try:
            iterator = iter(first)
            next(iterator)
            state = first.state_dict()
            iterator.close()
            restored.load_state_dict(state)
            iterator = iter(restored)
            if close_early:
                next(iterator)
                workers = list(restored._iterator._workers)
                iterator.close()
                assert all(not worker.is_alive() for worker in workers)
            else:
                list(iterator)
            restored._notify_epoch(1)
            actual = sorted(
                value for batch in restored for value in batch.tolist()
            )
            assert actual == list(range(24))
        finally:
            _close_dataloader_owner_resources(first)
            _close_dataloader_owner_resources(restored)

    @pytest.mark.parametrize("random_sampler", [False, True])
    @pytest.mark.parametrize("shared_sampler", [False, True])
    @pytest.mark.parametrize("external_epoch", [False, True])
    def test_direct_epoch_notifies_once_when_resumed(
        self, random_sampler, shared_sampler, external_epoch
    ):
        class Source(Dataset):
            def __init__(self, generator):
                self.generator = generator
                self.epoch = 0
                self.epochs = []
                self.decision = 0

            def __len__(self):
                return 20

            def __getitem__(self, index):
                return self.decision * 100 + index

            def set_epoch(self, epoch):
                self.epoch = epoch
                self.epochs.append(epoch)
                self.decision = torch.randint(
                    10000, (), generator=self.generator
                ).item()

        def build():
            generator = torch.Generator().manual_seed(17)
            source = Source(generator)
            if random_sampler:
                sampler_generator = (
                    generator
                    if shared_sampler
                    else torch.Generator().manual_seed(31)
                )
                sampler = torch.utils.data.RandomSampler(
                    source, generator=sampler_generator
                )
            else:
                sampler = torch.utils.data.SequentialSampler(source)
            loader = DataLoader(
                source, sampler=sampler, batch_size=2, generator=generator
            )
            return loader, source

        original, source = build()
        if external_epoch:
            source.set_epoch(1)
        else:
            original._notify_epoch(1)
        decision = source.decision
        start = original.state_dict()
        iterator = iter(original)
        prefix = next(iterator).tolist()
        saved = original.state_dict()
        suffix = [batch.tolist() for batch in iterator]
        assert source.epochs == [1]
        assert source.decision == decision
        assert all(value // 100 == decision for value in prefix)

        checkpoints = [(saved, suffix)]
        if not external_epoch:
            checkpoints.append((start, [prefix] + suffix))
        for state, expected in checkpoints:
            restored, restored_source = build()
            restored.load_state_dict(state)
            assert restored_source.epochs == [1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                actual = [batch.tolist() for batch in restored]
            assert restored_source.epochs == [1]
            assert len(actual) == len(expected)
            if random_sampler and shared_sampler:
                assert restored.sampler.generator is restored.generator

        already_notified, already_source = build()
        already_source.set_epoch(1)
        already_notified.load_state_dict(saved)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            list(already_notified)
        assert already_source.epochs == [1]

    def test_resume_runtime_does_not_retain_loader(self):
        loader = DataLoader(range(4), batch_size=2)
        runtime = loader._resume_runtime
        owner_ref = weakref.ref(loader)
        del loader
        gc.collect()
        assert owner_ref() is None
        assert not runtime._live_iterators()

    @pytest.mark.parametrize(
        "kind",
        ["map", "map_shared", "dict", "dict_numpy", "shared", "indexed"],
    )
    def test_inactive_loader_pickle_and_spawn_keep_private_state(self, kind):
        generator = torch.Generator().manual_seed(17)
        torch.rand(5, generator=generator)
        if kind == "map":
            loader = DataLoader(ArrayDataset(list(range(8))), batch_size=2)
        elif kind == "map_shared":
            loader = DataLoader(
                ArrayDataset(list(range(8))),
                batch_size=2,
                shuffle=True,
                generator=generator,
            )
        elif kind == "indexed":
            dataset = DatasetWithIndices(
                ArrayDataset(list(range(8)))
            ).to_iterable_dataset(shuffle=True, generator=generator)
            loader = DataLoader(dataset, batch_size=2, generator=generator)
        else:
            numpy_generator = np.random.default_rng(17)
            numpy_generator.random(5)
            dataset_generator = (
                generator
                if kind == "shared"
                else numpy_generator
                if kind == "dict_numpy"
                else None
            )
            dataset = DictIterableDataset(
                [ArrayDatasetItem(data=list(range(8)))],
                shuffle=True,
                generator=dataset_generator,
            )
            loader = DataLoader(
                dataset,
                batch_size=2,
                generator=generator if kind == "shared" else None,
            )
        before = loader.state_dict()
        cloned = pickle.loads(pickle.dumps(loader))
        assert cloned.state_dict() == before
        if kind in ("shared", "indexed"):
            assert (
                cloned.dataset._generator
                if kind == "shared"
                else cloned.dataset.indice_sampler.generator
            ) is cloned.generator
        if kind == "map_shared":
            assert cloned.sampler.generator is cloned.generator
        context = mp.get_context("spawn")
        results = context.Queue()
        process = context.Process(
            target=_consume_loader_in_spawn, args=(loader, results)
        )
        try:
            process.start()
            batches, shared = results.get(timeout=30)
            process.join(timeout=30)
            assert process.exitcode == 0
            assert sorted(
                value for batch in batches for value in batch
            ) == list(range(8))
            assert shared == (kind in ("shared", "indexed"))
            assert loader.state_dict() == before
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            results.close()

    def test_pickle_preserves_closed_and_pending_positions(self):
        def build():
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(12)))],
                    shuffle=True,
                    generator=torch.Generator().manual_seed(9),
                ),
                batch_size=2,
            )

        loader = build()
        iterator = iter(loader)
        next(iterator)
        state = loader.state_dict()
        iterator.close()
        cloned = pickle.loads(pickle.dumps(loader))
        assert cloned.state_dict() == state
        restored = build()
        restored.load_state_dict(state)
        cloned_pending = pickle.loads(pickle.dumps(restored))
        assert cloned_pending.state_dict() == state
        assert [batch.tolist() for batch in cloned_pending] == [
            batch.tolist() for batch in restored
        ]

    def test_pickle_rejects_active_iterators_and_kept_worker_pools(self):
        loader = DataLoader(ArrayDataset(list(range(8))), batch_size=2)
        iterator = iter(loader)
        with pytest.raises(RuntimeError, match="active iterator"):
            pickle.dumps(loader)
        iterator.close()
        pickle.dumps(loader)

        persistent = DataLoader(
            ArrayDataset(list(range(8))),
            batch_size=2,
            num_workers=1,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        iterator = iter(persistent)
        try:
            next(iterator)
            iterator.close()
            with pytest.raises(
                RuntimeError, match="active iterator|persistent pool"
            ):
                pickle.dumps(persistent)
        finally:
            _close_dataloader_owner_resources(persistent)

    def test_nonpersistent_loader_rebuilds_with_updated_native_options(self):
        loader = DataLoader(ArrayDataset(list(range(8))), batch_size=2)
        assert [batch.tolist() for batch in loader] == [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]
        loader.collate_fn = _plus_hundred
        generator = torch.Generator().manual_seed(19)
        expected_generator = torch.Generator().set_state(generator.get_state())
        loader.generator = generator
        assert [batch.tolist() for batch in loader] == [
            [100, 101],
            [102, 103],
            [104, 105],
            [106, 107],
        ]
        torch.empty((), dtype=torch.int64).random_(
            generator=expected_generator
        )
        assert torch.equal(
            generator.get_state(), expected_generator.get_state()
        )
        loader.num_workers = 1
        loader.multiprocessing_context = "spawn"
        assert [batch.tolist() for batch in loader] == [
            [100, 101],
            [102, 103],
            [104, 105],
            [106, 107],
        ]

    def test_explicit_generator_supports_changed_worker_context(self):
        generator = torch.Generator().manual_seed(17)
        loader = DataLoader(
            DictIterableDataset(
                [ArrayDatasetItem(data=list(range(12)))],
                shuffle=True,
                generator=generator,
            ),
            batch_size=2,
        )
        assert len(list(loader)) == 6
        loader.num_workers = 2
        loader.multiprocessing_context = "spawn"
        loader.worker_init_fn = _resume_worker_init
        assert sum(len(batch) for batch in loader) == 12
        state = loader.state_dict()
        assert len(state["worker_items"]) == 2
        assert state["recovery"]["accuracy"] == "unknown"

    def test_nonpersistent_iterators_keep_independent_carriers(self):
        loader = DataLoader(ArrayDataset(list(range(8))), batch_size=2)
        first = iter(loader)
        assert next(first).tolist() == [0, 1]
        loader.collate_fn = _plus_hundred
        second = iter(loader)
        assert next(second).tolist() == [100, 101]
        first.close()
        assert next(second).tolist() == [102, 103]
        second.close()

    def test_persistent_pool_rebuilds_after_teardown_with_new_options(self):
        loader = DataLoader(
            ArrayDataset(list(range(8))),
            batch_size=2,
            num_workers=1,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        try:
            assert [batch.tolist() for batch in loader] == [
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ]
            old_iterator = loader._iterator
            old_worker = old_iterator._workers[0]
            assert old_worker.is_alive()
            _close_dataloader_owner_resources(loader)
            assert loader._iterator is None
            assert not old_worker.is_alive()

            loader.collate_fn = _plus_hundred
            generator = torch.Generator().manual_seed(19)
            expected = torch.Generator().set_state(generator.get_state())
            loader.generator = generator
            assert [batch.tolist() for batch in loader] == [
                [100, 101],
                [102, 103],
                [104, 105],
                [106, 107],
            ]
            new_iterator = loader._iterator
            new_worker = new_iterator._workers[0]
            assert new_iterator is not old_iterator
            assert new_worker.pid != old_worker.pid
            assert new_worker.is_alive()
            torch.empty((), dtype=torch.int64).random_(generator=expected)
            assert torch.equal(generator.get_state(), expected.get_state())
        finally:
            _close_dataloader_owner_resources(loader)

    @pytest.mark.parametrize("kind", ["python", "numpy"])
    def test_custom_sampler_rng_is_used_without_breaking_iteration(self, kind):
        class CustomSampler(torch.utils.data.Sampler):
            def __init__(self):
                self.generator = (
                    random.Random(3)
                    if kind == "python"
                    else np.random.default_rng(3)
                )

            def __iter__(self):
                indices = list(range(12))
                self.generator.shuffle(indices)
                return iter(indices)

            def __len__(self):
                return 12

        def build():
            return DataLoader(
                ArrayDataset(list(range(12))),
                batch_size=2,
                sampler=CustomSampler(),
            )

        expected = [batch.tolist() for batch in build()]
        original = build()
        iterator = iter(original)
        prefix = [next(iterator).tolist() for _ in range(2)]
        state = original.state_dict()
        assert state["recovery"]["accuracy"] == "unknown"
        iterator.close()
        restored = build()
        restored.load_state_dict(state)
        with pytest.warns(UserWarning, match="replaying"):
            suffix = [batch.tolist() for batch in restored]
        assert prefix + suffix == expected

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_default_sampling_seed_follows_its_generator_identity(
        self, shuffle
    ):
        loader = DataLoader(ArrayDataset(list(range(8))), shuffle=shuffle)
        expected = 42 if shuffle else None
        assert loader._get_sampling_seed() == expected
        assert loader.state_dict()["seed"] == expected
        if shuffle:
            loader.sampler.generator = torch.Generator().manual_seed(42)
        else:
            loader.generator = torch.Generator().manual_seed(42)
        assert loader._get_sampling_seed() is None
        assert loader.state_dict()["seed"] is None

    def test_sequential_restore_without_sampling_seed(
        self, isolated_accelerate_state, request
    ):
        original_rng = torch.get_rng_state()
        request.addfinalizer(lambda: torch.set_rng_state(original_rng))
        accelerator = Accelerator(cpu=True)
        torch.manual_seed(11)
        source = DataLoader(range(12), batch_size=2)
        wrapper = accelerator.prepare(source)
        base = DataLoader._from_prepared_base(
            wrapper.base_dataloader, source_loader=source
        )
        iterator = iter(base)
        assert next(iterator).tolist() == [0, 1]
        state = base.state_dict()
        assert state["seed"] is None
        iterator.close()

        torch.manual_seed(99)
        new_source = DataLoader(range(12), batch_size=2)
        new_wrapper = accelerator.prepare(new_source)
        restored = DataLoader._from_prepared_base(
            new_wrapper.base_dataloader, source_loader=new_source
        )
        restored.load_state_dict(state)
        iterator = iter(restored)
        assert next(iterator).tolist() == [2, 3]
        assert restored.state_dict()["seed"] is None
        iterator.close()

    @pytest.mark.parametrize("seed", [42, 71])
    def test_explicit_advanced_map_generator_has_no_inferred_seed(self, seed):
        generator = torch.Generator().manual_seed(seed)
        torch.rand(7, generator=generator)
        loader = DataLoader(
            ArrayDataset(list(range(8))), shuffle=True, generator=generator
        )
        assert loader._get_sampling_seed() is None
        assert loader.state_dict()["seed"] is None
        assert loader.sampler.generator is generator

    def test_explicit_sampler_generator_has_no_inferred_seed(self):
        data = ArrayDataset(list(range(8)))
        generator = torch.Generator().manual_seed(42)
        torch.rand(7, generator=generator)
        sampler = torch.utils.data.RandomSampler(data, generator=generator)
        loader = DataLoader(data, sampler=sampler)
        assert loader._get_sampling_seed() is None
        assert loader.state_dict()["seed"] is None

    def test_explicit_iterable_state_keeps_destination_sampling_mode(self):
        generator = torch.Generator().manual_seed(42)
        torch.rand(7, generator=generator)
        explicit = DataLoader(
            DatasetWithIndices(
                ArrayDataset(list(range(8)))
            ).to_iterable_dataset(shuffle=True, generator=generator),
            batch_size=2,
        )
        iterator = iter(explicit)
        next(iterator)
        state = explicit.state_dict()
        iterator.close()
        assert state["seed"] is None
        assert state["recovery"]["strategy"] == "batch_replay"
        assert state["recovery"]["accuracy"] == "unknown"

        default = DataLoader(
            DatasetWithIndices(
                ArrayDataset(list(range(8)))
            ).to_iterable_dataset(shuffle=True),
            batch_size=2,
        )
        default.load_state_dict(state)
        assert default.state_dict()["seed"] == 42
        assert default.state_dict()["recovery"]["accuracy"] == "unknown"
        with pytest.warns(UserWarning, match="replaying"):
            list(default)
        assert default.state_dict()["seed"] == 42
        default._notify_epoch(1)
        assert default.state_dict()["seed"] == 42

    def test_default_map_generators_are_isolated_and_reproducible(self):
        def build():
            return DataLoader(
                ArrayDataset(list(range(24))), batch_size=2, shuffle=True
            )

        before = torch.get_rng_state().clone()
        first = [batch.tolist() for batch in build()]
        assert torch.equal(before, torch.get_rng_state())
        second = [batch.tolist() for batch in build()]
        assert first == second

    def test_custom_sampler_generator_supports_batch_replay(
        self,
    ):
        def build():
            generator = torch.Generator().manual_seed(23)
            torch.rand(5, generator=generator)
            sampler = torch.utils.data.WeightedRandomSampler(
                [1.0] * 24, 24, replacement=True, generator=generator
            )
            return DataLoader(
                ArrayDataset(list(range(24))),
                batch_size=2,
                sampler=sampler,
                generator=generator,
            )

        expected = [batch.tolist() for batch in build()]
        loader = build()
        iterator = iter(loader)
        prefix = [next(iterator).tolist() for _ in range(3)]
        state = loader.state_dict()
        assert state["recovery"]["accuracy"] == "unknown"
        iterator.close()
        resumed = build()
        resumed.load_state_dict(state)
        with pytest.warns(UserWarning, match="replaying"):
            suffix = [batch.tolist() for batch in resumed]
        assert prefix + suffix == expected

    @pytest.mark.parametrize("context", ["fork", "spawn"])
    def test_worker_init_with_explicit_generator_replays_with_unknown_accuracy(
        self, context
    ):
        if context not in mp.get_all_start_methods():
            pytest.skip(f"{context} is unavailable")

        def build():
            generator = torch.Generator().manual_seed(7)
            torch.rand(5, generator=generator)
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(32)))],
                    shuffle=True,
                    generator=generator,
                ),
                batch_size=2,
                num_workers=2,
                multiprocessing_context=context,
                worker_init_fn=_resume_worker_init,
            )

        def partition(batches):
            output = [[], []]
            for batch in batches:
                for item in batch.tolist():
                    output[int(item >= 16)].append(item)
            return output

        expected = partition(list(build()))
        original = build()
        iterator = iter(original)
        prefix = [next(iterator) for _ in range(4)]
        state = original.state_dict()
        assert state["recovery"]["accuracy"] == "unknown"
        iterator.close()
        resumed = build()
        resumed.load_state_dict(state)
        with pytest.warns(UserWarning, match="replaying"):
            suffix = list(resumed)
        assert sum(map(len, partition(prefix + suffix))) == sum(
            map(len, expected)
        )
        assert resumed.state_dict()["recovery"]["accuracy"] == "unknown"

    def test_offload_restoration_replays_actual_batches(self, tmp_path):
        def build():
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(32)))],
                    shuffle=True,
                    seed=42,
                    offload_num_batches=2,
                    offload_dir=str(tmp_path),
                ),
                batch_size=2,
            )

        expected = [batch.tolist() for batch in build()]
        original = build()
        iterator = iter(original)
        prefix = [next(iterator).tolist() for _ in range(4)]
        state = original.state_dict()
        iterator.close()
        resumed = build()
        resumed.load_state_dict(state)
        with pytest.warns(UserWarning, match="replaying"):
            suffix = [batch.tolist() for batch in resumed]
        assert prefix + suffix == expected
        assert resumed.state_dict()["recovery"]["strategy"] == "batch_replay"

    def test_replay_fails_if_saved_prefix_exceeds_source(self):
        def build():
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(4)))],
                    generator=torch.Generator().manual_seed(3),
                ),
                batch_size=2,
            )

        loader = build()
        state = loader.state_dict()
        state["delivered_batches"] = 3
        loader.load_state_dict(state)
        with pytest.warns(UserWarning, match="replaying"):
            with pytest.raises(RuntimeError, match="saved batch prefix"):
                list(loader)

    def test_changed_worker_layout_downgrades_pending_and_later_state(self):
        def build(num_workers):
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(24)))], seed=42
                ),
                batch_size=2,
                num_workers=num_workers,
            )

        original = build(0)
        iterator = iter(original)
        next(iterator)
        state = original.state_dict()
        iterator.close()
        resumed = build(1)
        resumed.load_state_dict(state)
        assert resumed.state_dict()["recovery"]["accuracy"] == "unknown"
        iterator = iter(resumed)
        with pytest.warns(UserWarning, match="replaying"):
            next(iterator)
        later = resumed.state_dict()
        iterator.close()
        assert later["recovery"]["accuracy"] == "unknown"
        again = build(1)
        again.load_state_dict(later)
        with pytest.warns(UserWarning, match="replaying"):
            list(again)
        assert again.state_dict()["recovery"]["accuracy"] == "unknown"

    def test_changed_generator_alias_is_accepted_for_replay(self):
        def build(shared):
            data_generator = torch.Generator().manual_seed(5)
            loader_generator = (
                data_generator if shared else torch.Generator().manual_seed(5)
            )
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(8)))],
                    generator=data_generator,
                ),
                generator=loader_generator,
            )

        state = build(True).state_dict()
        changed = build(False)
        changed.load_state_dict(state)
        assert changed.state_dict()["recovery"]["accuracy"] == "unknown"
        with pytest.warns(UserWarning, match="replaying"):
            assert len(list(changed)) == 8

    @pytest.mark.parametrize("in_order", [True, False])
    @pytest.mark.parametrize("dataset_batching", [False, True])
    def test_worker_delivery_vector_survives_two_restores(
        self, in_order, dataset_batching
    ):
        def build():
            return DataLoader(
                DictIterableDataset(
                    [
                        ArrayDatasetItem(data=list(range(start, start + 24)))
                        for start in (0, 100, 200)
                    ],
                    shuffle=True,
                    seed=42,
                    max_dataset_concurrency=2,
                ),
                batch_size=2,
                num_workers=2,
                in_order=in_order,
                use_dataset_side_batching=dataset_batching,
            )

        def partition(batches):
            result = [[], []]
            for batch in batches:
                for value in batch.tolist():
                    result[int(value % 100 >= 12)].append(value)
            return result

        reference = partition(list(build()))
        first = build()
        first_iter = iter(first)
        prefix = [next(first_iter) for _ in range(3)]
        first_state = first.state_dict()
        assert first_state["delivered_batches"] == 3
        assert sum(first_state["worker_items"]) == (
            3 if dataset_batching else 6
        )
        first_iter.close()

        second = build()
        second.load_state_dict(first_state)
        assert second.state_dict() == first_state
        second._notify_epoch(first_state["epoch"])
        assert second.state_dict() == first_state
        second_iter = iter(second)
        middle = [next(second_iter) for _ in range(2)]
        second_state = second.state_dict()
        assert second_state["delivered_batches"] == 5
        assert sum(second_state["worker_items"]) == (
            5 if dataset_batching else 10
        )
        second_iter.close()

        third = build()
        third.load_state_dict(second_state)
        suffix = list(third)
        actual = partition(prefix + middle + suffix)
        assert actual == reference

    def test_load_rejects_active_and_state_rejects_two_iterators(self):
        loader = DataLoader(
            DictIterableDataset(
                [ArrayDatasetItem(data=list(range(12)))], seed=42
            ),
            batch_size=2,
        )
        state = loader.state_dict()
        first = iter(loader)
        with pytest.raises(RuntimeError, match="active"):
            loader.load_state_dict(state)
        second = iter(loader)
        with pytest.raises(RuntimeError, match="concurrent"):
            loader.state_dict()
        first.close()
        second.close()

    @pytest.mark.parametrize("kind", ["torch", "numpy", "numpy_mt"])
    @pytest.mark.parametrize("shared", [False, True])
    def test_explicit_generator_replays_two_restores(self, kind, shared):
        if kind.startswith("numpy") and shared:
            pytest.skip("NumPy generator cannot be a Torch loader generator")

        def build():
            if kind == "torch":
                data_generator = torch.Generator().manual_seed(17)
                torch.rand(7, generator=data_generator)
            else:
                data_generator = (
                    np.random.Generator(np.random.MT19937(17))
                    if kind == "numpy_mt"
                    else np.random.default_rng(17)
                )
                data_generator.random(7)
            loader_generator = (
                data_generator if shared else torch.Generator().manual_seed(31)
            )
            return DataLoader(
                DictIterableDataset(
                    [
                        ArrayDatasetItem(data=list(range(start, start + 24)))
                        for start in (0, 100, 200)
                    ],
                    shuffle=True,
                    generator=data_generator,
                    max_dataset_concurrency=2,
                ),
                batch_size=2,
                generator=loader_generator,
            )

        def values(batches):
            return [item for batch in batches for item in batch.tolist()]

        reference = values(list(build()))
        first = build()
        iterator = iter(first)
        prefix = [next(iterator) for _ in range(3)]
        state = first.state_dict()
        assert state["recovery"]["accuracy"] == "unknown"
        iterator.close()

        second = build()
        second.load_state_dict(state)
        assert second.state_dict() == state
        iterator = iter(second)
        with pytest.warns(UserWarning, match="replaying"):
            middle = [next(iterator) for _ in range(2)]
        state = second.state_dict()
        iterator.close()

        third = build()
        third.load_state_dict(state)
        with pytest.warns(UserWarning, match="replaying"):
            suffix = list(third)
        assert len(values(prefix + middle + suffix)) == len(reference)
        assert third.state_dict()["recovery"]["accuracy"] == "unknown"

    @pytest.mark.parametrize("context", ["fork", "spawn"])
    @pytest.mark.parametrize("kind", ["torch", "numpy", "shared"])
    @pytest.mark.parametrize("rows", [1, 24])
    def test_persistent_generator_epoch_boundary_and_zero_worker(
        self, context, kind, rows
    ):
        if context not in mp.get_all_start_methods():
            pytest.skip(f"{context} is unavailable")

        def build():
            if kind == "numpy":
                data_generator = np.random.default_rng(17)
                data_generator.random(7)
            else:
                data_generator = torch.Generator().manual_seed(17)
                torch.rand(7, generator=data_generator)
            loader_generator = (
                data_generator
                if kind == "shared"
                else torch.Generator().manual_seed(31)
            )
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(rows)))],
                    shuffle=True,
                    generator=data_generator,
                ),
                batch_size=1,
                num_workers=2,
                persistent_workers=True,
                multiprocessing_context=context,
                generator=loader_generator,
            )

        def partition(batches):
            result = [[], []]
            split = (rows + 1) // 2
            for batch in batches:
                for value in batch.tolist():
                    result[int(value >= split)].append(value)
            return result

        original = build()
        try:
            list(original)
            original._notify_epoch(1)
            state = original.state_dict()
            assert state["epoch"] == 1
            assert state["delivered_batches"] == 0
            assert state["recovery"]["accuracy"] == "unknown"
            expected_epoch_1 = partition(list(original))
            original._notify_epoch(2)
            expected_epoch_2 = partition(list(original))

            restored = build()
            try:
                restored.load_state_dict(state)
                with pytest.warns(UserWarning, match="replaying"):
                    actual_epoch_1 = partition(list(restored))
                restored._notify_epoch(2)
                actual_epoch_2 = partition(list(restored))
                assert list(map(len, actual_epoch_1)) == list(
                    map(len, expected_epoch_1)
                )
                assert list(map(len, actual_epoch_2)) == list(
                    map(len, expected_epoch_2)
                )
            finally:
                _close_dataloader_owner_resources(restored)
        finally:
            _close_dataloader_owner_resources(original)

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_worker_init_downgrades_seeded_worker_recovery(self, num_workers):
        def build():
            return DataLoader(
                DictIterableDataset(
                    [ArrayDatasetItem(data=list(range(24)))],
                    shuffle=True,
                    seed=42,
                ),
                batch_size=2,
                num_workers=num_workers,
                worker_init_fn=_seeded_resume_worker_init,
                **(
                    {"multiprocessing_context": "spawn"} if num_workers else {}
                ),
            )

        original = build()
        iterator = iter(original)
        prefix = [next(iterator)]
        state = original.state_dict()
        expected = "unknown" if num_workers else "worker_local"
        assert state["recovery"]["accuracy"] == expected
        iterator.close()

        restored = build()
        restored.load_state_dict(state)
        assert restored.state_dict()["recovery"]["accuracy"] == expected
        if num_workers:
            with pytest.warns(UserWarning, match="replaying"):
                suffix = list(restored)
        else:
            suffix = list(restored)
        assert sum(len(batch) for batch in prefix + suffix) == 24
        assert restored.state_dict()["recovery"]["accuracy"] == expected

    def test_worker_init_change_only_downgrades_exported_capability(self):
        loader = DataLoader(
            DictIterableDataset(
                [ArrayDatasetItem(data=list(range(12)))], seed=42
            ),
            batch_size=2,
            num_workers=2,
        )
        assert loader.state_dict()["recovery"]["accuracy"] == "worker_local"
        loader.worker_init_fn = _seeded_resume_worker_init
        assert loader.state_dict()["recovery"]["accuracy"] == "unknown"
        loader.worker_init_fn = None
        assert loader.state_dict()["recovery"]["accuracy"] == "unknown"

    @pytest.mark.parametrize("kind", ["indexed", "dict"])
    def test_uncloned_iterable_subclass_keeps_behavior_and_state(self, kind):
        class CustomIndexed(IterableWithLenDataset):
            def __iter__(self):
                for value in super().__iter__():
                    yield value + self.marker

        class CustomDict(DictIterableDataset):
            def __iter__(self):
                for value in super().__iter__():
                    yield value + self.marker

        if kind == "indexed":
            dataset = CustomIndexed(ArrayDataset(list(range(4))))
        else:
            dataset = CustomDict(
                [ArrayDatasetItem(data=list(range(4)))],
                generator=torch.Generator().manual_seed(4),
            )
        dataset.marker = 10
        loader = DataLoader(dataset, batch_size=2)
        assert loader.dataset is dataset
        assert [batch.tolist() for batch in loader] == [[10, 11], [12, 13]]
        assert loader.state_dict()["recovery"]["accuracy"] == "unknown"

    @pytest.mark.parametrize("seedable", [False, True])
    def test_direct_map_restores_by_replaying_batches(self, seedable):
        from accelerate.data_loader import SeedableRandomSampler

        class Source(Dataset):
            def __init__(self):
                self.reads = []

            def __len__(self):
                return 12

            def __getitem__(self, index):
                self.reads.append(index)
                return index

        def build():
            source = Source()
            sampler = (
                SeedableRandomSampler(source, data_seed=17)
                if seedable
                else torch.utils.data.SequentialSampler(source)
            )
            return source, DataLoader(source, sampler=sampler, batch_size=2)

        _, original = build()
        iterator = iter(original)
        prefix = [next(iterator).tolist() for _ in range(2)]
        state = original.state_dict()
        assert state["recovery"]["strategy"] == "batch_replay"
        suffix = [batch.tolist() for batch in iterator]
        source, restored = build()
        restored.load_state_dict(state)
        assert restored.state_dict()["recovery"]["strategy"] == "batch_replay"
        assert restored.state_dict()["recovery"]["accuracy"] == "unknown"
        with pytest.warns(UserWarning, match="replaying"):
            actual = [batch.tolist() for batch in restored]
        assert actual == suffix
        assert source.reads == [
            value for batch in prefix + suffix for value in batch
        ]
