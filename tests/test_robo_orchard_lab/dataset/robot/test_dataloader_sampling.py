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

"""Tests for managed DataLoader sampling RNG ownership."""

import multiprocessing as mp
import pickle

import numpy as np
import pytest
import torch
from test_robo_orchard_lab.dataset.robot._dataset_ex_helpers import (
    ArrayDataset,
    ArrayDatasetItem,
)

from robo_orchard_lab.dataset.robot import (
    DataLoader,
)
from robo_orchard_lab.dataset.robot._prefetch import (
    _close_dataloader_owner_resources,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DictIterableDataset,
)


class TestManagedSamplingRng:
    """Framework sampling is independent of default and worker RNGs."""

    @staticmethod
    def _dataset(*, offload_dir=None):
        return DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(start, start + 24)))
                for start in (0, 100, 200)
            ],
            shuffle=True,
            seed=42,
            offload_num_batches=2 if offload_dir is not None else None,
            offload_dir=offload_dir,
        )

    def test_default_rng_is_not_used_by_managed_sampling(self):
        before = torch.get_rng_state().clone()
        first = self._dataset()
        sequence = list(first)
        assert torch.equal(before, torch.get_rng_state())
        torch.rand(17)
        replay = self._dataset()
        assert list(replay) == sequence
        assert sorted(sequence) == sorted(range(24)) + list(
            range(100, 124)
        ) + list(range(200, 224))

    def test_rank_epoch_and_complete_reiteration(self):
        first = self._dataset()
        rank0 = list(first)
        assert list(first) == rank0
        other_rank = self._dataset()
        other_rank._bind_sampling(1)
        assert list(other_rank) != rank0
        first.set_epoch(3)
        epoch3 = list(first)
        assert epoch3 != rank0
        first.set_epoch(0)
        assert list(first) == rank0
        first.set_epoch(3)
        assert list(first) == epoch3

    def test_epoch_notification_does_not_restart_active_iterator(self):
        reference = list(self._dataset())
        dataset = self._dataset()
        iterator = iter(dataset)
        prefix = [next(iterator) for _ in range(7)]
        dataset.set_epoch(0)
        assert prefix + list(iterator) == reference

    def test_pickle_and_clone_keep_independent_sampling_owners(self):
        dataset = self._dataset()
        dataset.set_epoch(5)
        reference = list(dataset)
        restored = pickle.loads(pickle.dumps(dataset))
        assert list(restored) == reference
        restored.set_epoch(0)
        assert list(dataset) == reference
        loader = DataLoader(
            dataset, batch_size=4, use_dataset_side_batching=True
        )
        try:
            rows = [int(v) for batch in loader for v in batch.reshape(-1)]
            same = DataLoader(
                self._dataset(), batch_size=4, use_dataset_side_batching=True
            )
            try:
                same.dataset.set_epoch(5)
                assert rows == [
                    int(v) for batch in same for v in batch.reshape(-1)
                ]
            finally:
                _close_dataloader_owner_resources(same)
        finally:
            _close_dataloader_owner_resources(loader)

    @pytest.mark.parametrize("batching", [False, True])
    def test_separate_loaders_do_not_share_epoch_with_source_or_peer(
        self, batching
    ):
        dataset = self._dataset()
        dataset.set_epoch(3)
        source_reference = list(dataset)
        loader_a = DataLoader(
            dataset, batch_size=4, use_dataset_side_batching=batching
        )
        loader_b = DataLoader(
            dataset, batch_size=4, use_dataset_side_batching=batching
        )

        def collect(loader):
            return [int(v) for batch in loader for v in batch.reshape(-1)]

        try:
            reference_a = collect(loader_a)
            reference_b = collect(loader_b)
            assert reference_a == reference_b
            loader_a.dataset.set_epoch(1)
            assert collect(loader_a) != reference_a
            assert collect(loader_b) == reference_b
            assert list(dataset) == source_reference
            dataset.set_epoch(2)
            assert collect(loader_b) == reference_b
        finally:
            _close_dataloader_owner_resources(loader_a)
            _close_dataloader_owner_resources(loader_b)

    def test_shards_do_not_share_epoch_with_source_or_peer(self):
        source = self._dataset()
        source.set_epoch(3)
        first = source.shard(2, 0)
        second = source.shard(2, 1)
        source_rows, first_rows, second_rows = (
            list(dataset) for dataset in (source, first, second)
        )
        first.set_epoch(1)
        assert list(first) != first_rows
        assert source.epoch == second.epoch == 3
        assert list(source) == source_rows
        assert list(second) == second_rows
        source.set_epoch(2)
        assert first.epoch == 1
        assert second.epoch == 3
        assert list(second) == second_rows

    def test_resampling_ignores_processing_default_rng(self, monkeypatch):
        def build():
            return DictIterableDataset(
                [
                    ArrayDatasetItem(data=list(range(i, i + 24)))
                    for i in (0, 100, 200)
                ],
                shuffle=True,
                seed=0,
                resample_ratios=(1.5, 0.5, 1.0),
            )

        expected = list(build())
        original = ArrayDataset.__getitem__

        def noisy_read(self, index):
            torch.rand(3)
            np.random.random()
            return original(self, index)

        monkeypatch.setattr(ArrayDataset, "__getitem__", noisy_read)
        assert list(build()) == expected
        assert [
            sum(lo <= x < lo + 24 for x in expected) for lo in (0, 100, 200)
        ] == [36, 12, 24]

    @pytest.mark.parametrize("bad", [-1, True, 1.5])
    def test_rejects_invalid_seed(self, bad):
        with pytest.raises((ValueError, TypeError)):
            DictIterableDataset([], seed=bad)

    def test_explicit_generator_is_an_alternative_to_managed_seed(self):
        with pytest.raises(ValueError):
            DictIterableDataset([], seed=42, generator=torch.Generator())

    @pytest.mark.parametrize(
        "generator_factory",
        [
            lambda: torch.Generator().manual_seed(42),
            lambda: np.random.default_rng(42),
        ],
        ids=["torch", "numpy"],
    )
    def test_explicit_generator_advances_across_epoch_notifications(
        self, generator_factory
    ):
        def build():
            return DictIterableDataset(
                [
                    ArrayDatasetItem(data=list(range(start, start + 24)))
                    for start in (0, 100, 200)
                ],
                shuffle=True,
                generator=generator_factory(),
            )

        reference = build()
        first, second = list(reference), list(reference)
        assert first != second
        dataset = build()
        dataset.set_epoch(3)
        assert list(dataset) == first
        dataset.set_epoch(3)
        assert list(dataset) == second

    @pytest.mark.parametrize("context", ["fork", "spawn"])
    @pytest.mark.parametrize("batching", [False, True])
    @pytest.mark.parametrize("offload", [False, True])
    def test_persistent_workers_receive_epoch_and_ignore_worker_seed(
        self, tmp_path, context, batching, offload
    ):
        if context not in mp.get_all_start_methods():
            pytest.skip(f"{context} is unavailable")
        cache = str(tmp_path / "offload") if offload else None
        traces = []
        for worker_seed in (11, 99):
            dataset = self._dataset(offload_dir=cache)
            loader = DataLoader(
                dataset,
                batch_size=4,
                num_workers=2,
                persistent_workers=True,
                multiprocessing_context=context,
                use_dataset_side_batching=batching,
                generator=torch.Generator().manual_seed(worker_seed),
            )
            epochs = []
            try:
                for epoch in (0, 1, 0):
                    loader.dataset.set_epoch(epoch)
                    rows = [
                        int(v) for batch in loader for v in batch.reshape(-1)
                    ]
                    epochs.append(rows)
                assert epochs[0] == epochs[2]
                assert epochs[0] != epochs[1]
                assert sorted(epochs[0]) == sorted(range(24)) + list(
                    range(100, 124)
                ) + list(range(200, 224))
                traces.append(epochs)
            finally:
                _close_dataloader_owner_resources(loader)
        assert traces[0] == traces[1]
