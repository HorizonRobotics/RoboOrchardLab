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

"""Tests for DataLoader iterator ownership and worker cleanup."""

import gc
import multiprocessing as mp
import threading
import time
import weakref
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.utils import DataLoaderConfiguration
from test_robo_orchard_lab.dataset.robot._dataset_ex_helpers import (
    ArrayDataset,
    ArrayDatasetItem,
    _get_dataloader_multiprocessing_context,
)
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    IterableDataset as TorchIterableDataset,
)

from robo_orchard_lab.dataset.robot import (
    DataLoader,
    IterableDatasetMixin,
    IterableWithLenDataset,
    ShuffleConfig,
)
from robo_orchard_lab.dataset.robot._prefetch import (
    DataloaderCloseReason,
    _close_dataloader_iterator,
    _close_dataloader_owner_resources,
    close_dataloader_resources,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DictIterableDataset,
)
from robo_orchard_lab.dataset.sampler import (
    DEFAULT_SAMPLING_SEED,
)


class TestCreatePrefetchIterator:
    def test_close_closes_single_process_closeable_dataset_iterator(self):
        class _CloseTrackingIterator:
            def __init__(self) -> None:
                self._items = iter([0, 1, 2])
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self) -> int:
                return next(self._items)

            def close(self) -> None:
                self.closed = True

        class _CloseTrackingDataset(TorchIterableDataset):
            def __init__(self) -> None:
                self.last_iter: Any | None = None

            def __iter__(self):
                self.last_iter = _CloseTrackingIterator()
                return self.last_iter

        dataset = _CloseTrackingDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        dataloader_iter = iter(dataloader)

        assert next(dataloader_iter) == 0

        _close_dataloader_iterator(cast(Any, dataloader_iter))

        assert cast(Any, dataset).last_iter.closed is True

    def test_close_keeps_persistent_workers_reusable(self):
        num_workers = 1
        dataloader = DataLoader(
            ArrayDataset(data=list(range(16))),
            batch_size=4,
            num_workers=num_workers,
            persistent_workers=True,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        iterator = iter(dataloader)
        first_batch = cast(torch.Tensor, next(iterator))

        _close_dataloader_iterator(cast(Any, iterator))

        iterator = iter(dataloader)
        second_batch = cast(torch.Tensor, next(iterator))

        assert first_batch.tolist() == [0, 1, 2, 3]
        assert second_batch.tolist() == [0, 1, 2, 3]

    def test_close_resources_shutdowns_persistent_workers_for_early_break(
        self,
    ):
        num_workers = 1
        dataloader = DataLoader(
            ArrayDataset(data=list(range(16))),
            batch_size=4,
            num_workers=num_workers,
            persistent_workers=True,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        iterator = iter(dataloader)
        first_batch = cast(torch.Tensor, next(iterator))

        close_dataloader_resources(
            dataloader,
            iterator,
            reason=DataloaderCloseReason.EARLY_BREAK,
        )

        assert getattr(dataloader, "_iterator", None) is None
        iterator = iter(dataloader)
        second_batch = cast(torch.Tensor, next(iterator))
        close_dataloader_resources(
            dataloader,
            iterator,
            reason=DataloaderCloseReason.EPOCH_EXHAUSTED,
        )

        assert first_batch.tolist() == [0, 1, 2, 3]
        assert second_batch.tolist() == [0, 1, 2, 3]

    def test_close_owner_resources_shutdowns_kept_persistent_iterator(self):
        """Final teardown closes persistent workers kept across epochs."""
        num_workers = 1
        dataloader = DataLoader(
            ArrayDataset(data=list(range(16))),
            batch_size=4,
            num_workers=num_workers,
            persistent_workers=True,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        iterator = iter(dataloader)
        first_batch = cast(torch.Tensor, next(iterator))
        close_dataloader_resources(
            dataloader,
            iterator,
            reason=DataloaderCloseReason.EPOCH_EXHAUSTED,
        )

        assert getattr(dataloader, "_iterator", None) is not None

        _close_dataloader_owner_resources(dataloader)

        assert getattr(dataloader, "_iterator", None) is None
        iterator = iter(dataloader)
        second_batch = cast(torch.Tensor, next(iterator))
        close_dataloader_resources(
            dataloader,
            iterator,
            reason=DataloaderCloseReason.TRAINER_TEARDOWN,
        )

        assert first_batch.tolist() == [0, 1, 2, 3]
        assert second_batch.tolist() == [0, 1, 2, 3]

    @pytest.mark.parametrize(
        "reason",
        [
            DataloaderCloseReason.EPOCH_EXHAUSTED,
            DataloaderCloseReason.COORDINATED_EPOCH_END,
        ],
    )
    def test_epoch_close_reuses_actual_persistent_worker_pool(self, reason):
        loader = DataLoader(
            ArrayDataset(data=list(range(8))),
            batch_size=2,
            num_workers=1,
            persistent_workers=True,
            multiprocessing_context=_get_dataloader_multiprocessing_context(1),
        )
        workers = []
        try:
            iterator = iter(loader)
            assert [v for batch in iterator for v in batch.tolist()] == list(
                range(8)
            )
            native = loader._iterator
            assert native is loader._dataloader._iterator
            workers = list(native._workers)
            assert all(worker.is_alive() for worker in workers)
            close_dataloader_resources(loader, iterator, reason=reason)
            assert loader._iterator is native

            iterator = iter(loader)
            assert next(iterator).tolist() == [0, 1]
            assert loader._iterator is native
            assert all(worker.is_alive() for worker in workers)
            close_dataloader_resources(loader, iterator, reason=reason)
        finally:
            _close_dataloader_owner_resources(loader)
        assert loader._iterator is None
        assert all(not worker.is_alive() for worker in workers)

    def test_loading_state_closes_kept_persistent_pool(self):
        loader = DataLoader(
            ArrayDataset(data=list(range(8))),
            batch_size=2,
            num_workers=1,
            persistent_workers=True,
            multiprocessing_context=_get_dataloader_multiprocessing_context(1),
        )
        try:
            iterator = iter(loader)
            assert next(iterator).tolist() == [0, 1]
            close_dataloader_resources(
                loader, iterator, reason=DataloaderCloseReason.EPOCH_EXHAUSTED
            )
            native = loader._iterator
            workers = list(native._workers)
            state = loader.state_dict()
            loader.load_state_dict(state)
            assert loader._iterator is None
            assert all(not worker.is_alive() for worker in workers)
            assert [v for batch in loader for v in batch.tolist()] == list(
                range(2, 8)
            )
            assert loader._iterator is None

            loader._notify_epoch(1)
            iterator = iter(loader)
            assert next(iterator).tolist() == [0, 1]
            assert loader._iterator is not None
            close_dataloader_resources(
                loader, iterator, reason=DataloaderCloseReason.TRAINER_TEARDOWN
            )
        finally:
            _close_dataloader_owner_resources(loader)


class TestPreparedLoaderFailureCleanup:
    @pytest.mark.parametrize("fail_index", [0, 2])
    def test_failed_generator_closes_only_its_owned_workers(self, fail_index):
        """Retained error tracebacks release workers and leave peers active."""
        values = list(range(8))
        values[fail_index] = None
        loaders = [
            prepare_data_loader(
                TorchDataLoader(
                    ArrayDataset(data=data),
                    batch_size=1,
                    num_workers=1,
                    persistent_workers=False,
                    multiprocessing_context=(
                        _get_dataloader_multiprocessing_context(1)
                    ),
                    generator=torch.Generator().manual_seed(0),
                ),
                device=torch.device("cpu"),
                num_processes=1,
                process_index=0,
                put_on_device=True,
                dispatch_batches=False,
                split_batches=False,
                even_batches=False,
            )
            for data in (list(range(16)), values)
        ]
        peer, failed = loaders
        gradient_state = peer.gradient_state
        initial_active_loader = gradient_state.active_dataloader
        peer_iter, failed_iter = iter(peer), iter(failed)
        primary_exc = None
        baseline_pids = {process.pid for process in mp.active_children()}
        try:
            assert next(peer_iter).tolist() == [0]
            peer_pids = {process.pid for process in mp.active_children()}
            with pytest.raises(TypeError, match="NoneType") as caught:
                list(failed_iter)
            primary_exc = caught.value
            assert failed_iter.gi_frame is None
            failed_pids = {
                process.pid for process in mp.active_children()
            } - peer_pids
            assert failed_pids
            # The same iterator code/dataset with a different owner is not
            # enough to claim the exited generator's retained worker iterator.
            close_dataloader_resources(
                SimpleNamespace(dataset=failed.dataset),
                failed_iter,
                reason=DataloaderCloseReason.EXCEPTION_ABORT,
                primary_exc=primary_exc,
            )
            assert failed_pids <= {
                process.pid for process in mp.active_children()
            }
            close_dataloader_resources(
                failed,
                failed_iter,
                reason=DataloaderCloseReason.EXCEPTION_ABORT,
                primary_exc=primary_exc,
            )
            assert {
                process.pid for process in mp.active_children()
            } == peer_pids
            assert next(peer_iter).tolist() == [1]
        finally:
            close_dataloader_resources(
                failed,
                failed_iter,
                reason=DataloaderCloseReason.EXCEPTION_ABORT,
                primary_exc=primary_exc,
            )
            close_dataloader_resources(
                peer, peer_iter, reason=DataloaderCloseReason.EARLY_BREAK
            )
        assert {
            process.pid for process in mp.active_children()
        } == baseline_pids
        assert gradient_state.active_dataloader is initial_active_loader


class TestDataLoaderEarlyBreakCleanup:
    def _count_prefetch_threads(self) -> int:
        return sum(
            thread.name == "dataset-prefetch-producer"
            for thread in threading.enumerate()
        )

    def _iterate_with_early_break(
        self,
        dataloader: DataLoader,
        max_batches: int,
    ) -> list[Any]:
        dataloader_iter = iter(dataloader)
        collected_batches = []
        try:
            for batch_idx, batch in enumerate(dataloader_iter):
                collected_batches.append(batch)
                if batch_idx + 1 >= max_batches:
                    break
        finally:
            _close_dataloader_iterator(cast(Any, dataloader_iter))

        return collected_batches

    def _wait_for_prefetch_threads(self, expected_count: int) -> None:
        for _ in range(40):
            if self._count_prefetch_threads() == expected_count:
                return
            time.sleep(0.05)

        assert self._count_prefetch_threads() == expected_count

    def _active_child_count(self) -> int:
        return len(mp.active_children())

    def _wait_for_active_child_count(self, expected_count: int) -> None:
        for _ in range(60):
            if self._active_child_count() == expected_count:
                return
            time.sleep(0.05)

        assert self._active_child_count() == expected_count

    def _build_dataset(
        self,
        dataset_kind: str,
    ) -> IterableDatasetMixin:
        shuffle = ShuffleConfig(
            shuffle=True,
            chunk_size=4,
            prefetch_factor=2,
        )
        if dataset_kind == "iterable":
            return IterableWithLenDataset(
                ArrayDataset(data=list(range(32))),
                shuffle=shuffle,
            )

        return DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(6))),
                ArrayDatasetItem(data=list(range(100, 106))),
                ArrayDatasetItem(data=list(range(200, 206))),
            ],
            shuffle=shuffle,
            max_dataset_concurrency=2,
        )

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_dict_iterable_repeated_early_break_releases_dynamic_sub_iterators(
        self,
        monkeypatch,
        use_dataset_side_batching: bool,
    ):
        tracked_state = {"created": 0, "finalized": 0}
        live_generators = weakref.WeakSet()
        original_iter = IterableWithLenDataset.__iter__

        def mark_finalized() -> None:
            tracked_state["finalized"] += 1

        def tracked_iter(self):
            generator = original_iter(self)
            tracked_state["created"] += 1
            live_generators.add(generator)
            weakref.finalize(generator, mark_finalized)
            return generator

        monkeypatch.setattr(
            IterableWithLenDataset,
            "__iter__",
            tracked_iter,
        )

        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(3))),
                ArrayDatasetItem(data=list(range(100, 103))),
                ArrayDatasetItem(data=list(range(200, 203))),
            ],
            shuffle=False,
            max_dataset_concurrency=1,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2 if use_dataset_side_batching else 1,
            num_workers=0,
            use_dataset_side_batching=use_dataset_side_batching,
        )

        for _ in range(5):
            batches = self._iterate_with_early_break(
                dataloader,
                max_batches=4,
            )
            assert batches
            gc.collect()
            assert len(live_generators) == 0
            assert tracked_state["finalized"] == tracked_state["created"]

        assert tracked_state["created"] >= 10

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_repeated_early_break_keeps_historical_iterators_clean(
        self,
        use_dataset_side_batching: bool,
    ):
        baseline_threads = self._count_prefetch_threads()
        shuffle = ShuffleConfig(
            shuffle=True,
            chunk_size=4,
            prefetch_factor=2,
        )
        batch_size = 2 if use_dataset_side_batching else 1
        dataset = IterableWithLenDataset(
            ArrayDataset(data=list(range(32))),
            shuffle=shuffle,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=shuffle,
            use_dataset_side_batching=use_dataset_side_batching,
        )

        first_values = []
        for _ in range(6):
            batches = self._iterate_with_early_break(
                dataloader,
                max_batches=2,
            )
            assert batches
            first_batch = batches[0]
            if isinstance(first_batch, torch.Tensor):
                first_values.append(first_batch.tolist())
            else:
                first_values.append(list(first_batch))
            self._wait_for_prefetch_threads(baseline_threads)

        assert len(first_values) == 6

    @pytest.mark.parametrize("dataset_kind", ["iterable", "dict"])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    @pytest.mark.parametrize(
        "num_workers,persistent_workers",
        [
            (0, False),
            (1, False),
            (1, True),
            (2, False),
            (2, True),
        ],
    )
    def test_repeated_early_break_keeps_dataloader_reusable(
        self,
        dataset_kind: str,
        use_dataset_side_batching: bool,
        num_workers: int,
        persistent_workers: bool,
    ):
        baseline_child_count = self._active_child_count()
        dataloader_kwargs = {
            "batch_size": 2 if use_dataset_side_batching else 1,
            "num_workers": num_workers,
            "use_dataset_side_batching": use_dataset_side_batching,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = persistent_workers
            dataloader_kwargs["multiprocessing_context"] = (
                _get_dataloader_multiprocessing_context(num_workers)
            )

        dataloader = DataLoader(
            self._build_dataset(dataset_kind),
            **dataloader_kwargs,
        )
        expected_cycle_child_count = baseline_child_count
        if num_workers > 0 and persistent_workers:
            expected_cycle_child_count += num_workers

        for _ in range(4):
            batches = self._iterate_with_early_break(
                dataloader,
                max_batches=2,
            )
            assert batches
            self._wait_for_active_child_count(expected_cycle_child_count)

        del dataloader
        gc.collect()
        self._wait_for_active_child_count(baseline_child_count)


@pytest.mark.parametrize("workers", [0, 1])
def test_failed_lab_snapshot_closes_owned_iterator_and_preserves_peer(
    monkeypatch, workers
):
    from accelerate.state import AcceleratorState, GradientState, PartialState

    from robo_orchard_lab.utils.accelerate import (
        _configure_prepared_loader_recovery,
    )

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )
    options = dict(batch_size=1, num_workers=workers)
    if workers:
        options.update(
            persistent_workers=True, multiprocessing_context="spawn"
        )
    sources = [
        DataLoader(
            DictIterableDataset(
                [ArrayDatasetItem(data=list(range(start, start + 8)))],
                shuffle=False,
                seed=DEFAULT_SAMPLING_SEED,
            ),
            **options,
        )
        for start in (0, 100)
    ]
    loaders = []
    for source in sources:
        prepared = accelerator.prepare(source)
        _configure_prepared_loader_recovery(accelerator, prepared, source)
        assert isinstance(prepared.base_dataloader, DataLoader)
        loaders.append(prepared)
    peer, failed = loaders
    peer_iter, failed_iter = iter(peer), iter(failed)
    primary = None
    try:
        assert next(peer_iter).tolist() == [0]

        def fail_snapshot():
            raise RuntimeError("injected owned snapshot failure")

        monkeypatch.setattr(
            failed.base_dataloader, "state_dict", fail_snapshot
        )
        with pytest.raises(
            RuntimeError, match="owned snapshot failure"
        ) as caught:
            next(failed_iter)
        primary = caught.value
        assert failed_iter.gi_frame is None
        runtime = failed.base_dataloader._resume_runtime
        assert runtime._live_iterators()
        native = failed.base_dataloader._iterator
        failed_workers = [] if native is None else list(native._workers)
        close_dataloader_resources(
            SimpleNamespace(dataset=failed.dataset),
            failed_iter,
            reason=DataloaderCloseReason.EXCEPTION_ABORT,
            primary_exc=primary,
        )
        assert runtime._live_iterators()
        close_dataloader_resources(
            failed,
            failed_iter,
            reason=DataloaderCloseReason.EXCEPTION_ABORT,
            primary_exc=primary,
        )
        assert not runtime._live_iterators()
        assert all(not worker.is_alive() for worker in failed_workers)
        assert next(peer_iter).tolist() == [1]
    finally:
        close_dataloader_resources(
            failed,
            failed_iter,
            reason=DataloaderCloseReason.EXCEPTION_ABORT,
            primary_exc=primary,
        )
        close_dataloader_resources(
            peer, peer_iter, reason=DataloaderCloseReason.EARLY_BREAK
        )
