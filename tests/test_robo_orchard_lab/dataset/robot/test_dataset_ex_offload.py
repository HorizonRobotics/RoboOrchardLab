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

"""Finite offload plans preserve sampling, ownership and reader lifetimes."""

import gc
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import weakref
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from pydantic import PrivateAttr
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.robot._prefetch import (
    DataloaderCloseReason,
    close_dataloader_resources,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    BatchLoaderConfig,
    DataLoader,
    DatasetItem,
    DatasetItemGroup,
    DatasetWithIndices,
    DictIterableDataset,
    IterableWithLenDataset,
    ShardConfig,
    ShuffleConfig,
    _OffloadCursor,
)
from robo_orchard_lab.dataset.sampler import IndiceTable

__all__ = []


def record_event(path, kind, source):
    if path:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(
                fd, (json.dumps([kind, source, os.getpid()]) + "\n").encode()
            )
        finally:
            os.close(fd)


class TrackedReader(Dataset):
    live = 0
    high_water = 0
    events: list[tuple] = []
    references: list[weakref.ReferenceType] = []

    def __init__(self, rows, source=0, log_path=None, fail_at=None):
        self.rows = rows
        self.source = source
        self.log_path = log_path
        self.fail_at = fail_at
        self.closed = False
        self.reads = []
        type(self).live += 1
        type(self).high_water = max(type(self).high_water, type(self).live)
        type(self).references.append(weakref.ref(self))
        type(self).events.append(("open", source))
        record_event(log_path, "open", source)

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        assert not self.closed
        if index == self.fail_at:
            raise RuntimeError("fixture read failure")
        return {"source": self.source, "row": index}

    def __getitems__(self, indices):
        self.reads.extend(indices)
        type(self).events.append(("read", self.source, tuple(indices)))
        return [self[index] for index in indices]

    def close(self):
        if not self.closed:
            self.closed = True
            type(self).live -= 1
            type(self).events.append(("close", self.source))
            record_event(self.log_path, "close", self.source)


class TrackedItem(DatasetItem[TrackedReader]):
    class_type: ClassType[TrackedReader] = TrackedReader
    rows: int
    source: int = 0
    root: str = ""
    split: str | None = None
    log_path: str | None = None
    fail_at: int | None = None
    pause_path: str | None = None
    count_reads: bool = False
    _selection_cache: IndiceTable | None = PrivateAttr(default=None)
    index_references: ClassVar[list[weakref.ReferenceType]] = []

    def _selection(self):
        if self.split is None:
            return None
        if self._selection_cache is None:
            self._selection_cache = IndiceTable(
                str(Path(self.root) / (self.split + ".arrow"))
            )
            self.index_references.append(weakref.ref(self._selection_cache))
        return self._selection_cache

    def get_dataset_row_num(self):
        selection = self._selection()
        if selection is not None:
            return len(selection)
        if self.count_reads:
            reader = self._create_dataset()
            try:
                return len(reader)
            finally:
                reader.close()
        return self.rows

    def _create_dataset(self):
        return TrackedReader(
            self.rows, self.source, self.log_path, self.fail_at
        )

    def create_dataset(self, shard_config):
        record_event(self.log_path, "base", self.source)
        selection = self._selection()
        if selection is None:
            view = super().create_dataset(shard_config)
        else:
            view = DatasetWithIndices(self._create_dataset(), selection)
            if self.is_sharded:
                view = view.shard(
                    self.num_shards, self.shard_id, **shard_config.to_dict()
                )
        if self.pause_path and Path(self.pause_path).exists():
            Path(self.pause_path + ".entered").touch()
            while Path(self.pause_path).exists():
                time.sleep(0.02)
        return view


class TrackedGroup(DatasetItemGroup):
    rows: int
    root: str

    def iter_dataset_items(self):
        for source, split in enumerate(("train", "val")):
            yield TrackedItem(
                rows=self.rows, source=source, root=self.root, split=split
            )


@pytest.fixture(autouse=True)
def reader_tracking():
    TrackedReader.live = 0
    TrackedReader.high_water = 0
    TrackedReader.events = []
    TrackedReader.references = []
    TrackedItem.index_references = []
    yield
    assert TrackedReader.live == 0


def flatten(batches):
    return [
        (source, row)
        for batch in batches
        for source, row in zip(
            batch["source"].tolist(), batch["row"].tolist(), strict=True
        )
    ]


def consume_item(item, cache):
    list(DictIterableDataset([item], offload_num_batches=2, offload_dir=cache))


class TestIterationSources:
    @pytest.mark.parametrize("offload", [False, True])
    def test_factory_keeps_readers_plans_and_rng_lazy(self, offload, tmp_path):
        cache = tmp_path / "cache"
        generator = torch.Generator().manual_seed(17)
        dataset = DictIterableDataset(
            [TrackedItem(rows=12)],
            generator=generator,
            offload_num_batches=1 if offload else None,
            offload_dir=str(cache),
        )
        _ = dataset.total_iterator_length
        before = generator.get_state().clone()
        global_before = torch.get_rng_state().clone()

        with dataset._iteration_sources() as (make_iter, cursors, targets):
            assert targets == [12]
            iterator = make_iter(0)
            assert iterator is not None
            assert TrackedReader.events == []
            assert not list(cache.glob("**/*.arrow"))
            assert torch.equal(generator.get_state(), before)
            assert torch.equal(torch.get_rng_state(), global_before)
            if offload:
                assert cursors is not None
                assert not cursors[0].initialized
            else:
                assert cursors is None
            iterator.close()

        assert not list(cache.glob("*/iteration-*"))
        assert cache.exists() == offload

    @pytest.mark.parametrize("offload", [False, True])
    def test_empty_worker_admission_keeps_original_mode_rules(
        self, offload, tmp_path, monkeypatch
    ):
        generator = torch.Generator().manual_seed(17)
        dataset = DictIterableDataset(
            [TrackedItem(rows=1)],
            generator=generator,
            offload_num_batches=1 if offload else None,
            offload_dir=str(tmp_path),
        )
        _ = dataset.total_iterator_length
        before = generator.get_state().clone()
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda: SimpleNamespace(num_workers=2, id=1),
        )

        with dataset._iteration_sources() as (make_iter, cursors, targets):
            iterator = make_iter(0)
            if offload:
                assert cursors is not None and cursors[0].exhausted
                assert iterator is None
            else:
                assert iterator is not None
                iterator.close()
            assert TrackedReader.events == []
            assert not list(tmp_path.glob("**/*.arrow"))
            assert torch.equal(generator.get_state(), before)

    @pytest.mark.parametrize("offload", [False, True])
    @pytest.mark.parametrize(
        "outcome",
        ["exhaustion", "early_close", "read_failure", "close_failure"],
    )
    def test_directory_cleanup_follows_successful_reader_cleanup(
        self, offload, outcome, tmp_path, monkeypatch
    ):
        import shutil

        cache = tmp_path / "cache"
        dataset = DictIterableDataset(
            [
                TrackedItem(
                    rows=4, fail_at=1 if outcome == "read_failure" else None
                ),
                TrackedItem(rows=3, source=1),
            ],
            max_dataset_concurrency=1,
            offload_num_batches=1 if offload else None,
            offload_dir=str(cache),
        )
        original_close = TrackedReader.close
        original_rmtree = shutil.rmtree
        removed = []

        def checked_close(reader):
            if offload:
                assert list(cache.glob("*/iteration-*"))
            original_close(reader)
            if outcome == "close_failure":
                raise OSError("fixture close failure")

        def checked_rmtree(path, *args, **kwargs):
            assert TrackedReader.live == 0
            removed.append(str(path))
            return original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(TrackedReader, "close", checked_close)
        monkeypatch.setattr(shutil, "rmtree", checked_rmtree)
        iterator = iter(dataset)
        if outcome == "read_failure":
            with pytest.raises(RuntimeError, match="fixture read failure"):
                list(iterator)
        elif outcome == "close_failure":
            with pytest.raises(OSError, match="fixture close failure"):
                list(iterator)
        elif outcome == "early_close":
            assert next(iterator) == {"source": 0, "row": 0}
            iterator.close()
        else:
            assert Counter(
                (row["source"], row["row"]) for row in iterator
            ) == {
                **{(0, index): 1 for index in range(4)},
                **{(1, index): 1 for index in range(3)},
            }

        assert TrackedReader.live == 0
        assert len(removed) == int(offload and outcome == "exhaustion")
        assert bool(list(cache.glob("*/iteration-*"))) == (
            offload and outcome != "exhaustion"
        )

    @pytest.mark.parametrize("offload", [False, True])
    @pytest.mark.parametrize("prefetch", [False, True])
    @pytest.mark.parametrize(
        "outcome",
        [
            "early_close",
            "read_failure",
            "close_failure",
            "early_close_with_close_failure",
            "read_failure_with_close_failure",
        ],
    )
    def test_interruption_closes_all_resident_readers(
        self, offload, prefetch, outcome, tmp_path, monkeypatch, caplog
    ):
        from robo_orchard_lab.dataset.robot import dataset_ex

        # Two admission draws, then service each resident once.
        choices = iter([0, 0, 0, 1])
        prefetchers = []
        original_prefetch = dataset_ex.create_prefetch_iterator

        def track_prefetch(*args, **kwargs):
            iterator = original_prefetch(*args, **kwargs)
            prefetchers.append(iterator)
            return iterator

        monkeypatch.setattr(
            dataset_ex, "create_prefetch_iterator", track_prefetch
        )
        dataset = DictIterableDataset(
            [TrackedItem(rows=128, source=source) for source in range(2)],
            max_dataset_concurrency=2,
            shuffle=ShuffleConfig(
                shuffle=True, chunk_size=1 if prefetch else None
            ),
            generator=torch.Generator().manual_seed(17),
            offload_num_batches=64 if offload else None,
            offload_dir=str(tmp_path),
        )
        monkeypatch.setattr(
            dataset, "_sample_dataset_index", lambda _: next(choices, 0)
        )
        iterator = iter(dataset)
        try:
            first_source = next(iterator)["source"]
            assert next(iterator)["source"] != first_source
            assert TrackedReader.live == TrackedReader.high_water == 2
            assert len(prefetchers) == (2 if prefetch else 0)
            assert all(
                entry._producer_thread.is_alive() for entry in prefetchers
            )

            primary = RuntimeError("armed read failure")
            secondary = OSError("armed close failure")
            original_getitems = TrackedReader.__getitems__
            original_close = TrackedReader.close
            close_attempts = []

            def checked_getitems(reader, indices):
                if reader.source == first_source and outcome.startswith(
                    "read_failure"
                ):
                    raise primary
                return original_getitems(reader, indices)

            def checked_close(reader):
                if offload:
                    assert list(tmp_path.glob("*/iteration-*"))
                close_attempts.append(reader.source)
                original_close(reader)
                if reader.source == first_source and outcome.endswith(
                    "close_failure"
                ):
                    raise secondary

            monkeypatch.setattr(
                TrackedReader, "__getitems__", checked_getitems
            )
            monkeypatch.setattr(TrackedReader, "close", checked_close)
            if outcome.startswith("early_close"):
                iterator.close()
            elif outcome.startswith("read_failure"):
                with pytest.raises(RuntimeError) as error:
                    list(iterator)
                assert error.value is primary
            else:
                with pytest.raises(OSError) as error:
                    list(iterator)
                assert error.value is secondary

            assert Counter(close_attempts) == {0: 1, 1: 1}
            assert TrackedReader.live == 0
            assert bool(list(tmp_path.glob("*/iteration-*"))) == offload
            if "with_close_failure" in outcome:
                assert any(
                    record.exc_info and record.exc_info[1] is secondary
                    for record in caplog.records
                )
            for entry in prefetchers:
                # A producer error reaches the consumer before its finalizer
                # necessarily finishes; verify bounded eventual shutdown.
                entry._producer_thread.join(timeout=2)
                assert not entry._producer_thread.is_alive()
                assert not entry._state.incoming_queue
                assert not entry._ready_queue
                assert not entry._shuffle_reservoir
                assert not entry._shuffle_pending_incoming
        finally:
            iterator.close()


class TestOffloadCursor:
    @pytest.mark.parametrize(
        "outcome",
        ["exhaustion", "early_close", "read_failure", "close_failure"],
    )
    def test_segment_commits_only_after_successful_cleanup(
        self, outcome, tmp_path, monkeypatch
    ):
        cursor = _OffloadCursor(
            cache_root=str(tmp_path),
            plan_path=str(tmp_path / "plan.arrow"),
            total_rows=12,
            segment_rows=4,
        )
        original_close = TrackedReader.close

        def checked_close(reader):
            assert cursor.cursor == 0
            original_close(reader)
            if outcome == "close_failure":
                raise OSError("fixture close failure")

        monkeypatch.setattr(TrackedReader, "close", checked_close)
        generator = torch.Generator().manual_seed(17)
        before = generator.get_state().clone()
        iterator = cursor.iter_segment(
            TrackedItem(
                rows=12, fail_at=2 if outcome == "read_failure" else None
            ),
            shard_kwargs=ShardConfig(),
            shuffle=ShuffleConfig(shuffle=False),
            generator=generator,
            batch_loader_kwargs=None,
            resample_ratio=1.0,
            expected_rows=12,
        )
        assert TrackedReader.events == []
        assert not Path(cursor.plan_path).exists()
        assert torch.equal(generator.get_state(), before)

        if outcome == "read_failure":
            with pytest.raises(RuntimeError, match="fixture read failure"):
                list(iterator)
        elif outcome == "close_failure":
            with pytest.raises(OSError, match="fixture close failure"):
                list(iterator)
        elif outcome == "early_close":
            assert next(iterator) == {"source": 0, "row": 0}
            assert cursor.cursor == 0
            iterator.close()
        else:
            assert [row["row"] for row in iterator] == [0, 1, 2, 3]

        assert cursor.cursor == (4 if outcome == "exhaustion" else 0)
        assert not cursor.exhausted
        assert TrackedReader.live == 0
        assert Path(cursor.plan_path).is_file()

    def test_completed_segments_release_readers_and_index_tables(
        self, tmp_path, monkeypatch
    ):
        index_references = []
        original_init = IndiceTable.__init__

        def track_indices(table, *args, **kwargs):
            original_init(table, *args, **kwargs)
            index_references.append(weakref.ref(table))

        monkeypatch.setattr(IndiceTable, "__init__", track_indices)
        cursor = _OffloadCursor(
            cache_root=str(tmp_path),
            plan_path=str(tmp_path / "plan.arrow"),
            total_rows=12,
            segment_rows=4,
        )
        for start in (0, 4, 8):
            iterator = cursor.iter_segment(
                TrackedItem(rows=12),
                shard_kwargs=ShardConfig(),
                shuffle=ShuffleConfig(shuffle=False),
                generator=torch.Generator().manual_seed(17),
                batch_loader_kwargs=None,
                resample_ratio=1.0,
                expected_rows=12,
            )
            assert [row["row"] for row in iterator] == list(
                range(start, start + 4)
            )
            assert cursor.cursor == start + 4
            assert TrackedReader.live == 0
            gc.collect()
            assert all(ref() is None for ref in TrackedReader.references)
            assert all(ref() is None for ref in index_references)
        assert index_references
        assert cursor.exhausted


class TestOffloadPlans:
    @pytest.mark.parametrize("numpy_rng", [False, True])
    @pytest.mark.parametrize("chunk_size", [None, 3, 7])
    @pytest.mark.parametrize("ratio", [0.5, 1.0, 2.5])
    @pytest.mark.parametrize("batched", [False, True])
    def test_complete_shuffle_before_finite_reads(
        self, numpy_rng, chunk_size, ratio, batched, tmp_path
    ):
        reference_reader = TrackedReader(23)
        reader = TrackedReader(23)
        config = ShuffleConfig(shuffle=True, chunk_size=chunk_size)
        try:
            expected = list(
                IterableWithLenDataset(
                    reference_reader,
                    shuffle=config,
                    generator=np.random.default_rng(17)
                    if numpy_rng
                    else torch.Generator().manual_seed(17),
                    resample_ratio=ratio,
                )
            )
            iterable = IterableWithLenDataset(
                reader,
                shuffle=config,
                generator=np.random.default_rng(17)
                if numpy_rng
                else torch.Generator().manual_seed(17),
                resample_ratio=ratio,
                batch_loader_kwargs=BatchLoaderConfig(batch_size=4)
                if batched
                else None,
            )
            stream, reservoir = iterable._prepare_worker_index_stream()
            path = str(tmp_path / "plan.arrow")
            assert IndiceTable.save_index_stream(path, stream) == len(expected)
            plan = IndiceTable(path)
            assert list(plan) == reference_reader.reads
            actual = []
            for start in range(0, len(plan), 8):
                part = list(
                    iterable._iter_prepared_indices(
                        plan.take(slice(start, start + 8)), reservoir
                    )
                )
                actual.extend(
                    flatten(part)
                    if batched
                    else [(row["source"], row["row"]) for row in part]
                )
            assert reader.reads == reference_reader.reads
            assert Counter(actual) == Counter(
                (row["source"], row["row"]) for row in expected
            )
        finally:
            reference_reader.close()
            reader.close()

    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("drop_last", [False, True])
    def test_modes_counts_and_residency(
        self, shuffle, batched, drop_last, tmp_path
    ):
        dataset = DictIterableDataset(
            [TrackedItem(rows=23), TrackedItem(rows=11, source=1)],
            shuffle=ShuffleConfig(
                shuffle=shuffle, chunk_size=3, prefetch_factor=3
            ),
            max_dataset_concurrency=1,
            offload_num_batches=2,
            offload_dir=str(tmp_path),
        )
        loader = DataLoader(
            dataset,
            batch_size=3,
            drop_last=drop_last,
            use_dataset_side_batching=batched,
        )
        batches = list(loader)
        assert len(loader) == len(batches)
        counts = Counter(source for source, _ in flatten(batches))
        assert sum(counts.values()) == (
            30 if drop_last and batched else 33 if drop_last else 34
        )
        if not drop_last:
            assert counts == {0: 23, 1: 11}
        assert TrackedReader.high_water == 1
        assert not list(tmp_path.glob("*/iteration-*"))
        assert len(list(tmp_path.glob("*/base/*.arrow"))) == 2
        gc.collect()
        assert all(ref() is None for ref in TrackedReader.references)

    def test_rotation_bounds_input_before_virtual_prefetch(self, tmp_path):
        dataset = DictIterableDataset(
            [TrackedItem(rows=17), TrackedItem(rows=17, source=1)],
            max_dataset_concurrency=1,
            offload_num_batches=2,
            offload_dir=str(tmp_path),
        )
        iterator = iter(dataset)
        try:
            assert next(iterator) == {"source": 0, "row": 0}
            assert [
                event for event in TrackedReader.events if event[0] == "read"
            ] == [("read", 0, (0, 1))]
            assert next(iterator)["source"] == 0
            assert next(iterator)["source"] == 1
            assert TrackedReader.events[:4] == [
                ("open", 0),
                ("read", 0, (0, 1)),
                ("close", 0),
                ("open", 1),
            ]
        finally:
            iterator.close()
        assert TrackedReader.live == 0
        assert len(list(tmp_path.glob("*/iteration-*"))) == 1

    @pytest.mark.parametrize("value", [0, -1, True, 1.5])
    def test_invalid_residency(self, value):
        with pytest.raises(ValueError, match="positive integer"):
            DictIterableDataset([], offload_num_batches=value)

    def test_read_failure_retains_only_owned_files(self, tmp_path):
        dataset = DictIterableDataset(
            [TrackedItem(rows=12, fail_at=3)],
            offload_num_batches=2,
            offload_dir=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="fixture read failure"):
            list(dataset)
        assert len(list(tmp_path.glob("*/iteration-*"))) == 1
        assert len(list(tmp_path.glob("*/base/*.arrow"))) == 1

    def test_metadata_queries_do_not_open_reader_outside_full_pool(
        self, tmp_path
    ):
        dataset = DictIterableDataset(
            [TrackedItem(rows=17, count_reads=True)],
            max_dataset_concurrency=1,
            offload_num_batches=2,
            offload_dir=str(tmp_path),
        )
        iterator = iter(dataset)
        try:
            next(iterator)
            assert TrackedReader.live == 1
            assert dataset.total_dataset_length == 17
            assert dataset.get_total_batch_num(0, 3, False) == 6
            assert "total" in dataset.summary()
            assert TrackedReader.high_water == 1
        finally:
            iterator.close()


class TestOffloadIdentity:
    @pytest.mark.parametrize("offload", [False, True])
    @pytest.mark.parametrize("grouped", [False, True])
    @pytest.mark.parametrize(
        "outcome", ["exhaustion", "early_close", "read_failure"]
    )
    def test_prewarmed_input_selection_is_not_retained(
        self, offload, grouped, outcome, tmp_path
    ):
        class ExplicitGroup(DatasetItemGroup):
            members: list[TrackedItem]

            def iter_dataset_items(self):
                for item in self.members:
                    yield item.replace()

        anchors = list(range(1, 16, 2))
        IndiceTable(anchors).save_to_file(str(tmp_path / "train.arrow"))

        def make_mixture():
            item = TrackedItem(
                rows=16,
                root=str(tmp_path),
                split="train",
                fail_at=3 if outcome == "read_failure" else None,
            )
            assert item.get_dataset_row_num() == len(anchors)
            original_selection = item._selection_cache
            selection_ref = weakref.ref(original_selection)
            source = ExplicitGroup(members=[item]) if grouped else item
            dataset = DictIterableDataset(
                [source],
                offload_num_batches=2 if offload else None,
                offload_dir=str(tmp_path / "offload"),
            )
            assert item._selection_cache is original_selection
            if not offload:
                assert dataset.logical_dataset_items[0] is source
            return dataset, selection_ref

        dataset, selection_ref = make_mixture()
        gc.collect()
        assert (selection_ref() is None) == offload
        assert dataset.total_iterator_length == len(anchors)
        iterator = iter(dataset)
        try:
            if outcome == "read_failure":
                with pytest.raises(RuntimeError, match="fixture read failure"):
                    list(iterator)
            elif outcome == "early_close":
                assert next(iterator) == {"source": 0, "row": anchors[0]}
            else:
                assert [row["row"] for row in iterator] == anchors
        finally:
            iterator.close()
        gc.collect()
        assert (selection_ref() is None) == offload
        assert TrackedReader.live == 0

    def test_default_key_describes_base_selection_without_materializing(self):
        item = TrackedItem(rows=7)
        key = item.get_index_cache_key()
        assert isinstance(key, str)
        assert (
            key == item.replace(name="view").shard(3, 1).get_index_cache_key()
        )
        assert key != item.replace(rows=8).get_index_cache_key()
        assert key != item.replace(split="train").get_index_cache_key()
        item._selection_cache = IndiceTable([2, 4])
        assert key == item.get_index_cache_key()
        assert not TrackedReader.events
        assert not TrackedItem.index_references

    def test_default_key_is_stable_across_process_hash_seeds(self):
        code = (
            "from robo_orchard_lab.dataset.robot import RODatasetItem; "
            "print(RODatasetItem(dataset_path='source').get_index_cache_key())"
        )
        keys = [
            subprocess.check_output(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": seed},
                text=True,
                timeout=30,
            ).strip()
            for seed in ("1", "987")
        ]
        assert keys[0] == keys[1]

    def test_custom_key_supports_opaque_configuration(self, tmp_path):
        class OpaqueItem(TrackedItem):
            payload: object

            def get_index_cache_key(self) -> str:
                return f"fixture:v1/../\n{self.rows}:{self.source}"

        for payload in (object(), lambda value: value):
            item = OpaqueItem(rows=7, payload=payload)
            dataset = DictIterableDataset(
                [item], offload_num_batches=1, offload_dir=str(tmp_path)
            )
            assert [row["row"] for row in dataset] == list(range(7))
        assert len(list(tmp_path.glob("*/base/*.arrow"))) == 1

    @pytest.mark.parametrize("value", [None, 42, b"bytes", {}])
    def test_invalid_key_rejected_before_reader_creation(
        self, value, tmp_path
    ):
        class InvalidKeyItem(TrackedItem):
            def get_index_cache_key(self) -> str:
                return value

        dataset = DictIterableDataset(
            [InvalidKeyItem(rows=3)],
            offload_num_batches=1,
            offload_dir=str(tmp_path),
        )
        with pytest.raises(TypeError, match="get_index_cache_key"):
            list(dataset)
        assert not TrackedReader.events

    def test_relative_selection_paths_include_resolution_context(
        self, tmp_path, monkeypatch
    ):
        cache = tmp_path / "offload"
        for name, anchors in (("run-a", [1, 3]), ("run-b", [2, 4])):
            root = tmp_path / name
            root.mkdir()
            IndiceTable(anchors).save_to_file(str(root / "train.arrow"))
            monkeypatch.chdir(root)
            dataset = DictIterableDataset(
                [TrackedItem(rows=12, root=".", split="train")],
                offload_num_batches=1,
                offload_dir=str(cache),
            )
            assert [row["row"] for row in dataset] == anchors
        assert len(list(cache.glob("*/base/*.arrow"))) == 2

    def test_local_config_subclass_keeps_its_known_type(self, tmp_path):
        class LocalItem(TrackedItem):
            pass

        dataset = DictIterableDataset(
            [LocalItem(rows=4)],
            offload_num_batches=1,
            offload_dir=str(tmp_path),
        )
        assert [row["row"] for row in dataset] == [0, 1, 2, 3]

    def test_split_identity_and_count_cache_lifetime(self, tmp_path):
        IndiceTable([1, 3, 8, 11]).save_to_file(str(tmp_path / "train.arrow"))
        IndiceTable([2, 9]).save_to_file(str(tmp_path / "val.arrow"))
        group = TrackedGroup(rows=12, root=str(tmp_path))
        cache = tmp_path / "offload"
        dataset = DictIterableDataset(
            [group], offload_num_batches=1, offload_dir=str(cache)
        )
        assert dataset.total_dataset_length == 6
        assert dataset.total_iterator_length == 6
        assert dataset.get_total_batch_num(2, 2, False) == 4
        assert "total" in dataset.summary()
        assert all(
            item._selection_cache is None for item in dataset.dataset_items
        )
        assert [(row["source"], row["row"]) for row in dataset] == [
            (0, 1),
            (1, 2),
            (0, 3),
            (1, 9),
            (0, 8),
            (0, 11),
        ]
        assert len(list(cache.glob("*/base/*.arrow"))) == 2
        gc.collect()
        assert all(ref() is None for ref in TrackedItem.index_references)

    def test_duplicate_items_share_base_but_not_progress(self, tmp_path):
        log = str(tmp_path / "events.jsonl")
        item = TrackedItem(rows=9, log_path=log)
        dataset = DictIterableDataset(
            [item, item], offload_num_batches=2, offload_dir=str(tmp_path)
        )
        for _ in range(2):
            assert Counter(row["row"] for row in dataset) == {
                row: 2 for row in range(9)
            }
        events = [
            json.loads(line) for line in Path(log).read_text().splitlines()
        ]
        assert sum(kind == "base" for kind, *_ in events) == 1
        assert len(list(tmp_path.glob("*/base/*.arrow"))) == 1

    @pytest.mark.parametrize("workers", [0, 1, 2])
    @pytest.mark.parametrize("batched", [False, True])
    def test_rank_and_worker_ownership(self, workers, batched, tmp_path):
        combined = []
        for rank in range(2):
            dataset = DictIterableDataset(
                [TrackedItem(rows=23)],
                shuffle=True,
                shard_kwargs=ShardConfig(shard_strategy=None),
                offload_num_batches=1,
                offload_dir=str(tmp_path),
            ).shard(2, rank)
            loader = DataLoader(
                dataset,
                batch_size=3,
                num_workers=workers,
                multiprocessing_context="spawn" if workers else None,
                use_dataset_side_batching=batched,
            )
            batches = list(loader)
            assert len(batches) == len(loader)
            combined.extend(row for _, row in flatten(batches))
        assert Counter(combined) == {row: 1 for row in range(23)}
        assert len(list(tmp_path.glob("*/base/*.arrow"))) == 1

    def test_persistent_workers_reset_and_publish_once(self, tmp_path):
        log = str(tmp_path / "events.jsonl")
        dataset = DictIterableDataset(
            [TrackedItem(rows=31, log_path=log)],
            shuffle=ShuffleConfig(
                shuffle=True, chunk_size=5, prefetch_factor=2
            ),
            offload_num_batches=2,
            offload_dir=str(tmp_path),
        )
        loader = DataLoader(
            dataset,
            batch_size=3,
            num_workers=2,
            multiprocessing_context="spawn",
            persistent_workers=True,
        )
        try:
            epochs = [flatten(list(loader)), flatten(list(loader))]
            for epoch in epochs:
                assert Counter(row for _, row in epoch) == {
                    row: 1 for row in range(31)
                }
            assert epochs[0] != epochs[1]
            assert not list(tmp_path.glob("*/iteration-*"))
        finally:
            close_dataloader_resources(
                loader,
                loader._iterator,
                reason=DataloaderCloseReason.TRAINER_TEARDOWN,
            )
        events = [
            json.loads(line) for line in Path(log).read_text().splitlines()
        ]
        assert sum(kind == "base" for kind, *_ in events) == 1
        assert sum(kind == "open" for kind, *_ in events) == sum(
            kind == "close" for kind, *_ in events
        )

    @pytest.mark.parametrize("strategy", [None, "pad_last", "drop_last"])
    @pytest.mark.parametrize("batched", [False, True])
    def test_static_rank_shards_and_resampling(
        self, strategy, batched, tmp_path
    ):
        for rank in range(2):
            item = TrackedItem(rows=29).shard(2, 1)
            kwargs = dict(
                datasets=[item],
                shard_kwargs=ShardConfig(shard_strategy=strategy),
                shuffle=True,
                generator=torch.Generator().manual_seed(29),
                resample_ratios=2.5,
            )
            reference = DictIterableDataset(**kwargs).shard(2, rank)
            offloaded = DictIterableDataset(
                **kwargs, offload_num_batches=1, offload_dir=str(tmp_path)
            ).shard(2, rank)
            rows = flatten(
                list(
                    DataLoader(
                        offloaded,
                        batch_size=4,
                        use_dataset_side_batching=batched,
                    )
                )
            )
            baseline = list(reference)
            assert len(rows) == len(baseline)
            assert {row for _, row in rows} == {row["row"] for row in baseline}

    @pytest.mark.parametrize("rows", [0, 1, 2])
    def test_empty_tiny_worker_targets(self, rows, tmp_path):
        dataset = DictIterableDataset(
            [TrackedItem(rows=rows)],
            shuffle=True,
            resample_ratios=0.5,
            offload_num_batches=2,
            offload_dir=str(tmp_path),
        )
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=2,
            multiprocessing_context="spawn",
        )
        assert sum(len(batch["row"]) for batch in loader) == round(rows * 0.5)

    def test_default_temp_directory_and_disabled_path(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        ordinary = DictIterableDataset([TrackedItem(rows=7)])
        assert len(list(ordinary)) == 7
        assert not list(tmp_path.iterdir())
        offloaded = DictIterableDataset(
            [TrackedItem(rows=7)], offload_num_batches=2
        )
        assert len(list(offloaded)) == 7
        assert (
            len(list(tmp_path.glob("robo-orchard-offload-*/*/base/*.arrow")))
            == 1
        )

    def test_real_rodataset_callable_transform_and_reopen(
        self, tmp_path, monkeypatch
    ):
        import datasets

        from robo_orchard_lab.dataset.robot.dataset import (
            RODataset,
            RODatasetItem,
        )
        from robo_orchard_lab.dataset.robot.packaging import (
            DataFrame,
            DatasetPackaging,
            EpisodeData,
            EpisodeMeta,
            EpisodePackaging,
        )

        class Rows(EpisodePackaging):
            @property
            def features(self):
                return datasets.Features({"value": datasets.Value("int64")})

            def generate_episode_meta(self):
                return EpisodeMeta(episode=EpisodeData())

            def generate_frames(self):
                for index in range(5):
                    yield DataFrame(features={"value": index})

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        root = str(tmp_path / "source")
        episode = Rows()
        DatasetPackaging(episode.features, database_driver="sqlite").packaging(
            [episode], root, fail_fast=True
        )
        closed = []
        close = RODataset.close

        def track_close(reader):
            closed.append(weakref.ref(reader))
            close(reader)

        monkeypatch.setattr(RODataset, "close", track_close)
        cache = tmp_path / "offload"
        keys = []
        for offset in (0, 10):
            item = RODatasetItem(
                dataset_path=root,
                transform=lambda row, shift=offset: {
                    "value": row["value"] + shift
                },
            )
            keys.append(item.get_index_cache_key())
            dataset = DictIterableDataset(
                [item], offload_num_batches=1, offload_dir=str(cache)
            )
            assert [row["value"] for row in dataset] == list(
                range(offset, offset + 5)
            )
        assert len(closed) == 10
        assert keys[0] == keys[1]
        gc.collect()
        assert all(ref() is None for ref in closed)
        assert len(list(cache.glob("*/base/*.arrow"))) == 1

    def test_failed_base_write_does_not_publish_and_can_retry(
        self, tmp_path, monkeypatch
    ):
        cache = tmp_path / "offload"
        dataset = DictIterableDataset(
            [TrackedItem(rows=7)],
            offload_num_batches=2,
            offload_dir=str(cache),
        )
        original = IndiceTable.save_to_file

        def failed_write(table, path, reload=False):
            Path(path).write_bytes(b"incomplete")
            raise OSError("fixture write failure")

        with monkeypatch.context() as patch:
            patch.setattr(IndiceTable, "save_to_file", failed_write)
            with pytest.raises(OSError, match="fixture write failure"):
                list(dataset)
        assert not list(cache.glob("*/base/*.arrow"))
        assert not list(cache.glob("*/base/*.tmp"))
        assert TrackedReader.live == 0
        assert IndiceTable.save_to_file is original
        assert len(list(dataset)) == 7
        assert len(list(cache.glob("*/base/*.arrow"))) == 1

    def test_exited_base_writer_releases_process_lock(self, tmp_path):
        pause = tmp_path / "pause"
        pause.touch()
        cache = tmp_path / "offload"
        item = TrackedItem(rows=7, pause_path=str(pause))
        process = mp.get_context("spawn").Process(
            target=consume_item, args=(item, str(cache)), daemon=True
        )
        process.start()
        try:
            deadline = time.monotonic() + 30
            while not Path(str(pause) + ".entered").exists():
                assert process.is_alive()
                assert time.monotonic() < deadline
                time.sleep(0.02)
            process.terminate()
            process.join(10)
            assert not process.is_alive()
            assert not list(cache.glob("*/base/*.arrow"))
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)
            pause.unlink()
        assert (
            len(
                list(
                    DictIterableDataset(
                        [item], offload_num_batches=2, offload_dir=str(cache)
                    )
                )
            )
            == 7
        )
        assert len(list(cache.glob("*/base/*.arrow"))) == 1


class TestWeightedAdmission:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("capacity", [1, 2, 3])
    def test_exact_cyclic_row_flow(self, batch_size, capacity, monkeypatch):
        # Enumerate the complete tiny closed-chain state space. Probabilities
        # come from the production admission/service paths, not a replay copy.
        targets = [20, 50, 10]
        lengths = [6, 9, 1]
        units = [list(range(length, 0, -batch_size)) for length in lengths]
        dataset = DictIterableDataset(
            [TrackedItem(rows=n, source=i) for i, n in enumerate(targets)],
            shuffle=True,
            max_dataset_concurrency=capacity,
            batch_loader_kwargs=BatchLoaderConfig(batch_size=batch_size),
        )
        dataset._total_indices_length = targets
        cursors = {
            i: _OffloadCursor("", "", n, n) for i, n in enumerate(lengths)
        }
        draws = []

        def choose(weights):
            draws.append(weights.copy())
            return 0

        monkeypatch.setattr(dataset, "_sample_dataset_index", choose)
        states = [
            (group, phase)
            for group in combinations(range(3), capacity)
            for phase in product(*(range(len(units[i])) for i in group))
        ]
        state_index = {state: i for i, state in enumerate(states)}
        matrix = np.zeros((len(states), len(states)))
        stationary = []
        flow = np.zeros(3)
        for state_idx, (group, phase) in enumerate(states):
            remaining = {
                i: units[i][k] for i, k in zip(group, phase, strict=True)
            }
            active = [(i, iter(())) for i in group]
            service = dataset._prepare_dataset_for_iter(
                active, [], lambda _: iter(()), targets, remaining, cursors
            )
            v = np.array(
                [targets[i] / min(batch_size, remaining[i]) for i in group]
            )
            a = np.array(targets) / np.array(lengths)
            mass = v.sum() * np.prod(a[list(group)] / v)
            stationary.append(mass)
            for pos, i in enumerate(group):
                unit = min(batch_size, remaining[i])
                flow[i] += mass * service[pos] * unit
                if phase[pos] + 1 < len(units[i]):
                    next_phase = list(phase)
                    next_phase[pos] += 1
                    destination = (group, tuple(next_phase))
                    matrix[state_idx, state_index[destination]] += service[pos]
                    continue
                survivors = [j for j in group if j != i]
                pending = [j for j in range(3) if j not in survivors]
                survivor_remaining = {j: remaining[j] for j in survivors}
                draws.clear()
                dataset._prepare_dataset_for_iter(
                    [(j, iter(())) for j in survivors],
                    pending.copy(),
                    lambda _: iter(()),
                    targets,
                    survivor_remaining,
                    cursors,
                )
                assert len(draws) == 1
                for j, admission in zip(pending, draws[0], strict=True):
                    new_group = tuple(sorted([*survivors, j]))
                    phase_by_source = dict(zip(group, phase, strict=True))
                    phase_by_source[j] = 0
                    destination = (
                        new_group,
                        tuple(phase_by_source[k] for k in new_group),
                    )
                    matrix[state_idx, state_index[destination]] += (
                        service[pos] * admission
                    )
        stationary = np.array(stationary)
        stationary /= stationary.sum()
        np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(stationary @ matrix, stationary, atol=1e-14)
        np.testing.assert_allclose(
            flow / flow.sum(), np.array(targets) / sum(targets), atol=1e-14
        )
        assert not TrackedReader.events

    @pytest.mark.parametrize("offload", [None, 1, 2])
    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("rng", ["torch", "numpy"])
    def test_full_quotas_and_short_yields(
        self, offload, drop_last, rng, tmp_path
    ):
        targets = [0, 1, 2, 7, 18]

        def run():
            generator = (
                torch.Generator().manual_seed(39)
                if rng == "torch"
                else np.random.default_rng(39)
            )
            dataset = DictIterableDataset(
                [TrackedItem(rows=n, source=i) for i, n in enumerate(targets)],
                shuffle=True,
                generator=generator,
                max_dataset_concurrency=2,
                offload_num_batches=offload,
                offload_dir=str(tmp_path),
                batch_loader_kwargs=BatchLoaderConfig(
                    batch_size=4, drop_last=drop_last
                ),
            )
            return flatten(list(dataset))

        first = run()
        assert first == run()
        expected = {
            i: n // 4 * 4 if drop_last else n
            for i, n in enumerate(targets)
            if n >= 4 or not drop_last
        }
        assert Counter(source for source, _ in first) == Counter(expected)
        assert TrackedReader.high_water <= 2
        assert not list(tmp_path.glob("*/iteration-*"))

    @pytest.mark.parametrize("offload", [None, 1])
    def test_zero_output_still_validates_metadata(self, offload, tmp_path):
        class BadMetadata(TrackedItem):
            def get_dataset_row_num(self):
                return self.rows + 1

        dataset = DictIterableDataset(
            [BadMetadata(rows=2)],
            shuffle=True,
            offload_num_batches=offload,
            offload_dir=str(tmp_path),
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=8, drop_last=True
            ),
        )
        with pytest.raises(RuntimeError, match="row metadata"):
            list(dataset)
        assert TrackedReader.live == 0

    @pytest.mark.parametrize("close_after_last", [False, True])
    def test_last_yield_does_not_commit_until_resumed(
        self, close_after_last, tmp_path, monkeypatch
    ):
        from contextlib import contextmanager

        dataset = DictIterableDataset(
            [TrackedItem(rows=2)],
            shuffle=True,
            offload_num_batches=2,
            offload_dir=str(tmp_path),
        )
        original = dataset._iteration_sources
        captured = []

        @contextmanager
        def capture():
            with original() as sources:
                captured.append(sources[1][0])
                yield sources

        monkeypatch.setattr(dataset, "_iteration_sources", capture)
        iterator = iter(dataset)
        next(iterator)
        next(iterator)
        assert captured[0].cursor == 0
        if close_after_last:
            iterator.close()
            assert captured[0].cursor == 0
        else:
            with pytest.raises(StopIteration):
                next(iterator)
            assert captured[0].cursor == 2

    @pytest.mark.parametrize("difference", [-1, 1])
    def test_segment_output_mismatch_does_not_commit(
        self, difference, tmp_path, monkeypatch
    ):
        original = IterableWithLenDataset._iter_prepared_indices

        def wrong_count(reader, indices, generator):
            if difference < 0:
                indices = indices.take(slice(0, len(indices) - 1))
            yield from original(reader, indices, generator)
            if difference > 0:
                yield {"source": 0, "row": 99}

        monkeypatch.setattr(
            IterableWithLenDataset, "_iter_prepared_indices", wrong_count
        )
        cursor = _OffloadCursor(
            str(tmp_path), str(tmp_path / "plan.arrow"), 4, 2
        )
        iterator = cursor.iter_segment(
            TrackedItem(rows=4),
            shard_kwargs=ShardConfig(),
            shuffle=ShuffleConfig(shuffle=False),
            generator=torch.Generator().manual_seed(1),
            batch_loader_kwargs=None,
            resample_ratio=1.0,
            expected_rows=4,
        )
        with pytest.raises(RuntimeError, match="output quota"):
            list(iterator)
        assert cursor.cursor == 0

    @pytest.mark.parametrize("rng", ["torch", "numpy"])
    def test_scheduling_uses_only_explicit_rng(self, rng):
        generator = (
            torch.Generator().manual_seed(11)
            if rng == "torch"
            else np.random.default_rng(11)
        )
        dataset = DictIterableDataset([], shuffle=True, generator=generator)
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        weights = np.array([0.1, 0.9], dtype=np.float64)
        draws = [dataset._sample_dataset_index(weights) for _ in range(50)]
        assert set(draws) == {0, 1}
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert all(
            np.array_equal(a, b)
            for a, b in zip(numpy_before, np.random.get_state(), strict=True)
        )

    @pytest.mark.parametrize("offload", [None, 2])
    def test_worker_targets_reuse_metadata_counts(
        self, offload, tmp_path, monkeypatch
    ):
        dataset = DictIterableDataset(
            [TrackedItem(rows=5), TrackedItem(rows=11, source=1)],
            offload_num_batches=offload,
            offload_dir=str(tmp_path),
        )
        assert dataset.total_iterator_length == 16
        monkeypatch.setattr(
            TrackedItem,
            "get_sharded_row_num",
            lambda *args, **kwargs: pytest.fail("duplicate metadata count"),
        )
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda: SimpleNamespace(num_workers=2, id=1),
        )
        with dataset._iteration_sources() as (_, _, targets):
            assert targets == [2, 5]
