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

"""Tests for prefetch ordering, shuffle windows, and producer teardown."""

import gc
import threading
import time
import weakref
from typing import Any, cast

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from test_robo_orchard_lab.dataset.robot._dataset_ex_helpers import (
    ArrayDatasetItem,
)

import robo_orchard_lab.dataset.robot._prefetch as prefetch_module
from robo_orchard_lab.dataset.robot import (
    DataLoader,
    ShuffleConfig,
)
from robo_orchard_lab.dataset.robot._prefetch import (
    _close_dataloader_iterator,
    create_prefetch_iterator,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DictIterableDataset,
)
from robo_orchard_lab.utils.accelerate import (
    configure_data_loader_for_accelerate,
)


class TestCreatePrefetchIterator:
    def _count_prefetch_threads(self) -> int:
        return sum(
            thread.name == "dataset-prefetch-producer"
            for thread in threading.enumerate()
        )

    def test_waits_for_k_plus_t_candidates_before_first_partition(self):
        # K=12 uses an effective T=2. The producer must publish the replacement
        # chunk atomically, so thirteen live-source candidates are
        # insufficient.
        allow_last_chunk_item = threading.Event()
        first_item_ready = threading.Event()
        yielded_items = []

        def blocking_iter():
            yield from range(13)
            if not allow_last_chunk_item.wait(timeout=1):
                raise TimeoutError(
                    "Timed out waiting for the last chunk item."
                )
            yield 13

        generator = torch.Generator()
        generator.manual_seed(0)
        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=12,
            shuffle=True,
            generator=generator,
        )

        def consume_first_item():
            yielded_items.append(next(prefetch_iter))
            first_item_ready.set()

        consumer_thread = threading.Thread(target=consume_first_item)
        consumer_thread.start()

        # The producer has one item in its reserved local chunk. It must not
        # publish or wake the consumer until the second item is available.
        assert not first_item_ready.wait(timeout=0.1)

        allow_last_chunk_item.set()
        assert first_item_ready.wait(timeout=1)
        assert yielded_items[0] in range(14)

        consumer_thread.join(timeout=1)
        remaining_items = list(prefetch_iter)
        assert sorted(yielded_items + remaining_items) == list(range(14))

    def test_chunk_partition_matches_one_torch_randperm(self):
        actual_generator = torch.Generator().manual_seed(0)
        prefetch_iter = create_prefetch_iterator(
            iter(range(14)),
            prefetch_size=12,
            shuffle=True,
            generator=actual_generator,
        )

        actual_outputs = [next(prefetch_iter) for _ in range(2)]

        expected_generator = torch.Generator().manual_seed(0)
        pool = list(range(14))
        indices = torch.randperm(14, generator=expected_generator).tolist()
        assert actual_outputs == [pool[index] for index in indices[:2]]
        assert cast(Any, prefetch_iter)._shuffle_reservoir == [
            pool[index] for index in indices[2:]
        ]
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )
        cast(Any, prefetch_iter).close()

    def test_chunk_partition_matches_one_numpy_permutation(self):
        actual_generator = np.random.default_rng(0)
        prefetch_iter = create_prefetch_iterator(
            iter(range(14)),
            prefetch_size=12,
            shuffle=True,
            generator=actual_generator,
        )

        actual_outputs = [next(prefetch_iter) for _ in range(2)]

        expected_generator = np.random.default_rng(0)
        pool = list(range(14))
        indices = expected_generator.permutation(14).tolist()
        assert actual_outputs == [pool[index] for index in indices[:2]]
        assert cast(Any, prefetch_iter)._shuffle_reservoir == [
            pool[index] for index in indices[2:]
        ]
        assert (
            actual_generator.bit_generator.state
            == expected_generator.bit_generator.state
        )
        cast(Any, prefetch_iter).close()

    def test_partial_eof_chunk_partitions_only_available_candidates(self):
        actual_generator = torch.Generator().manual_seed(11)
        prefetch_iter = create_prefetch_iterator(
            iter(range(13)),
            prefetch_size=12,
            shuffle=True,
            generator=actual_generator,
        )

        actual_output = next(prefetch_iter)

        expected_generator = torch.Generator().manual_seed(11)
        pool = list(range(13))
        indices = torch.randperm(13, generator=expected_generator).tolist()
        assert actual_output == pool[indices[0]]
        assert cast(Any, prefetch_iter)._shuffle_reservoir == [
            pool[index] for index in indices[1:]
        ]
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )
        cast(Any, prefetch_iter).close()

    def test_non_divisible_k_waits_for_full_live_partition_chunk(self):
        allow_last_chunk_item = threading.Event()
        first_item_ready = threading.Event()
        outputs: list[int] = []

        def blocking_iter():
            yield from range(14)
            if not allow_last_chunk_item.wait(timeout=1):
                raise TimeoutError(
                    "Timed out waiting for the last chunk item."
                )
            yield 14

        actual_generator = torch.Generator().manual_seed(12)
        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=13,
            shuffle=True,
            generator=actual_generator,
        )

        def consume_first_item():
            outputs.append(next(prefetch_iter))
            first_item_ready.set()

        consumer_thread = threading.Thread(target=consume_first_item)
        consumer_thread.start()
        assert not first_item_ready.wait(timeout=0.1)

        allow_last_chunk_item.set()
        assert first_item_ready.wait(timeout=1)
        consumer_thread.join(timeout=1)

        expected_generator = torch.Generator().manual_seed(12)
        pool = list(range(15))
        indices = torch.randperm(15, generator=expected_generator).tolist()
        assert outputs == [pool[indices[0]]]
        assert list(cast(Any, prefetch_iter)._ready_queue) == [
            pool[indices[1]]
        ]
        assert cast(Any, prefetch_iter)._shuffle_reservoir == [
            pool[index] for index in indices[2:]
        ]
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )
        assert sorted(outputs + list(prefetch_iter)) == list(range(15))

    def test_non_divisible_k_reuses_partial_queue_without_deadlock(self):
        source_length = 200
        prefetch_iter = create_prefetch_iterator(
            iter(range(source_length)),
            prefetch_size=38,
            shuffle=True,
            generator=torch.Generator().manual_seed(13),
        )
        consume_done = threading.Event()
        outputs: list[int] = []
        errors: list[BaseException] = []

        def consume_all():
            try:
                outputs.extend(prefetch_iter)
            except BaseException as exc:
                errors.append(exc)
            finally:
                consume_done.set()

        consumer_thread = threading.Thread(target=consume_all)
        consumer_thread.start()
        if not consume_done.wait(timeout=2):
            cast(Any, prefetch_iter).close(timeout=1)
        consumer_thread.join(timeout=1)

        assert consume_done.is_set()
        assert errors == []
        assert sorted(outputs) == list(range(source_length))

    @pytest.mark.parametrize("prefetch_size", range(6, 129))
    def test_chunk_layout_keeps_two_chunks_within_1_5k(
        self,
        prefetch_size: int,
    ):
        prefetch_iter = create_prefetch_iterator(
            iter(range(1000)),
            prefetch_size=prefetch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        state = cast(Any, prefetch_iter)._state

        chunk_size = state.producer_chunk_size
        incoming_capacity = state.buffer_capacity
        headroom = prefetch_size // 2
        expected_chunk_size = min(16, headroom // 3)
        expected_capacity = (
            (headroom - expected_chunk_size) // expected_chunk_size
        ) * expected_chunk_size
        assert chunk_size == expected_chunk_size
        assert incoming_capacity == expected_capacity
        assert incoming_capacity >= 2 * chunk_size
        assert incoming_capacity % chunk_size == 0
        assert (
            prefetch_size + chunk_size + incoming_capacity
            <= prefetch_size + prefetch_size // 2
        )

        cast(Any, prefetch_iter).close()

    @pytest.mark.parametrize("prefetch_size", range(2, 6))
    def test_small_k_scalar_fallback_shares_1_5k_headroom(
        self,
        prefetch_size: int,
    ):
        produced_count = 0

        def counted_iter():
            nonlocal produced_count
            for item in range(100):
                produced_count += 1
                yield item

        prefetch_iter = create_prefetch_iterator(
            iter(counted_iter()),
            prefetch_size=prefetch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        state = cast(Any, prefetch_iter)._state
        assert state.producer_chunk_size == 1
        assert state.buffer_capacity == prefetch_size // 2
        assert cast(Any, prefetch_iter)._shuffle_uses_shared_headroom_credit

        outputs = []
        for item in prefetch_iter:
            outputs.append(item)
            assert produced_count - len(outputs) <= (
                prefetch_size + prefetch_size // 2
            )

        assert sorted(outputs) == list(range(100))

    def test_small_k_releases_credit_only_on_the_next_request(self):
        produced_count = 0

        def counted_iter():
            nonlocal produced_count
            for item in range(100):
                produced_count += 1
                yield item

        prefetch_iter = create_prefetch_iterator(
            iter(counted_iter()),
            prefetch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        state = cast(Any, prefetch_iter)._state

        first_item = next(prefetch_iter)
        assert first_item in range(3)
        assert produced_count == 3
        with state.condition:
            assert state.consumer_reserved_size == 1
            assert len(state.incoming_queue) == 0

        time.sleep(0.05)
        assert produced_count == 3

        second_item = next(prefetch_iter)
        assert second_item in range(4)
        cast(Any, prefetch_iter).close()

    def test_chunk_credits_keep_total_materialization_within_1_5k(self):
        produced_count = 0

        def counted_iter():
            nonlocal produced_count
            for item in range(300):
                produced_count += 1
                yield item

        prefetch_iter = create_prefetch_iterator(
            iter(counted_iter()),
            prefetch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        state = cast(Any, prefetch_iter)._state

        for _ in range(100):
            if produced_count == 48:
                break
            time.sleep(0.01)
        assert produced_count == 48
        with state.condition:
            assert state.producer_chunk_size == 16
            assert state.buffer_capacity == 48
            assert len(state.incoming_queue) == 48
            assert state.producer_reserved_size == 0

        yielded_item = next(prefetch_iter)
        assert yielded_item in range(192)

        # The yielded item plus K reservoir, T-1 active outputs, and N shared
        # producer-side items account for exactly 1.5K materialized samples.
        for _ in range(100):
            if produced_count == 192:
                break
            time.sleep(0.01)
        assert produced_count == 192
        time.sleep(0.05)
        assert produced_count == 192
        with state.condition:
            iterator_owned = (
                len(cast(Any, prefetch_iter)._shuffle_reservoir)
                + len(cast(Any, prefetch_iter)._ready_queue)
                + len(cast(Any, prefetch_iter)._shuffle_pending_incoming)
                + len(state.incoming_queue)
                + state.producer_reserved_size
            )
        assert iterator_owned == 191
        assert iterator_owned + 1 <= 128 + 128 // 2

        del yielded_item
        cast(Any, prefetch_iter).close()

    def test_slow_producer_does_not_drain_shuffle_reservoir(self):
        allow_next_item = threading.Event()
        producer_waiting = threading.Event()
        second_item_ready = threading.Event()
        yielded_items: list[int] = []

        def blocking_iter():
            yield from range(14)
            producer_waiting.set()
            if not allow_next_item.wait(timeout=1):
                raise TimeoutError("Timed out waiting for the next item.")
            yield 14
            yield 15

        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=12,
            shuffle=True,
            generator=torch.Generator().manual_seed(2),
        )

        yielded_items.append(next(prefetch_iter))
        assert producer_waiting.wait(timeout=1)
        assert len(cast(Any, prefetch_iter)._ready_queue) == 1

        # The second output was already partitioned with the first one, so a
        # slow producer cannot add another per-sample handoff here.
        yielded_items.append(next(prefetch_iter))

        def consume_third_item():
            yielded_items.append(next(prefetch_iter))
            second_item_ready.set()

        consumer_thread = threading.Thread(target=consume_third_item)
        consumer_thread.start()
        assert not second_item_ready.wait(timeout=0.1)

        allow_next_item.set()
        assert second_item_ready.wait(timeout=1)
        consumer_thread.join(timeout=1)

        remaining_items = list(prefetch_iter)
        assert sorted(yielded_items + remaining_items) == list(range(16))

    @pytest.mark.parametrize("source_length", [0, 1, 3, 4, 5, 6, 20])
    def test_streaming_shuffle_preserves_input_multiset(
        self,
        source_length: int,
    ):
        prefetch_iter = create_prefetch_iterator(
            iter(range(source_length)),
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(3),
        )

        assert sorted(prefetch_iter) == list(range(source_length))

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    def test_producer_timing_does_not_change_shuffle_order(
        self,
        generator_kind: str,
    ):
        def collect(delay: float) -> list[int]:
            def source_iter():
                for item in range(30):
                    if delay > 0 and item % 3 == 0:
                        time.sleep(delay)
                    yield item

            if generator_kind == "torch":
                generator: torch.Generator | np.random.Generator = (
                    torch.Generator().manual_seed(4)
                )
            else:
                generator = np.random.default_rng(4)
            return list(
                create_prefetch_iterator(
                    iter(source_iter()),
                    prefetch_size=4,
                    shuffle=True,
                    generator=generator,
                )
            )

        assert collect(delay=0.0) == collect(delay=0.001)

    def test_torch_chunk_rng_matches_repeated_randperm(self):
        prefetch_size = 12
        chunk_size = 2
        output_count = 4
        actual_generator = torch.Generator().manual_seed(5)
        prefetch_iter = create_prefetch_iterator(
            iter(range(100)),
            prefetch_size=prefetch_size,
            shuffle=True,
            generator=actual_generator,
        )

        actual_outputs = [next(prefetch_iter) for _ in range(output_count)]
        cast(Any, prefetch_iter).close()

        expected_generator = torch.Generator().manual_seed(5)
        reservoir = list(range(prefetch_size))
        expected_outputs = []
        for chunk_start in range(
            prefetch_size,
            prefetch_size + output_count,
            chunk_size,
        ):
            pool = reservoir + list(
                range(chunk_start, chunk_start + chunk_size)
            )
            indices = torch.randperm(
                len(pool),
                generator=expected_generator,
            ).tolist()
            expected_outputs.extend(
                pool[index] for index in indices[:chunk_size]
            )
            reservoir = [pool[index] for index in indices[chunk_size:]]

        assert actual_outputs == expected_outputs
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    def test_chunk_rng_matches_reference_for_all_supported_k(
        self,
        generator_kind: str,
    ):
        for prefetch_size in range(2, 129):
            headroom = prefetch_size // 2
            chunk_size = min(16, headroom // 3) if headroom >= 3 else 1
            source = list(range(prefetch_size + 2 * chunk_size + 1))
            if generator_kind == "torch":
                actual_generator: torch.Generator | np.random.Generator = (
                    torch.Generator().manual_seed(17)
                )
                expected_generator: torch.Generator | np.random.Generator = (
                    torch.Generator().manual_seed(17)
                )
            else:
                actual_generator = np.random.default_rng(17)
                expected_generator = np.random.default_rng(17)

            actual = list(
                create_prefetch_iterator(
                    iter(source),
                    prefetch_size=prefetch_size,
                    shuffle=True,
                    generator=actual_generator,
                )
            )

            reservoir = source[:prefetch_size]
            expected = []
            for offset in range(prefetch_size, len(source), chunk_size):
                incoming = source[offset : offset + chunk_size]
                pool = reservoir + incoming
                if isinstance(expected_generator, torch.Generator):
                    indices = torch.randperm(
                        len(pool),
                        generator=expected_generator,
                    ).tolist()
                else:
                    indices = expected_generator.permutation(
                        len(pool)
                    ).tolist()
                expected.extend(
                    pool[index] for index in indices[: len(incoming)]
                )
                reservoir = [pool[index] for index in indices[len(incoming) :]]

            if isinstance(expected_generator, torch.Generator):
                tail_indices = torch.randperm(
                    len(reservoir),
                    generator=expected_generator,
                ).tolist()
            else:
                tail_indices = expected_generator.permutation(
                    len(reservoir)
                ).tolist()
            expected.extend(reservoir[index] for index in tail_indices)

            assert actual == expected, prefetch_size
            if isinstance(actual_generator, torch.Generator):
                assert isinstance(expected_generator, torch.Generator)
                assert torch.equal(
                    actual_generator.get_state(),
                    expected_generator.get_state(),
                ), prefetch_size
            else:
                assert isinstance(expected_generator, np.random.Generator)
                assert (
                    actual_generator.bit_generator.state
                    == expected_generator.bit_generator.state
                ), prefetch_size

    def test_early_close_commits_the_whole_chunk_rng_draw(self):
        actual_generator = torch.Generator().manual_seed(6)
        prefetch_iter = create_prefetch_iterator(
            iter(range(100)),
            prefetch_size=12,
            shuffle=True,
            generator=actual_generator,
        )

        next(prefetch_iter)
        cast(Any, prefetch_iter).close()

        expected_generator = torch.Generator().manual_seed(6)
        torch.randperm(14, generator=expected_generator)
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )

    def test_prefetches_next_window_while_consuming_current_window(self):
        # After the first window is handed to the consumer, the producer should
        # immediately start filling the next window instead of waiting for the
        # current one to be fully drained.
        attempted_refill = threading.Event()
        allow_tail_items = threading.Event()

        def blocking_iter():
            yield 0
            yield 1
            attempted_refill.set()
            if not allow_tail_items.wait(timeout=1):
                raise TimeoutError("Timed out waiting for tail items.")
            yield 2
            yield 3

        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        # Reaching this event means the producer has already advanced to the
        # next refill stage while the current window is still being consumed.
        assert attempted_refill.wait(timeout=1)

        allow_tail_items.set()
        remaining_items = list(prefetch_iter)
        assert remaining_items == [1, 2, 3]

    def test_close_stops_prefetch_thread_waiting_on_full_queue(self):
        # Closing the generator early should notify the producer and let the
        # background thread exit instead of remaining blocked on a full queue.
        item_requests = 0

        def blocking_iter():
            nonlocal item_requests
            item_requests += 1
            yield 0
            item_requests += 1
            yield 1
            item_requests += 1
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        for _ in range(40):
            if item_requests == 2:
                break
            time.sleep(0.05)
        assert item_requests == 2

        cast(Any, prefetch_iter).close()

        for _ in range(20):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    def test_close_waits_for_short_inflight_prefetch_item(
        self,
        caplog,
        monkeypatch: pytest.MonkeyPatch,
    ):
        tail_item_started = threading.Event()

        monkeypatch.setattr(
            prefetch_module,
            "_PREFETCH_CLOSE_JOIN_TIMEOUT_SEC",
            0.02,
        )
        caplog.set_level("INFO")

        def slow_tail_iter():
            yield 0
            yield 1
            tail_item_started.set()
            time.sleep(0.1)
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = create_prefetch_iterator(
            iter(slow_tail_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        assert tail_item_started.wait(timeout=1)

        cast(Any, prefetch_iter).close()

        assert self._count_prefetch_threads() == baseline_threads
        assert "continuing the soft close wait" in caplog.text
        assert "exited and was joined" in caplog.text

    def test_close_warns_when_inflight_prefetch_item_blocks(self, caplog):
        tail_item_started = threading.Event()
        allow_tail_item = threading.Event()

        def blocked_tail_iter():
            yield 0
            yield 1
            tail_item_started.set()
            allow_tail_item.wait()
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = create_prefetch_iterator(
            iter(blocked_tail_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        assert tail_item_started.wait(timeout=1)

        start = time.monotonic()
        cast(Any, prefetch_iter).close(timeout=0.2)
        elapsed = time.monotonic() - start

        assert elapsed < 0.7
        assert "close() is returning while the producer remains alive" in (
            caplog.text
        )

        allow_tail_item.set()
        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    @pytest.mark.parametrize("observation_path", ["next", "close"])
    def test_producer_error_is_raised_before_blocking_source_close(
        self,
        observation_path: str,
    ):
        class _ErrorThenBlockingCloseIterator:
            def __init__(self) -> None:
                self.next_started = threading.Event()
                self.allow_error = threading.Event()
                self.close_started = threading.Event()
                self.allow_close = threading.Event()

            def __iter__(self):
                return self

            def __next__(self) -> int:
                self.next_started.set()
                if not self.allow_error.wait(timeout=1):
                    raise TimeoutError("Timed out waiting to raise error.")
                raise RuntimeError("primary source error")

            def close(self) -> None:
                self.close_started.set()
                if not self.allow_close.wait(timeout=1):
                    raise TimeoutError("Timed out waiting to close source.")
                raise RuntimeError("secondary source close error")

        source_iter = _ErrorThenBlockingCloseIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(6),
        )
        assert source_iter.next_started.wait(timeout=1)

        observer_done = threading.Event()
        observer_errors: list[BaseException] = []

        def observe_producer_error():
            try:
                if observation_path == "next":
                    next(prefetch_iter)
                else:
                    cast(Any, prefetch_iter).close(timeout=2.0)
            except BaseException as exc:
                observer_errors.append(exc)
            finally:
                observer_done.set()

        observer_thread = threading.Thread(target=observe_producer_error)
        if observation_path == "next":
            observer_thread.start()
            time.sleep(0.05)
            source_iter.allow_error.set()
            assert source_iter.close_started.wait(timeout=1)
        else:
            source_iter.allow_error.set()
            assert source_iter.close_started.wait(timeout=1)
            observer_thread.start()

        try:
            assert observer_done.wait(timeout=0.5)
            assert len(observer_errors) == 1
            assert isinstance(observer_errors[0], RuntimeError)
            assert str(observer_errors[0]) == "primary source error"
            if observation_path == "next":
                start = time.monotonic()
                cast(Any, prefetch_iter).close(timeout=2.0)
                assert time.monotonic() - start < 0.5
        finally:
            source_iter.allow_close.set()
            observer_thread.join(timeout=1)
            cast(Any, prefetch_iter).close(timeout=1)

    def test_close_observes_producer_error_arriving_during_wait(self):
        class _LateErrorThenBlockingCloseIterator:
            def __init__(self) -> None:
                self.next_started = threading.Event()
                self.allow_error = threading.Event()
                self.close_started = threading.Event()
                self.allow_close = threading.Event()

            def __iter__(self):
                return self

            def __next__(self) -> int:
                self.next_started.set()
                if not self.allow_error.wait(timeout=2):
                    raise TimeoutError("Timed out waiting to raise error.")
                raise RuntimeError("late primary source error")

            def close(self) -> None:
                self.close_started.set()
                if not self.allow_close.wait(timeout=2):
                    raise TimeoutError("Timed out waiting to close source.")

        source_iter = _LateErrorThenBlockingCloseIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(9),
        )
        assert source_iter.next_started.wait(timeout=1)

        close_done = threading.Event()
        close_errors: list[BaseException] = []

        def close_prefetch_iter():
            try:
                cast(Any, prefetch_iter).close(timeout=2.0)
            except BaseException as exc:
                close_errors.append(exc)
            finally:
                close_done.set()

        close_thread = threading.Thread(target=close_prefetch_iter)
        close_thread.start()
        state = cast(Any, prefetch_iter)._state
        for _ in range(100):
            with state.condition:
                if state.consumer_closed:
                    break
            time.sleep(0.01)
        with state.condition:
            assert state.consumer_closed

        source_iter.allow_error.set()
        try:
            assert source_iter.close_started.wait(timeout=1)
            assert close_done.wait(timeout=0.5)
            assert len(close_errors) == 1
            assert isinstance(close_errors[0], RuntimeError)
            assert str(close_errors[0]) == "late primary source error"
        finally:
            source_iter.allow_close.set()
            close_thread.join(timeout=1)
            cast(Any, prefetch_iter).close(timeout=1)

    def test_close_releases_buffered_sample_references(self):
        class _Payload:
            pass

        payload_refs: list[weakref.ReferenceType[_Payload]] = []

        def payload_iter():
            for _ in range(100):
                payload = _Payload()
                payload_refs.append(weakref.ref(payload))
                yield payload

        prefetch_iter = create_prefetch_iterator(
            iter(payload_iter()),
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
        )
        yielded_item = next(prefetch_iter)
        for _ in range(40):
            if len(payload_refs) == 6:
                break
            time.sleep(0.01)
        assert len(payload_refs) == 6

        del yielded_item
        cast(Any, prefetch_iter).close()
        gc.collect()

        assert all(payload_ref() is None for payload_ref in payload_refs)

    def test_close_releases_producer_local_sample_before_source_close(self):
        class _Payload:
            pass

        class _OnePayloadThenBlockingCloseIterator:
            def __init__(self) -> None:
                self.produced = False
                self.payload_ref: weakref.ReferenceType[_Payload] | None = None
                self.close_started = threading.Event()
                self.allow_close = threading.Event()

            def __iter__(self):
                return self

            def __next__(self) -> _Payload:
                if self.produced:
                    raise StopIteration
                self.produced = True
                payload = _Payload()
                self.payload_ref = weakref.ref(payload)
                return payload

            def close(self) -> None:
                self.close_started.set()
                self.allow_close.wait()

        source_iter = _OnePayloadThenBlockingCloseIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(10),
        )
        state = cast(Any, prefetch_iter)._state
        for _ in range(100):
            with state.condition:
                if state.incoming_queue:
                    break
            time.sleep(0.01)
        with state.condition:
            assert len(state.incoming_queue) == 1
        assert source_iter.payload_ref is not None

        cast(Any, prefetch_iter).close(
            raise_producer_errors=False,
            timeout=0.0,
        )
        try:
            assert source_iter.close_started.wait(timeout=1)
            gc.collect()
            assert source_iter.payload_ref() is None
        finally:
            source_iter.allow_close.set()
            cast(Any, prefetch_iter).close(timeout=1)

    def test_producer_error_releases_buffered_sample_references(self):
        class _Payload:
            pass

        payload_refs: list[weakref.ReferenceType[_Payload]] = []

        def failing_payload_iter():
            for _ in range(7):
                payload = _Payload()
                payload_refs.append(weakref.ref(payload))
                yield payload
            raise RuntimeError("payload source failed")

        prefetch_iter = create_prefetch_iterator(
            iter(failing_payload_iter()),
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(8),
        )

        with pytest.raises(RuntimeError, match="payload source failed"):
            while True:
                yielded_item = next(prefetch_iter)
                del yielded_item

        cast(Any, prefetch_iter).close()
        gc.collect()

        assert all(payload_ref() is None for payload_ref in payload_refs)

    def test_close_logs_producer_error_in_gc_close_path(self, caplog):
        baseline_threads = self._count_prefetch_threads()

        def failing_iter():
            yield 0
            yield 1
            raise RuntimeError("producer failed during close")

        prefetch_iter = create_prefetch_iterator(
            iter(failing_iter()),
            prefetch_size=3,
            shuffle=False,
            generator=None,
        )

        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            time.sleep(0.05)

        cast(Any, prefetch_iter).close(raise_producer_errors=False)

        assert "producer-side exception" in caplog.text

    def test_close_calls_source_iter_close_from_producer_thread(self):
        class _CloseTrackingIterator:
            def __init__(self) -> None:
                self.item_count = 0
                self.window_ready = threading.Event()
                self.closed = threading.Event()
                self.close_thread_name: str | None = None

            def __iter__(self):
                return self

            def __next__(self) -> int:
                if self.item_count >= 2:
                    raise StopIteration
                value = self.item_count
                self.item_count += 1
                if self.item_count == 2:
                    self.window_ready.set()
                return value

            def close(self) -> None:
                self.close_thread_name = threading.current_thread().name
                self.closed.set()

        source_iter = _CloseTrackingIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert source_iter.window_ready.wait(timeout=1)
        cast(Any, prefetch_iter).close()

        assert source_iter.closed.is_set()
        assert source_iter.close_thread_name == "dataset-prefetch-producer"

    def test_source_iter_close_error_is_raised_without_timeout(self):
        class _FailingCloseIterator:
            def __init__(self) -> None:
                self.item_count = 0
                self.window_ready = threading.Event()
                self.closed = threading.Event()

            def __iter__(self):
                return self

            def __next__(self) -> int:
                if self.item_count >= 2:
                    raise StopIteration
                value = self.item_count
                self.item_count += 1
                if self.item_count == 2:
                    self.window_ready.set()
                return value

            def close(self) -> None:
                self.closed.set()
                raise RuntimeError("source close failed")

        source_iter = _FailingCloseIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert source_iter.window_ready.wait(timeout=1)
        with pytest.raises(RuntimeError, match="source close failed"):
            cast(Any, prefetch_iter).close(timeout=0.2)

        assert source_iter.closed.is_set()

    def test_prefetch_thread_start_failure_closes_source_iterator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class _CloseTrackingIterator:
            def __init__(self) -> None:
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self) -> int:
                return 0

            def close(self) -> None:
                self.closed = True

        class _FailingThread:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("can't start new thread")

        source_iter = _CloseTrackingIterator()
        monkeypatch.setattr(
            prefetch_module.threading,
            "Thread",
            _FailingThread,
        )

        with pytest.raises(RuntimeError, match="can't start new thread"):
            create_prefetch_iterator(
                source_iter,
                prefetch_size=2,
                shuffle=False,
                generator=None,
            )

        assert source_iter.closed is True

    def test_second_close_consumes_error_after_first_close_timeout(
        self,
        caplog,
    ):
        source_blocked = threading.Event()
        unblock_source = threading.Event()

        def blocked_then_failing_iter():
            yield 0
            yield 1
            source_blocked.set()
            unblock_source.wait()
            raise RuntimeError("producer failed after timeout")

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = create_prefetch_iterator(
            iter(blocked_then_failing_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        assert source_blocked.wait(timeout=1)
        cast(Any, prefetch_iter).close(timeout=0.05)
        assert "close() is returning while the producer remains alive" in (
            caplog.text
        )

        unblock_source.set()
        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            time.sleep(0.05)

        with pytest.raises(
            RuntimeError,
            match="producer failed after timeout",
        ):
            cast(Any, prefetch_iter).close(timeout=1.0)

    def test_close_closes_nested_prefetch_iterator_but_not_wrapper_state(
        self,
    ):
        baseline_threads = self._count_prefetch_threads()
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(16))),
                ArrayDatasetItem(data=list(range(100, 116))),
            ],
            shuffle=ShuffleConfig(
                shuffle=True,
                chunk_size=4,
                prefetch_factor=2,
            ),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=dataset._shuffle,
        )
        accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(
                dispatch_batches=False,
                split_batches=False,
                even_batches=False,
            )
        )
        configure_data_loader_for_accelerate(
            accelerator=accelerator,
            data_loader=dataloader,
        )
        dataloader = accelerator.prepare(dataloader)
        baseline_ref_count = sum(
            ref is not None
            for ref in accelerator.gradient_state.dataloader_references
        )

        dataloader_iter = iter(dataloader)
        next(dataloader_iter)

        _close_dataloader_iterator(dataloader_iter)
        # `_close_dataloader_iterator()` only tears down iterator-owned
        # resources. The prepared wrapper still owns Accelerate state cleanup.
        assert accelerator.gradient_state.in_dataloader
        assert (
            sum(
                ref is not None
                for ref in accelerator.gradient_state.dataloader_references
            )
            > baseline_ref_count
        )
        dataloader.end()
        assert not accelerator.gradient_state.in_dataloader
        assert (
            sum(
                ref is not None
                for ref in accelerator.gradient_state.dataloader_references
            )
            == baseline_ref_count
        )

        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    def test_raises_producer_error_without_draining_ready_queue(self):
        # If the producer fails after the current window has been handed off,
        # the consumer should observe that failure on the next pull instead of
        # silently draining the rest of the ready queue first.
        fail_now = threading.Event()
        failure_branch_reached = threading.Event()

        def failing_iter():
            yield 0
            yield 1
            if not fail_now.wait(timeout=1):
                raise TimeoutError("Timed out waiting to trigger failure.")
            failure_branch_reached.set()
            raise RuntimeError("producer failed")

        prefetch_iter = create_prefetch_iterator(
            iter(failing_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        fail_now.set()
        # Wait until the producer has actually reached the failing branch so
        # the next `next()` call deterministically checks error propagation.
        assert failure_branch_reached.wait(timeout=1)
        with pytest.raises(RuntimeError, match="producer failed"):
            next(prefetch_iter)
