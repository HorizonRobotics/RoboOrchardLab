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

from __future__ import annotations
import inspect
import os
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from types import GeneratorType
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from robo_orchard_core.utils.logging import LoggerManager
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data._utils.fetch import _IterableDatasetFetcher
from torch.utils.data.dataloader import (
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter,
)

logger = LoggerManager().get_child(__name__)

__all__ = [
    "DataloaderCloseReason",
    "create_prefetch_iterator",
    "close_iterators_best_effort",
    "close_dataloader_resources",
]

_PREFETCH_CLOSE_JOIN_TIMEOUT_SEC = 1.0
_DEFAULT_PREFETCH_CLOSE_HARD_TIMEOUT_SEC = 60.0
_PREFETCH_CLOSE_HARD_TIMEOUT_ENV = (
    "ROBO_ORCHARD_DATASET_PREFETCH_CLOSE_HARD_TIMEOUT_SEC"
)
_SHUFFLE_PREFETCH_CHUNK_SIZE = 16


class DataloaderCloseReason(Enum):
    """Reason a training loop is closing active dataloader resources."""

    EPOCH_EXHAUSTED = "epoch_exhausted"
    COORDINATED_EPOCH_END = "coordinated_epoch_end"
    COORDINATED_EARLY_STOP = "coordinated_early_stop"
    MAX_STEP_END = "max_step_end"
    EARLY_BREAK = "early_break"
    EXCEPTION_ABORT = "exception_abort"
    TRAINER_TEARDOWN = "trainer_teardown"


_KEEP_PERSISTENT_WORKER_REASONS = {
    DataloaderCloseReason.EPOCH_EXHAUSTED,
    DataloaderCloseReason.COORDINATED_EPOCH_END,
}


def create_prefetch_iterator(
    source_iter: Iterator[Any],
    prefetch_size: int,
    shuffle: bool,
    generator: torch.Generator | np.random.Generator | None,
    *,
    hard_close_timeout: float | None = None,
    gc_close_timeout: float = 0.0,
) -> Iterator[Any]:
    """Wrap an iterator with close-aware prefetching.

    Use this in dataset iteration paths that need producer/consumer overlap
    without owning a full ``DataLoader`` lifecycle. When ``prefetch_size`` is
    greater than one, the returned iterator owns a producer thread; callers
    that may stop early should close it directly or pass it to
    ``close_iterators_best_effort``.

    The threaded wrapper is the sole caller of ``source_iter`` and releases
    its own buffered sample references during cleanup. It cannot interrupt a
    ``source_iter.__next__()`` or ``source_iter.close()`` call that is already
    running; the source remains responsible for eventually returning from
    those calls and for releasing resources retained inside the source itself.

    Args:
        source_iter (Iterator[Any]): Iterator to consume from the producer
            thread.
        prefetch_size (int): FIFO handoff window size or streaming shuffle
            reservoir size. A value of one returns ``source_iter`` unchanged.
        shuffle (bool): Whether to use chunked streaming reservoir shuffle.
        generator (torch.Generator | np.random.Generator | None): Random
            generator used when ``shuffle`` is enabled. When ``None`` and
            shuffle is enabled, a local torch generator is created.
        hard_close_timeout (float | None, optional): Maximum explicit close
            wait in seconds before logging a warning and returning. Default is
            ``None``, which reads the hidden environment override and otherwise
            uses the module default. The historical parameter name is retained
            for compatibility; reaching the deadline is not a hard failure.
        gc_close_timeout (float, optional): Best-effort wait in seconds used
            by finalization paths. Default is ``0.0``.

    Returns:
        Iterator[Any]: ``source_iter`` when no producer thread is needed, or
            a close-aware prefetch iterator otherwise.

    Raises:
        ValueError: If ``prefetch_size`` is not positive or a shuffle
            generator has an unsupported type.
    """
    if prefetch_size <= 0:
        raise ValueError("prefetch_size must be greater than 0.")

    if prefetch_size == 1:
        return source_iter

    if shuffle and generator is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

    return _ThreadedPrefetchIterator(
        source_iter=source_iter,
        prefetch_size=prefetch_size,
        shuffle=shuffle,
        generator=generator,
        hard_close_timeout=hard_close_timeout,
        gc_close_timeout=gc_close_timeout,
    )


def close_iterators_best_effort(
    iterators: Iterable[Any],
    *,
    primary_exc: BaseException | None = None,
    shutdown_persistent_workers: bool = False,
) -> None:
    """Close iterator-like resources while preserving primary failures.

    Use this for dataset-internal iterators, prefetch iterators, and active
    PyTorch dataloader iterators when the caller does not own a full
    dataloader lifecycle. If ``primary_exc`` is set, cleanup errors are
    logged and suppressed so the original failure remains visible; otherwise
    cleanup errors are raised.

    Args:
        iterators (Iterable[Any]): Iterator-like objects to close. Objects
            with ``close()``, Python generators, and PyTorch dataloader
            iterators receive specialized cleanup.
        primary_exc (BaseException | None, optional): Exception already being
            propagated by the caller. Default is ``None``.
        shutdown_persistent_workers (bool, optional): Whether PyTorch
            persistent worker iterators should be shut down. Default is
            ``False``.

    Raises:
        BaseException: A cleanup error when no ``primary_exc`` is active.
        RuntimeError: If multiple cleanup errors occur and no ``primary_exc``
            is active.
    """
    errors: list[BaseException] = []
    visited: set[int] = set()
    for iterator in iterators:
        try:
            _close_single_iterator(
                iterator,
                visited,
                shutdown_persistent_workers=shutdown_persistent_workers,
            )
        except BaseException as exc:
            errors.append(exc)

    if not errors:
        return
    if primary_exc is not None:
        _log_close_errors(primary_exc, errors)
        return
    _raise_close_errors(errors)


def close_dataloader_resources(
    dataloader: Any,
    dataloader_iter: Any,
    *,
    reason: DataloaderCloseReason,
    primary_exc: BaseException | None = None,
) -> None:
    """Close active dataloader resources after a training loop stops.

    Use this from trainer-level code that owns both the dataloader object and
    its active iterator. Besides closing the iterator stack, this ends
    prepared dataloader wrappers and clears persistent-worker owner state when
    the close reason means workers should not be reused.

    Args:
        dataloader (Any): Original or prepared dataloader owner.
        dataloader_iter (Any): Active iterator returned by
            ``iter(dataloader)``.
        reason (DataloaderCloseReason): Why the iterator is being closed.
            Natural epoch exhaustion keeps persistent workers reusable;
            early-stop, max-step, exception, and teardown reasons shut them
            down.
        primary_exc (BaseException | None, optional): Exception already being
            propagated by the training body. Default is ``None``. When set,
            cleanup failures are logged instead of replacing it.

    Raises:
        BaseException: A cleanup error when no ``primary_exc`` is active.
        RuntimeError: If multiple cleanup errors occur and no ``primary_exc``
            is active.
    """

    shutdown_persistent_workers = reason not in _KEEP_PERSISTENT_WORKER_REASONS
    dataloader_owners = list(_iter_dataloader_owners(dataloader))
    errors: list[BaseException] = []
    try:
        close_iterators_best_effort(
            [dataloader_iter],
            primary_exc=primary_exc,
            shutdown_persistent_workers=shutdown_persistent_workers,
        )
    except BaseException as exc:
        errors.append(exc)

    try:
        _end_prepared_dataloader_wrappers(dataloader_owners)
    except BaseException as exc:
        errors.append(exc)

    if shutdown_persistent_workers:
        try:
            _clear_persistent_dataloader_owner_iterators(dataloader_owners)
        except BaseException as exc:
            errors.append(exc)

    if not errors:
        return
    if primary_exc is not None:
        _log_close_errors(primary_exc, errors)
        return
    _raise_close_errors(errors)


def _close_dataloader_owner_resources(
    dataloader: Any,
    *,
    primary_exc: BaseException | None = None,
) -> None:
    """Shut down dataloader resources kept alive by owner objects.

    This is the trainer-teardown counterpart of ``close_dataloader_resources``.
    It only closes existing owner-held PyTorch dataloader iterators; it never
    calls ``iter(dataloader)`` and therefore cannot create workers during
    cleanup. Use it after normal epoch-level closes have intentionally kept
    persistent workers reusable.

    Args:
        dataloader (Any): Original or prepared dataloader owner.
        primary_exc (BaseException | None, optional): Exception already being
            propagated by the caller. Default is ``None``. When set, cleanup
            failures are logged instead of replacing it.

    Raises:
        BaseException: A cleanup error when no ``primary_exc`` is active.
        RuntimeError: If multiple cleanup errors occur and no ``primary_exc``
            is active.
    """

    dataloader_owners = list(_iter_dataloader_owners(dataloader))
    owner_iterators = [
        owner._iterator
        for owner in dataloader_owners
        if isinstance(owner, TorchDataLoader)
        and getattr(owner, "_iterator", None) is not None
    ]

    errors: list[BaseException] = []
    try:
        close_iterators_best_effort(
            owner_iterators,
            primary_exc=primary_exc,
            shutdown_persistent_workers=True,
        )
    except BaseException as exc:
        errors.append(exc)

    try:
        _end_prepared_dataloader_wrappers(dataloader_owners)
    except BaseException as exc:
        errors.append(exc)

    try:
        _clear_persistent_dataloader_owner_iterators(dataloader_owners)
    except BaseException as exc:
        errors.append(exc)

    if not errors:
        return
    if primary_exc is not None:
        _log_close_errors(primary_exc, errors)
        return
    _raise_close_errors(errors)


def _get_prefetch_close_hard_timeout_sec() -> float:
    value = os.environ.get(_PREFETCH_CLOSE_HARD_TIMEOUT_ENV)
    if value is None:
        return _DEFAULT_PREFETCH_CLOSE_HARD_TIMEOUT_SEC

    try:
        timeout = float(value)
    except ValueError:
        warnings.warn(
            f"Ignoring invalid {_PREFETCH_CLOSE_HARD_TIMEOUT_ENV}={value!r}; "
            "falling back to "
            f"{_DEFAULT_PREFETCH_CLOSE_HARD_TIMEOUT_SEC:.1f}s.",
            UserWarning,
        )
        return _DEFAULT_PREFETCH_CLOSE_HARD_TIMEOUT_SEC

    if timeout <= 0:
        warnings.warn(
            f"Ignoring non-positive {_PREFETCH_CLOSE_HARD_TIMEOUT_ENV}="
            f"{value!r}; falling back to "
            f"{_DEFAULT_PREFETCH_CLOSE_HARD_TIMEOUT_SEC:.1f}s.",
            UserWarning,
        )
        return _DEFAULT_PREFETCH_CLOSE_HARD_TIMEOUT_SEC

    return timeout


@dataclass(slots=True)
class _ThreadedPrefetchState:
    source_iter: Iterator[Any]
    buffer_capacity: int
    consumer_wakeup_size: int
    producer_chunk_size: int
    incoming_queue: deque[Any] = field(default_factory=deque)
    condition: threading.Condition = field(default_factory=threading.Condition)
    consumer_close_event: threading.Event = field(
        default_factory=threading.Event
    )
    producer_reserved_size: int = 0
    consumer_reserved_size: int = 0
    producer_done: bool = False
    close_requested: bool = False
    close_completed: bool = False
    consumer_closed: bool = False
    producer_error: BaseException | None = None
    producer_error_handled: bool = False


def _close_source_iter_if_supported(source_iter: Iterator[Any]) -> None:
    close = getattr(source_iter, "close", None)
    if callable(close):
        close()


def _prefetch_producer_loop(state: _ThreadedPrefetchState) -> None:
    local_chunk: list[Any] = []
    try:
        while True:
            with state.condition:
                while (
                    len(state.incoming_queue)
                    + state.producer_reserved_size
                    + state.consumer_reserved_size
                    + state.producer_chunk_size
                    > state.buffer_capacity
                    and not state.consumer_closed
                ):
                    state.condition.wait()
                if state.consumer_closed:
                    return
                state.producer_reserved_size = state.producer_chunk_size

            source_exhausted = False
            for _ in range(state.producer_chunk_size):
                if state.consumer_close_event.is_set():
                    break
                try:
                    item = next(state.source_iter)
                except StopIteration:
                    source_exhausted = True
                    break
                try:
                    local_chunk.append(item)
                finally:
                    # The chunk owns the transferred reference. Do not also
                    # retain the last item in the producer frame.
                    del item

            with state.condition:
                state.producer_reserved_size = 0
                if state.consumer_closed:
                    local_chunk.clear()
                    return
                previous_size = len(state.incoming_queue)
                state.incoming_queue.extend(local_chunk)
                local_chunk.clear()
                if (
                    previous_size
                    < state.consumer_wakeup_size
                    <= len(state.incoming_queue)
                ):
                    state.condition.notify()

            if source_exhausted:
                return
    except BaseException as exc:
        with state.condition:
            state.producer_reserved_size = 0
            local_chunk.clear()
            if (
                state.producer_error is None
                and not state.producer_error_handled
            ):
                state.producer_error = exc
            # Surface the primary source error before source close, which may
            # block independently in the producer thread.
            state.condition.notify_all()
    finally:
        # Release producer-local samples before source_iter.close(), which may
        # block independently. A source that permanently blocks or retains its
        # own resources remains outside this wrapper's cleanup boundary.
        local_chunk.clear()
        with state.condition:
            state.producer_reserved_size = 0
        try:
            _close_source_iter_if_supported(state.source_iter)
        except BaseException as exc:
            with state.condition:
                if (
                    state.producer_error is None
                    and not state.producer_error_handled
                ):
                    state.producer_error = exc
                    state.condition.notify_all()
        finally:
            with state.condition:
                state.producer_done = True
                state.condition.notify_all()


class _ThreadedPrefetchIterator:
    """Iterator that overlaps source iteration with foreground consumption.

    This class is the resource-owning implementation behind
    ``create_prefetch_iterator``. It starts one producer thread during
    construction. The FIFO path preserves fixed-window handoff behavior. The
    shuffle path keeps a consumer-owned ``prefetch_size`` reservoir. The
    producer publishes chunks of up to 16 samples, while the consumer applies
    one random partition to each reservoir-plus-chunk candidate set. Chunk and
    producer capacity are selected so the reservoir, active consumer chunk,
    shared queue, and producer-local reservation stay within
    ``prefetch_size + prefetch_size // 2`` materialized samples. The chunked
    path keeps producer capacity at no less than two complete chunks. Shuffle
    sizes below six use a scalar chunk and retain its active credit until the
    output is handed to the caller, because the chunked capacity constraints
    are not simultaneously feasible there.

    The iterator owns the producer thread and the wrapped ``source_iter``.
    Normal exhaustion closes the producer path before raising
    ``StopIteration``. Callers that stop early must call ``close()`` directly
    or close it through ``close_iterators_best_effort``; ``__del__`` provides
    only best-effort finalization and must not be treated as the primary
    lifecycle path. Producer-side exceptions are re-raised on the consumer
    thread from ``__next__`` or ``close()`` so failures are not silently lost.
    Cleanup releases all prefetch-owned sample references, but it cannot
    interrupt an already-running ``source_iter.__next__()`` or
    ``source_iter.close()`` call. The source remains responsible for
    eventually returning and releasing resources it retains internally.

    Args:
        source_iter (Iterator[Any]): Iterator consumed by the producer thread.
        prefetch_size (int): FIFO window size or streaming shuffle reservoir
            size.
        shuffle (bool): Whether to use chunked streaming reservoir shuffle.
        generator (torch.Generator | np.random.Generator | None): Random
            generator used when ``shuffle`` is enabled.
        hard_close_timeout (float | None, optional): Maximum explicit close
            wait in seconds before logging a warning and returning. Default is
            ``None``, which defers to the module timeout policy. The historical
            parameter name is retained for compatibility; reaching the deadline
            is not a hard failure.
        gc_close_timeout (float, optional): Best-effort finalizer close wait
            in seconds. Default is ``0.0``.

    Raises:
        ValueError: If a shuffle generator has an unsupported type.
    """

    def __init__(
        self,
        source_iter: Iterator[Any],
        prefetch_size: int,
        shuffle: bool,
        generator: torch.Generator | np.random.Generator | None,
        *,
        hard_close_timeout: float | None = None,
        gc_close_timeout: float = 0.0,
    ) -> None:
        if shuffle and not isinstance(
            generator,
            (torch.Generator, np.random.Generator),
        ):
            raise ValueError(
                "Generator must be either a torch.Generator or a "
                "numpy.random.Generator."
            )

        shuffle_uses_shared_headroom_credit = False
        if shuffle:
            shuffle_headroom = prefetch_size // 2
            if shuffle_headroom >= 3:
                shuffle_chunk_size = min(
                    _SHUFFLE_PREFETCH_CHUNK_SIZE,
                    shuffle_headroom // 3,
                )
                available_producer_capacity = (
                    shuffle_headroom - shuffle_chunk_size
                )
                incoming_capacity = (
                    available_producer_capacity // shuffle_chunk_size
                ) * shuffle_chunk_size
                if incoming_capacity < 2 * shuffle_chunk_size:
                    raise RuntimeError(
                        "Shuffle producer capacity must hold at least two "
                        "complete chunks."
                    )
            else:
                shuffle_chunk_size = 1
                incoming_capacity = shuffle_headroom
                shuffle_uses_shared_headroom_credit = True
        else:
            shuffle_chunk_size = 1
            incoming_capacity = prefetch_size
        self._state = _ThreadedPrefetchState(
            source_iter=source_iter,
            buffer_capacity=incoming_capacity,
            consumer_wakeup_size=(
                shuffle_chunk_size if shuffle else prefetch_size
            ),
            producer_chunk_size=shuffle_chunk_size,
        )
        self._prefetch_size = prefetch_size
        self._shuffle = shuffle
        self._shuffle_uses_shared_headroom_credit = (
            shuffle_uses_shared_headroom_credit
        )
        self._generator = generator
        self._ready_queue: deque[Any] = deque()
        self._ready_queue_consumer_credits = 0
        self._deferred_consumer_credit = 0
        self._shuffle_reservoir: list[Any] = []
        self._shuffle_pending_incoming: list[Any] = []
        self._hard_close_timeout = hard_close_timeout
        self._gc_close_timeout = gc_close_timeout
        self._producer_thread = threading.Thread(
            target=_prefetch_producer_loop,
            args=(self._state,),
            name="dataset-prefetch-producer",
            daemon=True,
        )
        try:
            self._producer_thread.start()
        except BaseException:
            with self._state.condition:
                self._state.close_requested = True
                self._state.consumer_closed = True
                self._state.close_completed = True
                self._state.consumer_close_event.set()
            try:
                _close_source_iter_if_supported(source_iter)
            except BaseException:
                logger.warning(
                    "Ignoring source iterator close error after prefetch "
                    "producer thread failed to start.",
                    exc_info=True,
                )
            raise

    def __iter__(self) -> _ThreadedPrefetchIterator:
        return self

    def __next__(self) -> Any:
        self._release_deferred_consumer_credit()
        self._raise_if_producer_failed()
        if self._shuffle:
            return self._next_shuffled()
        return self._next_fifo()

    def _next_fifo(self) -> Any:
        """Yield from the legacy fixed-window FIFO path."""
        state = self._state
        while not self._ready_queue:
            error: BaseException | None = None
            should_stop = False
            with state.condition:
                while (
                    len(state.incoming_queue) < state.consumer_wakeup_size
                    and not state.producer_done
                    and state.producer_error is None
                    and not state.consumer_closed
                ):
                    state.condition.wait()

                error = self._take_producer_error_and_close_locked()
                if error is None:
                    if state.consumer_closed:
                        should_stop = True
                    elif state.incoming_queue:
                        queue_was_full = (
                            len(state.incoming_queue) == state.buffer_capacity
                        )
                        self._ready_queue.extend(state.incoming_queue)
                        state.incoming_queue.clear()
                        if queue_was_full:
                            state.condition.notify()
                    else:
                        should_stop = True

            if error is not None:
                self._clear_consumer_buffers()
                raise error
            if should_stop:
                self.close()
                raise StopIteration

        item = self._ready_queue.popleft()
        self._raise_if_producer_failed()
        return item

    def _next_shuffled(self) -> Any:
        """Yield one item from a consumer-owned random-partition chunk."""
        while not self._ready_queue:
            if len(self._shuffle_reservoir) < self._prefetch_size:
                incoming_chunk = self._take_next_incoming_chunk(
                    self._state.consumer_wakeup_size
                )
                if incoming_chunk:
                    reservoir_missing = self._prefetch_size - len(
                        self._shuffle_reservoir
                    )
                    fill_size = min(reservoir_missing, len(incoming_chunk))
                    self._shuffle_reservoir.extend(incoming_chunk[:fill_size])
                    self._shuffle_pending_incoming.extend(
                        incoming_chunk[fill_size:]
                    )
                    continue

                if self._shuffle_reservoir:
                    self._queue_shuffled_reservoir_tail()
                    continue

                self.close()
                raise StopIteration

            needed = self._state.consumer_wakeup_size - len(
                self._shuffle_pending_incoming
            )
            incoming_chunk = self._take_next_incoming_chunk(
                needed,
                hold_consumer_credit=(
                    self._shuffle_uses_shared_headroom_credit
                ),
            )
            if incoming_chunk:
                self._shuffle_pending_incoming.extend(incoming_chunk)
                if (
                    len(self._shuffle_pending_incoming)
                    == self._state.consumer_wakeup_size
                ):
                    self._partition_shuffle_chunk(
                        self._shuffle_pending_incoming
                    )
                    self._shuffle_pending_incoming.clear()
                continue

            # A live source only commits full T-sized partitions. Normal EOF
            # is the sole case that can commit the remaining partial chunk.
            if self._shuffle_pending_incoming:
                self._partition_shuffle_chunk(self._shuffle_pending_incoming)
                self._shuffle_pending_incoming.clear()
                continue

            if self._shuffle_reservoir:
                self._queue_shuffled_reservoir_tail()
                continue

            self.close()
            raise StopIteration

        ret = self._ready_queue.popleft()
        if self._ready_queue_consumer_credits:
            self._ready_queue_consumer_credits -= 1
            # Returning from this frame is the handoff boundary. Release the
            # credit only when the caller requests another item; releasing it
            # here would let the producer refill while ``ret`` is still held
            # by this iterator frame and temporarily exceed K + floor(K / 2).
            self._deferred_consumer_credit += 1
        self._raise_if_producer_failed()
        return ret

    def _take_next_incoming_chunk(
        self,
        requested_size: int,
        *,
        hold_consumer_credit: bool = False,
    ) -> list[Any]:
        """Take one FIFO chunk, or return empty on normal exhaustion."""
        state = self._state
        error: BaseException | None = None
        consumer_closed = False
        with state.condition:
            while (
                not state.incoming_queue
                and not state.producer_done
                and state.producer_error is None
                and not state.consumer_closed
            ):
                state.condition.wait()

            error = self._take_producer_error_and_close_locked()
            if error is None and state.incoming_queue:
                take_size = min(
                    requested_size,
                    len(state.incoming_queue),
                )
                incoming_chunk = [
                    state.incoming_queue.popleft() for _ in range(take_size)
                ]
                if hold_consumer_credit:
                    state.consumer_reserved_size += take_size
                # Normal chunks use a distinct consumer-active T credit, so
                # producer-side capacity can be reused immediately. The small
                # scalar fallback instead keeps this shared credit reserved.
                state.condition.notify()
                return incoming_chunk
            consumer_closed = state.consumer_closed

        if error is not None:
            self._clear_consumer_buffers()
            raise error
        if consumer_closed:
            self._clear_consumer_buffers()
        return []

    def _release_consumer_credit(self, released_size: int) -> None:
        """Release scalar credit after outputs leave the iterator."""
        state = self._state
        with state.condition:
            if state.consumer_reserved_size < released_size:
                raise RuntimeError(
                    "Cannot release more shuffle consumer credit than is "
                    "currently reserved."
                )
            state.consumer_reserved_size -= released_size
            state.condition.notify()

    def _release_deferred_consumer_credit(self) -> None:
        """Release credit retained across the previous caller handoff."""
        if not self._deferred_consumer_credit:
            return
        released_size = self._deferred_consumer_credit
        self._deferred_consumer_credit = 0
        self._release_consumer_credit(released_size)

    def _partition_shuffle_chunk(self, incoming_chunk: list[Any]) -> None:
        """Randomly split K+t candidates into t outputs and K survivors."""
        if len(self._shuffle_reservoir) != self._prefetch_size:
            raise RuntimeError(
                "Shuffle reservoir must be full before chunk partitioning."
            )
        output_size = len(incoming_chunk)
        pool = self._shuffle_reservoir + incoming_chunk
        indices = self._draw_random_permutation(len(pool))
        self._ready_queue.extend(
            pool[index] for index in indices[:output_size]
        )
        if self._shuffle_uses_shared_headroom_credit:
            self._ready_queue_consumer_credits += output_size
        self._shuffle_reservoir = [
            pool[index] for index in indices[output_size:]
        ]

    def _queue_shuffled_reservoir_tail(self) -> None:
        """Move the normally exhausted reservoir into random output order."""
        indices = self._draw_random_permutation(len(self._shuffle_reservoir))
        self._ready_queue.extend(
            self._shuffle_reservoir[index] for index in indices
        )
        self._shuffle_reservoir.clear()

    def _draw_random_permutation(self, size: int) -> list[int]:
        """Draw one no-replacement permutation for a committed chunk."""
        generator = self._generator
        if isinstance(generator, np.random.Generator):
            return generator.permutation(size).tolist()
        if isinstance(generator, torch.Generator):
            return torch.randperm(size, generator=generator).tolist()
        raise RuntimeError("Shuffle generator was not initialized.")

    def close(
        self,
        *,
        raise_producer_errors: bool = True,
        timeout: float | None = None,
    ) -> None:
        state = self._state
        with state.condition:
            self._mark_consumer_closed_locked()
            pending_error = self._take_producer_error_locked()
            producer_error_handled = state.producer_error_handled
            close_completed = state.close_completed
        self._clear_consumer_buffers()
        if close_completed:
            return

        if pending_error is not None and raise_producer_errors:
            if not self._producer_thread.is_alive():
                with state.condition:
                    state.close_completed = True
            raise pending_error

        # A primary producer error has already been raised by __next__ or this
        # close call. Do not let a producer-owned source close delay that
        # failure through the extended close wait.
        if producer_error_handled and raise_producer_errors:
            if not self._producer_thread.is_alive():
                with state.condition:
                    state.close_completed = True
            return

        if pending_error is not None:
            self._log_producer_error(pending_error)

        if not raise_producer_errors:
            join_timeout = (
                self._gc_close_timeout if timeout is None else timeout
            )
            self._producer_thread.join(timeout=max(0.0, join_timeout))
            if self._producer_thread.is_alive():
                logger.warning(
                    "Prefetch producer thread did not exit during "
                    "best-effort GC close."
                )
                return
            self._log_unhandled_producer_error()
            with state.condition:
                state.close_completed = True
            return

        close_timeout = (
            self._hard_close_timeout if timeout is None else timeout
        )
        if close_timeout is None:
            close_timeout = _get_prefetch_close_hard_timeout_sec()
        close_timeout = max(0.0, close_timeout)
        soft_timeout = min(_PREFETCH_CLOSE_JOIN_TIMEOUT_SEC, close_timeout)
        close_started_at = time.monotonic()

        with state.condition:
            state.condition.wait_for(
                lambda: (
                    state.producer_error is not None or state.producer_done
                ),
                timeout=soft_timeout,
            )
            pending_error = self._take_producer_error_locked()
            producer_done = state.producer_done

        if pending_error is not None:
            raise pending_error

        used_extended_wait = not producer_done and close_timeout > soft_timeout
        if used_extended_wait:
            logger.info(
                "Prefetch producer thread is still exiting after %.1fs; "
                "continuing the soft close wait for up to %.1fs. This alone "
                "does not indicate a resource leak.",
                soft_timeout,
                close_timeout,
            )
            remaining_timeout = max(
                0.0,
                close_timeout - (time.monotonic() - close_started_at),
            )
            with state.condition:
                state.condition.wait_for(
                    lambda: (
                        state.producer_error is not None or state.producer_done
                    ),
                    timeout=remaining_timeout,
                )
                pending_error = self._take_producer_error_locked()
                producer_done = state.producer_done

            if pending_error is not None:
                raise pending_error

        if not producer_done:
            logger.warning(
                "Prefetch producer thread did not exit within %.1fs; close() "
                "is returning while the producer remains alive. The wrapped "
                "source must eventually return from its in-flight next() or "
                "close() call.",
                close_timeout,
            )
            return

        # producer_done is the producer's last state transition. A join is now
        # non-blocking in practice and releases the finished thread resources.
        self._producer_thread.join()
        with state.condition:
            state.close_completed = True
        if used_extended_wait:
            logger.info(
                "Prefetch producer thread exited and was joined after %.1fs "
                "during close().",
                time.monotonic() - close_started_at,
            )

    def _take_producer_error(self) -> BaseException | None:
        with self._state.condition:
            return self._take_producer_error_locked()

    def _raise_if_producer_failed(self) -> None:
        error: BaseException | None = None
        with self._state.condition:
            error = self._take_producer_error_and_close_locked()
        if error is None:
            return
        self._clear_consumer_buffers()
        raise error

    def _take_producer_error_and_close_locked(
        self,
    ) -> BaseException | None:
        error = self._take_producer_error_locked()
        if error is not None:
            self._mark_consumer_closed_locked()
        return error

    def _take_producer_error_locked(self) -> BaseException | None:
        if (
            self._state.producer_error is None
            or self._state.producer_error_handled
        ):
            return None
        error = self._state.producer_error
        self._state.producer_error = None
        self._state.producer_error_handled = True
        return error

    def _mark_consumer_closed_locked(self) -> None:
        state = self._state
        state.close_requested = True
        state.consumer_closed = True
        state.consumer_close_event.set()
        state.incoming_queue.clear()
        state.consumer_reserved_size = 0
        state.condition.notify_all()

    def _clear_consumer_buffers(self) -> None:
        self._ready_queue.clear()
        self._ready_queue_consumer_credits = 0
        self._deferred_consumer_credit = 0
        self._shuffle_reservoir.clear()
        self._shuffle_pending_incoming.clear()

    def _log_unhandled_producer_error(self) -> None:
        error = self._take_producer_error()
        if error is None:
            return
        self._log_producer_error(error)

    def _log_producer_error(self, error: BaseException) -> None:
        logger.warning(
            "Ignoring producer-side exception during best-effort prefetch "
            "iterator close.",
            exc_info=(type(error), error, error.__traceback__),
        )

    def __del__(self) -> None:
        try:
            self.close(
                raise_producer_errors=False,
                timeout=self._gc_close_timeout,
            )
        except BaseException:
            logger.warning(
                "Ignoring exception during prefetch iterator finalization.",
                exc_info=True,
            )


def _raise_close_errors(errors: list[BaseException]) -> None:
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    message = "; ".join(repr(error) for error in errors)
    raise RuntimeError(
        "Multiple iterator close errors occurred: " + message
    ) from errors[0]


def _log_close_errors(
    primary_exc: BaseException,
    errors: list[BaseException],
) -> None:
    for error in errors:
        logger.warning(
            "Suppressing iterator close error because another exception is "
            "already being propagated: %r",
            primary_exc,
            exc_info=(type(error), error, error.__traceback__),
        )


def _close_single_iterator(
    iterator: Any,
    _visited: set[int],
    *,
    shutdown_persistent_workers: bool = False,
) -> None:
    if isinstance(
        iterator,
        (
            GeneratorType,
            _SingleProcessDataLoaderIter,
            _MultiProcessingDataLoaderIter,
        ),
    ):
        _close_dataloader_iterator(
            iterator,
            _visited=_visited,
            shutdown_persistent_workers=shutdown_persistent_workers,
        )
        return

    close = getattr(iterator, "close", None)
    if callable(close):
        iterator_id = id(iterator)
        if iterator_id in _visited:
            return
        _visited.add(iterator_id)
        close()


def _iter_dataloader_owners(dataloader: Any) -> Iterator[Any]:
    seen: set[int] = set()
    stack = [dataloader]
    while stack:
        owner = stack.pop()
        owner_id = id(owner)
        if owner_id in seen:
            continue
        seen.add(owner_id)
        yield owner

        for attr_name in (
            "base_dataloader",
            "dataloader",
            "_dataloader",
            "data_loader",
        ):
            child = getattr(owner, attr_name, None)
            if child is not None:
                stack.append(child)


def _end_prepared_dataloader_wrappers(owners: Iterable[Any]) -> None:
    for owner in owners:
        end = getattr(owner, "end", None)
        if callable(end):
            end()


def _clear_persistent_dataloader_owner_iterators(
    owners: Iterable[Any],
) -> None:
    for owner in owners:
        if isinstance(owner, TorchDataLoader) and hasattr(owner, "_iterator"):
            owner._iterator = None


def _close_dataloader_iterator(
    dataloader_iter: (
        GeneratorType
        | _SingleProcessDataLoaderIter
        | _MultiProcessingDataLoaderIter
    ),
    _visited: set[int] | None = None,
    *,
    shutdown_persistent_workers: bool = False,
) -> None:
    """Close a dataloader iterator and the nested iterator layers it owns.

    This helper only tears down resources owned by the active iterator stack.
    Prepared-wrapper lifecycle state such as `accelerate`'s
    `DataLoaderStateMixin` must be ended separately by the owner that
    prepared the dataloader.
    """

    if _visited is None:
        _visited = set()

    iterator_id = id(dataloader_iter)
    if iterator_id in _visited:
        return
    _visited.add(iterator_id)

    if isinstance(dataloader_iter, GeneratorType):
        generator_locals = inspect.getgeneratorlocals(dataloader_iter)
        errors: list[BaseException] = []
        for nested_iter_name in ("dataloader_iter", "main_iterator"):
            nested_dataloader_iter = generator_locals.get(nested_iter_name)
            if isinstance(
                nested_dataloader_iter,
                (
                    GeneratorType,
                    _SingleProcessDataLoaderIter,
                    _MultiProcessingDataLoaderIter,
                ),
            ):
                try:
                    _close_dataloader_iterator(
                        nested_dataloader_iter,
                        _visited,
                        shutdown_persistent_workers=(
                            shutdown_persistent_workers
                        ),
                    )
                except BaseException as exc:
                    errors.append(exc)
        try:
            dataloader_iter.close()
        except BaseException as exc:
            errors.append(exc)
        _raise_close_errors(errors)
        return

    if isinstance(dataloader_iter, _SingleProcessDataLoaderIter):
        if not isinstance(
            dataloader_iter._dataset_fetcher, _IterableDatasetFetcher
        ):
            return
        dataset_iter = dataloader_iter._dataset_fetcher.dataset_iter
        _close_single_iterator(
            dataset_iter,
            _visited,
            shutdown_persistent_workers=shutdown_persistent_workers,
        )
        return

    if isinstance(dataloader_iter, _MultiProcessingDataLoaderIter) and (
        not dataloader_iter._persistent_workers or shutdown_persistent_workers
    ):
        dataloader_iter._shutdown_workers()
