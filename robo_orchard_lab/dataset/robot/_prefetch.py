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
import copy
import inspect
import os
import threading
import warnings
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

    Args:
        source_iter (Iterator[Any]): Iterator to consume from the producer
            thread.
        prefetch_size (int): Number of items to buffer before each consumer
            handoff. A value of one returns ``source_iter`` unchanged.
        shuffle (bool): Whether to shuffle each prefetch window before
            yielding it.
        generator (torch.Generator | np.random.Generator | None): Random
            generator used when ``shuffle`` is enabled. When ``None`` and
            shuffle is enabled, a local torch generator is created.
        hard_close_timeout (float | None, optional): Maximum explicit close
            wait in seconds. Default is ``None``, which reads the hidden
            environment override and otherwise uses the module default.
        gc_close_timeout (float, optional): Best-effort wait in seconds used
            by finalization paths. Default is ``0.0``.

    Returns:
        Iterator[Any]: ``source_iter`` when no producer thread is needed, or
            a close-aware prefetch iterator otherwise.

    Raises:
        ValueError: If ``prefetch_size`` is not positive.
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


def _shuffle_prefetch_buffer(
    queue: list[Any],
    generator: torch.Generator | np.random.Generator,
) -> list[Any]:
    if isinstance(generator, np.random.Generator):
        ret = copy.copy(queue)
        generator.shuffle(ret)
        return ret
    if isinstance(generator, torch.Generator):
        indices = torch.randperm(len(queue), generator=generator).tolist()
        return [queue[i] for i in indices]
    raise ValueError(
        "Generator must be either a torch.Generator or a "
        "numpy.random.Generator."
    )


@dataclass(slots=True)
class _ThreadedPrefetchState:
    source_iter: Iterator[Any]
    prefetch_size: int
    shuffle: bool
    generator: torch.Generator | np.random.Generator | None
    queue: list[Any] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)
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
    try:
        while True:
            with state.condition:
                while (
                    len(state.queue) >= state.prefetch_size
                    and not state.consumer_closed
                ):
                    state.condition.wait()
                if state.consumer_closed:
                    return

            try:
                item = next(state.source_iter)
            except StopIteration:
                return

            with state.condition:
                if state.consumer_closed:
                    return
                state.queue.append(item)
                state.condition.notify_all()
    except BaseException as exc:
        with state.condition:
            state.producer_error = exc
    finally:
        try:
            _close_source_iter_if_supported(state.source_iter)
        except BaseException as exc:
            with state.condition:
                if state.producer_error is None:
                    state.producer_error = exc
        finally:
            with state.condition:
                state.producer_done = True
                state.condition.notify_all()


class _ThreadedPrefetchIterator:
    """Iterator that overlaps source iteration with foreground consumption.

    This class is the resource-owning implementation behind
    ``create_prefetch_iterator``. It starts one producer thread during
    construction, buffers up to ``prefetch_size`` items, and optionally
    shuffles each ready window before the foreground iterator yields it.

    The iterator owns the producer thread and the wrapped ``source_iter``.
    Normal exhaustion closes the producer path before raising
    ``StopIteration``. Callers that stop early must call ``close()`` directly
    or close it through ``close_iterators_best_effort``; ``__del__`` provides
    only best-effort finalization and must not be treated as the primary
    lifecycle path. Producer-side exceptions are re-raised on the consumer
    thread from ``__next__`` or ``close()`` so failures are not silently lost.

    Args:
        source_iter (Iterator[Any]): Iterator consumed by the producer thread.
        prefetch_size (int): Maximum number of pending items in the producer
            buffer.
        shuffle (bool): Whether each producer window is shuffled before
            foreground consumption.
        generator (torch.Generator | np.random.Generator | None): Random
            generator used when ``shuffle`` is enabled.
        hard_close_timeout (float | None, optional): Maximum explicit close
            wait in seconds. Default is ``None``, which defers to the module
            timeout policy.
        gc_close_timeout (float, optional): Best-effort finalizer close wait
            in seconds. Default is ``0.0``.
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
        self._state = _ThreadedPrefetchState(
            source_iter=source_iter,
            prefetch_size=prefetch_size,
            shuffle=shuffle,
            generator=generator,
        )
        self._ready_queue: list[Any] = []
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
        state = self._state
        error = self._take_producer_error()
        if error is not None:
            raise error

        while not self._ready_queue:
            should_stop = False
            with state.condition:
                if state.consumer_closed:
                    should_stop = True
                while (
                    not should_stop
                    and len(state.queue) < state.prefetch_size
                    and not state.producer_done
                    and state.producer_error is None
                    and not state.consumer_closed
                ):
                    state.condition.wait()

                error = self._take_producer_error_locked()
                if error is not None:
                    raise error

                if not should_stop and (
                    state.consumer_closed
                    or (len(state.queue) == 0 and state.producer_done)
                ):
                    should_stop = True

                if not should_stop:
                    self._ready_queue = state.queue
                    state.queue = []
                    state.condition.notify_all()

            if should_stop:
                self.close()
                raise StopIteration

            if state.shuffle:
                assert state.generator is not None
                self._ready_queue = _shuffle_prefetch_buffer(
                    self._ready_queue,
                    state.generator,
                )

        error = self._take_producer_error()
        if error is not None:
            raise error
        return self._ready_queue.pop(0)

    def close(
        self,
        *,
        raise_on_timeout: bool = True,
        timeout: float | None = None,
    ) -> None:
        state = self._state
        with state.condition:
            state.close_requested = True
            state.consumer_closed = True
            state.condition.notify_all()
            if state.close_completed:
                return

        if not raise_on_timeout:
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

        hard_timeout = self._hard_close_timeout if timeout is None else timeout
        if hard_timeout is None:
            hard_timeout = _get_prefetch_close_hard_timeout_sec()
        hard_timeout = max(0.0, hard_timeout)
        soft_timeout = min(_PREFETCH_CLOSE_JOIN_TIMEOUT_SEC, hard_timeout)
        self._producer_thread.join(timeout=soft_timeout)
        if self._producer_thread.is_alive():
            logger.warning(
                "Prefetch producer thread did not exit within %.1fs; "
                "waiting up to %.1fs before failing close().",
                soft_timeout,
                hard_timeout,
            )
            self._producer_thread.join(
                timeout=max(0.0, hard_timeout - soft_timeout)
            )
        if self._producer_thread.is_alive():
            raise RuntimeError(
                "Prefetch producer thread did not exit during close() "
                f"within {hard_timeout:.1f}s."
            )

        error = self._take_producer_error()
        with state.condition:
            state.close_completed = True
        if error is not None:
            raise error

    def _take_producer_error(self) -> BaseException | None:
        with self._state.condition:
            return self._take_producer_error_locked()

    def _take_producer_error_locked(self) -> BaseException | None:
        if (
            self._state.producer_error is None
            or self._state.producer_error_handled
        ):
            return None
        self._state.producer_error_handled = True
        return self._state.producer_error

    def _log_unhandled_producer_error(self) -> None:
        error = self._take_producer_error()
        if error is None:
            return
        logger.warning(
            "Ignoring producer-side exception during best-effort prefetch "
            "iterator close.",
            exc_info=(type(error), error, error.__traceback__),
        )

    def __del__(self) -> None:
        try:
            self.close(raise_on_timeout=False, timeout=self._gc_close_timeout)
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
