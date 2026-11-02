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

import copy
import random
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.data_loader import (
    DataLoaderDispatcher,
    DataLoaderShard,
    IterableDatasetShard,
    SeedableRandomSampler,
    skip_first_batches,
)
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    RandomSampler,
)

from robo_orchard_lab.dataset.robot._prefetch import (
    DataloaderCloseReason,
    _close_dataloader_owner_resources,
    close_dataloader_resources,
)

__all__ = [
    "PreparedDataLoaderIteration",
    "bind_prepared_data_loader_sampling",
    "configure_data_loader_for_accelerate",
    "iterate_prepared_loader",
    "prepare_data_loader",
]


class _PreparedRecovery(TypedDict):
    """Adapter-only state for a prepared owner; never a checkpoint payload."""

    sampler_state: torch.Tensor | None
    replay: int
    resume_pending: bool
    training_rng: tuple | None
    """Temporary process RNG for first-fetch recovery, never serialized."""


def _configure_prepared_loader_recovery(accelerator, prepared, source) -> None:
    """Bind recovery to existing loader owners after the combined prepare.

    Lab's runtime owns its processed-position snapshot. Only unsupported
    prepared bases use the adapter's output-replay state; no batch or reader
    is retained in either checkpoint payload.
    """
    lab = _attach_prepared_lab_loader(prepared, source)
    bind_prepared_data_loader_sampling(accelerator, prepared)
    prepared._lab_recovery = _PreparedRecovery(
        sampler_state=None,
        replay=0,
        resume_pending=False,
        training_rng=None,
    )
    if lab:
        _commit_prepared_loader_position(prepared, unstarted=True)


def _capture_prepared_loader_recovery(
    data_loader,
    *,
    epoch: int | None = None,
    processed_batches: int | None = None,
) -> dict:
    """Capture data recovery, checking any supplied progress expectations.

    A Lab snapshot must match the expected epoch and processed batch count
    when supplied. Ordinary loaders retain their sampler-origin fallback;
    requesting zero processed batches can capture an unstarted sampler.
    Missing Lab snapshots retain the existing empty-state behavior. Sampling
    seed metadata is read from its current owner at capture time.

    Raises:
        RuntimeError: A present Lab snapshot disagrees with expected progress.
    """
    recovery = data_loader._lab_recovery
    if hasattr(
        getattr(data_loader, "base_dataloader", data_loader), "_resume_runtime"
    ):
        state = copy.deepcopy(
            data_loader.base_dataloader._resume_runtime._prepared_checkpoint
        )
        if state is not None and (
            (epoch is not None and state.get("epoch") != epoch)
            or (
                processed_batches is not None
                and state.get("delivered_batches") != processed_batches
            )
        ):
            raise RuntimeError("Checkpoint data and progress do not match.")
        return {"loader_state": state}
    sampler_state = recovery["sampler_state"]
    if sampler_state is None and processed_batches == 0:
        generator = _private_random_sampler_generator(data_loader)
        if generator is not None:
            sampler_state = generator.get_state()
    return {
        "loader_state": None,
        "sampler_state": sampler_state,
        "seed": _identified_data_seed(data_loader),
    }


def _load_prepared_loader_recovery(
    accelerator, data_loader, rank_states, *, epoch, processed_batches
) -> None:
    """Stage per-rank data recovery without starting an iterator or workers."""
    data_loader._lab_recovery["training_rng"] = (
        (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        )
        if processed_batches > 0
        else None
    )
    if rank_states and len(rank_states) != accelerator.num_processes:
        raise RuntimeError("Checkpoint rank topology does not match.")
    if rank_states:
        saved_state = rank_states[accelerator.process_index]["loader_state"]
        if saved_state is not None and (
            saved_state.get("epoch") != epoch
            or saved_state.get("delivered_batches") != processed_batches
        ):
            rank_states = []
    if not rank_states:
        warnings.warn(
            "Checkpoint has no current DataLoader recovery state; resuming "
            "by replaying the epoch prefix. Data order and position may "
            "differ from the original run.",
            UserWarning,
        )
        base = getattr(data_loader, "base_dataloader", data_loader)
        if hasattr(base, "_resume_runtime"):
            state = base.state_dict()
            state["epoch"] = epoch
            state["delivered_batches"] = processed_batches
            state["recovery"] = {
                "strategy": "batch_replay",
                "accuracy": "unknown",
                "reason": "checkpoint has no DataLoader snapshot",
            }
            base.load_state_dict(state)
            _commit_prepared_loader_position(data_loader, unstarted=True)
        else:
            recovery = data_loader._lab_recovery
            generator = _private_random_sampler_generator(data_loader)
            recovery["sampler_state"] = (
                generator.get_state() if generator is not None else None
            )
            recovery["resume_pending"] = True
            recovery["replay"] = processed_batches
        return
    saved = rank_states[accelerator.process_index]
    recovery = data_loader._lab_recovery
    state = saved["loader_state"]
    if state is not None:
        if not hasattr(
            getattr(data_loader, "base_dataloader", data_loader),
            "_resume_runtime",
        ):
            raise RuntimeError(
                "Checkpoint requires a compatible Lab DataLoader."
            )
        data_loader.base_dataloader.load_state_dict(state)
        _commit_prepared_loader_position(data_loader, unstarted=True)
    else:
        _restore_identified_data_seed(data_loader, saved.get("seed"))
        sampler_state = saved.get("sampler_state")
        generator = _private_random_sampler_generator(data_loader)
        if (generator is None) != (sampler_state is None):
            raise RuntimeError(
                "Checkpoint private sampler state does not match the "
                "prepared DataLoader."
            )
        recovery["sampler_state"] = sampler_state
        recovery["replay"] = processed_batches
        recovery["resume_pending"] = True


def _commit_prepared_loader_position(data_loader, *, unstarted=False) -> None:
    """Latch prepared delivery only after a complete optimizer window.

    Accelerate's prepared snapshot already excludes its one-batch lookahead.
    Keeping the latch on the data owner also excludes a fetched peer-EOF batch
    which the trainer never processed. Direct DataLoader.state_dict semantics
    remain delivery-based and do not depend on this trainer commit boundary.
    """
    if not hasattr(
        getattr(data_loader, "base_dataloader", data_loader), "_resume_runtime"
    ):
        return
    base = data_loader.base_dataloader
    state = copy.deepcopy(
        base.state_dict() if unstarted else data_loader.dl_state_dict
    )
    state.pop("_iterator_finished", None)
    base._resume_runtime._prepared_checkpoint = state
    if unstarted:
        data_loader.dl_state_dict = copy.deepcopy(state)


def _set_prepared_loader_epoch(data_loader, epoch) -> None:
    """Advance inactive data owners before an epoch checkpoint is saved."""
    recovery = data_loader._lab_recovery
    if hasattr(
        getattr(data_loader, "base_dataloader", data_loader), "_resume_runtime"
    ):
        data_loader.base_dataloader._notify_epoch(epoch)
        data_loader.iteration = epoch
        _commit_prepared_loader_position(data_loader, unstarted=True)
    else:
        generator = _private_random_sampler_generator(data_loader)
        recovery["sampler_state"] = (
            generator.get_state() if generator is not None else None
        )


@dataclass(slots=True)
class PreparedDataLoaderIteration(Iterator[Any]):
    """Iterate prepared training data with coordinated stopping and cleanup.

    Obtain this object from :func:`iterate_prepared_loader`. Iterate over it
    normally, call :meth:`commit` after a completed optimizer window, and call
    :meth:`should_stop` after each processed batch. The caller owns training
    computation and decides when training should end. All ranks must use the
    same iteration and stop-check sequence.

    The enclosing context owns cleanup on exhaustion, early exit or error.
    Read failures notify ranks that reach the next stop check before the
    original error is raised; this cannot release peers blocked elsewhere.
    """

    _accelerator: Accelerator
    _data_loader: DataLoader
    _iterator: Iterator[Any] | None = None
    _init_error: BaseException | None = None
    _close_reason: DataloaderCloseReason = DataloaderCloseReason.EARLY_BREAK
    _finished: bool = False

    def __iter__(self) -> "PreparedDataLoaderIteration":
        return self

    def __next__(self) -> Any:
        if self._finished:
            raise StopIteration
        fetch_error = self._init_error
        self._init_error = None
        exhausted = False
        try:
            if fetch_error is not None:
                raise fetch_error
            if self._iterator is None:
                raise RuntimeError("Dataloader iterator was not created.")
            batch = next(self._iterator)
        except StopIteration:
            exhausted = True
        except BaseException as exc:
            fetch_error = exc

        recovery = self._data_loader._lab_recovery
        training_rng = recovery.get("training_rng")
        if fetch_error is None and training_rng is not None:
            python_state, numpy_state, torch_state, cuda_states = training_rng
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            recovery["training_rng"] = None

        if exhausted or fetch_error is not None:
            self._accelerator.set_trigger()
        stopped = self._accelerator.check_trigger()
        if fetch_error is not None:
            raise fetch_error
        if stopped or exhausted:
            self._close_reason = (
                DataloaderCloseReason.EPOCH_EXHAUSTED
                if exhausted
                else DataloaderCloseReason.COORDINATED_EPOCH_END
            )
            self._finished = True
            raise StopIteration
        return batch

    def commit(self) -> None:
        """Latch the processed Lab position after an optimizer window.

        Call before checkpoint hooks run, including windows whose parameter
        update was skipped by AMP. This makes the data position available to
        checkpointing; it does not save a checkpoint or record trainer steps.
        """
        _commit_prepared_loader_position(self._data_loader)

    def should_stop(self, *, local_stop: bool = False) -> bool:
        """Coordinate stopping after a processed batch on every rank.

        Args:
            local_stop: Whether this rank reached its training limit. If
                true, notify peers and close this epoch at the training
                limit; otherwise check for a peer's stop request.

        Returns:
            Whether the caller should break the epoch iteration.
        """
        if local_stop:
            self._close_reason = DataloaderCloseReason.MAX_STEP_END
            self._accelerator.set_trigger()
        stopped = self._accelerator.check_trigger()
        if stopped and not local_stop:
            self._close_reason = DataloaderCloseReason.COORDINATED_EPOCH_END
        self._finished = stopped or local_stop
        return self._finished


@contextmanager
def iterate_prepared_loader(
    accelerator: Accelerator, data_loader: DataLoader, *, epoch: int
) -> Iterator[PreparedDataLoaderIteration]:
    """Manage one epoch of prepared data for a custom training loop.

    Use a loader returned by :func:`prepare_data_loader`. This context binds
    the epoch, applies staged recovery, coordinates read failures and EOF,
    and closes the iterator on every exit. Source ordering, sharding and
    accumulation remain owned by the existing loader and Accelerate.
    Iterator-creation failures surface on the first read so every rank
    participates in the same pre-batch stop check.

    Args:
        accelerator: The accelerator that prepared the loader.
        data_loader: A loader prepared once through the Lab wrapper, or
            the trainer's combined preparation path.
        epoch: Epoch to bind before creating the iterator.

    Yields:
        PreparedDataLoaderIteration: Iterate, commit completed optimizer
        windows, and check stopping after each processed batch::

            with iterate_prepared_loader(
                accelerator, loader, epoch=0
            ) as batches:
                for batch in batches:
                    # Run training and commit completed optimizer windows.
                    if batches.should_stop(local_stop=reached_limit):
                        break

    Raises:
        BaseException: The original setup, reading or training error. Cleanup
            retains it as the primary error.
    """
    session = PreparedDataLoaderIteration(accelerator, data_loader)
    active_loader = data_loader
    primary_exc: BaseException | None = None
    try:
        try:
            recovery = data_loader._lab_recovery
            base = getattr(data_loader, "base_dataloader", data_loader)
            if hasattr(base, "_resume_runtime"):
                base._notify_epoch(epoch)
            else:
                generator = _private_random_sampler_generator(data_loader)
                if recovery["resume_pending"] and generator is not None:
                    state = recovery["sampler_state"]
                    if state is not None:
                        generator.set_state(state)
                recovery["sampler_state"] = (
                    generator.get_state() if generator is not None else None
                )
                recovery["resume_pending"] = False
            if hasattr(base, "_resume_runtime") and isinstance(
                data_loader, DataLoaderShard
            ):
                # HF's nested sampler branch omits the dataset notification.
                # Notify once; all other business epoch calls remain HF's.
                nested = getattr(base.batch_sampler, "batch_sampler", None)
                sampler = getattr(nested, "sampler", None)
                if hasattr(sampler, "set_epoch") and hasattr(
                    base.dataset, "set_epoch"
                ):
                    base.dataset.set_epoch(epoch)
            if hasattr(data_loader, "iteration"):
                # Prepared wrappers call set_epoch inside iter().
                data_loader.iteration = epoch
            elif epoch > 0 and hasattr(data_loader, "set_epoch"):
                data_loader.set_epoch(epoch)
            if recovery["replay"]:
                if hasattr(data_loader, "set_epoch"):
                    data_loader.set_epoch(epoch)
                active_loader = skip_first_batches(
                    data_loader, num_batches=recovery["replay"]
                )
                if hasattr(active_loader, "iteration"):
                    active_loader.iteration = epoch
            recovery["replay"] = 0

        except BaseException as exc:
            # A partially prepared wrapper is closed before notifying peers,
            # while the original loader remains the outer cleanup owner.
            if active_loader is not data_loader:
                try:
                    _close_dataloader_owner_resources(
                        active_loader, primary_exc=exc
                    )
                except BaseException as cleanup_exc:
                    exc = cleanup_exc
                active_loader = data_loader
            session._init_error = exc
        else:
            try:
                session._iterator = iter(active_loader)
            except BaseException as exc:
                session._init_error = exc
        yield session
    except BaseException as exc:
        primary_exc = exc
        raise
    finally:
        reason = (
            DataloaderCloseReason.EXCEPTION_ABORT
            if primary_exc is not None
            else session._close_reason
        )
        session._finished = True
        try:
            close_dataloader_resources(
                active_loader,
                session._iterator,
                reason=reason,
                primary_exc=primary_exc,
            )
        finally:
            if active_loader is not data_loader:
                _close_dataloader_owner_resources(
                    active_loader, primary_exc=sys.exc_info()[1]
                )


def _bind_managed_map_generators(
    data_loader: DataLoader, *, rank: int | None = None
) -> None:
    """Isolate absent standard map-loader RNGs before Accelerate prepares them.

    A missing worker generator uses an isolated run-seed/rank stream, while
    a missing RandomSampler generator uses seed 42. Explicit generators and
    aliases remain untouched; unknown loader subclasses retain their behavior.
    """
    from robo_orchard_lab.dataset.robot.dataset_ex import (
        DataLoader as LabDataLoader,
    )

    if type(data_loader) not in (DataLoader, LabDataLoader):
        return
    from robo_orchard_lab.dataset.robot._resume import (
        _bind_default_worker_generator,
        _bind_private_sampler_generator,
    )

    _bind_default_worker_generator(data_loader, rank)
    if isinstance(data_loader.dataset, IterableDataset):
        return
    sampler = getattr(
        data_loader.batch_sampler, "sampler", data_loader.sampler
    )
    _bind_private_sampler_generator(sampler)


def _prepared_sampler_owner(
    data_loader: DataLoader, *, fallback: Any = None
) -> Any:
    """Resolve the sampler behind Accelerate batch-sampler wrappers.

    Unwrap batch samplers before selecting their sampler. This differs from
    the native runtime's first generator-owning sampler lookup: a custom
    batch sampler may carry its own unrelated generator. ``fallback`` is
    used only when the unwrapped node has no sampler attribute.
    """
    node = getattr(data_loader, "batch_sampler", None)
    while hasattr(node, "batch_sampler"):
        node = node.batch_sampler
    return getattr(node, "sampler", fallback)


def _private_random_sampler_generator(data_loader):
    """Find only the standard sampler generator created by the framework."""
    base = getattr(data_loader, "base_dataloader", data_loader)
    sampler = _prepared_sampler_owner(base)
    if type(sampler) is RandomSampler:
        generator = sampler.generator
        if (
            generator is not None
            and getattr(sampler, "_lab_default_generator", None) is generator
        ):
            return generator
    return None


def _identified_data_seed(data_loader: DataLoader) -> int | None:
    """Return only a known sampling owner's actual integer seed."""
    from robo_orchard_lab.dataset.robot.dataset_ex import (
        DataLoader as LabDataLoader,
    )

    base = getattr(data_loader, "base_dataloader", data_loader)
    if isinstance(base, LabDataLoader):
        return base._get_sampling_seed()
    dataset = data_loader.dataset
    if isinstance(dataset, IterableDatasetShard):
        dataset = dataset.dataset
    seed = getattr(dataset, "_sampling_seed", None)
    if type(seed) is int:
        return seed
    sampler = _prepared_sampler_owner(data_loader)
    if isinstance(sampler, SeedableRandomSampler):
        return sampler.initial_seed
    if (
        type(sampler) is RandomSampler
        and sampler.generator is not None
        and sampler.generator
        is getattr(sampler, "_lab_default_generator", None)
    ):
        return sampler.generator.initial_seed()
    return None


def _restore_identified_data_seed(
    data_loader: DataLoader, seed: int | None
) -> None:
    """Restore a recorded seed on a known fallback sampling owner."""
    if seed is None:
        return
    if type(seed) is not int:
        raise RuntimeError("Checkpoint sampling seed must be an integer.")
    from robo_orchard_lab.dataset.robot.dataset_ex import DictIterableDataset

    dataset = data_loader.dataset
    if isinstance(dataset, IterableDatasetShard):
        dataset = dataset.dataset
    if isinstance(dataset, DictIterableDataset):
        if dataset._sampling_seed is None:
            raise RuntimeError("Checkpoint sampling seed has no seeded owner.")
        dataset._sampling_seed = seed
        return
    sampler = _prepared_sampler_owner(data_loader)
    if isinstance(sampler, SeedableRandomSampler):
        sampler.initial_seed = seed


def _retain_prepared_generator_owners(
    prepared: DataLoader, source: DataLoader
) -> None:
    """Retain source-owned worker and private sampler streams after prepare."""
    base = getattr(prepared, "base_dataloader", None)
    if not isinstance(base, DataLoader):
        return
    if (
        isinstance(prepared, DataLoaderDispatcher)
        and type(base) is DataLoader
        and base.generator is None
        and source.generator is not None
    ):
        base.generator = source.generator
    source_sampler = getattr(source.batch_sampler, "sampler", source.sampler)
    sampler = _prepared_sampler_owner(base, fallback=base.sampler)
    owned = getattr(source_sampler, "_lab_default_generator", None)
    if owned is not None and getattr(sampler, "generator", None) is owned:
        sampler._lab_default_generator = owned


def _attach_prepared_lab_loader(
    prepared_loader: DataLoader,
    source_loader: DataLoader,
) -> bool:
    """Attach the Lab state owner to a supported, unstarted prepared loader.

    The existing Accelerate wrapper remains registered and owns sharding,
    lookahead, device placement and gradient state. Unsupported wrappers keep
    their original base and use the trainer's output-level replay path.
    """
    from robo_orchard_lab.dataset.robot.dataset_ex import (
        DataLoader as LabDataLoader,
        DictIterableDataset,
    )

    _retain_prepared_generator_owners(prepared_loader, source_loader)
    if not isinstance(source_loader, LabDataLoader) or not isinstance(
        source_loader.dataset, DictIterableDataset
    ):
        return False
    if not isinstance(prepared_loader, DataLoaderShard):
        return False
    if prepared_loader.use_stateful_dataloader:
        return False
    base = prepared_loader.base_dataloader
    if isinstance(base.dataset, IterableDatasetShard):
        return False
    prepared_loader.base_dataloader = LabDataLoader._from_prepared_base(
        base, source_loader=source_loader
    )
    prepared_loader.dl_state_dict = (
        prepared_loader.base_dataloader.state_dict()
    )
    return True


_DISPATCHER_SLOW_PATH_WARNING = (
    "The prepared dataloader fell back to DataLoaderDispatcher. "
    "This dispatcher-based path is very inefficient for "
    "IterableDatasetMixin in multi-process training. To avoid this "
    "slow path, keep `dispatch_batches=False` and avoid "
    "`put_on_device=True` in the accelerate prepare path for "
    "iterable datasets."
)

_ITERABLE_SHARD_SLOW_PATH_WARNING = (
    "The prepared dataloader fell back to accelerate "
    "IterableDatasetShard. This shard wrapper is very inefficient "
    "for IterableDatasetMixin. To avoid this slow path, ensure the "
    "dataset exposes an `n_shards` value larger than the current "
    "number of processes so accelerate can use dataset-native "
    "sharding."
)


def _warn_and_reset_dataloader_config(
    dataloader_config: object,
    field_name: str,
    expected_value: object,
    warning_message: str,
) -> None:
    field_value = getattr(dataloader_config, field_name)
    if field_value == expected_value:
        return

    warnings.warn(
        warning_message
        + f" Found {field_name}={field_value!r}; "
        + f"reset it to {expected_value!r}.",
        UserWarning,
    )
    setattr(dataloader_config, field_name, expected_value)


def _warn_if_prepare_falls_back_to_slow_path(
    data_loader: object,
) -> None:
    if isinstance(data_loader, DataLoaderDispatcher):
        warnings.warn(_DISPATCHER_SLOW_PATH_WARNING, UserWarning)
        return

    if not isinstance(data_loader, DataLoaderShard):
        return

    base_dataloader = getattr(data_loader, "base_dataloader", None)
    if base_dataloader is None:
        return

    if isinstance(
        getattr(base_dataloader, "dataset", None),
        IterableDatasetShard,
    ):
        warnings.warn(_ITERABLE_SHARD_SLOW_PATH_WARNING, UserWarning)


def bind_prepared_data_loader_sampling(
    accelerator: Accelerator, data_loader: DataLoader
) -> bool:
    """Bind managed sampling to the final prepared loader before iteration.

    Native shards use global-rank-specific streams. ``IterableDatasetShard``
    reads a common stream on every rank before assigning rows, so its private
    sampling seed omits rank. Dispatcher and other datasets retain their own
    behavior. Only explicitly seeded ``DictIterableDataset`` owners are bound;
    RNG synchronization and user-owned Generators retain their behavior.
    The configure helper binds a separate default worker stream to global rank.
    Call after a combined ``accelerator.prepare`` and before iteration.
    Repeated calls are safe. Accelerate's dispatcher discards the source
    loader generator; this function alone cannot restore it without the source.
    Use the Lab prepare wrapper or trainer to retain that generator.

    Args:
        accelerator: Existing runtime supplying the final global process rank.
        data_loader: Final loader returned by Accelerate prepare.

    Returns:
        bool: Whether this loader uses managed native or fallback sampling.

    Raises:
        ValueError: If an owner already bound to a fallback is reused directly
            for native sampling. A new RoboOrchard DataLoader clones it into
            an independent native owner.
    """
    from robo_orchard_lab.dataset.robot.dataset_ex import DictIterableDataset

    dataset = getattr(data_loader, "dataset", None)
    if isinstance(data_loader, DataLoaderDispatcher):
        return False
    if isinstance(dataset, IterableDatasetShard):
        owner = dataset.dataset
        if not isinstance(owner, DictIterableDataset):
            return False
        owner = owner._bind_sampling(None)
        if owner is None:
            return False
        dataset.dataset = owner
        return True
    if not isinstance(dataset, DictIterableDataset):
        return False
    return dataset._bind_sampling(accelerator.process_index) is not None


def configure_data_loader_for_accelerate(
    accelerator: Accelerator,
    data_loader: DataLoader,
) -> bool:
    """Normalize dataloader settings before calling `Accelerator.prepare`.

    Recommended usage:
        Use the trainer or the Lab ``prepare_data_loader`` wrapper to retain
        default worker RNG ownership, including with dispatcher loaders.
        The configure/prepare/bind sequence below only binds sampling; a raw
        Accelerate dispatcher discards the source worker generator.

    Pattern:
        `configure_data_loader_for_accelerate(accelerator, dataloader)`
        `dataloader, model, optimizer, scheduler = accelerator.prepare(`
        `    dataloader, model, optimizer, scheduler`
        `)`
        `bind_prepared_data_loader_sampling(accelerator, dataloader)`

    Returns:
        bool: Whether the prepared dataloader should be checked for
            IterableDataset slow-path warnings.
    """

    from robo_orchard_lab.dataset.robot.dataset_ex import (
        DataLoader as LabDataLoader,
        IterableDatasetMixin,
    )

    if type(data_loader) in (DataLoader, LabDataLoader):
        _bind_managed_map_generators(
            data_loader, rank=accelerator.process_index
        )
    dataset = data_loader.dataset
    if not (
        isinstance(dataset, IterableDatasetMixin)
        and accelerator.num_processes > 1
    ):
        return False

    if dataset.shard_kwargs.shard_strategy is None:
        warnings.warn(
            "The dataset is an iterable dataset and the shard strategy "
            "is not set for multi-process training. This may lead to "
            "unbalanced data loading and potential system hang. "
            "Reset the shard strategy to 'pad_last'. ",
            UserWarning,
        )
        dataset.shard_kwargs.shard_strategy = "pad_last"
    _warn_and_reset_dataloader_config(
        dataloader_config=accelerator.dataloader_config,
        field_name="dispatch_batches",
        expected_value=False,
        warning_message=(
            "Using IterableDatasetMixin with multi-process training and "
            "dispatch_batches != False will lead to inefficient data "
            "loading."
        ),
    )
    _warn_and_reset_dataloader_config(
        dataloader_config=accelerator.dataloader_config,
        field_name="even_batches",
        expected_value=False,
        warning_message=(
            "even_batches in accelerator dataloader config is not "
            "supported for IterableDataset. Set drop_last in the "
            "dataloader instead if you need to drop the last incomplete "
            "batch."
        ),
    )
    _warn_and_reset_dataloader_config(
        dataloader_config=accelerator.dataloader_config,
        field_name="split_batches",
        expected_value=False,
        warning_message=(
            "Using IterableDatasetMixin with multi-process training and "
            "split_batches != False will lead to inefficient data "
            "loading."
        ),
    )
    return True


def prepare_data_loader(
    accelerator: Accelerator,
    data_loader: DataLoader,
    **kwargs,
) -> DataLoader:
    """Prepare the dataloader using accelerator.prepare_data_loader.

    This function is a wrapper around accelerator.prepare_data_loader to handle
    the case when the dataset is `IterableDatasetMixin`.

    Warning:
        This function already calls `accelerator.prepare_data_loader()`.
        Do not pass the returned dataloader to `accelerator.prepare(...)`
        again, or it may be wrapped and sharded a second time.

        For combined preparation with model, optimizer and scheduler, use the
        trainer. The raw configure/prepare/bind sequence binds sampling but
        cannot retain a source worker generator discarded by a dispatcher.

    Recommended usage:
        Use this helper only when the dataloader must be prepared on its own.

    Pattern:
        `dataloader = prepare_data_loader(accelerator, dataloader)`

    Avoid:
        `dataloader = prepare_data_loader(accelerator, dataloader)`
        `dataloader, model, optimizer, scheduler = accelerator.prepare(`
        `    dataloader, model, optimizer, scheduler`
        `)`

    """

    should_check_slow_path = configure_data_loader_for_accelerate(
        accelerator=accelerator,
        data_loader=data_loader,
    )
    ret = accelerator.prepare_data_loader(data_loader, **kwargs)
    _configure_prepared_loader_recovery(accelerator, ret, data_loader)

    if should_check_slow_path:
        _warn_if_prepare_falls_back_to_slow_path(ret)

    return ret
