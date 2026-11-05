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

"""Private recovery primitives shared by Lab datasets and loaders."""

from __future__ import annotations
import copy
import hashlib
import inspect
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator

import torch
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    IterableDataset as TorchIterableDataset,
)

from robo_orchard_lab.dataset.robot._prefetch import (
    close_iterators_best_effort,
)
from robo_orchard_lab.dataset.sampler import DEFAULT_SAMPLING_SEED


def _bind_default_worker_generator(
    loader: TorchDataLoader, rank: int | None = None
) -> None:
    """Bind a private worker RNG without changing explicit generator owners.

    The run seed is captured on first binding. A rank may be corrected before
    that private stream advances, but an already-used stream cannot be rebound.
    """
    if rank is None:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 0
        )
    generator = getattr(loader, "_default_worker_generator", None)
    if generator is None:
        if loader.generator is not None:
            return
        generator = torch.Generator()
        loader.generator = generator
        loader._default_worker_generator = generator
        loader._worker_run_seed = torch.initial_seed()
        loader._worker_seed_rank = None
    elif loader.generator is not generator:
        return
    previous_rank = loader._worker_seed_rank
    if previous_rank == rank:
        return
    if previous_rank is not None:
        expected = torch.Generator().manual_seed(
            _worker_seed(loader._worker_run_seed, previous_rank)
        )
        if not torch.equal(generator.get_state(), expected.get_state()):
            raise RuntimeError("Cannot change rank of an advanced worker RNG.")
    generator.manual_seed(_worker_seed(loader._worker_run_seed, rank))
    loader._worker_seed_rank = rank


def _worker_seed(run_seed: int, rank: int) -> int:
    """Derive a rank-specific worker seed in its own RNG domain."""
    identity = f"robo-orchard-worker-v1\0{run_seed}\0{rank}".encode("ascii")
    return int.from_bytes(
        hashlib.blake2b(identity, digest_size=8).digest(), "big"
    ) & (2**63 - 1)


if TYPE_CHECKING:
    from robo_orchard_lab.dataset.robot.dataset_ex import DataLoader


_TORCH_DATALOADER_INIT_SIGNATURE = inspect.signature(TorchDataLoader.__init__)


_SKIPPED_ITEM = object()


@dataclass(slots=True)
class _IterationSkip:
    """Remaining native yields to advance without materializing samples."""

    remaining: int


def _validate_skip_n_items(skip_n_items: int) -> None:
    if (
        isinstance(skip_n_items, bool)
        or not isinstance(skip_n_items, int)
        or skip_n_items < 0
    ):
        raise ValueError("skip_n_items must be a non-negative integer.")


def _replay_skipped_outputs(
    iterator: Iterator[Any], count: int
) -> Iterator[Any]:
    """Replay and discard outputs when index-only skipping is unavailable."""

    primary_exc: BaseException | None = None
    try:
        for _ in range(count):
            try:
                next(iterator)
            except StopIteration:
                return
        yield from iterator
    except BaseException as exc:
        primary_exc = exc
        raise
    finally:
        close_iterators_best_effort([iterator], primary_exc=primary_exc)


@dataclass(slots=True)
class _SkipCollate:
    """Avoid invoking user collation while a native batch is skipped."""

    context: _IterationSkip
    collate_fn: Callable

    def __call__(self, batch: Any) -> Any:
        if self.context.remaining:
            return _SKIPPED_ITEM
        return self.collate_fn(batch)


@dataclass(slots=True)
class _DeliveryCollate:
    """Attach worker and input-count metadata inside the native loader."""

    collate_fn: Callable
    auto_collation: bool

    def __call__(self, batch: Any) -> tuple[Any, int, int]:
        worker = torch.utils.data.get_worker_info()
        count = len(batch) if self.auto_collation else 1
        return self.collate_fn(batch), worker.id if worker else 0, count


def _bind_private_sampler_generator(sampler: Any) -> None:
    """Initialize absent standard sampler streams and record their owner.

    SeedableRandomSampler otherwise creates its generator on first iteration.
    Eager initialization uses its own initial_seed. Explicit generators retain
    their identities and advancing/reseeding semantics.
    """
    from accelerate.data_loader import SeedableRandomSampler

    if type(sampler) not in (
        torch.utils.data.RandomSampler,
        SeedableRandomSampler,
    ):
        return
    if sampler.generator is None:
        seed = (
            sampler.initial_seed
            if type(sampler) is SeedableRandomSampler
            else DEFAULT_SAMPLING_SEED
        )
        device = (
            torch.get_default_device()
            if type(sampler) is SeedableRandomSampler
            and hasattr(torch, "get_default_device")
            else "cpu"
        )
        sampler.generator = torch.Generator(device=device).manual_seed(seed)
        sampler._lab_default_generator = sampler.generator


def _native_loader_kwargs(
    loader: TorchDataLoader, *, collate_fn: Callable | None = None
) -> dict[str, Any]:
    """Copy normalized native options without resharding or cloning."""
    kwargs: dict[str, Any] = {
        "dataset": loader.dataset,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn if collate_fn is None else collate_fn,
        "pin_memory": loader.pin_memory,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": loader.generator,
        "prefetch_factor": loader.prefetch_factor,
        "persistent_workers": loader.persistent_workers,
        "pin_memory_device": loader.pin_memory_device,
    }
    if "in_order" in _TORCH_DATALOADER_INIT_SIGNATURE.parameters:
        kwargs["in_order"] = getattr(loader, "in_order", True)
    if isinstance(loader.dataset, TorchIterableDataset):
        kwargs.update(batch_size=loader.batch_size, drop_last=loader.drop_last)
    elif loader.batch_size is None and loader.batch_sampler is not None:
        kwargs["batch_sampler"] = loader.batch_sampler
    else:
        kwargs.update(
            batch_size=loader.batch_size,
            sampler=loader.sampler,
            drop_last=loader.drop_last,
        )
    return kwargs


class _LoaderResumeRuntime:
    """Own delivery and recovery state without retaining the loader.

    Methods receive their loader explicitly; the native carrier and persistent
    iterator remain loader-owned for the existing cleanup traversal.
    """

    def __init__(
        self,
        loader: DataLoader,
        *,
        worker_prefix_source_known: bool,
        prepared: bool = False,
    ):
        loader._dataloader = None
        self._active_iterators: weakref.WeakSet = weakref.WeakSet()
        self._ambiguous_position = False
        self._pending_resume: dict[str, Any] | None = None
        self._delivery_epoch = int(getattr(loader.dataset, "epoch", 0))
        self._delivered_batches = 0
        self._worker_items = [0] * max(loader.num_workers, 1)
        self._prepared_checkpoint: dict[str, Any] | None = None
        # Accelerate owns business set_epoch calls for a prepared base; the
        # runtime only stages its delivery epoch before checkpoint capture.
        self._prepared = prepared
        self._worker_prefix_source_known = worker_prefix_source_known
        _bind_default_worker_generator(loader)
        if not isinstance(loader.dataset, TorchIterableDataset):
            sampler = self._sampler_owner(loader)
            _bind_private_sampler_generator(sampler)
        self._recovery = self._fresh_recovery(loader)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_active_iterators"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._active_iterators = weakref.WeakSet()

    def _fresh_recovery(self, loader: DataLoader) -> dict[str, str]:
        """Describe what this loader can reconstruct before it is resumed."""
        from robo_orchard_lab.dataset.robot.dataset_ex import (
            DictIterableDataset,
            IterableWithLenDataset,
            _uses_sample_prefetch,
        )

        if isinstance(
            loader.dataset, (DictIterableDataset, IterableWithLenDataset)
        ):
            if (
                type(loader.dataset) is not DictIterableDataset
                or not self._worker_prefix_source_known
                or loader.dataset._sampling_seed is None
            ):
                return {
                    "strategy": "batch_replay",
                    "accuracy": "unknown",
                    "reason": "worker generator origins are not captured",
                }
            if loader.num_workers > 0 and loader.worker_init_fn is not None:
                return {
                    "strategy": "batch_replay",
                    "accuracy": "unknown",
                    "reason": "worker initialization may change the data plan",
                }
            if (
                loader.dataset._shuffle.shuffle
                and not loader.dataset.sampling_strategy.supports_epoch_replay
            ):
                return {
                    "strategy": "batch_replay",
                    "accuracy": "unknown",
                    "reason": "sampling strategy does not declare replay",
                }
            # Dataset-side batching still prefetches in its unbatched view.
            if (
                loader.dataset.offload_num_batches is not None
                or _uses_sample_prefetch(loader.dataset._shuffle, None)
            ):
                return {
                    "strategy": "batch_replay",
                    "accuracy": "unknown",
                    "reason": "offload or sample prefetch requires replay",
                }
            return {
                "strategy": "worker_prefix",
                "accuracy": "worker_local",
                "reason": "",
            }
        return {
            "strategy": "batch_replay",
            "accuracy": "unknown",
            "reason": "worker-local recovery is unavailable for this source",
        }

    def _live_iterators(self) -> list[Iterator[Any]]:
        return [
            iterator
            for iterator in self._active_iterators
            if inspect.getgeneratorstate(iterator) != inspect.GEN_CLOSED
        ]

    def _layout(self, loader: DataLoader) -> dict[str, Any]:
        from robo_orchard_lab.dataset.robot.dataset_ex import (
            DictIterableDataset,
        )
        from robo_orchard_lab.dataset.robot.sampling_strategy import (
            DefaultMixtureSamplingStrategy,
        )

        layout = {
            "num_workers": loader.num_workers,
            "in_order": getattr(loader, "in_order", True),
            "outer_batch_size": loader.batch_size,
            "outer_drop_last": loader.drop_last,
            "native_unit": "batch"
            if loader._uses_dataset_batch_loader
            else "sample",
            "effective_batch_size": loader._effective_batch_size,
            "effective_drop_last": loader._effective_drop_last,
        }
        if (
            isinstance(loader.dataset, DictIterableDataset)
            and loader.dataset._shuffle.shuffle
        ):
            layout["sampling_strategy"] = (
                loader.dataset.sampling_strategy.model_dump(mode="json")
            )
            if (
                loader.dataset.sampling_strategy.class_type
                is DefaultMixtureSamplingStrategy
            ):
                layout["sampling_policy_version"] = (
                    "remaining-quota-initial-jitter-v1"
                )
        return layout

    def _sampler_owner(self, loader: DataLoader) -> Any:
        node = (
            loader.batch_sampler
            if loader.batch_sampler is not None
            else loader.sampler
        )
        seen: set[int] = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            if isinstance(node, torch.utils.data.Sampler) and (
                hasattr(node, "generator")
                or isinstance(node, torch.utils.data.SequentialSampler)
            ):
                return node
            nested = getattr(node, "batch_sampler", None)
            node = (
                nested
                if nested is not None
                else getattr(node, "sampler", None)
            )
        return None

    def _get_sampling_seed(self, loader: DataLoader) -> int | None:
        """Return a known sampling seed, checking the actual owner identity."""
        from robo_orchard_lab.dataset.robot.dataset_ex import (
            DictIterableDataset,
            IterableWithLenDataset,
        )

        if isinstance(
            loader.dataset, (DictIterableDataset, IterableWithLenDataset)
        ):
            return loader.dataset._sampling_seed
        if isinstance(loader.dataset, TorchIterableDataset):
            return None
        from accelerate.data_loader import SeedableRandomSampler

        sampler = self._sampler_owner(
            loader,
        )
        if type(sampler) is SeedableRandomSampler:
            return (
                sampler.initial_seed
                if type(sampler.initial_seed) is int
                else None
            )
        if (
            type(sampler) is torch.utils.data.RandomSampler
            and sampler.generator is not None
            and sampler.generator
            is getattr(sampler, "_lab_default_generator", None)
        ):
            return DEFAULT_SAMPLING_SEED
        return None

    def _can_resume_worker_prefix(
        self, loader: DataLoader, state: dict[str, Any]
    ) -> bool:
        """Check that the saved worker frontier is still reconstructible."""
        offsets = state.get("worker_items")
        return (
            self._fresh_recovery(
                loader,
            )["strategy"]
            == "worker_prefix"
            and state.get("recovery", {}).get("strategy") == "worker_prefix"
            and state.get("recovery", {}).get("accuracy") == "worker_local"
            and state.get("layout")
            == self._layout(
                loader,
            )
            and isinstance(offsets, list)
            and len(offsets) == max(loader.num_workers, 1)
        )

    def _constrain_recovery(
        self, loader: DataLoader, state: dict[str, Any]
    ) -> dict[str, str]:
        """Limit a saved capability to what the current loader supports."""
        saved = state["recovery"]
        strategy = saved["strategy"]
        if strategy == "worker_prefix" and self._can_resume_worker_prefix(
            loader, state
        ):
            return saved.copy()
        if strategy == "batch_replay" and saved["accuracy"] == "unknown":
            return saved.copy()
        return {
            "strategy": "batch_replay",
            "accuracy": "unknown",
            "reason": "the saved data position cannot be reconstructed",
        }

    def state_dict(self, loader: DataLoader) -> dict[str, Any]:
        """Copy the current delivery position without starting data loading.

        The worker vector counts native dataset outputs, not produced or
        prefetched outputs. Worker-local recovery requires a managed seed and
        unchanged data, topology and iteration configuration. It does not
        preserve cross-worker delivery order or user processing state. Other
        iterable configurations report unknown accuracy and replay batches.
        A loader with concurrent iterators has no unique recovery position.
        """
        if self._ambiguous_position or len(self._live_iterators()) > 1:
            raise RuntimeError(
                "concurrent iterators have no unique delivery position"
            )
        if self._pending_resume is not None:
            self._pending_resume["recovery"] = self._constrain_recovery(
                loader, self._pending_resume
            )
            return copy.deepcopy(self._pending_resume)
        seed = self._get_sampling_seed(
            loader,
        )
        state = {
            "schema": 1,
            "epoch": self._delivery_epoch,
            "delivered_batches": self._delivered_batches,
            "worker_items": (
                self._worker_items.copy()
                if isinstance(loader.dataset, TorchIterableDataset)
                else None
            ),
            "layout": self._layout(
                loader,
            ),
            "seed": seed,
            "recovery": self._recovery.copy(),
        }
        self._recovery = self._constrain_recovery(loader, state)
        state["recovery"] = self._recovery.copy()
        return state

    def load_state_dict(
        self, loader: DataLoader, state: dict[str, Any]
    ) -> None:
        """Validate and stage one restoration for the next iterator.

        Active iterators are rejected. A kept persistent pool is closed before
        the next restoration so that its old worker dataset copies cannot use
        the newly loaded position.
        """
        from robo_orchard_lab.dataset.robot.dataset_ex import (
            DictIterableDataset,
            IterableWithLenDataset,
        )

        if self._live_iterators():
            raise RuntimeError("cannot load while an iterator is active")
        state = copy.deepcopy(state)
        state.pop("_iterator_finished", None)
        if state.get("schema") != 1:
            raise ValueError("unsupported DataLoader state schema")
        epoch = state.get("epoch")
        delivered = state.get("delivered_batches")
        for name, value in (
            ("epoch", epoch),
            ("delivered_batches", delivered),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"invalid DataLoader state {name}")
        worker_items = state.get("worker_items")
        if worker_items is not None and (
            not isinstance(worker_items, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in worker_items
            )
        ):
            raise ValueError("invalid DataLoader worker_items")
        if not isinstance(state.get("layout"), dict):
            raise ValueError("missing DataLoader layout")
        recovery = state.get("recovery")
        if (
            not isinstance(recovery, dict)
            or recovery.get("strategy")
            not in {"worker_prefix", "batch_replay"}
            or recovery.get("accuracy") not in {"worker_local", "unknown"}
            or not isinstance(recovery.get("reason"), str)
        ):
            raise ValueError("invalid DataLoader recovery capability")
        seed = state.get("seed")
        if recovery["strategy"] == "worker_prefix" and seed is None:
            raise ValueError("checkpoint lacks the managed sampling seed")
        if isinstance(loader.dataset, DictIterableDataset):
            if loader.dataset._shuffle.shuffle:
                current_layout = self._layout(loader)
                if any(
                    state["layout"].get(key) != current_layout.get(key)
                    for key in (
                        "sampling_strategy",
                        "sampling_policy_version",
                    )
                ):
                    raise ValueError(
                        "DataLoader sampling strategy differs from the "
                        "checkpoint; the saved sample order cannot be "
                        "replayed."
                    )
            if seed is not None and (
                type(seed) is not int or not 0 <= seed < 2**64
            ):
                raise ValueError("invalid managed sampling seed")
        elif isinstance(loader.dataset, IterableWithLenDataset):
            if seed is not None and (
                type(seed) is not int or not 0 <= seed < 2**64
            ):
                raise ValueError("invalid iterable sampling seed")
        elif isinstance(loader.dataset, TorchIterableDataset):
            if seed is not None:
                raise ValueError("invalid generic iterable sampling seed")
        else:
            from accelerate.data_loader import SeedableRandomSampler

            sampler = self._sampler_owner(
                loader,
            )
            if type(sampler) is SeedableRandomSampler:
                if type(seed) is not int or not -(2**63) <= seed < 2**64:
                    raise ValueError(
                        "checkpoint lacks a valid seedable sampler seed"
                    )
            elif seed is not None and (
                type(seed) is not int
                or seed != DEFAULT_SAMPLING_SEED
                or type(sampler)
                not in (
                    torch.utils.data.RandomSampler,
                    torch.utils.data.SequentialSampler,
                )
            ):
                raise ValueError("invalid default map sampling seed")
        state["recovery"] = self._constrain_recovery(loader, state)
        if loader._dataloader is not None:
            from robo_orchard_lab.dataset.robot._prefetch import (
                _close_dataloader_owner_resources,
            )

            _close_dataloader_owner_resources(loader._dataloader)
            loader._dataloader = None
            loader._iterator = None
        if isinstance(loader.dataset, DictIterableDataset):
            if loader.dataset._sampling_seed is not None and seed is not None:
                loader.dataset._sampling_seed = seed
        elif isinstance(loader.dataset, IterableWithLenDataset):
            if loader.dataset._sampling_seed is not None and seed is not None:
                loader.dataset._sampling_seed = seed
        elif not isinstance(loader.dataset, TorchIterableDataset):
            sampler = self._sampler_owner(
                loader,
            )
            if type(sampler) is SeedableRandomSampler:
                sampler.initial_seed = seed
                if not self._prepared:
                    sampler.set_epoch(epoch)
        # An explicit generator remains caller-owned even if the source and
        # destination use different sampling modes. Report the current owner.
        state["seed"] = self._get_sampling_seed(loader)
        self._delivery_epoch = epoch
        self._pending_resume = state
        self._ambiguous_position = False
        if (
            not self._prepared
            and getattr(loader.dataset, "epoch", None) != epoch
            and hasattr(loader.dataset, "set_epoch")
        ):
            loader.dataset.set_epoch(epoch)

    def _notify_epoch(self, loader: DataLoader, epoch: int) -> None:
        """Notify an inactive loader of an epoch without starting workers."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if self._live_iterators():
            raise RuntimeError(
                "cannot change epoch while an iterator is active"
            )
        self._set_delivery_epoch(loader, epoch)

    def _set_delivery_epoch(self, loader: DataLoader, epoch: int) -> None:
        """Reset delivery metadata for a new epoch."""
        if epoch == self._delivery_epoch:
            return
        self._delivery_epoch = epoch
        self._delivered_batches = 0
        self._worker_items = [0] * max(loader.num_workers, 1)
        self._pending_resume = None
        self._recovery = self._fresh_recovery(
            loader,
        )
        if (
            not self._prepared
            and getattr(loader.dataset, "epoch", None) != epoch
            and hasattr(loader.dataset, "set_epoch")
        ):
            loader.dataset.set_epoch(epoch)

    def iterate(self, loader: DataLoader) -> Iterator[Any]:
        if self._live_iterators():
            self._ambiguous_position = True
        elif self._ambiguous_position:
            self._ambiguous_position = False
        iterator = self._iterate_deliveries(
            loader,
        )
        self._active_iterators.add(iterator)
        return iterator

    def _iterate_deliveries(self, loader: DataLoader) -> Iterator[Any]:
        """Own one native iterator while recording actual delivered outputs."""
        state = self._pending_resume
        replay_batches = 0
        restore_offsets = None
        if state is None:
            epoch = (
                self._delivery_epoch
                if self._prepared
                else int(
                    getattr(loader.dataset, "epoch", self._delivery_epoch)
                )
            )
            if epoch != self._delivery_epoch:
                self._set_delivery_epoch(loader, epoch)
            self._delivered_batches = 0
            self._worker_items = [0] * max(loader.num_workers, 1)
            self._recovery = self._fresh_recovery(
                loader,
            )
        else:
            state["recovery"] = self._constrain_recovery(loader, state)
            offsets = state.get("worker_items")
            fast = self._can_resume_worker_prefix(loader, state)
            if fast:
                restore_offsets = offsets.copy()
                self._delivered_batches = state["delivered_batches"]
                self._worker_items = offsets.copy()
                self._recovery = {
                    "strategy": "worker_prefix",
                    "accuracy": "worker_local",
                    "reason": "",
                }
            else:
                replay_batches = state["delivered_batches"]
                self._delivered_batches = 0
                self._worker_items = [0] * max(loader.num_workers, 1)
                self._recovery = {
                    "strategy": "batch_replay",
                    "accuracy": "unknown",
                    "reason": "worker position could not be reconstructed",
                }
                warnings.warn(
                    "Resuming by replaying the batch prefix; the original "
                    "worker order and sample suffix may differ.",
                    UserWarning,
                )
            self._pending_resume = None
        carrier = loader._dataloader
        if carrier is None or getattr(carrier, "_iterator", None) is None:
            tagged = _DeliveryCollate(
                loader.collate_fn, loader._auto_collation
            )
            native_kwargs = _native_loader_kwargs(loader, collate_fn=tagged)
            carrier = TorchDataLoader(**native_kwargs)
            loader._dataloader = carrier
        dataloader_iter: Iterator[Any] | None = None
        primary_exc: BaseException | None = None
        try:
            if restore_offsets is not None:
                loader.dataset._resume_worker_offsets = restore_offsets
            try:
                dataloader_iter = iter(carrier)
            finally:
                if restore_offsets is not None:
                    loader.dataset._resume_worker_offsets = None
            if carrier.persistent_workers:
                # Keep the native PyTorch owner contract visible on the Lab
                # loader as well as on its delivery-metadata carrier.
                loader._iterator = dataloader_iter
            for batch, worker_id, native_count in dataloader_iter:
                self._delivered_batches += 1
                if isinstance(loader.dataset, TorchIterableDataset):
                    self._worker_items[worker_id] += native_count
                if replay_batches:
                    replay_batches -= 1
                    continue
                yield batch
            if replay_batches:
                raise RuntimeError(
                    "DataLoader exhausted before the saved batch prefix "
                    "could be replayed."
                )
        except GeneratorExit:
            raise
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            if dataloader_iter is not None:
                shutdown_persistent = (
                    state is not None or primary_exc is not None
                )
                close_iterators_best_effort(
                    [dataloader_iter],
                    primary_exc=primary_exc,
                    shutdown_persistent_workers=shutdown_persistent,
                )
                if shutdown_persistent:
                    if getattr(carrier, "_iterator", None) is dataloader_iter:
                        carrier._iterator = None
                    if loader._iterator is dataloader_iter:
                        loader._iterator = None
                if (state is not None or not carrier.persistent_workers) and (
                    loader._dataloader is carrier
                ):
                    loader._dataloader = None
