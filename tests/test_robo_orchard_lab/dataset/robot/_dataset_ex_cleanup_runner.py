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

import argparse
import inspect
import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from types import GeneratorType

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.robot import DatasetItem
from robo_orchard_lab.dataset.robot._prefetch import (
    DataloaderCloseReason,
    _close_dataloader_owner_resources,
    close_dataloader_resources,
    close_iterators_best_effort,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader,
    DictIterableDataset,
    IterableWithLenDataset,
    ShuffleConfig,
)
from robo_orchard_lab.utils.accelerate import (
    prepare_data_loader as repo_prepare_data_loader,
)


class ArrayDataset(Dataset):
    def __init__(self, data: list[int]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> int:
        return self.data[idx]


class ArrayDatasetItem(DatasetItem[ArrayDataset]):
    class_type: ClassType[ArrayDataset] = ArrayDataset

    data: list[int]

    def get_dataset_row_num(self) -> int:
        return len(self.data)

    def _create_dataset(self) -> ArrayDataset:
        return ArrayDataset(self.data)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise repeated early-break dataloader cleanup and emit "
            "resource snapshots."
        )
    )
    parser.add_argument(
        "--dataset-kind",
        choices=["iterable", "dict"],
        required=True,
    )
    parser.add_argument(
        "--prepare-mode",
        choices=["none", "raw", "wrapped"],
        default="none",
    )
    parser.add_argument(
        "--use-dataset-side-batching",
        choices=["0", "1"],
        required=True,
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--persistent-workers",
        choices=["0", "1"],
        default="0",
    )
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--batches-per-cycle", type=int, default=2)
    parser.add_argument(
        "--close-mode",
        choices=["early-break", "epoch-exhausted"],
        default="early-break",
    )
    parser.add_argument(
        "--pin-memory",
        choices=["0", "1"],
        default="0",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def _get_worker_context(num_workers: int) -> str | None:
    if num_workers <= 0:
        return None

    start_methods = mp.get_all_start_methods()
    override = os.environ.get("ROBO_ORCHARD_TEST_DATALOADER_MP_CONTEXT")
    if override:
        if override not in start_methods:
            raise ValueError(
                "Unsupported multiprocessing context "
                f"{override!r}. Available contexts: {start_methods}."
            )
        return override

    if "fork" in start_methods:
        return "fork"
    if "forkserver" in start_methods:
        return "forkserver"
    if "spawn" in start_methods:
        return "spawn"
    return None


def _count_prefetch_threads() -> int:
    return sum(
        thread.name == "dataset-prefetch-producer"
        for thread in threading.enumerate()
    )


def _count_open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def _child_pids() -> list[int]:
    return sorted(child.pid for child in mp.active_children())


def _snapshot() -> dict[str, object]:
    return {
        "child_pids": _child_pids(),
        "fd_count": _count_open_fds(),
        "prefetch_threads": _count_prefetch_threads(),
    }


def _wait_for_state(
    expected_child_pids: list[int],
    expected_prefetch_threads: int,
    timeout: float = 5.0,
) -> dict[str, object]:
    deadline = time.time() + timeout
    last_snapshot = _snapshot()
    while time.time() < deadline:
        last_snapshot = _snapshot()
        if (
            last_snapshot["child_pids"] == expected_child_pids
            and last_snapshot["prefetch_threads"] == expected_prefetch_threads
        ):
            return last_snapshot
        time.sleep(0.05)

    return last_snapshot


def _build_dataset(
    dataset_kind: str,
):
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


def _prepare_dataloader(
    prepare_mode: str,
    dataloader: DataLoader,
) -> tuple[Accelerator | None, object]:
    if prepare_mode == "none":
        return None, dataloader

    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False,
            even_batches=False,
            split_batches=False,
        ),
    )
    if prepare_mode == "raw":
        return accelerator, accelerator.prepare_data_loader(dataloader)

    return accelerator, repo_prepare_data_loader(accelerator, dataloader)


def _collect_pin_memory_flags(
    owner: object,
    seen: set[int] | None = None,
) -> list[bool]:
    """Find PyTorch iterator pin-memory flags through wrapper layers."""
    if owner is None:
        return []
    if seen is None:
        seen = set()
    owner_id = id(owner)
    if owner_id in seen:
        return []
    seen.add(owner_id)

    flags: list[bool] = []
    pin_memory = getattr(owner, "_pin_memory", None)
    if isinstance(pin_memory, bool):
        flags.append(pin_memory)

    if isinstance(owner, GeneratorType):
        generator_locals = inspect.getgeneratorlocals(owner)
        for nested_name in ("dataloader_iter", "main_iterator"):
            flags.extend(
                _collect_pin_memory_flags(
                    generator_locals.get(nested_name),
                    seen,
                )
            )

    for attr_name in (
        "_iterator",
        "base_dataloader",
        "dataloader",
        "_dataloader",
        "data_loader",
    ):
        flags.extend(
            _collect_pin_memory_flags(getattr(owner, attr_name, None), seen)
        )

    return flags


def _pin_memory_observed(*owners: object) -> bool:
    """Return whether any active iterator is using the pin-memory path."""
    flags: list[bool] = []
    seen: set[int] = set()
    for owner in owners:
        flags.extend(_collect_pin_memory_flags(owner, seen))
    return any(flags)


def _iterate_with_early_break(
    dataloader: object,
    max_batches: int,
) -> tuple[int, bool]:
    dataloader_iter = iter(dataloader)
    pin_memory_observed = _pin_memory_observed(dataloader, dataloader_iter)
    batch_count = 0
    try:
        for batch_idx, _batch in enumerate(dataloader_iter):
            pin_memory_observed = pin_memory_observed or _pin_memory_observed(
                dataloader, dataloader_iter
            )
            batch_count += 1
            if batch_idx + 1 >= max_batches:
                break
    finally:
        # This subprocess probe is limited to iterator-owned resource cleanup.
        # Prepared-wrapper state such as Accelerate's DataLoaderStateMixin is
        # intentionally left to the wrapper owner and is not asserted here.
        close_iterators_best_effort([dataloader_iter])

    return batch_count, pin_memory_observed


def _iterate_until_exhausted(dataloader: object) -> tuple[int, bool]:
    """Consume a full epoch and close it as normal epoch exhaustion."""
    dataloader_iter = iter(dataloader)
    pin_memory_observed = _pin_memory_observed(dataloader, dataloader_iter)
    batch_count = 0
    try:
        for _batch in dataloader_iter:
            pin_memory_observed = pin_memory_observed or _pin_memory_observed(
                dataloader, dataloader_iter
            )
            batch_count += 1
    finally:
        close_dataloader_resources(
            dataloader,
            dataloader_iter,
            reason=DataloaderCloseReason.EPOCH_EXHAUSTED,
        )

    return batch_count, pin_memory_observed


def _iterate_for_cycle(
    dataloader: object,
    *,
    close_mode: str,
    max_batches: int,
) -> tuple[int, bool]:
    """Run one cleanup probe cycle for the requested close mode."""
    if close_mode == "early-break":
        return _iterate_with_early_break(
            dataloader,
            max_batches=max_batches,
        )
    return _iterate_until_exhausted(dataloader)


def main() -> int:
    args = _parse_args()
    use_dataset_side_batching = args.use_dataset_side_batching == "1"
    persistent_workers = (
        args.persistent_workers == "1" and args.num_workers > 0
    )
    pin_memory = args.pin_memory == "1"

    dataloader_kwargs = {
        "batch_size": 2 if use_dataset_side_batching else 1,
        "num_workers": args.num_workers,
        "use_dataset_side_batching": use_dataset_side_batching,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["multiprocessing_context"] = _get_worker_context(
            args.num_workers
        )

    dataloader = DataLoader(
        _build_dataset(args.dataset_kind),
        **dataloader_kwargs,
    )
    accelerator, prepared_dataloader = _prepare_dataloader(
        args.prepare_mode,
        dataloader,
    )

    baseline = _snapshot()
    per_cycle: list[dict[str, object]] = []
    persistent_child_pids: list[int] | None = None
    expected_prefetch_threads = int(baseline["prefetch_threads"])
    pin_memory_observed = False

    for _ in range(args.cycles):
        batch_count, cycle_pin_memory_observed = _iterate_for_cycle(
            prepared_dataloader,
            close_mode=args.close_mode,
            max_batches=args.batches_per_cycle,
        )
        pin_memory_observed = pin_memory_observed or cycle_pin_memory_observed
        if persistent_workers:
            current_child_pids = _child_pids()
            if persistent_child_pids is None:
                persistent_child_pids = current_child_pids
            expected_child_pids = persistent_child_pids
        else:
            expected_child_pids = list(baseline["child_pids"])

        cycle_snapshot = _wait_for_state(
            expected_child_pids=expected_child_pids,
            expected_prefetch_threads=expected_prefetch_threads,
        )
        cycle_snapshot["batch_count"] = batch_count
        cycle_snapshot["pin_memory_observed"] = cycle_pin_memory_observed
        per_cycle.append(cycle_snapshot)

    if args.close_mode == "epoch-exhausted":
        _close_dataloader_owner_resources(prepared_dataloader)

    del prepared_dataloader
    del dataloader
    if accelerator is not None:
        accelerator.end_training()
    del accelerator

    after_cleanup = _wait_for_state(
        expected_child_pids=list(baseline["child_pids"]),
        expected_prefetch_threads=expected_prefetch_threads,
    )

    payload = {
        "dataset_kind": args.dataset_kind,
        "prepare_mode": args.prepare_mode,
        "use_dataset_side_batching": use_dataset_side_batching,
        "num_workers": args.num_workers,
        "persistent_workers": persistent_workers,
        "close_mode": args.close_mode,
        "pin_memory_requested": pin_memory,
        "pin_memory_observed": pin_memory_observed,
        "baseline": baseline,
        "per_cycle": per_cycle,
        "after_cleanup": after_cleanup,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
