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
import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.robot import DatasetItem
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader,
    DictIterableDataset,
    IterableWithLenDataset,
    ShuffleConfig,
    _close_dataloader_iterator,
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


def _iterate_with_early_break(dataloader: object, max_batches: int) -> int:
    dataloader_iter = iter(dataloader)
    batch_count = 0
    try:
        for batch_idx, _batch in enumerate(dataloader_iter):
            batch_count += 1
            if batch_idx + 1 >= max_batches:
                break
    finally:
        # This subprocess probe is limited to iterator-owned resource cleanup.
        # Prepared-wrapper state such as Accelerate's DataLoaderStateMixin is
        # intentionally left to the wrapper owner and is not asserted here.
        _close_dataloader_iterator(dataloader_iter)

    return batch_count


def main() -> int:
    args = _parse_args()
    use_dataset_side_batching = args.use_dataset_side_batching == "1"
    persistent_workers = (
        args.persistent_workers == "1" and args.num_workers > 0
    )

    dataloader_kwargs = {
        "batch_size": 2 if use_dataset_side_batching else 1,
        "num_workers": args.num_workers,
        "use_dataset_side_batching": use_dataset_side_batching,
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

    for _ in range(args.cycles):
        batch_count = _iterate_with_early_break(
            prepared_dataloader,
            max_batches=args.batches_per_cycle,
        )
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
        per_cycle.append(cycle_snapshot)

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
        "baseline": baseline,
        "per_cycle": per_cycle,
        "after_cleanup": after_cleanup,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
