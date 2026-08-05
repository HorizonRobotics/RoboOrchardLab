# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

"""Manually benchmark scalar and batched multi-row index sampling.

This is an H1 diagnostic, not a pytest performance assertion. It measures only
the row-sampler index-read boundary on a real RODataset index table; it
intentionally excludes frame payload, sidecar, DataLoader, and model work.
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset

from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.row_sampler import (
    CachedIndexDataset,
    ColumnIndexOffsetSampler,
    ColumnIndexOffsetSamplerConfig,
    DeltaTimestampSampler,
    DeltaTimestampSamplerConfig,
    MultiRowSampler,
)

__all__ = ["main"]


class _CountingIndexDataset:
    """Wrap an index dataset and count list-index reads for one measurement."""

    def __init__(self, dataset: HFDataset) -> None:
        self._dataset = dataset
        self.read_call_count = 0

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        self.read_call_count += 1
        return self._dataset.__getitems__(indices)

    def reset_read_count(self) -> None:
        """Reset the counter before measuring one scalar or batch variant."""
        self.read_call_count = 0


def _parse_offsets(raw_offsets: str) -> list[int | None]:
    offsets: list[int | None] = []
    for raw_offset in raw_offsets.split(","):
        normalized_offset = raw_offset.strip().lower()
        if normalized_offset in {"none", "null"}:
            offsets.append(None)
        else:
            offsets.append(int(normalized_offset))
    if not offsets:
        raise ValueError("offsets must contain at least one value.")
    return offsets


def _parse_delta_timestamps(raw_delta_timestamps: str) -> list[float]:
    """Parse comma-separated timestamp deltas in seconds."""
    delta_timestamps = [
        float(raw_delta_timestamp.strip())
        for raw_delta_timestamp in raw_delta_timestamps.split(",")
    ]
    if not delta_timestamps:
        raise ValueError("delta timestamps must contain at least one value.")
    return delta_timestamps


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, got {raw_value!r}."
        )
    return value


def _nonnegative_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer, got {raw_value!r}."
        )
    return value


def _make_index_batches(
    *,
    dataset_length: int,
    batch_size: int,
    batch_count: int,
    seed: int,
) -> list[list[int]]:
    if batch_size > dataset_length:
        raise ValueError(
            "batch_size cannot exceed index dataset length: "
            f"{batch_size} > {dataset_length}."
        )
    random_generator = random.Random(seed)
    return [
        random_generator.sample(range(dataset_length), k=batch_size)
        for _ in range(batch_count)
    ]


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return an inclusive linear-interpolated percentile."""
    if not values:
        raise ValueError("Cannot calculate a percentile of no values.")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def _measure_variant(
    *,
    name: str,
    run_batch: Callable[[list[int]], object],
    index_dataset: _CountingIndexDataset,
    batches: Sequence[list[int]],
) -> dict[str, Any]:
    index_dataset.reset_read_count()
    batch_latencies_s: list[float] = []
    total_row_count = 0
    start_time = time.perf_counter()
    for batch in batches:
        batch_start_time = time.perf_counter()
        run_batch(batch)
        batch_latencies_s.append(time.perf_counter() - batch_start_time)
        total_row_count += len(batch)
    elapsed_s = time.perf_counter() - start_time
    return {
        "name": name,
        "elapsed_s": elapsed_s,
        "rows_per_s": total_row_count / elapsed_s,
        "index_read_call_count": index_dataset.read_call_count,
        "batch_latency_p50_s": _percentile(batch_latencies_s, 0.5),
        "batch_latency_p95_s": _percentile(batch_latencies_s, 0.95),
    }


def _run_measurement_pair(
    *,
    run_scalar_batch: Callable[[list[int]], object],
    run_batch: Callable[[list[int]], object],
    index_dataset: _CountingIndexDataset,
    batches: Sequence[list[int]],
    candidate_first: bool,
) -> dict[str, Any]:
    variants: dict[str, Callable[[list[int]], object]] = {
        "scalar": run_scalar_batch,
        "batch": run_batch,
    }
    variant_order = (
        ("batch", "scalar") if candidate_first else ("scalar", "batch")
    )
    measurements = {
        name: _measure_variant(
            name=name,
            run_batch=variants[name],
            index_dataset=index_dataset,
            batches=batches,
        )
        for name in variant_order
    }
    scalar_elapsed_s = measurements["scalar"]["elapsed_s"]
    batch_elapsed_s = measurements["batch"]["elapsed_s"]
    return {
        "variant_order": variant_order,
        "scalar": measurements["scalar"],
        "batch": measurements["batch"],
        "batch_over_scalar_elapsed_ratio": batch_elapsed_s / scalar_elapsed_s,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument(
        "--sampler",
        choices=("offset", "delta"),
        default="offset",
        help="Sampler family to benchmark.",
    )
    parser.add_argument("--column", default="joints")
    parser.add_argument(
        "--offsets",
        default=",".join(str(index) for index in range(33)),
        help="Comma-separated offsets; default matches RobotWin action rows.",
    )
    parser.add_argument(
        "--delta-timestamps",
        default="0,0.04",
        help=(
            "Comma-separated timestamp deltas in seconds for --sampler delta."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Timestamp matching tolerance in seconds for --sampler delta.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--batch-count", type=_positive_int, default=100)
    parser.add_argument(
        "--warmup-batch-count",
        type=_nonnegative_int,
        default=20,
    )
    parser.add_argument("--repeat-count", type=_positive_int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-cross-episode",
        action="store_true",
        help="Disable the normal same-episode candidate requirement.",
    )
    return parser.parse_args()


def main() -> None:
    """Run balanced scalar-versus-batch sampler measurements and print JSON."""
    args = _parse_args()
    if args.sampler == "offset":
        offsets = _parse_offsets(args.offsets)
        sampler: ColumnIndexOffsetSampler | DeltaTimestampSampler = (
            ColumnIndexOffsetSamplerConfig(
                column_offsets={args.column: offsets},
                force_in_episode=not args.allow_cross_episode,
            )()
        )
    else:
        delta_timestamps = _parse_delta_timestamps(args.delta_timestamps)
        sampler = DeltaTimestampSamplerConfig(
            column_delta_ts={args.column: delta_timestamps},
            tolerance=args.tolerance,
        )()

    with RODataset(dataset_path=str(args.dataset_path)) as dataset:
        index_dataset = _CountingIndexDataset(dataset.index_dataset)
        batches = _make_index_batches(
            dataset_length=len(index_dataset),
            batch_size=args.batch_size,
            batch_count=args.batch_count,
            seed=args.seed,
        )
        warmup_batches = _make_index_batches(
            dataset_length=len(index_dataset),
            batch_size=args.batch_size,
            batch_count=args.warmup_batch_count,
            seed=args.seed + 1,
        )
        if args.sampler == "offset":

            def run_scalar_batch(batch: list[int]) -> object:
                return [
                    sampler.sample_row_idx(index_dataset, index)
                    for index in batch
                ]

            def run_batch(batch: list[int]) -> object:
                return sampler.sample_row_idx_batch(index_dataset, batch)

        else:

            def run_scalar_batch(batch: list[int]) -> object:
                return MultiRowSampler.sample_row_idx_batch(
                    sampler,
                    CachedIndexDataset(index_dataset),
                    batch,
                )

            def run_batch(batch: list[int]) -> object:
                return sampler.sample_row_idx_batch(
                    CachedIndexDataset(index_dataset),
                    batch,
                )

        for warmup_index, warmup_batch in enumerate(warmup_batches):
            (run_batch if warmup_index % 2 else run_scalar_batch)(warmup_batch)

        pairs = [
            _run_measurement_pair(
                run_scalar_batch=run_scalar_batch,
                run_batch=run_batch,
                index_dataset=index_dataset,
                batches=batches,
                candidate_first=repeat_index % 2 == 1,
            )
            for repeat_index in range(args.repeat_count)
        ]

    report = {
        "dataset_path": str(args.dataset_path.resolve()),
        "sampler": args.sampler,
        "column": args.column,
        "offsets": offsets if args.sampler == "offset" else None,
        "force_in_episode": (
            not args.allow_cross_episode if args.sampler == "offset" else None
        ),
        "delta_timestamps": (
            delta_timestamps if args.sampler == "delta" else None
        ),
        "tolerance": args.tolerance if args.sampler == "delta" else None,
        "batch_size": args.batch_size,
        "batch_count": args.batch_count,
        "warmup_batch_count": args.warmup_batch_count,
        "repeat_count": args.repeat_count,
        "seed": args.seed,
        "python_version": sys.version,
        "pairs": pairs,
        "median_batch_over_scalar_elapsed_ratio": _percentile(
            [pair["batch_over_scalar_elapsed_ratio"] for pair in pairs],
            0.5,
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
