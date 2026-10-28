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
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.robot import DatasetItem
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader,
    DictIterableDataset,
    ShardConfig,
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
            "Run a real accelerate multi-process trace for "
            "DictIterableDataset scheduling tests."
        )
    )
    parser.add_argument(
        "--prepare-mode",
        choices=["raw", "wrapped"],
        required=True,
        help="Choose raw Accelerator.prepare_data_loader or the repo wrapper.",
    )
    parser.add_argument(
        "--use-dataset-side-batching",
        choices=["0", "1"],
        required=True,
        help="Whether to enable dataset-side batching in the test dataloader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers to use per rank.",
    )
    parser.add_argument(
        "--shard-strategy",
        choices=["none", "drop_last", "pad_last"],
        default="none",
        help="Shard strategy to apply before accelerate prepares the loader.",
    )
    parser.add_argument(
        "--offload-num-batches",
        type=int,
        default=None,
        help="Enable finite offload segments in the native-sharding trace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where each rank writes its trace JSON.",
    )
    parser.add_argument(
        "--rng-path", choices=["native", "fallback", "dispatcher"]
    )
    return parser.parse_args()


def _first_value(batch):
    if isinstance(batch, torch.Tensor):
        values = batch.tolist()
        return values[0] if isinstance(values, list) else values
    if isinstance(batch, list):
        return batch[0]
    if isinstance(batch, tuple):
        return batch[0]
    return batch


def _jsonable_batch_values(batch):
    if isinstance(batch, torch.Tensor):
        return batch.tolist()
    if isinstance(batch, list | tuple):
        return [_jsonable_batch_values(value) for value in batch]
    return batch


def _get_base_dataset(loader):
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        return dataset
    base_dataloader = getattr(loader, "base_dataloader", None)
    if base_dataloader is not None:
        return base_dataloader.dataset
    raise RuntimeError(
        f"Unable to locate dataset for prepared loader: {loader!r}"
    )


def _get_worker_context(num_workers: int):
    if num_workers <= 0:
        return None

    start_methods = mp.get_all_start_methods()
    if "fork" in start_methods:
        return "fork"
    if "forkserver" in start_methods:
        return "forkserver"
    if "spawn" in start_methods:
        return "spawn"
    return None


def _run_managed_rng(args, accelerator):
    from robo_orchard_lab.dataset.robot._prefetch import (
        _close_dataloader_owner_resources,
    )
    from robo_orchard_lab.utils.accelerate import (
        bind_prepared_data_loader_sampling,
        configure_data_loader_for_accelerate,
    )

    path = args.rng_path
    if path == "fallback":
        # Force the upstream low-shard branch, including loader-created clones.
        # This process is dedicated to one path; production hints stay intact.
        DictIterableDataset.n_shards = property(lambda self: 1)
    dataset = DictIterableDataset(
        [
            ArrayDatasetItem(data=list(range(start, start + 32)))
            for start in (0, 100, 200)
        ],
        shuffle=True,
        seed=42,
        shard_kwargs=ShardConfig(shard_strategy="pad_last"),
        offload_num_batches=args.offload_num_batches,
        offload_dir=str(args.output_dir / "offload"),
    )
    accelerator.dataloader_config.dispatch_batches = path == "dispatcher"
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        use_dataset_side_batching=args.use_dataset_side_batching == "1",
        multiprocessing_context=_get_worker_context(args.num_workers),
    )
    if path != "dispatcher":
        configure_data_loader_for_accelerate(accelerator, loader)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    prepared, _, _, _ = accelerator.prepare(
        loader, model, optimizer, scheduler
    )
    source_seed = loader.dataset._sampling_seed
    source_state = loader.dataset._generator.get_state().clone()
    rng_state = torch.get_rng_state().clone()
    bound = bind_prepared_data_loader_sampling(accelerator, prepared)
    assert bound is (path != "dispatcher")
    assert loader.dataset._sampling_seed == source_seed
    assert torch.equal(loader.dataset._generator.get_state(), source_state)
    assert torch.equal(torch.get_rng_state(), rng_state)
    owner = (
        prepared.dataset.dataset if path == "fallback" else prepared.dataset
    )
    if path == "fallback":
        assert owner is not loader.dataset
    epoch = owner._sampling_epoch.value
    rank = owner._sampling_rank
    bind_prepared_data_loader_sampling(accelerator, prepared)
    new_owner = (
        prepared.dataset.dataset if path == "fallback" else prepared.dataset
    )
    assert new_owner._sampling_epoch.value == epoch
    assert new_owner._sampling_rank == rank
    assert torch.equal(torch.get_rng_state(), rng_state)
    traces = []
    try:
        for epoch in (0, 1, 0):
            prepared.set_epoch(epoch)
            traces.append(
                [int(v) for batch in prepared for v in batch.reshape(-1)]
            )
    finally:
        _close_dataloader_owner_resources(prepared)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank_{accelerator.process_index}.json").write_text(
        json.dumps(
            {
                "rank": accelerator.process_index,
                "prepared_type": type(prepared).__name__,
                "dataset_type": type(prepared.dataset).__name__,
                "epochs": traces,
            }
        )
    )
    accelerator.wait_for_everyone()
    return 0


def main() -> int:
    args = _parse_args()
    shard_strategy = (
        None if args.shard_strategy == "none" else args.shard_strategy
    )
    use_dataset_side_batching = args.use_dataset_side_batching == "1"

    accelerator = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False,
            even_batches=False,
            split_batches=False,
        ),
    )
    if args.rng_path is not None:
        return _run_managed_rng(args, accelerator)
    generator = torch.Generator()
    generator.manual_seed(123)
    dataset = DictIterableDataset(
        [
            ArrayDatasetItem(data=list(range(21))),
            ArrayDatasetItem(data=list(range(100, 105))),
        ],
        shuffle=True,
        generator=generator,
        offload_num_batches=args.offload_num_batches,
        offload_dir=str(args.output_dir / "offload"),
        shard_kwargs=ShardConfig(
            contiguous=True,
            shard_strategy=shard_strategy,
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4 if use_dataset_side_batching else 1,
        num_workers=args.num_workers,
        use_dataset_side_batching=use_dataset_side_batching,
        multiprocessing_context=_get_worker_context(args.num_workers),
    )

    if args.prepare_mode == "raw":
        prepared = accelerator.prepare_data_loader(dataloader)
    else:
        prepared = repo_prepare_data_loader(
            accelerator,
            dataloader,
        )

    trace = []
    value_trace = []
    for batch in prepared:
        trace.append("A" if _first_value(batch) < 100 else "B")
        value_trace.append(_jsonable_batch_values(batch))
        if len(trace) >= 20:
            break

    base_dataset = _get_base_dataset(prepared)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": accelerator.process_index,
        "trace": trace,
        "value_trace": value_trace,
        "shard_strategy": base_dataset.shard_kwargs.shard_strategy,
        "total_indices_length": getattr(
            base_dataset,
            "_total_indices_length",
            None,
        ),
        "prepared_type": type(prepared).__name__,
        "use_dataset_side_batching": use_dataset_side_batching,
        "num_workers": args.num_workers,
        "offload_num_batches": base_dataset.offload_num_batches,
    }
    output_file = args.output_dir / f"rank_{accelerator.process_index}.json"
    output_file.write_text(json.dumps(payload), encoding="utf-8")
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
