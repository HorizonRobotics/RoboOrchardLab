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

"""Two-rank CPU worker for the checkpoint/resume integration tests.

The launcher starts a fresh interpreter per rank and per save/load phase.
Faults are injected at real data, loader-state, or RNG-file boundaries; the
trainer, prepared loader, checkpoint hook, and distributed backend are real.
"""

from __future__ import annotations
import argparse
import faulthandler
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import (
    BatchSampler,
    DataLoader as TorchDataLoader,
    Dataset,
    IterableDataset,
    SequentialSampler,
    get_worker_info,
)

from robo_orchard_lab.dataset.robot import DatasetItem, DictIterableDataset
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader,
    IterableWithLenDataset,
    ShardConfig,
)
from robo_orchard_lab.pipeline.hooks.checkpoint import SaveCheckpointConfig
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
    ResumeCheckpointConfig,
)
from robo_orchard_lab.processing.step_processor import SimpleStepProcessor


class TraceDataset(Dataset[torch.Tensor]):
    """A real map source with worker reads observable after teardown."""

    def __init__(self, data: list[int], trace_dir: str, scenario: str):
        self.data = data
        self.trace_dir = Path(trace_dir)
        self.scenario = scenario

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        rank = int(os.environ["RANK"])
        if self.scenario == "fetch_error" and rank == 1:
            (self.trace_dir / "fetch_fault_rank1.txt").write_text(
                "injected source read failure\n", encoding="utf-8"
            )
            raise RuntimeError("INJECTED_SOURCE_READ_FAILURE")
        worker = get_worker_info()
        worker_id = -1 if worker is None else worker.id
        value = self.data[index]
        with (
            self.trace_dir / f"reads_rank{rank}_worker{worker_id}.jsonl"
        ).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps([value, worker_id]) + "\n")
        return torch.tensor([value, worker_id], dtype=torch.float32)


class TracedBatchSampler(BatchSampler):
    """Record entry into a custom sampler with four real source batches."""

    def __init__(
        self, dataset: Dataset[torch.Tensor], trace_dir: Path, rank: int
    ) -> None:
        super().__init__(
            SequentialSampler(dataset), batch_size=1, drop_last=False
        )
        self.trace_dir = trace_dir
        self.rank = rank

    def __iter__(self):
        (self.trace_dir / f"custom_batch_iter_rank{self.rank}.txt").write_text(
            "custom batch sampler entered\n", encoding="utf-8"
        )
        return super().__iter__()


class TraceDatasetItem(DatasetItem[TraceDataset]):
    class_type: ClassType[TraceDataset] = TraceDataset

    data: list[int]
    trace_dir: str
    scenario: str

    def get_dataset_row_num(self) -> int:
        return len(self.data)

    def _create_dataset(self) -> TraceDataset:
        return TraceDataset(self.data, self.trace_dir, self.scenario)


class EofOrErrorDataset(IterableDataset[torch.Tensor]):
    """Expose an actual source EOF on rank 0 and source error on rank 1."""

    def __init__(self, trace_dir: str):
        self.trace_dir = Path(trace_dir)

    def __iter__(self):
        rank = int(os.environ["RANK"])
        if rank == 0:
            (self.trace_dir / "source_eof_rank0.txt").write_text(
                "source returned EOF\n", encoding="utf-8"
            )
            yield from ()
            return
        (self.trace_dir / "source_error_rank1.txt").write_text(
            "source raised\n", encoding="utf-8"
        )
        raise RuntimeError("INJECTED_SOURCE_ERROR_WITH_PEER_EOF")


class RecordingProcessor(SimpleStepProcessor):
    """Call the real model and record inputs seen by the trainer."""

    def __init__(self, trace_dir: Path, rank: int, wait_for_prefetch: bool):
        super().__init__()
        self.trace_dir = trace_dir
        self.rank = rank
        self.wait_for_prefetch = wait_for_prefetch
        self.calls: list[list[int]] = []

    def forward(self, model: torch.nn.Module, batch: torch.Tensor):
        self.calls.extend(
            [
                [int(row[0]), int(row[1])]
                for row in batch.detach().cpu().tolist()
            ]
        )
        if self.wait_for_prefetch and len(self.calls) == 1:
            # Wait for real workers, not a fake prefetch or a rank barrier.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                count = sum(
                    len(path.read_text(encoding="utf-8").splitlines())
                    for path in self.trace_dir.glob(
                        f"reads_rank{self.rank}_worker*.jsonl"
                    )
                )
                if count >= 4:
                    break
                time.sleep(0.02)
        outputs = model(batch.float() / 128.0)
        return outputs, (outputs - 0.5).square().mean()


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _progress(trainer: HookBasedTrainer) -> dict[str, int]:
    progress = trainer.trainer_progress_state
    return {
        "epoch": progress.epoch_id,
        "global_step": progress.global_step_id,
        "epoch_step": progress.step_id,
        "micro_epoch": progress.micro_step.epoch_step_id,
    }


def _checkpoint_seed(trainer: HookBasedTrainer) -> int | None:
    """Read the actual per-rank seed captured in checkpoint progress."""
    rank = trainer.accelerator.process_index
    return next(
        (
            state.get("seed")
            for state in trainer.trainer_progress_state.rank_states
            if state["rank"] == rank
        ),
        None,
    )


def _loader_state(trainer: HookBasedTrainer) -> dict[str, Any] | None:
    # Only the frozen schema is inspected; RNG payloads need not become JSON.
    base = getattr(trainer.dataloader, "base_dataloader", None)
    if base is None or not callable(getattr(base, "state_dict", None)):
        return None
    state = trainer.dataloader.state_dict()
    if not isinstance(state, dict):
        raise TypeError("Prepared loader state is not a dict")
    return {
        key: state.get(key)
        for key in (
            "schema",
            "epoch",
            "delivered_batches",
            "worker_items",
            "layout",
            "recovery",
        )
    }


def _parameter_sum(trainer: HookBasedTrainer) -> float:
    model = trainer.accelerator.unwrap_model(trainer.model)
    return sum(
        parameter.detach().double().sum().item()
        for parameter in model.parameters()
    )


def _inject_snapshot_error(trainer: HookBasedTrainer, trace_dir: Path) -> None:
    """Fail the state read on the real prepared wrapper, not a fake loader.

    Integration must verify that the final snapshot reader reaches this seam;
    the test fails if the injection does not fire.
    """

    def fail_state_dict(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        (trace_dir / "snapshot_fault_rank1.txt").write_text(
            "injected state read failure\n", encoding="utf-8"
        )
        raise RuntimeError("INJECTED_SNAPSHOT_READ_FAILURE")

    trainer.dataloader.base_dataloader.state_dict = fail_state_dict


def _observe_static_shard_and_iterator(trace_dir: Path) -> None:
    """Record calls to the real shard and iterable dataset."""
    real_shard = IterableWithLenDataset.shard
    real_iter = IterableWithLenDataset.__iter__

    def observe_shard(self: Any, *args: Any, **kwargs: Any):
        result = real_shard(self, *args, **kwargs)
        rank = int(os.environ["RANK"])
        (trace_dir / f"shard_rank{rank}.json").write_text(
            json.dumps({"rows": result.total_iterator_length}) + "\n",
            encoding="utf-8",
        )
        return result

    def observe_iter(self: Any):
        rank = int(os.environ["RANK"])
        (trace_dir / f"iterator_rank{rank}.txt").write_text(
            "real iterable dataset entered\n", encoding="utf-8"
        )
        return real_iter(self)

    IterableWithLenDataset.shard = observe_shard
    IterableWithLenDataset.__iter__ = observe_iter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=(
            "save",
            "resume",
            "resume_save",
            "fetch_error",
            "eof_fetch_error",
            "snapshot_error",
            "uneven_g1",
            "custom_batch_g2",
            "map_full",
            "map_save",
            "map_resume",
        ),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--in-order", choices=("true", "false"), default="true"
    )
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def _log_phase(phase: str) -> None:
    """Emit rank-local progress retained by the launcher on failure."""
    print(
        f"[checkpoint-resume] rank={os.environ['RANK']} pid={os.getpid()} "
        f"monotonic={time.monotonic():.3f} phase={phase}",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    args = _parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    result: dict[str, Any] = {
        "rank": rank,
        "pid": os.getpid(),
        "scenario": args.scenario,
        "status": "error",
        "trainer_started": False,
    }
    trainer: HookBasedTrainer | None = None
    processor = RecordingProcessor(
        args.run_dir,
        rank,
        wait_for_prefetch=args.scenario == "save" and args.workers > 0,
    )
    try:
        faulthandler.dump_traceback_later(30, repeat=True, exit=False)
        _log_phase("accelerator_init_begin")
        accelerator = Accelerator(
            cpu=True,
            gradient_accumulation_steps=2
            if args.scenario == "custom_batch_g2"
            else 1,
            step_scheduler_with_optimizer=False,
            project_config=ProjectConfiguration(
                project_dir=str(args.project_dir),
                automatic_checkpoint_naming=True,
                total_limit=None,
            ),
            dataloader_config=DataLoaderConfiguration(
                dispatch_batches=False,
                even_batches=False,
                split_batches=False,
            ),
        )
        _log_phase("accelerator_init_end")
        result["world_size"] = accelerator.num_processes
        result["backend"] = (
            str(dist.get_backend()) if dist.is_initialized() else None
        )
        is_uneven = args.scenario == "uneven_g1"
        is_custom_batch = args.scenario == "custom_batch_g2"
        is_map = args.scenario in {"map_full", "map_save", "map_resume"}
        if is_uneven:
            _observe_static_shard_and_iterator(args.run_dir)
        if args.scenario == "eof_fetch_error":
            dataset: Any = EofOrErrorDataset(str(args.run_dir))
        elif is_uneven:
            dataset = IterableWithLenDataset(
                TraceDataset(list(range(3)), str(args.run_dir), args.scenario),
                shuffle=False,
                shard_kwargs=ShardConfig(contiguous=True, shard_strategy=None),
            )
        elif is_custom_batch:
            dataset = TraceDataset(
                list(range(4)), str(args.run_dir), args.scenario
            )
        elif is_map:
            dataset = TraceDataset(
                list(range(32)), str(args.run_dir), args.scenario
            )
        else:
            dataset = DictIterableDataset(
                [
                    TraceDatasetItem(
                        data=list(range(64)),
                        trace_dir=str(args.run_dir),
                        scenario=args.scenario,
                    )
                ],
                shuffle=False,
                seed=73,
                shard_kwargs=ShardConfig(shard_strategy="pad_last"),
            )
        loader_options: dict[str, Any] = {
            "batch_size": 1,
            "num_workers": args.workers,
            "in_order": args.in_order == "true",
        }
        if args.workers:
            loader_options.update(
                persistent_workers=True,
                prefetch_factor=2,
                multiprocessing_context="spawn",
            )
        if is_custom_batch:
            if args.workers:
                raise ValueError(
                    "Custom sampler acceptance uses num_workers=0"
                )
            custom_batch_sampler = TracedBatchSampler(
                dataset, args.run_dir, rank
            )
            loader = TorchDataLoader(
                dataset, batch_sampler=custom_batch_sampler, num_workers=0
            )
        elif is_map:
            if args.workers:
                raise ValueError("Map acceptance uses num_workers=0")
            loader = TorchDataLoader(dataset, batch_size=1, shuffle=True)
        else:
            loader = DataLoader(dataset, **loader_options)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        fault_scenario = args.scenario in {
            "fetch_error",
            "eof_fetch_error",
            "snapshot_error",
        }
        checkpoint_hook = SaveCheckpointConfig(
            save_epoch_freq=1 if fault_scenario else None,
            save_step_freq=None if fault_scenario else 1_000_000,
            save_when_loop_end=args.scenario
            in {"save", "resume_save", "map_save"},
            save_model=False,
        )
        resume_from = (
            None
            if args.checkpoint is None
            else ResumeCheckpointConfig(resume_from=str(args.checkpoint))
        )
        whole_epoch = args.scenario in {
            "resume",
            "map_full",
            "map_resume",
            "uneven_g1",
        }
        max_step = (
            None
            if whole_epoch
            else {
                "save": 2,
                "resume_save": 4,
                "map_save": 2,
            }.get(args.scenario, 1)
        )
        _log_phase("trainer_init_begin")
        trainer = HookBasedTrainer(
            accelerator=accelerator,
            model=model,
            dataloader=loader,
            batch_processor=processor,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            hooks=[checkpoint_hook],
            max_step=max_step,
            max_epoch=1 if whole_epoch else None,
            resume_from=resume_from,
        )
        _log_phase("trainer_init_end")
        result["start_progress"] = _progress(trainer)
        result["start_parameters"] = _parameter_sum(trainer)
        if is_map:
            result["start_dataloader_seed"] = _checkpoint_seed(trainer)
        if args.scenario in {
            "save",
            "resume",
            "resume_save",
            "map_full",
            "map_save",
            "map_resume",
        }:
            result["start_loader_state"] = _loader_state(trainer)
        if args.scenario == "snapshot_error" and rank == 1:
            _inject_snapshot_error(trainer, args.run_dir)
        result["trainer_started"] = True
        _log_phase("trainer_run_begin")
        trainer()
        _log_phase("trainer_run_end")
        result["end_progress"] = _progress(trainer)
        result["end_parameters"] = _parameter_sum(trainer)
        if is_map:
            result["end_dataloader_seed"] = _checkpoint_seed(trainer)
        if args.scenario in {
            "save",
            "resume",
            "resume_save",
            "map_full",
            "map_save",
            "map_resume",
        }:
            result["end_loader_state"] = _loader_state(trainer)
        result["status"] = "ok"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        if trainer is not None:
            result["end_progress"] = _progress(trainer)
    finally:
        try:
            result["processor_calls"] = processor.calls
            deadline = time.monotonic() + 3.0
            while (
                time.monotonic() < deadline
                and multiprocessing.active_children()
            ):
                time.sleep(0.05)
            result["active_children"] = [
                child.pid
                for child in multiprocessing.active_children()
                if child.is_alive()
            ]
            (args.run_dir / f"result_rank{rank}.json").write_text(
                json.dumps(result, default=_json_default, indent=2) + "\n",
                encoding="utf-8",
            )
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
