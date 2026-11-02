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

"""Shared test models, loader inputs, and Trainer construction helpers."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedType,
    ProjectConfiguration,
)
from robo_orchard_core.utils.config import ClassType
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from robo_orchard_lab.dataset.robot import (
    DatasetItem,
    DictIterableDataset,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    DataLoader as RODataLoader,
)
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
)
from robo_orchard_lab.processing.step_processor import (
    SimpleStepProcessor,
)

__all__ = [
    "ArrayDatasetItem",
    "DummyBatchProcessor",
    "PrepareOnlyDeepSpeedAccelerator",
    "SimpleModel",
    "TensorDataset",
    "_adamw_deepspeed_config",
    "_deepspeed_topology_config",
    "_epoch_checkpoint_trainer",
    "build_micro_step_hook_args",
    "dummy_trainer",
]


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class DummyBatchProcessor(SimpleStepProcessor):
    def forward(self, model: torch.nn.Module, batch: torch.Tensor):
        if (
            self.accelerator is not None
            and batch.device != self.accelerator.device
        ):
            batch = batch.to(self.accelerator.device)

        outputs = model(batch)
        loss = torch.mean((outputs - 1) ** 2)
        return outputs, loss


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[int]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        value = self.data[idx]
        return torch.tensor([value, value], dtype=torch.float32)


class ArrayDatasetItem(DatasetItem[ArrayDataset]):
    class_type: ClassType[ArrayDataset] = ArrayDataset

    data: list[int]

    def get_dataset_row_num(self) -> int:
        return len(self.data)

    def _create_dataset(self) -> ArrayDataset:
        return ArrayDataset(self.data)


class PrepareOnlyDeepSpeedAccelerator:
    distributed_type = DistributedType.DEEPSPEED
    process_index = 0

    def __init__(self, deepspeed_config: dict[str, object]) -> None:
        self._models: list[torch.nn.Module] = []
        self.deepspeed_plugin = SimpleNamespace(
            deepspeed_config=deepspeed_config,
            gradient_clipping=deepspeed_config.get("gradient_clipping", None),
        )
        self.prepare_calls = 0
        self.prepared_args: tuple[object, ...] = ()
        self.registered_for_checkpointing: list[object] = []

    def prepare(self, *args):
        self.prepare_calls += 1
        self.prepared_args = args
        return args

    def register_for_checkpointing(self, obj: object) -> None:
        self.registered_for_checkpointing.append(obj)


def _adamw_deepspeed_config() -> dict[str, Any]:
    return {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "weight_decay": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
    }


def _deepspeed_topology_config() -> dict[str, Any]:
    return {"zero_optimization": {"stage": 0}}


def build_micro_step_hook_args(
    trainer: HookBasedTrainer,
    *,
    is_optimizer_step_boundary: bool,
) -> PipelineHookArgs:
    current_micro_step = (
        trainer.trainer_progress_state.preview_next_micro_step()
    )
    trainer.trainer_progress_state.commit_micro_step(current_micro_step)
    hook_args = PipelineHookArgs(
        accelerator=trainer.accelerator,
        micro_step=trainer.trainer_progress_state.micro_step,
        is_optimizer_step_boundary=is_optimizer_step_boundary,
    )
    trainer.trainer_progress_state.sync_pipeline_hook_arg(hook_args)
    hook_args.is_optimizer_step_boundary = is_optimizer_step_boundary
    return hook_args


@pytest.fixture(scope="function")
def dummy_trainer():
    model = SimpleModel()
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor([[0.5, 0.5], [0.1, 0.2]], dtype=torch.float32)
        ),
    )

    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(
        device_placement=True,
        step_scheduler_with_optimizer=False,
    )
    batch_processor = DummyBatchProcessor()

    trainer = HookBasedTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=batch_processor,
        max_epoch=1,
    )

    return trainer


def _epoch_checkpoint_trainer(
    root,
    *,
    lab_loader,
    events,
    batches,
    resume=None,
    max_epoch=None,
    max_step=None,
    fail_after_checkpoint=False,
    accumulation_steps=1,
    dataset_size=4,
    save_epoch_freq=1,
    save_step_freq=None,
    save_when_loop_end=False,
    shuffle=False,
    generator=None,
):
    from robo_orchard_lab.pipeline.hooks.checkpoint import SaveCheckpointConfig
    from robo_orchard_lab.pipeline.hooks.validation import ValidationHookConfig
    from robo_orchard_lab.pipeline.training.hook_based_trainer import (
        ResumeCheckpointConfig,
    )

    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=accumulation_steps,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        project_config=ProjectConfiguration(
            project_dir=str(root.parent), automatic_checkpoint_naming=True
        ),
    )
    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    dataset = (
        DictIterableDataset(
            [ArrayDatasetItem(data=list(range(dataset_size)))], seed=42
        )
        if lab_loader
        else ArrayDataset(list(range(dataset_size)))
    )
    loader = (RODataLoader if lab_loader else DataLoader)(
        dataset, batch_size=1, shuffle=shuffle, generator=generator
    )

    class Processor(DummyBatchProcessor):
        def forward(self, model, batch):
            batches.append(int(batch[0, 0]))
            return super().forward(model, batch)

    observer = PipelineHooks()

    def after_epoch(args):
        events.append(("after", args.epoch_id))
        if fail_after_checkpoint:
            raise RuntimeError("later epoch hook failed")

    observer.register_hook(
        "on_epoch", HookContext.from_callable(after=after_epoch)
    )
    return HookBasedTrainer(
        accelerator=accelerator,
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=StepLR(optimizer, step_size=1),
        batch_processor=Processor(),
        max_epoch=max_epoch,
        max_step=max_step,
        validation=ValidationHookConfig(
            epoch_eval_freq=1,
            eval_callback=lambda model: events.append(("eval", None)),
        ),
        hooks=(
            [
                SaveCheckpointConfig(
                    save_root=None,
                    save_epoch_freq=save_epoch_freq,
                    save_step_freq=save_step_freq,
                    save_when_loop_end=save_when_loop_end,
                    save_model=False,
                ),
            ]
            if save_epoch_freq is not None or save_step_freq is not None
            else []
        )
        + [observer],
        resume_from=(
            ResumeCheckpointConfig(resume_from=str(resume)) if resume else None
        ),
    )
