# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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


from dataclasses import dataclass
from typing import Optional

import pytest
import torch
from accelerate import Accelerator
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Metric

from robo_orchard_lab.pipeline import SimpleTrainer
from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
)


@dataclass
class TrainerState:
    """A class to manage the state of the training process.

    Attributes:
        epoch (int): The current epoch in the training process.
        step (int): The current step within the current epoch.
        global_step (int): The total number of steps taken across all epochs.
    """

    epoch: int = 0
    step: int = 0
    global_step: int = 0


class DummyPipelineHook(PipelineHooks):
    def __init__(self):
        super().__init__()

        self._on_loop_begin_cnt = 0
        self._on_loop_end_cnt = 0
        self._on_epoch_begin_cnt = 0
        self._on_epoch_end_cnt = 0
        self._on_step_begin_cnt = 0
        self._on_step_end_cnt = 0

        self.register_hook(
            "on_loop",
            HookContext.from_callable(
                before=self.on_loop_begin, after=self.on_loop_end
            ),
        )
        self.register_hook(
            "on_epoch",
            HookContext.from_callable(
                before=self.on_epoch_begin, after=self.on_epoch_end
            ),
        )
        self.register_hook(
            "on_step",
            HookContext.from_callable(
                before=self.on_step_begin, after=self.on_step_end
            ),
        )
        self._on_epoch_begin_state: Optional[TrainerState] = None
        self._on_epoch_end_state: Optional[TrainerState] = None

        self._on_step_begin_state: Optional[TrainerState] = None
        self._on_step_end_state: Optional[TrainerState] = None

    def on_loop_begin(self, args: PipelineHookArgs):
        self._on_loop_begin_cnt += 1
        print("Loop begin")

    def on_loop_end(self, args: PipelineHookArgs):
        self._on_loop_end_cnt += 1
        print("Loop end")

    def on_epoch_begin(self, args: PipelineHookArgs):
        self._on_epoch_begin_cnt += 1
        self._on_epoch_begin_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )
        print("Epoch begin. begin state: ", self._on_epoch_begin_state)

    def on_epoch_end(self, args: PipelineHookArgs):
        self._on_epoch_end_cnt += 1
        self._on_epoch_end_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )
        print("Epoch end. end state: ", self._on_epoch_end_state)
        print("Checking trainer state...")
        assert self._on_epoch_begin_state is not None
        assert (
            self._on_epoch_end_state.epoch == self._on_epoch_begin_state.epoch
        )
        # for epoch with step number > 1, the global step should be
        # different from the epoch step
        assert (
            self._on_epoch_end_state.global_step
            != self._on_epoch_begin_state.global_step
        )
        assert self._on_epoch_end_state.step != self._on_epoch_begin_state.step

    def on_step_begin(self, args: PipelineHookArgs):
        self._on_step_begin_cnt += 1
        self._on_step_begin_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )
        print("Step begin. begin state: ", self._on_step_begin_state)

    def on_step_end(self, args: PipelineHookArgs):
        self._on_step_end_cnt += 1
        self._on_step_end_state = TrainerState(
            epoch=args.epoch_id,
            step=args.step_id,
            global_step=args.global_step_id,
        )
        print("Step end. end state: ", self._on_step_end_state)
        print("Checking trainer state...")
        assert self._on_step_begin_state is not None
        assert self._on_step_end_state.step == self._on_step_begin_state.step
        assert (
            self._on_step_end_state.global_step
            == self._on_step_begin_state.global_step
        )
        assert self._on_step_end_state.epoch == self._on_step_begin_state.epoch


class DummyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.value = 0

    def update(self, preds, targets):
        self.value += 1

    def compute(self):
        return self.value


class SimpleModel(torch.nn.Module):
    """A simple neural network model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class DummyBatchProcessor(SimpleBatchProcessor):
    """A simple batch processor for testing."""

    def forward(self, model: torch.nn.Module, batch: torch.Tensor):
        if (
            self.accelerator is not None
            and batch.device != self.accelerator.device
        ):
            batch = batch.to(self.accelerator.device)

        outputs = model(batch)
        loss = torch.mean((outputs - 1) ** 2)  # Mean squared error loss
        return outputs, loss


# the fixture scope should be function, not session!
@pytest.fixture(scope="function")
def dummy_trainer():
    """Fixture to create a dummy trainer with a real model and optimizer."""
    model = SimpleModel()
    dataloader = [
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        torch.tensor([[0.1, 0.2]], dtype=torch.float32),
    ]
    optimizer = SGD(params=model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    accelerator = Accelerator(device_placement=True)
    batch_processor = DummyBatchProcessor()

    trainer = SimpleTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=batch_processor,
        max_epoch=1,
    )

    return trainer


def test_trainer_initialization(dummy_trainer):
    """Test trainer initialization."""
    assert dummy_trainer.max_epoch == 1
    # assert dummy_trainer.lr_scheduler_step_at == "step"


def test_training_loop(dummy_trainer: SimpleTrainer):
    """Test training loop execution."""
    # Spy on hook methods

    print("Old hook: ", dummy_trainer.hooks)
    hook = DummyPipelineHook()
    print("New hook: ", hook)

    dummy_trainer.hooks = hook

    # Run the training loop
    dummy_trainer()

    assert dummy_trainer.dataloader is not None

    # Verify that hooks were called the expected number of times
    assert hook._on_loop_begin_cnt == 1
    assert hook._on_epoch_begin_cnt == 1
    assert hook._on_step_begin_cnt == len(dummy_trainer.dataloader)
    assert hook._on_step_end_cnt == len(dummy_trainer.dataloader)
    assert hook._on_epoch_end_cnt == 1
    assert hook._on_loop_end_cnt == 1


def test_optimizer_and_scheduler(dummy_trainer):
    """Test optimizer and scheduler behavior."""
    initial_lr = dummy_trainer.optimizer.param_groups[0]["lr"]
    dummy_trainer()
    final_lr = dummy_trainer.optimizer.param_groups[0]["lr"]

    # Check that the learning rate scheduler updated the learning rate
    assert final_lr < initial_lr


def test_metric_computation(dummy_trainer):
    """Test metrics update and computation."""
    dummy_trainer()
    if dummy_trainer.metric is not None:
        dummy_trainer.metric.compute()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
