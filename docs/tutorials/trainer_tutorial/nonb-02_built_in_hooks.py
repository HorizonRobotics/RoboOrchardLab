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

# ruff: noqa: E501 D415 D205 E402

"""Leveraging Built-in Hooks for Common Training Tasks
==========================================================

Hooks are the primary way to add custom behavior to the training or evaluation loop without modifying its source code.
They are called at specific points during training or evaluation (e.g., end of epoch, after a micro step, or after a completed accumulation window).
"""

# %%
# Reuse the code from previous tutorial
# ------------------------------------------------------
#

import logging
import os
from typing import Any, Optional, Tuple

import torch
from robo_orchard_core.utils.logging import LoggerManager

logger = LoggerManager(level=logging.INFO).get_child(__name__)

from pydantic import Field
from robo_orchard_core.utils.cli import SettingConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


class DatasetConfig(SettingConfig):
    """Configuration for the dataset.

    This is a example configuration for the ImageNet dataset.
    """

    data_root: Optional[str] = Field(
        description="Image dataset directory.", default=None
    )

    pipeline_test: bool = Field(
        description="Whether or not use dummy data for fast pipeline test.",
        default=False,
    )

    dummy_train_imgs: int = Field(
        description="Number of dummy training images.",
        default=8,
    )

    dummy_val_imgs: int = Field(
        description="Number of dummy validation images.",
        default=4,
    )

    def __post_init__(self):
        if self.pipeline_test is False and self.data_root is None:
            raise ValueError(
                "data_root must be specified when pipeline_test is False."
            )

    def get_dataset(self) -> Tuple[Dataset, Dataset]:
        if self.pipeline_test:
            train_dataset = datasets.FakeData(
                self.dummy_train_imgs,
                (3, 64, 64),
                1000,
                transforms.ToTensor(),
            )
            val_dataset = datasets.FakeData(
                self.dummy_val_imgs, (3, 64, 64), 1000, transforms.ToTensor()
            )
        else:
            assert self.data_root is not None
            train_dataset = datasets.ImageFolder(
                os.path.join(self.data_root, "train"),
                transform=transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                ),
            )

            val_dataset = datasets.ImageFolder(
                os.path.join(self.data_root, "val"),
                transform=transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                ),
            )
        return train_dataset, val_dataset


class TrainerConfig(SettingConfig):
    """Configuration for the trainer.

    This is an example configuration for training a ResNet18 model
    on ImageNet. Only a few parameters are set here for demonstration
    purposes.
    """

    dataset: DatasetConfig = Field(
        description="Dataset configuration. Need to be set by user.",
    )

    batch_size: int = Field(
        description="Batch size for training.",
        default=128,
    )

    num_workers: int = Field(
        description="Number of workers for data loading.",
        default=4,
    )

    max_epoch: int = Field(
        description="Maximum number of epochs for training.",
        default=90,
    )

    workspace_root: str = Field(
        description="Workspace root directory.",
        default="./workspace/",
    )


cfg = TrainerConfig(
    dataset=DatasetConfig(pipeline_test=True),
    batch_size=2,
    max_epoch=1,
    num_workers=0,
    workspace_root="./workspace/tutorial2/",
)

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

accelerator = Accelerator(
    cpu=True,
    step_scheduler_with_optimizer=False,
    project_config=ProjectConfiguration(
        project_dir=cfg.workspace_root,
        logging_dir=os.path.join(cfg.workspace_root, "logs"),
        automatic_checkpoint_naming=True,
        total_limit=32,  # Max checkpoints to keep
    ),
)


from robo_orchard_lab.processing.step_processor import SimpleStepProcessor


class MyBatchProcessor(SimpleStepProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()  # Define your loss

    def forward(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        # unpack batch by yourself
        images, target = batch

        # Accelerator has already moved batch to the correct device
        output = model(images)
        loss = self.criterion(output, target) if self.need_backward else None

        return output, loss  # Returns model output and loss


from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

train_dataset, _ = cfg.dataset.get_dataset()
train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=False,
)

model = models.resnet18()

optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

hooks = []

# %%
# Core concepts of the hook System
# ------------------------------------------------------------------------
#
# * :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHookArgs`: A dataclass holding the hook-visible context
#   (accelerator, epoch/step IDs, batch data, and model outputs). Hooks read
#   losses only when the model explicitly publishes them in ``model_outputs``;
#   the backward tensor is never a hook value. The deprecated
#   ``reduced_backward_loss`` constructor input and ``reduce_loss`` alias are
#   retained only to fail with a clear migration error when read.
#
# * :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHooksConfig`: The entire :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHooks` setup,
#   including which individual hooks are active and their parameters, can often be defined via Pydantic configurations.
#   This offers great flexibility and reproducibility. For instance, a main
#   experiment config might point to a :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHooksConfig`,
#   which in turn might specify a list of individual hook configurations (e.g.,
#   :py:class:`~robo_orchard_lab.pipeline.hooks.metric.MetricTrackerConfig`,
#   :py:class:`~robo_orchard_lab.pipeline.hooks.checkpoint.SaveCheckpointConfig`).
#
# ``global_step_id`` and ``step_id`` count completed gradient-accumulation
# windows, including an AMP-skipped parameter update. The separate
# ``is_optimizer_step_committed`` flag reports whether parameters were actually
# updated. Built-in step frequencies use completed windows; scheduler stepping
# retains its optimizer-success condition.
#

# %%
# Some built-in hooks
# ------------------------------------------------------------------------
#
# To streamline common training tasks and reduce boilerplate code, **robo_orchard_lab**
# provides a collection of pre-built hooks. These hooks are implemented using the same
# underlying :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHooks` and
# :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.HookContext` mechanisms discussed
# previously.
#
# Typically, you interact with these built-in hooks by instantiating their
# corresponding Pydantic configuration classes (e.g., ``MetricTrackerConfig``,
# ``StatsMonitorConfig``, ``SaveCheckpointConfig``). These config objects allow you
# to customize the behavior of the hook and are then used to create the actual
# hook instances, which are subsequently added to your main ``HookBasedTrainer``'s
# list of active hooks.
#
# In the following subsections, we will explore how to configure and use some of
# the most common built-in hooks provided by the framework. While these cover
# frequent use cases, remember that you can always create your own custom hooks
# (as shown in the "(Advanced) Creating Your First Custom Hook" section) for more
# specialized requirements.
#

# %%
# LossTracker: Aggregate model-output losses
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :py:class:`~robo_orchard_lab.pipeline.hooks.loss_tracker.LossTracker` reads
# loss entries from ``model_outputs`` after each micro-step. It keeps one local
# detached sum and count per key, then performs a single packed distributed
# reduction only when a completed accumulation window reaches ``step_log_freq``.
# Therefore ``accumulate_grad_steps=1`` remains numerically compatible with the
# old per-update reduction, while multi-micro-step accumulation and AMP-skipped
# updates use the completed-window cadence.
#
# The first update after each logging-window reset defines the complete loss-key
# set. Every later update in that window must provide the same keys on every
# rank; the tracker does not synchronize key presence or infer missing values.
# A logging boundary or an exception resets the window, so the next window may
# use a different complete key set. An AMP skip alone does not reset it. The
# separate backward-loss return value is never read by this hook. A custom
# ``update`` override must call ``super().update`` to contribute to the
# canonical aggregation.

# For a model whose ``model_outputs`` is a loss dictionary, enable the tracker
# with ``LossTrackerConfig(step_log_freq=64)`` and append that config to
# ``hooks``. The ResNet example below returns a tensor, so it intentionally
# does not register this dictionary-loss hook.


# %%
# MetricTracker: Track on performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# :py:class:`~robo_orchard_lab.pipeline.hooks.metric.MetricTracker` is a specialized hook for handling metrics.
# It takes a list of :py:class:`~robo_orchard_lab.pipeline.hooks.metric.MetricEntry` objects. Each MetricEntry defines:
#
# * names: How the metric will be logged.
#
# * metric: An instance of a torchmetrics metric (or any compatible metric object).
#
# :py:class:`~robo_orchard_lab.pipeline.hooks.metric.MetricTracker` is an abstarct class, you should inherit it
# and implement the :py:meth:`~robo_orchard_lab.pipeline.hooks.metric.MetricTracker.update_metric` method, which is called by the trainer to update these metrics with batch outputs and targets.
# Metrics are updated from micro-batch outputs, while ``step_log_freq`` is evaluated on completed accumulation windows.
# Trainer checkpoints do not save ``LossTracker`` or ``MetricTracker``
# aggregation state. When an aggregation interval crosses a restart, logged
# values can cover only post-resume updates until the next reset.
#

from torchmetrics import Accuracy as AccuracyMetric

from robo_orchard_lab.pipeline.hooks import (
    MetricEntry,
    MetricTracker,
    MetricTrackerConfig,
)


class MyMetricTracker(MetricTracker):
    def update_metric(self, batch: Any, model_outputs: Any):
        _, targets = batch
        for metric_i in self.metrics:
            metric_i(model_outputs, targets)


class MyMetricTrackerConfig(MetricTrackerConfig):
    """An example metric tracker config."""

    # note: bind MyMetricTracker
    class_type: type[MyMetricTracker] = MyMetricTracker


metric_tracker = MyMetricTrackerConfig(
    metric_entrys=[
        MetricEntry(
            names=["top1_acc"],
            metric=AccuracyMetric(
                task="multiclass", num_classes=1000, top_k=1
            ),
        ),
        MetricEntry(
            names=["top5_acc"],
            metric=AccuracyMetric(
                task="multiclass", num_classes=1000, top_k=5
            ),
        ),
    ],
    step_log_freq=64,
    log_main_process_only=False,
)

hooks.append(metric_tracker)


# %%
# StatsMonitor: Logging Training Vitals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :py:class:`~robo_orchard_lab.pipeline.hooks.stats.StatsMonitor` publishes
# ``Performance/samples_per_second`` (samples/sec),
# ``Performance/step_time_seconds`` (seconds), and, when the remaining time can
# be estimated, ``Performance/estimated_remaining_seconds`` (seconds). When an
# optimizer is available it also publishes ``LR/group{idx}`` for each parameter
# group. ``step_log_freq`` controls the completed-window cadence for
# these metrics.
#

from robo_orchard_lab.pipeline.hooks import StatsMonitorConfig

stats = StatsMonitorConfig(step_log_freq=64)

hooks.append(stats)

# %%
# SaveCheckpoint: Saving Your Progress
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :py:class:`~robo_orchard_lab.pipeline.hooks.checkpoint.SaveCheckpointConfig` is responsible for triggering model checkpoint saves.
# It calls :py:meth:`~accelerate.Accelerator.save_state` internally.
# ``save_step_freq`` counts completed optimizer-step boundaries, including AMP
# skips. Trainer/data checkpoints do not restore an unfinished gradient
# accumulation window.
# In ``HookBasedTrainer``, an epoch save runs after the epoch hooks complete.
# On natural exhaustion it records the next epoch's starting position; an early
# ``max_step`` stop retains the current epoch and consumed position. Earlier
# checkpoint formats do not identify this boundary reliably, and arbitrary
# user hooks are not guaranteed to execute exactly once across a restart.
# Checkpoints use Accelerate's directory naming and retention behavior. Saving
# is best effort: interruption can leave a partial directory, and automatic
# retention may remove an older checkpoint before the next save completes.
# Choose a checkpoint known to have saved successfully for resume; the latest
# directory name does not prove that its contents are complete.
# If training, a hook or saving fails, checkpoint behavior is undefined; peers
# are not guaranteed to complete checkpoint operations or receive that error.
# The trainer attempts DataLoader cleanup through its normal teardown.
#
# Pass :py:class:`~robo_orchard_lab.pipeline.training.hook_based_trainer.ResumeCheckpointConfig`
# to ``HookBasedTrainer(resume_from=...)`` to restore training and data progress.
# This training entrypoint validates recovery capability and starts a new
# accumulation window after the complete native load. The generic
# ``utils.huggingface.accelerator_load_state`` delegates to Accelerate and
# preserves its loaded phase, even with registered Trainer progress.
# Data recovery applies the saved position once; do not additionally skip
# batches. Keep the rank/worker layout compatible with the checkpoint. A
# resumed accumulation window starts at a boundary; changing the accumulation
# size is allowed but can change the subsequent training trajectory. Increasing
# the training limit is supported, while a limit that is already reached ends
# without processing another batch.
#
# With multi-rank gradient accumulation, configure distributed sampling so
# ranks reach compatible synchronization boundaries. Uneven inputs can make
# ranks synchronize on different batches and hang.
# By default, a short epoch tail completes an accumulation window. If an
# Accelerate gradient accumulation plugin sets ``sync_with_dataloader=False``,
# windows can span epochs and a final unfinished window is not applied.
#
# Data recovery on the normal path is best effort, including ``in_order=False``.
# In supported recovery configurations, a checkpoint without a current
# DataLoader snapshot, such as a native save after training advances, resumes
# by replaying the epoch prefix with a warning; data order and position may
# differ.
# A native save during an unfinished accumulation window cannot restore its
# gradients. Resume warns and starts a new window after the consumed batches.
# Trainer/data resume currently excludes DeepSpeed/FSDP, node-local checkpoint
# layouts, and external stateful loaders. Accelerate restores global RNG and
# invokes registered serialization hooks. After loading, the trainer preserves
# that RNG state across a nonzero data skip before processing the next batch or
# ending the epoch. Random draws by hooks before the first resumed batch can
# also be rolled back; user hooks and random transforms are not guaranteed to
# replay exactly. Resumed tail-batch metrics follow Accelerate's normal
# behavior without a separate correction. Old progress formats require a new
# run from separately loadable model weights.
#

from robo_orchard_lab.pipeline.hooks import SaveCheckpointConfig

save_checkpoint = SaveCheckpointConfig(save_step_freq=1024)

hooks.append(save_checkpoint)

# %%
# Orchestrating the Training
# ------------------------------------------------------------------------
#

from robo_orchard_lab.pipeline import HookBasedTrainer

trainer = HookBasedTrainer(
    model=model,
    dataloader=train_dataloader,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    accelerator=accelerator,
    batch_processor=MyBatchProcessor(need_backward=True),
    max_epoch=cfg.max_epoch,
    hooks=hooks,
)

# %%
# Show hooks
#

print(trainer.hooks)

# %%
# Begin training
#

trainer()


# %%
# All the checkpoints are saved to ``cfg.workspace``

from pathlib import Path

workspace_root = Path(cfg.workspace_root)
for path in sorted(workspace_root.rglob("*")):
    print(path.relative_to(workspace_root))
