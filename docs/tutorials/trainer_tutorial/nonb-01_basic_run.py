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


"""Configuration & Your First Basic Training Run
==========================================================
"""


# %%
# Prerequest: Import packages and configure logging
# ----------------------------------------------------------
#

import logging
import os
from typing import Any, Optional, Tuple

import torch

from robo_orchard_lab.utils import log_basic_config

log_basic_config(level=logging.INFO)
logger = logging.getLogger(__name__)


# %%
# Using pydantic for configuration
# ---------------------------------------------------------
# Our framework emphasizes configuration-driven training. Let's look at DatasetConfig
#

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
        default=1024,
    )

    dummy_val_imgs: int = Field(
        description="Number of dummy validation images.",
        default=256,
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
                (3, 224, 224),
                1000,
                transforms.ToTensor(),
            )
            val_dataset = datasets.FakeData(
                self.dummy_val_imgs, (3, 224, 224), 1000, transforms.ToTensor()
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


# %%
# DatasetConfig uses Pydantic Field to define parameters like data_root and pipeline_test.
# The pipeline_test flag is crucial. If True, it uses FakeData for a quick test run without needing the actual ImageNet dataset.
# get_dataset() method encapsulates the logic for creating train and validation datasets.

# %%
# TrainerConfig, which nests DatasetConfig:


class TrainerConfig(SettingConfig):
    """Configuration for the trainer.

    This is an example configuration for training a ResNet50 model
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
    max_epoch=5,
    num_workers=0,
    workspace_root="./workspace/tutorial1/",
)


# %%
# These Pydantic models can also parsed from command-line arguments using ``pydantic_from_argparse(TrainerConfig, parser)``. This means you can override any setting from the CLI, e.g., ``--batch_size 64`` or even nested ones like ``--dataset.data_root /path/to/my/data``.

# %%
# Understanding the Core trainer Components
# ---------------------------------------------------------
# Let's dive into the specific components from **robo_orchard_lab** that make this work.
#

# %%
# Accelerator: The Engine for Scale and Simplicity
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Accelerator from Hugging Face handles:
#
# * Device Agnosticism: Runs your code on CPU, single GPU, multiple GPUs, or TPUs with minimal changes.
#
# * Distributed Training: Automatically sets up distributed training if multiple GPUs are available.
#
# * Mixed Precision: Can enable AMP for faster training and reduced memory.
#
# * Checkpointing: Accelerator manage saving and loading checkpoints in an organized way.
#

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

accelerator = Accelerator(
    project_config=ProjectConfiguration(
        project_dir=cfg.workspace_root,
        logging_dir=os.path.join(cfg.workspace_root, "logs"),
        automatic_checkpoint_naming=True,
        total_limit=32,  # Max checkpoints to keep
    )
)


# %%
# Batch processor: Defining Per-Step Logic
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Instead of hard coding the forward pass and loss calculation within the trainer, we use a BatchProcessor.
# MyBatchProcessor inherits from :py:class:`~robo_orchard_lab.pipeline.batch_processor.simple.SimpleBatchProcessor`.
#
# You define your loss function (here, CrossEntropyLoss) in ``__init__``.
# ``forward`` contains the logic for a single step: getting model output and calculating loss.
# The need_backward flag allows this processor to be used for evaluation loops where backpropagation isn't needed.
#
# .. note::
#   Accelerator handles moving the model and batch to the correct device before forward is called by the trainer.
#

from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor


class MyBatchProcessor(SimpleBatchProcessor):
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


# %%
# Hooks: Customizing your pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Hooks are a powerful feature for injecting custom behaviors into the
# training or evaluation loop at predefined stages (e.g., end of an epoch,
# after a step), all without modifying the core engine code.
#
# This tutorial provides a brief introduction, demonstrating how to use the
# built-in :py:class:`~robo_orchard_lab.pipeline.hooks.stats.StatsMonitor` for basic logging.
# For a comprehensive guide on the hook system and creating custom hooks,
# please refer to the next tutorial in this series.
#

hooks = []

from robo_orchard_lab.pipeline.hooks import StatsMonitorConfig

stats = StatsMonitorConfig(step_log_freq=64)

hooks.append(stats)

# %%
# DataLoader, model, optimizer and learning rate scheduler
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

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

model = models.resnet50()

optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# %%
# Orchestrating the Training
# ---------------------------------------------------------
#
# :py:class:`~robo_orchard_lab.pipeline.hook_based_trainer.HookBasedTrainer` is the heart of your training loop.
# It takes all necessary PyTorch components (model, dataloader, optimizer, scheduler) and crucially,
# the accelerator and a batch_processor. The hooks list allows for powerful customization, which we have explored before.
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
# Begin training
#

trainer()
