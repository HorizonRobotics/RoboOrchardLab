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

"""(Advanced) Creating Custom Hooks for Tailored Logic
==========================================================
In many complex training scenarios, you might need to inject custom logic at various
points within the training loop (e.g., at the beginning/end of an epoch, or before/after
a training step). The **robo_orchard_lab** framework provides a powerful and flexible
Hook system based on :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHooks` to achieve this
without modifying the core training engine.

This tutorial will guide you through:

1. Understanding the core components: :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHooksConfig`, :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHooks`,
:py:class:`~robo_orchard_lab.pipeline.hooks.mixin.HookContext`, and :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHookArgs`.

2. Creating a custom hook class that bundles logging logic for different training stages.

3. Configuring and instantiating your custom hook.

4. (Simulated) Seeing how this hook would interact with a training engine.

Let's get started!
"""

# %%
# Reuse the code from previous tutorial
# ------------------------------------------------------
#

import logging
import os
from typing import Any, Optional, Tuple

import torch

from robo_orchard_lab.utils import log_basic_config

log_basic_config(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    workspace_root="./workspace/tutorial3/",
)

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

hooks = []

# %%
# Implementing a Custom Hook
# ------------------------------------------------------------------------
#
# Implementing a custom hook in ``robo_orchard_lab`` typically involves two key parts:
#
# 1.  **The Hook Implementation Class (e.g., ``MyHook``)**: This is a Python class
#     that inherits from :py:class:`~robo_orchard_lab.pipeline.hooks.mixin.PipelineHooks`.
#     It contains the actual logic that will be executed at different points (channels)
#     in the training/evaluation pipeline. In its ``__init__`` method, it registers
#     its methods (or other callables) to specific channels like "on_loop",
#     "on_epoch", or "on_step" using ``self.register_hook()``, often with the
#     help of ``HookContext.from_callable()``.
#
# 2.  **The Hook Configuration Class (e.g., ``MyHookConfig``)**: This is a Pydantic
#     class that inherits from ``robo_orchard_lab.pipeline.hooks.mixin.PipelineHooksConfig``.
#     Its primary roles are:
#
#     * To specify which Hook Implementation class should be instantiated (typically via a ``class_type`` attribute).
#
#     * To define and validate any parameters that the Hook Implementation needs (e.g., logging frequencies, file paths, thresholds).
#
# **The Relationship and Benefits**:
#
# This separation of implementation (logic) from configuration (parameters) is a
# core design principle that offers several advantages:
#
# * **Configurability & Reusability**: The same hook implementation (e.g., ``MyHook``) can be reused in different experiments with different behaviors simply by
#   providing different configurations (e.g., changing ``log_step_freq`` in ``MyHookConfig``). You don't need to change the Python code of the hook itself.
#
# * **Clarity & Maintainability**: The hook's logic is cleanly encapsulated in its
#   class, while its parameters are explicitly defined and validated by its
#   Pydantic config class.
#
# * **Type Safety**: Pydantic ensures that the parameters passed to your hook are of the correct type and meet any defined validation criteria.
#
# * **Integration with the Framework**: When you instantiate the configuration class
#   (e.g., ``my_hook_cfg = MyHookConfig()``), you get an object that knows how to
#   create the actual hook instance (e.g., by calling ``my_hook_cfg()``, it can
#   instantiate ``MyHook`` and pass itself as the ``cfg`` argument to ``MyHook``'s
#   ``__init__``). This allows the framework to manage hooks declaratively through
#   larger configuration files.
#
# In the following subsections, we will first define the implementation for ``MyHook``
# and then its corresponding configuration class ``MyHookConfig``.
#

# %%
# Defining Your Custom Hook Implementation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's define ``MyHook``. It inherits from ``PipelineHooks``.
# In its ``__init__`` method, it takes its configuration (``MyHookConfig``)
# and registers its own internal methods as callbacks to different channels
# using ``self.register_hook()`` and ``HookContext.from_callable()``.
#
# ``HookContext.from_callable(before=..., after=...)`` is a convenient way to create
# a ``HookContext`` object where its ``on_enter`` method will call the ``before``
# function, and its ``on_exit`` method will call the ``after`` function.
#

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)


class MyHook(PipelineHooks):
    """A custom hook that logs messages at the beginning and end of loops, epochs, and steps, based on configured frequencies."""

    def __init__(self, cfg: "MyHookConfig"):
        super().__init__()
        self.cfg = cfg

        # Register loop-level hooks
        self.register_hook(
            channel="on_loop",
            hook=HookContext.from_callable(
                before=self._on_loop_begin, after=self._on_loop_end
            ),
        )

        # Register step-level hooks
        self.register_hook(
            channel="on_step",
            hook=HookContext.from_callable(
                before=self._on_step_begin, after=self._on_step_end
            ),
        )

        # Register epoch-level hooks
        self.register_hook(
            channel="on_epoch",
            hook=HookContext.from_callable(
                before=self._on_epoch_begin, after=self._on_epoch_end
            ),
        )

        logger.info(
            f"MyHook instance created with step_freq={self.cfg.log_step_freq}, epoch_freq={self.cfg.log_epoch_freq}"
        )

    def _on_loop_begin(self, args: PipelineHookArgs):
        logger.info("Begining loop")

    def _on_loop_end(self, args: PipelineHookArgs):
        logger.info("Ended loop")

    def _on_step_begin(self, args: PipelineHookArgs):
        # Note: step_id is 0-indexed. Adding 1 for 1-indexed frequency check.
        if (args.step_id + 1) % self.cfg.log_step_freq == 0:
            logger.info("Begining {}-th step".format(args.step_id))

    def _on_step_end(self, args: PipelineHookArgs):
        if (args.step_id + 1) % self.cfg.log_step_freq == 0:
            logger.info("Ended {}-th step".format(args.step_id))

    def _on_epoch_begin(self, args: PipelineHookArgs):
        # Note: epoch_id is 0-indexed. Adding 1 for 1-indexed frequency check.
        if (args.epoch_id + 1) % self.cfg.log_epoch_freq == 0:
            logger.info("Begining {}-th epoch".format(args.epoch_id))

    def _on_epoch_end(self, args: PipelineHookArgs):
        if (args.epoch_id + 1) % self.cfg.log_epoch_freq == 0:
            logger.info("Ended {}-th epoch".format(args.epoch_id))


# %%
# Defining Your Custom Hook Configuration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Then, we define a Pydantic configuration class for our custom hook.
# This class will inherit from ``PipelineHooksConfig`` and specify our custom hook
# class as its ``class_type``. It will also hold any parameters our hook needs,
# like logging frequencies.
#


class MyHookConfig(PipelineHooksConfig[MyHook]):
    class_type: type[MyHook] = MyHook
    log_step_freq: int = 5
    log_epoch_freq: int = 1


my_hook = MyHookConfig()

hooks.append(my_hook)


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
