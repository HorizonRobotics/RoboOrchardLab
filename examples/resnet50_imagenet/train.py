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

import argparse
import logging
import os
from multiprocessing import set_start_method
from typing import Any, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from pydantic import Field
from robo_orchard_core.utils.cli import SettingConfig, pydantic_from_argparse
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy as AccuracyMetric
from torchvision import datasets, models, transforms

from robo_orchard_lab.pipeline import HookBasedTrainer
from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor
from robo_orchard_lab.pipeline.hooks import (
    MetricEntry,
    MetricTracker,
    MetricTrackerConfig,
    SaveCheckpointConfig,
    StatsMonitorConfig,
)
from robo_orchard_lab.utils import log_basic_config
from robo_orchard_lab.utils.huggingface import (
    get_accelerate_project_last_checkpoint_id,
)

logger = logging.getLogger(__file__)


class MyMetricTracker(MetricTracker):
    """An example metric tracker."""

    def update_metric(self, batch: Any, model_outputs: Any):
        _, targets = batch
        for metric_i in self.metrics:
            metric_i(model_outputs, targets)


class MyMetricTrackerConfig(MetricTrackerConfig):
    """An example metric tracker config."""

    class_type: type[MetricTracker] = MyMetricTracker


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
        default=1281167,
    )

    dummy_val_imgs: int = Field(
        description="Number of dummy validation images.",
        default=50000,
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


class MyBatchProcessor(SimpleBatchProcessor):
    """A simple example for a batch processor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        model: torch.nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        images, target = batch

        output = model(images)

        if self.need_backward:
            loss = self.criterion(output, target)
        else:
            loss = None

        return output, loss


def main(cfg: TrainerConfig):
    train_dataset, val_dataset = cfg.dataset.get_dataset()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    _ = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    model = models.resnet50()

    optimizer = SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    workspace_root = cfg.workspace_root
    last_ckpt_iteration = get_accelerate_project_last_checkpoint_id(
        workspace_root
    )

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(
            project_dir=workspace_root,
            logging_dir=os.path.join(workspace_root, "logs"),
            automatic_checkpoint_naming=True,
            total_limit=32,
            iteration=last_ckpt_iteration + 1,
        ),
    )

    trainer = HookBasedTrainer(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=MyBatchProcessor(need_backward=True),
        max_epoch=cfg.max_epoch,
        hooks=[
            MyMetricTrackerConfig(
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
            ),
            StatsMonitorConfig(step_log_freq=64),
            SaveCheckpointConfig(save_step_freq=1024),
        ],
    )
    print(trainer.hooks)
    trainer()


if __name__ == "__main__":
    log_basic_config(level=logging.INFO)
    set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    try:
        args: TrainerConfig = pydantic_from_argparse(TrainerConfig, parser)
    except SystemExit as e:
        # Handle the case where the script is run with --help
        if e.code == 2:
            parser.print_help()
        exit(0)

    logger.info(
        "Starting training with config: %s",
        args.to_str(format="json", indent=2),
    )
    main(args)
