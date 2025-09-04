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
import json
import logging
import os
from multiprocessing import set_start_method

import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, is_initialized
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration
from utils import load_checkpoint, load_config

from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.pipeline import SimpleTrainer
from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor
from robo_orchard_lab.pipeline.hooks import (
    LossMovingAverageTrackerConfig,
    SaveCheckpointConfig,
    StatsMonitorConfig,
)
from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__file__)


class MyBatchProcessor(SimpleBatchProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, model, batch):
        output = model(batch)
        loss = sum([y.mean() for x, y in output.items() if "loss" in x])
        return output, loss


def main(args, accelerator):
    if_cluster = os.environ.get("CLUSTER") is not None
    config = load_config(args.config)
    build_model = config.build_model
    build_dataset = config.build_dataset
    build_optimizer = config.build_optimizer
    build_processor = config.build_processor
    config = config.config

    if args.kwargs is not None:
        if os.path.isfile(args.kwargs):
            kwargs = json.load(open(args.kwargs, "r"))
        else:
            kwargs = json.loads(args.kwargs)
        config.update(kwargs)

    if accelerator.is_main_process:
        logger.info("\n" + json.dumps(config, indent=4))

    processor = build_processor(config)
    with open(
        os.path.join(args.workspace, "robotwin2_0_processor.json"),
        "w",
    ) as fh:
        fh.write(processor.cfg.model_dump_json(indent=4))

    model = build_model(config)
    # save model config
    accelerator.register_save_state_pre_hook(
        model.accelerator_save_state_pre_hook
    )

    train_dataset, _ = build_dataset(config, lazy_init=True)

    num_workers = config.get("num_workers", 4)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch_dict,
        persistent_workers=num_workers > 0,
    )

    optimizer, lr_scheduler = build_optimizer(config, model)
    load_checkpoint(model, config.get("checkpoint"), accelerator)

    trainer = SimpleTrainer(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        grad_clip_mode="norm",
        grad_max_norm=10,
        batch_processor=MyBatchProcessor(need_backward=True),
        hooks=[
            StatsMonitorConfig(
                step_log_freq=config["step_log_freq"],
            ),
            LossMovingAverageTrackerConfig(
                step_log_freq=config["step_log_freq"]
            ),
            SaveCheckpointConfig(
                save_step_freq=config.get("save_step_freq"),
                save_epoch_freq=config.get("save_epoch_freq"),
            ),
        ],
        max_step=config.get("max_step"),
        step_eval_freq=config.get("step_eval_freq"),
        lr_scheduler_step_at="step",
        resume_from=config.get("resume_from"),
        resume_share_dir=(
            "/job_data/resume_from" if if_cluster else "./resume_from"
        ),
    )

    trainer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument(
        "--config", type=str, default="./config_sem_robotwin.py"
    )
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()

    workspace_root = args.workspace
    os.makedirs(workspace_root, exist_ok=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(
            project_dir=workspace_root,
            logging_dir=os.path.join(workspace_root, "logs"),
            automatic_checkpoint_naming=True,
            total_limit=3,
        ),
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
        ),
    )
    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    logger.info(f"if accelerator initialized:{is_initialized()}")
    logger.info(f"accelerator state: {AcceleratorState._shared_state}")
    set_start_method("spawn", force=True)
    main(args, accelerator)
