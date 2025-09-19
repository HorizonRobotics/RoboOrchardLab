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
import importlib
import json
import logging
import os
from datetime import timedelta
from multiprocessing import set_start_method

import requests
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, is_initialized
from accelerate.utils import (
    DataLoaderConfiguration,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)
from safetensors.torch import load_model

from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.dataset.embodiedscan.metrics import DetMetric
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


def load_config(config_file):
    assert config_file.endswith(".py")
    module_name = os.path.split(config_file)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class GetFile:
    def __init__(self, url):
        self.url = url

    def __enter__(self):
        if not self.url.startswith("http"):
            return self.url
        file_name = "_" + self.url.split("/")[-1]
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            with open(file_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        self.url = file_name
        return file_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def load_checkpoint(model, checkpoint=None, accelerator=None, **kwargs):
    if checkpoint is None:
        return

    logger.info(f"load checkpoint: {checkpoint}")
    with GetFile(checkpoint) as checkpoint:
        if checkpoint.endswith(".safetensors"):
            missing_keys, unexpected_keys = load_model(
                model, checkpoint, **kwargs
            )
        else:
            state_dict = torch.load(checkpoint, weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False, **kwargs
            )
        if accelerator is None or accelerator.is_main_process:
            logger.info(
                f"num of missing_keys: {len(missing_keys)},"
                f"num of unexpected_keys: {len(unexpected_keys)}"
            )
            logger.info(
                f"missing_keys:\n {missing_keys}\n"
                f"unexpected_keys:\n {unexpected_keys}"
            )


def main(args, accelerator):
    if_cluster = os.environ.get("CLUSTER") is not None
    config = load_config(args.config)
    build_model = config.build_model
    build_dataset = config.build_dataset
    build_optimizer = config.build_optimizer
    config = config.config

    if args.kwargs is not None:
        if os.path.isfile(args.kwargs):
            kwargs = json.load(open(args.kwargs, "r"))
        else:
            kwargs = json.loads(args.kwargs)
        config.update(kwargs)
    if args.eval_only:
        config["eval_only"] = True

    if accelerator.is_main_process:
        logger.info("\n" + json.dumps(config, indent=4))

    model = build_model(config)
    train_dataset, val_dataset = build_dataset(config, lazy_init=True)

    num_workers = config.get("num_workers", 4)
    if not config.get("eval_only"):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch_dict,
            persistent_workers=num_workers > 0,
        )
    else:
        train_dataloader = None
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_batch_dict,
        persistent_workers=False,
    )

    optimizer, lr_scheduler = build_optimizer(config, model)
    load_checkpoint(model, config.get("checkpoint"), accelerator)

    trainer = SimpleTrainer(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
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
        metric=DetMetric(gather_device="cpu"),
        max_step=config.get("max_step"),
        max_epoch=config.get("max_epoch"),
        step_eval_freq=config.get("step_eval_freq"),
        epoch_eval_freq=config.get("epoch_eval_freq"),
        lr_scheduler_step_at="step",
        resume_from=config.get("resume_from"),
        resume_share_dir=(
            "/job_data/resume_from" if if_cluster else "./resume_from"
        ),
    )

    if config.get("eval_only"):
        trainer.eval()
    else:
        trainer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--config", type=str, default="./config_bip3d_det.py")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()

    workspace_root = args.workspace
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
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ],
    )
    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    logger.info(f"if accelerator initialized:{is_initialized()}")
    logger.info(f"accelerator state: {AcceleratorState._shared_state}")
    set_start_method("spawn", force=True)
    main(args, accelerator)
