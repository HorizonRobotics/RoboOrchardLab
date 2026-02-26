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
import sys
from multiprocessing import set_start_method

from accelerate import Accelerator
from accelerate.state import AcceleratorState, is_initialized
from accelerate.utils import (
    DataLoaderConfiguration,
    ProjectConfiguration,
)

from robo_orchard_lab.pipeline import SimpleTrainer
from robo_orchard_lab.pipeline.batch_processor import SimpleBatchProcessor
from robo_orchard_lab.pipeline.hooks import (
    LossMovingAverageTrackerConfig,
    SaveCheckpointConfig,
    StatsMonitorConfig,
)
from robo_orchard_lab.utils import log_basic_config, seed_everything

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


logger = logging.getLogger(__file__)


def load_config(config_file):
    assert config_file.endswith(".py")
    module_name = os.path.split(config_file)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class MyBatchProcessor(SimpleBatchProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, model, batch):
        model_outputs = model(batch)
        loss = sum([y for x, y in model_outputs.items() if "loss" in x])

        return model_outputs, loss


def main(args, accelerator):
    config = load_config(args.config)

    get_trainer_cfg = config.get_trainer_cfg
    trainer_cfg = get_trainer_cfg()

    # load config from param
    trainer_cfg = trainer_cfg.model_copy(update=vars(args))
    if args.kwargs is not None:
        kwargs = json.loads(args.kwargs)
        trainer_cfg = trainer_cfg.model_copy(update=kwargs)

    # saving config to directory
    os.system(f"cp {args.config} {args.workspace_root}/")
    with open(os.path.join(args.workspace_root, "config.json"), "w") as f:
        json.dump(trainer_cfg.model_dump(), f, indent=4)

    if accelerator.is_main_process:
        logger.info("\n" + json.dumps(trainer_cfg.model_dump(), indent=4))

    build_dataset = config.build_dataset
    build_model = config.build_model
    build_optimizer = config.build_optimizer
    build_metric = config.build_metric
    build_processor = config.build_processor

    processor = build_processor(config)
    with open(
        os.path.join(args.workspace, "finegrasp_processor.json"),
        "w",
    ) as fh:
        fh.write(processor.cfg.model_dump_json(indent=4))

    (
        train_data_cfgs,
        val_data_cfgs,
        train_dataloader,
        val_dataloader,
    ) = build_dataset(trainer_cfg)

    if trainer_cfg.max_step is None:
        world_size = accelerator.num_processes
        steps_per_epoch = len(train_dataloader) // world_size
        trainer_cfg.world_size = world_size
        trainer_cfg.max_step = trainer_cfg.max_epoch * steps_per_epoch

    model_cfgs, model = build_model()
    optim_cfgs, optimizer, lr_scheduler = build_optimizer(model, trainer_cfg)

    eval_save_dir = os.path.join(
        args.workspace_root,
        "eval_results",
        model_cfgs.model_name,
        train_data_cfgs.camera,
    )
    metric_cfgs, metric = build_metric(model_cfgs, trainer_cfg, eval_save_dir)
    trainer = SimpleTrainer(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        batch_processor=MyBatchProcessor(need_backward=True),
        hooks=[
            StatsMonitorConfig(step_log_freq=trainer_cfg.step_log_freq),
            LossMovingAverageTrackerConfig(
                step_log_freq=trainer_cfg.step_log_freq
            ),
            SaveCheckpointConfig(
                save_step_freq=trainer_cfg.save_step_freq,
                save_epoch_freq=trainer_cfg.save_epoch_freq,
            ),
        ],
        metric=metric,
        max_step=trainer_cfg.max_step,
        max_epoch=trainer_cfg.max_epoch,
        lr_scheduler_step_at=trainer_cfg.lr_scheduler_step_at,
        step_eval_freq=trainer_cfg.step_eval_freq,
        epoch_eval_freq=trainer_cfg.epoch_eval_freq,
        resume_from=trainer_cfg.resume_from,
        resume_share_dir=trainer_cfg.resume_share_dir,
        grad_clip_mode=trainer_cfg.grad_clip_mode,
        grad_clip_value=trainer_cfg.grad_clip_value,
        grad_max_norm=trainer_cfg.grad_max_norm,
        grad_norm_type=trainer_cfg.grad_norm_type,
    )

    if args.eval_only:
        trainer.eval()
    else:
        trainer()
        trainer.eval()


if __name__ == "__main__":
    seed_everything(2025)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_finegrasp_minkunet.py",
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--workspace_root", type=str, default="./workspace")
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()

    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.workspace_root, "logs")

    os.makedirs(args.workspace_root, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    accelerator = Accelerator(
        log_with="tensorboard",
        step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(
            project_dir=args.workspace_root,
            logging_dir=args.logging_dir,
            automatic_checkpoint_naming=True,
            total_limit=32,
        ),
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
        ),
    )
    accelerator.init_trackers("tensorboard")
    logger.info(f"Save config to workspace dir {args.workspace_root}")

    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa E501
        level=logging.INFO,
    )
    set_start_method("spawn", force=True)
    logger.info(f"if accelerator initialized:{is_initialized()}")
    logger.info(f"accelerator state: {AcceleratorState._shared_state}")
    main(args, accelerator)
