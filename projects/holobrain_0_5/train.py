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


"""Train HoloBrain 0.5 on local RoboTwin or LIBERO LMDB data.

Run from the repository root with ``python -m projects.holobrain_0_5.train``.
The shared trainer owns device preparation and optimizer execution.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from projects.holobrain_0_5.configs.dataset_factory import collate_action_batch
from projects.holobrain_0_5.utils import (
    load_checkpoint,
    load_config,
    parse_kwargs,
)
from robo_orchard_lab.dataset.dataset_wrapper import (
    DistributedBatchFlagSampler,
)
from robo_orchard_lab.models.holobrain_0_5.pipeline import (
    HoloBrainInferencePipeline,
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.pipeline import SimpleTrainer
from robo_orchard_lab.pipeline.hooks import (
    LossTrackerConfig,
    SaveCheckpointConfig,
    StatsMonitorConfig,
)
from robo_orchard_lab.processing.step_processor import SimpleStepProcessor
from robo_orchard_lab.utils import log_basic_config

__all__ = ["train"]

logger = logging.getLogger(__name__)


class ActionStepProcessor(SimpleStepProcessor):
    """Reduce action losses and reject non-finite updates."""

    def forward(self, model, batch):
        output = model(batch)
        loss = sum(
            value.mean() for name, value in output.items() if "loss" in name
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite action training loss")
        return output, loss


def train(config_path, overrides, workspace, seed=0, logging_dir=None):
    """Train for max_step optimizer updates and export the resulting policy."""
    module = load_config(config_path)
    config = dict(module.config)
    config.update(overrides)
    set_seed(seed)
    output = Path(workspace).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = Path(logging_dir).resolve() if logging_dir else output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        log_with="tensorboard",
        # The sampler already yields per-rank batches. Keep the LR scheduler
        # in optimizer-update units instead of advancing once per process.
        step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(
            project_dir=str(output),
            logging_dir=str(logs),
            automatic_checkpoint_naming=True,
            total_limit=config.get("total_limit", 3),
        ),
    )
    accelerator.init_trackers("tensorboard")
    # After Accelerator: %rank reads the process group, which is only
    # initialized once the Accelerator exists.
    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        logger.info("\n" + json.dumps(config, indent=4))
        shutil.copytree(
            Path(config_path).resolve().parent,
            output / "configs",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (output / "training_config.json").write_text(
            json.dumps(config, indent=2)
        )
        logger.info("Saved config to workspace dir %s", output)
    model = module.build_model(config)
    load_checkpoint(model, config.get("checkpoint"))

    trainable_param = 0
    non_trainable_param = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_param += param.numel()
        else:
            non_trainable_param += param.numel()
    total_param = trainable_param + non_trainable_param
    logger.info(
        f"number of parameters: {total_param / 10**6:.2f}M, "
        f"trainable: {trainable_param / 10**6:.2f}M, "
        f"non-trainable: {non_trainable_param / 10**6:.2f}M"
    )

    accelerator.register_save_state_pre_hook(
        model.accelerator_save_state_pre_hook
    )
    # Before training: a job that is cut short still leaves a usable export.
    if accelerator.is_main_process:
        for name, processor in module.build_processors(config).items():
            processor.save(str(output), f"{name}_processor.json")
            pipeline = HoloBrainInferencePipeline(
                HoloBrainInferencePipelineCfg(processor=processor.cfg), model
            )
            pipeline.save_pipeline(
                str(output),
                inference_prefix=f"{name}_inference",
                required_empty=False,
                save_model=False,
            )
            logger.info("Exported %s_processor.json", name)
    dataset = module.build_training_dataset(config)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config["num_workers"],
        collate_fn=collate_action_batch,
        batch_sampler=DistributedBatchFlagSampler(
            dataset, config["batch_size"], drop_last=True
        ),
    )
    optimizer, scheduler = module.build_optimizer(config, model)
    trainer = SimpleTrainer(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        accelerator=accelerator,
        grad_clip_mode="norm",
        grad_max_norm=10,
        batch_processor=ActionStepProcessor(need_backward=True),
        hooks=[
            StatsMonitorConfig(
                step_log_freq=config["step_log_freq"],
                # Inference from the dataloader would under-report by
                # num_processes: DistributedBatchFlagSampler already yields
                # per-rank batches, so accelerate leaves it unwrapped and
                # DataLoaderShard.total_batch_size misses the process factor.
                batch_size=config["batch_size"],
            ),
            LossTrackerConfig(
                step_log_freq=config["step_log_freq"], log_total_loss=True
            ),
            SaveCheckpointConfig(
                save_step_freq=config.get("save_step_freq"),
                save_epoch_freq=None,
            ),
        ],
        max_step=config["max_step"],
        lr_scheduler_step_at="step",
        resume_from=config.get("resume_from"),
    )
    trainer()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(trainer.model)
        model.save_model(str(output), required_empty=False)
        completed_steps = trainer.trainer_progress_state.global_step_id
        (output / "training_result.json").write_text(
            json.dumps(
                {"optimizer_steps": completed_steps},
                indent=2,
            )
        )
    # Last: end_training destroys the process group, so every collective
    # above must already be done.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs/config_holobrain_0_5.py"),
    )
    parser.add_argument("--kwargs")
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--logging_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(
        args.config,
        parse_kwargs(args.kwargs),
        args.workspace,
        args.seed,
        args.logging_dir,
    )
