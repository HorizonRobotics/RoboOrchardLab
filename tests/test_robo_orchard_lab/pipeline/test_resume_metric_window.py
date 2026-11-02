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

"""Exercise the documented metric window boundary with real checkpoints."""

import logging

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader

from robo_orchard_lab.pipeline.hooks.checkpoint import SaveCheckpointConfig
from robo_orchard_lab.pipeline.hooks.loss_tracker import LossTrackerConfig
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    HookBasedTrainer,
    ResumeCheckpointConfig,
)
from robo_orchard_lab.processing.step_processor import SimpleStepProcessor


class _LossFromBatch(SimpleStepProcessor):
    def forward(self, model, batch):
        loss = model(batch).sum() * 0 + batch.mean()
        return {"loss": loss}, loss


@pytest.mark.parametrize("moving_average", [True, False])
def test_resume_reports_partial_metric_windows(
    tmp_path, caplog, moving_average, monkeypatch
):
    """A reset ends the partial interval; names and boundaries stay stable."""
    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})

    def run(stop, *, save=False, resume=None):
        accelerator = Accelerator(
            cpu=True,
            project_config=ProjectConfiguration(
                project_dir=str(tmp_path), automatic_checkpoint_naming=True
            ),
            step_scheduler_with_optimizer=False,
        )
        logs = []
        accelerator.log = lambda values, step=None: logs.append((step, values))
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        hooks = [
            LossTrackerConfig(step_log_freq=5, moving_average=moving_average)
        ]
        if save:
            hooks.append(
                SaveCheckpointConfig(
                    save_step_freq=2,
                    save_epoch_freq=None,
                    save_when_loop_end=False,
                    save_model=False,
                )
            )
        trainer = HookBasedTrainer(
            accelerator,
            model,
            DataLoader(torch.arange(1.0, 12.0).reshape(-1, 1), batch_size=1),
            _LossFromBatch(),
            optimizer,
            scheduler,
            hooks=hooks,
            max_step=stop,
            resume_from=resume,
        )
        trainer()
        return {step: values["Loss/loss"] for step, values in logs}

    with caplog.at_level(logging.WARNING):
        run(2, save=True)
        assert not any(
            "aggregation state" in r.message for r in caplog.records
        )
        checkpoint = tmp_path / "checkpoints" / "checkpoint_0"
        resumed = run(
            10, resume=ResumeCheckpointConfig(resume_from=str(checkpoint))
        )
        warnings = [
            r.message
            for r in caplog.records
            if "aggregation state" in r.message
        ]
        control = run(10)

    assert len(warnings) == 1
    assert "resume" in warnings[0].lower()
    assert resumed[5] == pytest.approx(4.0 if moving_average else 5.0)
    assert control[5] == pytest.approx(3.0 if moving_average else 5.0)
    assert resumed[10] == pytest.approx(control[10])
