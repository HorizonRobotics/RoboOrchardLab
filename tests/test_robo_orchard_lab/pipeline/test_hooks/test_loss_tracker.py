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

from unittest.mock import MagicMock

import pytest
import torch

from robo_orchard_lab.pipeline.hooks.loss_tracker import LossTrackerConfig
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs


def test_loss_tracker_tracks_detached_reduced_model_output_loss():
    """LossTracker should reduce detached loss values for logging."""

    tracker = LossTrackerConfig()()
    accelerator = MagicMock()
    accelerator.reduce.return_value = torch.tensor(2.0)
    model_outputs = {"loss": torch.tensor(4.0, requires_grad=True)}

    tracker.update(accelerator=accelerator, model_outputs=model_outputs)

    accelerator.reduce.assert_called_once()
    reduced_arg = accelerator.reduce.call_args.args[0]
    assert torch.equal(reduced_arg, torch.tensor(4.0))
    assert reduced_arg.requires_grad is False
    assert accelerator.reduce.call_args.kwargs == {"reduction": "mean"}
    assert tracker.cached_loss["loss"] == 2.0


def test_loss_tracker_skips_reduce_on_context_exception():
    tracker = LossTrackerConfig()()
    accelerator = MagicMock()
    args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(4.0, requires_grad=True)},
        exception=RuntimeError("body failed"),
    )

    tracker._on_step_end(args)

    accelerator.reduce.assert_not_called()
    assert tracker.cached_loss == {}


def test_loss_tracker_logs_on_committed_optimizer_step(mocker):
    """LossTracker should log after committed optimizer steps."""

    tracker = LossTrackerConfig(step_log_freq=2, log_total_loss=True)()
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.reduce.side_effect = [torch.tensor(2.0), torch.tensor(4.0)]
    mock_logger = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.loss_tracker.logger"
    )

    first_step_args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(4.0, requires_grad=True)},
        global_step_id=1,
        step_id=1,
        max_step=2,
        max_epoch=1,
    )
    with tracker.begin("on_step", first_step_args):
        pass

    first_commit_args = first_step_args.copy_with_updates(
        is_optimizer_step_committed=True
    )
    with tracker.begin("on_optimizer_step", first_commit_args):
        pass

    assert tracker.cached_loss["loss"] == 2.0
    accelerator.log.assert_not_called()

    second_step_args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(8.0, requires_grad=True)},
        global_step_id=2,
        step_id=2,
        max_step=2,
        max_epoch=1,
    )
    with tracker.begin("on_step", second_step_args):
        pass

    second_commit_args = second_step_args.copy_with_updates(
        is_optimizer_step_committed=True
    )
    with tracker.begin("on_optimizer_step", second_commit_args):
        pass

    accelerator.log.assert_any_call({"Loss/loss": 3.0}, step=2)
    accelerator.log.assert_any_call({"Loss/Total_Loss": 3.0}, step=2)
    assert tracker.cached_loss == {}
    mock_logger.info.assert_called_once()
    assert (
        mock_logger.info.call_args.args[0]
        == "Epoch[0/0] Step[2] GlobalStep[2/2]: "
        "loss[3.0000]\ttotal_loss[3.0000]"
    )


def test_loss_tracker_discards_uncommitted_optimizer_window():
    """Skipped optimizer boundaries should discard pending loss values."""

    tracker = LossTrackerConfig(step_log_freq=1)()
    accelerator = MagicMock()
    accelerator.reduce.return_value = torch.tensor(2.0)
    args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(4.0, requires_grad=True)},
    )

    with tracker.begin("on_step", args):
        pass
    assert tracker.cached_loss["loss"] == pytest.approx(2.0)

    with tracker.begin("on_optimizer_step", args):
        pass

    assert tracker.cached_loss == {}
