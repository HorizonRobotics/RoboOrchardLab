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
