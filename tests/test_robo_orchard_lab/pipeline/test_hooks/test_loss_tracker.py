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


def test_loss_tracker_reduces_detached_model_output_loss():
    """LossTracker should keep logging reductions out of the autograd graph."""

    tracker = LossTrackerConfig()()
    accelerator = MagicMock()
    reduce_calls: list[tuple[torch.Tensor, str, bool]] = []

    def fake_reduce(loss: torch.Tensor, reduction: str = "mean"):
        reduce_calls.append((loss.clone(), reduction, loss.requires_grad))
        return loss

    accelerator.reduce.side_effect = fake_reduce
    model_outputs = {"loss": torch.tensor(4.0, requires_grad=True)}

    tracker.update(accelerator=accelerator, model_outputs=model_outputs)

    assert len(reduce_calls) == 1
    assert torch.equal(reduce_calls[0][0], torch.tensor(4.0))
    assert reduce_calls[0][1] == "mean"
    assert reduce_calls[0][2] is False
    assert tracker.cached_loss["loss"] == 4.0
