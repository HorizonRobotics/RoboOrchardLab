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

from robo_orchard_lab.pipeline.hooks.loss_tracker import (
    LossTracker,
    LossTrackerConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs


def test_loss_tracker_caches_detached_model_output_loss_locally():
    """Only model-output losses should enter the local logging cache."""

    tracker = LossTrackerConfig()()
    model_outputs = {"loss": torch.tensor(4.0, requires_grad=True)}

    tracker.update(model_outputs)

    assert torch.equal(tracker._cached_loss_sums["loss"], torch.tensor(4.0))
    assert tracker._cached_loss_sums["loss"].requires_grad is False
    assert tracker._cached_loss_counts == {"loss": 1}


def test_loss_tracker_legacy_update_ignores_accelerator():
    """The old direct update form must not restore early reduction."""

    tracker = LossTrackerConfig()()
    accelerator = MagicMock()
    model_outputs = {"loss": torch.tensor(4.0, requires_grad=True)}

    tracker.update(accelerator, model_outputs)

    accelerator.reduce.assert_not_called()
    assert torch.equal(tracker._cached_loss_sums["loss"], torch.tensor(4.0))
    assert tracker._cached_loss_counts == {"loss": 1}


def test_loss_tracker_hook_preserves_legacy_update_override():
    """The hook still passes the accelerator to legacy tracker subclasses."""

    class LegacyLossTracker(LossTracker):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.received_accelerator = None

        def update(self, accelerator, model_outputs):
            self.received_accelerator = accelerator
            return super().update(accelerator, model_outputs)

    tracker = LegacyLossTracker(LossTrackerConfig())
    accelerator = MagicMock()
    args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(4.0)},
    )

    tracker._on_step_end(args)

    assert tracker.received_accelerator is accelerator
    assert tracker._cached_loss_counts == {"loss": 1}


@pytest.mark.parametrize("shape", [(1,), (1, 1)])
def test_loss_tracker_scalarizes_singleton_loss_tensors(shape):
    """One-element tensors are packed as scalar sum/count pairs."""

    tracker = LossTrackerConfig(step_log_freq=1)()
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.reduce.return_value = torch.tensor([[4.0, 1.0]])
    args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.full(shape, 4.0)},
        global_step_id=1,
        step_id=1,
        max_step=1,
        max_epoch=1,
    )

    with tracker.begin("on_step", args):
        pass
    with tracker.begin(
        "on_optimizer_step",
        args.copy_with_updates(is_optimizer_step_committed=True),
    ):
        pass

    assert tracker._cached_loss_sums == {}
    accelerator.reduce.assert_called_once()
    assert accelerator.reduce.call_args.args[0].shape == (1, 2)
    accelerator.log.assert_called_once_with({"Loss/loss": 4.0}, step=1)


def test_loss_tracker_legacy_update_accepts_keyword_arguments():
    """The former keyword call remains accepted during migration."""

    tracker = LossTrackerConfig()()
    accelerator = MagicMock()

    tracker.update(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(4.0)},
    )

    assert tracker._cached_loss_counts == {"loss": 1}


@pytest.mark.parametrize(
    "next_outputs",
    [
        {"loss": torch.tensor(2.0)},
        {"aux_loss": torch.tensor(2.0)},
    ],
)
def test_loss_tracker_rejects_loss_key_changes_within_window(next_outputs):
    """Every micro step must publish the same complete key set."""

    tracker = LossTrackerConfig()()
    tracker.update(
        {
            "loss": torch.tensor(1.0),
            "aux_loss": torch.tensor(3.0),
        }
    )

    with pytest.raises(ValueError, match="stable within a logging window"):
        tracker.update(next_outputs)

    assert tracker._cached_loss_sums == {}
    assert tracker._cached_loss_counts == {}
    assert tracker._expected_loss_keys is None


def test_loss_tracker_reinfers_loss_keys_after_window_reset():
    """A new logging window may establish a different complete key set."""

    tracker = LossTrackerConfig(step_log_freq=1)()
    accelerator = MagicMock()
    accelerator.is_main_process = False
    accelerator.reduce.return_value = torch.tensor([[1.0, 1.0]])
    first_args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(1.0)},
        global_step_id=1,
        is_optimizer_step_committed=True,
    )

    with tracker.begin("on_step", first_args):
        pass
    with tracker.begin("on_optimizer_step", first_args):
        pass

    tracker.update({"aux_loss": torch.tensor(2.0)})
    assert tracker._expected_loss_keys == frozenset({"aux_loss"})


def test_loss_tracker_skips_reduce_on_context_exception():
    tracker = LossTrackerConfig()()
    accelerator = MagicMock()
    tracker.update({"loss": torch.tensor(1.0)})
    args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(4.0, requires_grad=True)},
        exception=RuntimeError("body failed"),
    )

    tracker._on_step_end(args)

    accelerator.reduce.assert_not_called()
    assert tracker._cached_loss_sums == {}
    assert tracker._cached_loss_counts == {}
    assert tracker._expected_loss_keys is None


def test_loss_tracker_logs_on_committed_optimizer_step(mocker):
    """LossTracker should log after committed optimizer steps."""

    tracker = LossTrackerConfig(step_log_freq=2, log_total_loss=True)()
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.reduce.return_value = torch.tensor([[20.0, 4.0]])
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

    accelerator.reduce.assert_not_called()
    assert torch.equal(tracker._cached_loss_sums["loss"], torch.tensor(4.0))
    assert tracker._cached_loss_counts == {"loss": 1}
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

    accelerator.reduce.assert_called_once()
    local_stats = accelerator.reduce.call_args.args[0]
    assert torch.equal(local_stats, torch.tensor([[12.0, 2.0]]))
    assert accelerator.reduce.call_args.kwargs == {"reduction": "sum"}
    accelerator.log.assert_any_call({"Loss/loss": 5.0}, step=2)
    accelerator.log.assert_any_call({"Loss/Total_Loss": 5.0}, step=2)
    assert tracker._cached_loss_sums == {}
    assert tracker._cached_loss_counts == {}
    mock_logger.info.assert_called_once()
    assert (
        mock_logger.info.call_args.args[0]
        == "Epoch[0/0] Step[2] GlobalStep[2/2]: "
        "loss[5.0000]\ttotal_loss[5.0000]"
    )


def test_loss_tracker_discards_uncommitted_optimizer_window():
    """Skipped optimizer boundaries should discard pending loss values."""

    tracker = LossTrackerConfig(step_log_freq=1)()
    accelerator = MagicMock()
    args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={"loss": torch.tensor(4.0, requires_grad=True)},
    )

    with tracker.begin("on_step", args):
        pass
    assert torch.equal(tracker._cached_loss_sums["loss"], torch.tensor(4.0))
    assert tracker._cached_loss_counts == {"loss": 1}

    with tracker.begin("on_optimizer_step", args):
        pass

    accelerator.reduce.assert_not_called()
    assert tracker._cached_loss_sums == {}
    assert tracker._cached_loss_counts == {}


def test_loss_tracker_discards_window_when_reduction_fails():
    """A failed logging collective must not leave stale local state."""

    tracker = LossTrackerConfig(step_log_freq=1)()
    accelerator = MagicMock()
    accelerator.reduce.side_effect = RuntimeError("collective failed")
    tracker.update({"loss": torch.tensor(4.0)})

    args = PipelineHookArgs(
        accelerator=accelerator,
        global_step_id=1,
        is_optimizer_step_committed=True,
    )
    with pytest.raises(RuntimeError, match="collective failed"):
        tracker._on_optimizer_step_end(args)

    assert tracker._cached_loss_sums == {}
    assert tracker._cached_loss_counts == {}
    assert tracker._expected_loss_keys is None


def test_loss_tracker_packs_multiple_losses_into_one_reduction():
    """All loss sum/count pairs should share one logging collective."""

    tracker = LossTrackerConfig(step_log_freq=1)()
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.reduce.return_value = torch.tensor([[6.0, 2.0], [12.0, 2.0]])
    args = PipelineHookArgs(
        accelerator=accelerator,
        model_outputs={
            "loss": torch.tensor(4.0, requires_grad=True),
            "aux_loss": torch.tensor(2.0, requires_grad=True),
        },
        global_step_id=1,
        step_id=1,
        max_step=1,
        max_epoch=1,
    )

    with tracker.begin("on_step", args):
        pass
    with tracker.begin(
        "on_optimizer_step",
        args.copy_with_updates(is_optimizer_step_committed=True),
    ):
        pass

    accelerator.reduce.assert_called_once()
    assert torch.equal(
        accelerator.reduce.call_args.args[0],
        torch.tensor([[2.0, 1.0], [4.0, 1.0]]),
    )
    accelerator.log.assert_any_call({"Loss/aux_loss": 3.0}, step=1)
    accelerator.log.assert_any_call({"Loss/loss": 6.0}, step=1)


def test_loss_tracker_reduces_accumulation_window_once():
    """Multiple micro steps should produce one sum/count reduction."""

    tracker = LossTrackerConfig(step_log_freq=1)()
    accelerator = MagicMock()
    accelerator.is_main_process = False
    accelerator.reduce.return_value = torch.tensor([[10.0, 4.0]])
    args = PipelineHookArgs(accelerator=accelerator)

    for loss in [1.0, 2.0, 3.0, 4.0]:
        with tracker.begin(
            "on_step",
            args.copy_with_updates(model_outputs={"loss": torch.tensor(loss)}),
        ):
            pass

    with tracker.begin(
        "on_optimizer_step",
        args.copy_with_updates(
            global_step_id=1,
            is_optimizer_step_committed=True,
        ),
    ):
        pass

    accelerator.reduce.assert_called_once()
    assert torch.equal(
        accelerator.reduce.call_args.args[0],
        torch.tensor([[10.0, 4.0]]),
    )
    accelerator.log.assert_not_called()


def test_loss_tracker_without_moving_average_keeps_latest_loss():
    """Disabling moving average should retain only the latest local value."""

    tracker = LossTrackerConfig(step_log_freq=2, moving_average=False)()
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.reduce.return_value = torch.tensor([[8.0, 1.0]])

    for global_step_id, loss in [(1, 4.0), (2, 8.0)]:
        args = PipelineHookArgs(
            accelerator=accelerator,
            model_outputs={"loss": torch.tensor(loss)},
            global_step_id=global_step_id,
            step_id=global_step_id,
            max_step=2,
            max_epoch=1,
        )
        with tracker.begin("on_step", args):
            pass
        with tracker.begin(
            "on_optimizer_step",
            args.copy_with_updates(is_optimizer_step_committed=True),
        ):
            pass

    accelerator.reduce.assert_called_once()
    assert torch.equal(
        accelerator.reduce.call_args.args[0],
        torch.tensor([[8.0, 1.0]]),
    )
    accelerator.log.assert_any_call({"Loss/loss": 8.0}, step=2)
