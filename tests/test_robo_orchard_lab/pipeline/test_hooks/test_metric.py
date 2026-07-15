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

import logging
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from torchmetrics.classification import BinaryAccuracy

from robo_orchard_lab.pipeline.hooks.metric import (
    MetricEntry,
    MetricTracker,
    MetricTrackerConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs


class CustomMetricTracker(MetricTracker):
    def update_metric(self, batch, model_outputs):
        for metric in self.metrics:
            metric.update(model_outputs, batch)


@pytest.fixture(scope="function")
def mock_metric_entry():
    """Fixture to create a mock MetricEntry."""
    metric = BinaryAccuracy()
    names = ["accuracy"]
    return MetricEntry(names=names, metric=metric)


@pytest.fixture(scope="function")
def mock_metric_tracker(mock_metric_entry):
    """Fixture to create a MetricTracker with a single MetricEntry."""

    return MetricTrackerConfig(
        class_type=CustomMetricTracker,
        metric_entrys=[mock_metric_entry],
        reset_by="epoch",
        reset_freq=1,
        step_log_freq=2,
        epoch_log_freq=1,
        log_main_process_only=True,
    )()


@pytest.fixture(scope="function")
def mock_hook_args():
    """Fixture to create mock HookArgs."""
    accelerator = MagicMock()
    accelerator.device = "cpu"
    accelerator.is_main_process = True

    return PipelineHookArgs(
        accelerator=accelerator,
        epoch_id=0,
        step_id=0,
        global_step_id=0,
        batch=torch.tensor([1, 0, 1]),  # Mock labels
        model_outputs=torch.tensor([1, 0, 1]),  # Mock predictions
    )


@pytest.fixture(scope="function")
def temp_log_file():
    """Fixture to create a temporary log file for each test."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.close()
        yield temp_file.name


@pytest.fixture(scope="function")
def configure_per_process_logging(worker_id):
    """Configure logging to write to a per-process log file."""
    log_file = f"test_logs_worker_{worker_id}.log"
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    yield log_file  # Return the log file path for verification

    logger.removeHandler(handler)  # Clean up after the test


def test_metric_entry_initialization(mock_metric_entry):
    """Test MetricEntry initialization."""
    assert mock_metric_entry.names == ["accuracy"]
    assert isinstance(mock_metric_entry.metric, BinaryAccuracy)


def test_metric_entry_get(mock_metric_entry):
    """Test the get method of MetricEntry."""
    metric = mock_metric_entry.metric
    metric.update(
        torch.tensor([1, 0, 1]), torch.tensor([1, 0, 1])
    )  # Update before compute
    result = list(mock_metric_entry.get())
    assert len(result) == 1
    assert result[0][0] == "accuracy"
    assert result[0][1] == 1.0  # Accuracy should be 100%


def test_on_step_end(
    mock_metric_tracker: CustomMetricTracker,
    mock_hook_args: PipelineHookArgs,
    configure_per_process_logging,
):
    """Test the on_step_end method with logging."""
    mock_hook_args.step_id = 1
    mock_hook_args.global_step_id = 2
    mock_hook_args.is_optimizer_step_committed = True
    mock_metric_tracker._log = MagicMock()

    with mock_metric_tracker.begin(
        "on_batch", mock_hook_args
    ):  # Update before step_end
        pass

    # Trigger step-end logic with logging
    with mock_metric_tracker.begin(
        "on_optimizer_step", mock_hook_args
    ):  # Update before step_end
        pass

    mock_metric_tracker._log.assert_called_once()
    assert mock_metric_tracker._log.call_args.kwargs["prefix"] == (
        "Epoch[0] Step[1] GlobalStep[2]: "
    )


def test_metric_tracker_skips_batch_update_on_context_exception(
    mock_metric_tracker: CustomMetricTracker,
    mock_hook_args: PipelineHookArgs,
):
    mock_hook_args.model_outputs = None
    mock_hook_args.exception = RuntimeError("body failed")

    mock_metric_tracker._on_batch_end(mock_hook_args)


def test_on_epoch_end(
    mock_metric_tracker: CustomMetricTracker,
    mock_hook_args: PipelineHookArgs,
    configure_per_process_logging,
):
    """Test the on_epoch_end method with logging."""

    with mock_metric_tracker.begin(
        "on_batch", mock_hook_args
    ):  # Update before step_end
        pass

    # Trigger step-end logic with logging
    with mock_metric_tracker.begin(
        "on_optimizer_step", mock_hook_args
    ):  # Update before step_end
        pass

    # Verify log output in the temporary file
    # log_file = configure_per_process_logging
    # with open(log_file, "r") as f:
    #     logs = f.read()
    # assert "Epoch[0]: accuracy[tensor(1.)]" in logs


def test_reset_logic(
    mock_metric_tracker: CustomMetricTracker, mock_hook_args: PipelineHookArgs
):
    """Test that metrics are reset based on reset_by and reset_freq."""
    with mock_metric_tracker.begin(
        "on_batch", mock_hook_args
    ):  # Update before step_end
        pass

    # Verify metrics are updated
    metric = mock_metric_tracker.metrics[0]
    assert metric.compute() == 1.0

    # Trigger epoch-end to reset
    with mock_metric_tracker.begin(
        "on_epoch", mock_hook_args
    ):  # Update before step_end
        pass

    assert metric.compute() == 0.0


def test_metric_tracker_resets_step_window_on_skipped_boundary(
    mock_metric_entry,
    mock_hook_args: PipelineHookArgs,
):
    """Skipped optimizer boundaries should not leak step metrics."""

    tracker = MetricTrackerConfig(
        class_type=CustomMetricTracker,
        metric_entrys=[mock_metric_entry],
        reset_by="step",
        reset_freq=1,
        step_log_freq=1,
        epoch_log_freq=1,
    )()

    with tracker.begin("on_batch", mock_hook_args):
        pass
    assert tracker.metrics[0].compute() == 1.0

    mock_hook_args.is_optimizer_step_committed = False
    with tracker.begin("on_optimizer_step", mock_hook_args):
        pass

    assert tracker.metrics[0].compute() == 0.0


if __name__ == "__main__":
    pytest.main(["-s", __file__])
