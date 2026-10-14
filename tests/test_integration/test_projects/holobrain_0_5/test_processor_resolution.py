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


"""Coverage for locating a processor from a nested checkpoint directory."""

import pytest

from projects.holobrain_0_5.utils import resolve_processor_dir

__all__ = []

NAME = "robotwin2_0_processor.json"


def test_processor_next_to_model_wins(tmp_path):
    """A flat export must not be redirected to an unrelated ancestor."""
    (tmp_path / NAME).write_text("{}")
    model_dir = tmp_path / "checkpoints" / "checkpoint_1"
    model_dir.mkdir(parents=True)
    (model_dir / NAME).write_text("{}")

    assert resolve_processor_dir(model_dir, NAME) == model_dir


def test_checkpoint_dir_falls_back_to_workspace_root(tmp_path):
    """Training keeps the processor two levels above its checkpoints."""
    (tmp_path / NAME).write_text("{}")
    model_dir = tmp_path / "checkpoints" / "checkpoint_1"
    model_dir.mkdir(parents=True)

    assert resolve_processor_dir(model_dir, NAME) == tmp_path


def test_missing_processor_lists_both_candidates(tmp_path):
    model_dir = tmp_path / "checkpoints" / "checkpoint_1"
    model_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_processor_dir(model_dir, NAME)

    message = str(excinfo.value)
    assert str(model_dir / NAME) in message
    assert str(tmp_path / NAME) in message


def test_shallow_model_dir_does_not_raise_index_error():
    """A path with no grandparent must still fail cleanly."""
    with pytest.raises(FileNotFoundError):
        resolve_processor_dir("holobrain", NAME)
