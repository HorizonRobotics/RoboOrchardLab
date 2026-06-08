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

from __future__ import annotations
import os
import sys
from pathlib import Path

import pytest

from robo_orchard_lab.envs.robotwin import workspace

pytestmark = pytest.mark.sim_env


@pytest.fixture(autouse=True)
def _clear_robotwin_path_cache():
    workspace.config_robotwin_path.cache_clear()
    yield
    workspace.config_robotwin_path.cache_clear()


def test_config_robotwin_path_adds_env_path_to_sys_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    robotwin_root = tmp_path / "RoboTwin"
    robotwin_root.mkdir()
    monkeypatch.setenv("RoboTwin_PATH", str(robotwin_root))
    monkeypatch.setattr(sys, "path", list(sys.path))
    workspace.config_robotwin_path.cache_clear()

    assert workspace.config_robotwin_path() == str(robotwin_root)
    assert str(robotwin_root) in sys.path


def test_in_robotwin_workspace_restores_original_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    robotwin_root = tmp_path / "RoboTwin"
    original_cwd = tmp_path / "caller"
    robotwin_root.mkdir()
    original_cwd.mkdir()
    monkeypatch.chdir(original_cwd)
    monkeypatch.setenv("RoboTwin_PATH", str(robotwin_root))
    workspace.config_robotwin_path.cache_clear()

    with workspace.in_robotwin_workspace():
        assert os.getcwd() == str(robotwin_root)

    assert os.getcwd() == str(original_cwd)
