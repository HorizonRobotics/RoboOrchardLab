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
import importlib.util
from pathlib import Path

import pytest
import yaml

from robo_orchard_lab.envs.libero import bootstrap
from robo_orchard_lab.envs.libero.bootstrap import (
    build_libero_default_path_config,
    ensure_libero_config,
    get_libero_config_file,
    resolve_libero_benchmark_root,
)


def test_ensure_libero_config_writes_default_paths(tmp_path: Path) -> None:
    config_root = tmp_path / "libero_config"
    benchmark_root = tmp_path / "benchmark_root"
    benchmark_root.mkdir()

    config_file = ensure_libero_config(
        config_root=config_root,
        benchmark_root=benchmark_root,
    )

    assert config_file == get_libero_config_file(config_root=config_root)
    assert config_file.exists()

    with open(config_file, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    assert config == build_libero_default_path_config(
        benchmark_root=benchmark_root
    )


def test_resolve_libero_benchmark_root_stabilizes_missing_package_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_parent_missing(name: str):
        raise ModuleNotFoundError("No module named 'libero'")

    monkeypatch.setattr(importlib.util, "find_spec", _raise_parent_missing)

    with pytest.raises(
        ModuleNotFoundError,
        match=(
            r"libero\.libero is not installed; cannot prepare LIBERO config\."
        ),
    ):
        resolve_libero_benchmark_root()


def test_resolve_libero_benchmark_root_rejects_missing_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap.importlib.util,
        "find_spec",
        lambda name: None,
    )

    with pytest.raises(
        ModuleNotFoundError,
        match=(
            r"libero\.libero is not installed; cannot prepare LIBERO config\."
        ),
    ):
        resolve_libero_benchmark_root()
