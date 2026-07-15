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
import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_run_unit_tests_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent / "run_unit_tests.py"
    spec = importlib.util.spec_from_file_location(
        "robo_orchard_lab_test_run_unit_tests",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_runner_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path,
        test_root=tmp_path / "tests",
        pytest_config=tmp_path / "pytest.ini",
        coverage_config=tmp_path / ".coveragerc",
        merge_script=tmp_path / "merge_pytest_reports.py",
        allure_dir="allure_unittest",
        coverage_file=".coverage.ut",
        final_junit_xml="unittest.xml",
        final_html="unittest.html",
        non_sim_workers=8,
        sim_workers=4,
        dry_run=False,
    )


def test_pipeline_cleans_allure_dir_before_pytest_without_worker_clean(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runner = load_run_unit_tests_module()
    args = make_runner_args(tmp_path)
    allure_dir = tmp_path / args.allure_dir
    stale_result = allure_dir / "stale-result.json"
    stale_result.parent.mkdir(parents=True)
    stale_result.write_text("stale", encoding="utf-8")

    commands: list[list[str]] = []

    def record_command(
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        dry_run: bool,
    ) -> None:
        del cwd, env, dry_run
        commands.append(command)
        if command[1:3] == ["-m", "pytest"]:
            assert stale_result.exists() is False
            assert allure_dir.is_dir()

    monkeypatch.setattr(runner, "run_command", record_command)

    runner.run_pipeline(args)

    pytest_commands = [
        command for command in commands if command[1:3] == ["-m", "pytest"]
    ]
    assert len(pytest_commands) == 2
    for command in pytest_commands:
        assert "--clean-alluredir" not in command
