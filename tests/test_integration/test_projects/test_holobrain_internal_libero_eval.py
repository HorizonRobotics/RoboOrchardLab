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

import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def project_paths():
    repo_root = Path(__file__).resolve().parents[3]
    paths = [
        repo_root,
        repo_root / "projects/holobrain_internal/common",
        repo_root / "projects/holobrain_internal/libero",
    ]
    for path in paths:
        sys.path.insert(0, str(path))
    try:
        yield
    finally:
        for path in paths:
            sys.path.remove(str(path))


@pytest.fixture
def libero_eval_module(project_paths):
    sys.modules.pop("libero_eval", None)
    try:
        yield importlib.import_module("libero_eval")
    finally:
        sys.modules.pop("libero_eval", None)


@pytest.fixture
def libero_utils_module(project_paths):
    sys.modules.pop("libero_utils", None)
    try:
        yield importlib.import_module("libero_utils")
    finally:
        sys.modules.pop("libero_utils", None)


class DummyTaskSuite:
    n_tasks = 3


class DummyTask:
    def __init__(self, name):
        self.name = name


class DummyPlusTaskSuite:
    n_tasks = 3

    def get_task(self, task_id):
        return DummyTask(f"task_name_{task_id}")


def test_libero_eval_builds_task_list_for_selected_benchmark(
    libero_eval_module,
):
    benchmark_dict = {
        "libero_goal": DummyTaskSuite,
        "libero_plus_goal": DummyTaskSuite,
    }

    assert libero_eval_module.resolve_target_suites(
        "libero_plus_goal", benchmark_dict
    ) == ["libero_plus_goal"]
    assert libero_eval_module.build_task_list(
        "libero_plus_goal", -1, benchmark_dict
    ) == [
        ("libero_plus_goal", 0),
        ("libero_plus_goal", 1),
        ("libero_plus_goal", 2),
    ]
    assert libero_eval_module.build_task_list(
        "libero_plus_goal", 1, benchmark_dict
    ) == [("libero_plus_goal", 1)]


def test_libero_eval_allocates_multiple_processes_per_gpu(
    libero_eval_module,
):
    tasks = [
        ("libero_goal", 0),
        ("libero_goal", 1),
        ("libero_goal", 2),
        ("libero_goal", 3),
        ("libero_goal", 4),
    ]

    assert libero_eval_module.allocate_tasks_to_workers(
        tasks,
        num_gpus=2,
    ) == [
        (0, [("libero_goal", 0), ("libero_goal", 2), ("libero_goal", 4)]),
        (1, [("libero_goal", 1), ("libero_goal", 3)]),
    ]
    assert libero_eval_module.allocate_tasks_to_workers(
        tasks,
        num_gpus=2,
        processes_per_gpu=2,
    ) == [
        (0, [("libero_goal", 0), ("libero_goal", 4)]),
        (0, [("libero_goal", 1)]),
        (1, [("libero_goal", 2)]),
        (1, [("libero_goal", 3)]),
    ]


def test_libero_eval_resolves_custom_benchmark_module(
    libero_eval_module,
    monkeypatch,
):
    benchmark_module = ModuleType("fake_liberoplus_benchmark")
    benchmark_module.get_benchmark_dict = lambda: {
        "libero_plus_goal": DummyTaskSuite
    }
    monkeypatch.setitem(
        sys.modules, "fake_liberoplus_benchmark", benchmark_module
    )

    assert libero_eval_module._get_benchmark_dict(
        "libero_plus", "fake_liberoplus_benchmark"
    ) == {"libero_plus_goal": DummyTaskSuite}


def test_libero_eval_passes_benchmark_to_eval_policy(
    libero_eval_module,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLUSTER", raising=False)
    captured_commands = []

    def fake_run(command, env, stdout, stderr):
        captured_commands.append(command)
        output_dir = Path(command[command.index("--output_dir") + 1])
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({"success_rate": 0.75}, f)
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(libero_eval_module.subprocess, "run", fake_run)
    args = SimpleNamespace(
        benchmark="libero_plus",
        benchmark_module="fake_liberoplus_benchmark",
        num_trials_per_task=2,
        num_steps_wait=5,
        save_video=False,
        model_config="/tmp/model",
        model_prefix="model",
        vlm_ckpt_dir=None,
        urdf_dir="/tmp/urdf",
        model_processor="libero_processor",
    )

    results = libero_eval_module.eval_tasks(
        2,
        [("libero_plus_goal", 1)],
        args,
    )

    command = captured_commands[0]
    assert results == {("libero_plus_goal", "task_1"): 0.75}
    assert command[command.index("--benchmark") + 1] == "libero_plus"
    assert (
        command[command.index("--benchmark_module") + 1]
        == "fake_liberoplus_benchmark"
    )
    assert "--vlm_ckpt_dir" not in command
    assert command[command.index("--urdf_dir") + 1] == "/tmp/urdf"


def test_libero_plus_category_summary_uses_task_classification(
    libero_eval_module,
    tmp_path,
):
    classification_dir = tmp_path / "libero/libero/benchmark"
    classification_dir.mkdir(parents=True)
    with open(
        classification_dir / "task_classification.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "libero_goal": [
                    {
                        "id": 1,
                        "name": "task_name_0",
                        "category": "Camera Viewpoints",
                    },
                    {
                        "id": 2,
                        "name": "task_name_1",
                        "category": "Camera Viewpoints",
                    },
                    {
                        "id": 3,
                        "name": "task_name_2",
                        "category": "Sensor Noise",
                    },
                ]
            },
            f,
        )

    summary = libero_eval_module.summarize_libero_plus_categories(
        {
            ("libero_goal", "task_0"): 1.0,
            ("libero_goal", "task_1"): 0.0,
            ("libero_goal", "task_2"): 1.0,
        },
        {"libero_goal": DummyPlusTaskSuite},
        tmp_path,
    )

    assert summary["categories_detail"] == {
        "Camera Viewpoints": {
            "average_success_rate": 0.5,
            "num_tasks": 2,
        },
        "Sensor Noise": {
            "average_success_rate": 1.0,
            "num_tasks": 1,
        },
    }
    assert summary["average_success_rate_across_categories"] == 0.75
    assert summary["average_success_rate_all_tasks"] == pytest.approx(2 / 3)


def test_make_libero_config_path_supports_concurrent_calls(
    libero_utils_module,
    tmp_path,
):
    benchmark_root = tmp_path / "LIBERO-plus"
    (benchmark_root / "libero/libero").mkdir(parents=True)

    def make_config(_):
        return libero_utils_module.make_libero_config_path(
            "libero_plus",
            benchmark_root,
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        config_dirs = list(executor.map(make_config, range(64)))

    assert len(set(config_dirs)) == 1
    config_path = config_dirs[0] / "config.yaml"
    benchmark_data_root = benchmark_root / "libero/libero"
    assert config_path.read_text(encoding="utf-8") == "\n".join(
        [
            f"assets: {benchmark_data_root / 'assets'}",
            f"bddl_files: {benchmark_data_root / 'bddl_files'}",
            f"benchmark_root: {benchmark_data_root}",
            f"datasets: {benchmark_root / 'libero/datasets'}",
            f"init_states: {benchmark_data_root / 'init_files'}",
            "",
        ]
    )
