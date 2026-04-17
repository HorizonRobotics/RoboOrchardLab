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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ACCELERATE_TRACE_RUNNER = Path(__file__).with_name(
    "_dataset_ex_accelerate_trace_runner.py"
)


def _run_accelerate_trace_subprocess(
    tmp_path: Path,
    project_root: str,
    prepare_mode: str,
    use_dataset_side_batching: bool,
    num_workers: int = 0,
    shard_strategy: str | None = None,
) -> list[dict[str, object]]:
    shard_strategy_name = "none" if shard_strategy is None else shard_strategy
    output_dir = tmp_path / (
        f"{prepare_mode}_{int(use_dataset_side_batching)}"
        f"_{num_workers}_{shard_strategy_name}"
    )
    assert _ACCELERATE_TRACE_RUNNER.exists(), _ACCELERATE_TRACE_RUNNER

    env = os.environ.copy()
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "COVERAGE_PROCESS_START",
    ]:
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith("COV_CORE_"):
            env.pop(key, None)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        project_root
        if not existing_pythonpath
        else os.pathsep.join([project_root, existing_pythonpath])
    )
    env["CUDA_VISIBLE_DEVICES"] = ""

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        "2",
        "--main_process_port",
        # Let the distributed launcher atomically reserve an ephemeral port so
        # xdist workers do not race on a probe-then-release socket.
        "0",
        str(_ACCELERATE_TRACE_RUNNER),
        "--prepare-mode",
        prepare_mode,
        "--use-dataset-side-batching",
        "1" if use_dataset_side_batching else "0",
        "--num-workers",
        str(num_workers),
        "--shard-strategy",
        shard_strategy_name,
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        "accelerate subprocess failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    result_files = sorted(output_dir.glob("rank_*.json"))
    assert len(result_files) == 2, (
        "Expected results from two ranks\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n"
        f"files: {[str(path) for path in result_files]}"
    )
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in result_files
    ]


class TestDictIterableDatasetAccelerateLaunch:
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_raw_prepare_can_diverge_rank_schedule(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
        use_dataset_side_batching: bool,
    ):
        results = _run_accelerate_trace_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            prepare_mode="raw",
            use_dataset_side_batching=use_dataset_side_batching,
        )

        assert [result["shard_strategy"] for result in results] == [None, None]
        assert (
            results[0]["total_indices_length"]
            != results[1]["total_indices_length"]
        )
        assert results[0]["trace"] != results[1]["trace"]

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_wrapped_prepare_aligns_rank_schedule(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
        use_dataset_side_batching: bool,
    ):
        results = _run_accelerate_trace_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            prepare_mode="wrapped",
            use_dataset_side_batching=use_dataset_side_batching,
        )

        assert [result["shard_strategy"] for result in results] == [
            "pad_last",
            "pad_last",
        ]
        assert (
            results[0]["total_indices_length"]
            == results[1]["total_indices_length"]
        )
        assert results[0]["trace"] == results[1]["trace"]

    def test_wrapped_prepare_multi_worker_pad_last_stays_rank_aligned(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
    ):
        results = _run_accelerate_trace_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            prepare_mode="wrapped",
            use_dataset_side_batching=True,
            num_workers=2,
            shard_strategy="pad_last",
        )

        assert [result["shard_strategy"] for result in results] == [
            "pad_last",
            "pad_last",
        ]
        assert [result["num_workers"] for result in results] == [2, 2]
        assert (
            results[0]["total_indices_length"]
            == results[1]["total_indices_length"]
        )
        assert results[0]["trace"] == results[1]["trace"]
