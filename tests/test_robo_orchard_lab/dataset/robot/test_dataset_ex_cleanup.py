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

_CLEANUP_RUNNER = Path(__file__).with_name("_dataset_ex_cleanup_runner.py")


def _run_cleanup_subprocess(
    tmp_path: Path,
    project_root: str,
    dataset_kind: str,
    prepare_mode: str,
    use_dataset_side_batching: bool,
    num_workers: int,
    persistent_workers: bool,
    close_mode: str = "early-break",
    pin_memory: bool = False,
) -> dict[str, object]:
    output_path = tmp_path / (
        f"cleanup_{dataset_kind}_{prepare_mode}_"
        f"{int(use_dataset_side_batching)}_{num_workers}_"
        f"{int(persistent_workers)}_{close_mode}_"
        f"{int(pin_memory)}.json"
    )
    env = os.environ.copy()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        env.pop(key, None)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        project_root
        if not existing_pythonpath
        else os.pathsep.join([project_root, existing_pythonpath])
    )

    command = [
        sys.executable,
        str(_CLEANUP_RUNNER),
        "--dataset-kind",
        dataset_kind,
        "--prepare-mode",
        prepare_mode,
        "--use-dataset-side-batching",
        "1" if use_dataset_side_batching else "0",
        "--num-workers",
        str(num_workers),
        "--persistent-workers",
        "1" if persistent_workers else "0",
        "--close-mode",
        close_mode,
        "--pin-memory",
        "1" if pin_memory else "0",
        "--cycles",
        "4",
        "--batches-per-cycle",
        "2",
        "--output-path",
        str(output_path),
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
        "cleanup subprocess failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _assert_cleanup_payload(payload: dict[str, object]) -> None:
    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    per_cycle = payload["per_cycle"]
    assert isinstance(per_cycle, list)
    after_cleanup = payload["after_cleanup"]
    assert isinstance(after_cleanup, dict)

    baseline_child_pids = baseline["child_pids"]
    assert isinstance(baseline_child_pids, list)
    baseline_prefetch_threads = baseline["prefetch_threads"]
    assert isinstance(baseline_prefetch_threads, int)
    baseline_fd_count = baseline["fd_count"]
    assert isinstance(baseline_fd_count, int)

    num_workers = payload["num_workers"]
    assert isinstance(num_workers, int)
    persistent_workers = payload["persistent_workers"]
    assert isinstance(persistent_workers, bool)
    close_mode = payload["close_mode"]
    assert close_mode in {"early-break", "epoch-exhausted"}
    pin_memory_requested = payload["pin_memory_requested"]
    assert isinstance(pin_memory_requested, bool)
    pin_memory_observed = payload["pin_memory_observed"]
    assert isinstance(pin_memory_observed, bool)

    cycle_child_pid_sets: list[tuple[int, ...]] = []
    cycle_fd_counts: list[int] = []
    for cycle in per_cycle:
        assert isinstance(cycle, dict)
        if close_mode == "early-break":
            assert cycle["batch_count"] == 2
        else:
            assert cycle["batch_count"] > 0
        assert cycle["prefetch_threads"] == baseline_prefetch_threads
        child_pids = cycle["child_pids"]
        assert isinstance(child_pids, list)
        cycle_child_pid_sets.append(tuple(child_pids))
        fd_count = cycle["fd_count"]
        assert isinstance(fd_count, int)
        cycle_fd_counts.append(fd_count)

    if num_workers > 0 and persistent_workers:
        assert cycle_child_pid_sets
        assert cycle_child_pid_sets[0]
        assert all(
            child_pid_set == cycle_child_pid_sets[0]
            for child_pid_set in cycle_child_pid_sets[1:]
        )
    else:
        assert all(
            list(child_pid_set) == baseline_child_pids
            for child_pid_set in cycle_child_pid_sets
        )

    assert after_cleanup["child_pids"] == baseline_child_pids
    assert after_cleanup["prefetch_threads"] == baseline_prefetch_threads
    assert isinstance(after_cleanup["fd_count"], int)
    fd_tolerance = 8 + 8 * num_workers
    if pin_memory_requested:
        fd_tolerance += 8
    if cycle_fd_counts:
        assert max(cycle_fd_counts) - min(cycle_fd_counts) <= (
            4 + 2 * num_workers
        )

    if pin_memory_requested and persistent_workers:
        assert cycle_fd_counts
        assert after_cleanup["fd_count"] <= max(cycle_fd_counts)
    else:
        assert after_cleanup["fd_count"] <= baseline_fd_count + fd_tolerance
        if cycle_fd_counts:
            assert max(cycle_fd_counts) <= baseline_fd_count + fd_tolerance + 4


def _assert_pin_memory_observed(payload: dict[str, object]) -> None:
    pin_memory_requested = payload["pin_memory_requested"]
    assert isinstance(pin_memory_requested, bool)
    pin_memory_observed = payload["pin_memory_observed"]
    assert isinstance(pin_memory_observed, bool)
    if pin_memory_requested and not pin_memory_observed:
        pytest.skip(
            "pin_memory=True did not enable the PyTorch pin-memory iterator "
            "path in this environment."
        )
    assert pin_memory_observed is True


class TestDataLoaderEarlyBreakCleanupSubprocess:
    @pytest.mark.parametrize("dataset_kind", ["iterable", "dict"])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    @pytest.mark.parametrize(
        "num_workers,persistent_workers",
        [
            (0, False),
            (1, False),
            (1, True),
            (2, False),
            (2, True),
        ],
    )
    def test_unprepared_repeated_early_break_exits_cleanly(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
        dataset_kind: str,
        use_dataset_side_batching: bool,
        num_workers: int,
        persistent_workers: bool,
    ):
        payload = _run_cleanup_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            dataset_kind=dataset_kind,
            prepare_mode="none",
            use_dataset_side_batching=use_dataset_side_batching,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        _assert_cleanup_payload(payload)

    @pytest.mark.parametrize("prepare_mode", ["raw", "wrapped"])
    @pytest.mark.parametrize(
        "num_workers,persistent_workers",
        [
            (0, False),
            (1, False),
            (1, True),
            (2, False),
            (2, True),
        ],
    )
    def test_prepared_repeated_early_break_releases_iterator_resources(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
        prepare_mode: str,
        num_workers: int,
        persistent_workers: bool,
    ):
        payload = _run_cleanup_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            dataset_kind="dict",
            prepare_mode=prepare_mode,
            use_dataset_side_batching=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        _assert_cleanup_payload(payload)


class TestDataLoaderEpochExhaustionCleanupSubprocess:
    @pytest.mark.parametrize("dataset_kind", ["iterable", "dict"])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    @pytest.mark.parametrize("persistent_workers", [False, True])
    def test_unprepared_repeated_epoch_exhaustion_reuses_workers_cleanly(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
        dataset_kind: str,
        use_dataset_side_batching: bool,
        persistent_workers: bool,
    ):
        """Natural epoch end should reuse persistent workers."""
        payload = _run_cleanup_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            dataset_kind=dataset_kind,
            prepare_mode="none",
            use_dataset_side_batching=use_dataset_side_batching,
            num_workers=2,
            persistent_workers=persistent_workers,
            close_mode="epoch-exhausted",
            pin_memory=False,
        )

        _assert_cleanup_payload(payload)

    @pytest.mark.parametrize("prepare_mode", ["raw", "wrapped"])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_prepared_dict_epoch_exhaustion_reuses_workers_cleanly(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
        prepare_mode: str,
        use_dataset_side_batching: bool,
    ):
        """Prepared dataloader wrappers should survive normal epoch closes."""
        payload = _run_cleanup_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            dataset_kind="dict",
            prepare_mode=prepare_mode,
            use_dataset_side_batching=use_dataset_side_batching,
            num_workers=2,
            persistent_workers=True,
            close_mode="epoch-exhausted",
            pin_memory=False,
        )

        _assert_cleanup_payload(payload)

    @pytest.mark.parametrize("dataset_kind", ["iterable", "dict"])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_unprepared_epoch_exhaustion_observes_pin_memory_when_available(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
        dataset_kind: str,
        use_dataset_side_batching: bool,
    ):
        """Pin-memory coverage runs separately from basic worker reuse."""
        payload = _run_cleanup_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            dataset_kind=dataset_kind,
            prepare_mode="none",
            use_dataset_side_batching=use_dataset_side_batching,
            num_workers=2,
            persistent_workers=True,
            close_mode="epoch-exhausted",
            pin_memory=True,
        )

        _assert_cleanup_payload(payload)
        _assert_pin_memory_observed(payload)

    @pytest.mark.parametrize("prepare_mode", ["raw", "wrapped"])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_prepared_dict_epoch_exhaustion_observes_pin_memory_when_available(
        self,
        tmp_path: Path,
        PROJECT_ROOT: str,
        prepare_mode: str,
        use_dataset_side_batching: bool,
    ):
        """Prepared wrappers should expose pin-memory when the runtime does."""
        payload = _run_cleanup_subprocess(
            tmp_path=tmp_path,
            project_root=PROJECT_ROOT,
            dataset_kind="dict",
            prepare_mode=prepare_mode,
            use_dataset_side_batching=use_dataset_side_batching,
            num_workers=2,
            persistent_workers=True,
            close_mode="epoch-exhausted",
            pin_memory=True,
        )

        _assert_cleanup_payload(payload)
        _assert_pin_memory_observed(payload)
