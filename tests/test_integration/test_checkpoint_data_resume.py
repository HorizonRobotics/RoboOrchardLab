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

"""Real two-rank CPU checkpoint/resume acceptance tests.

Each phase is a separate torchrun invocation, including fresh-process reloads.
The tests fail if an intended fault does not reach the real injection seam.
"""

from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import torch
from accelerate.utils.constants import RNG_STATE_NAME

_RUNNER = Path(__file__).with_name("_checkpoint_data_resume_runner.py")


def _launch(
    project_root: str,
    run_dir: Path,
    project_dir: Path,
    scenario: str,
    *,
    in_order: bool = True,
    workers: int = 2,
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    """Run two real ranks with a launcher-owned port and bounded teardown."""
    assert _RUNNER.is_file(), _RUNNER
    run_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "COVERAGE_PROCESS_START",
    ):
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith("COV_CORE_"):
            env.pop(key, None)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        project_root
        if not existing_pythonpath
        else os.pathsep.join((project_root, existing_pythonpath))
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    # These ranks share one host. Avoid its hostname and automatic NIC choice;
    # keep the bootstrap store bound while the launcher owns its port.
    env["GLOO_SOCKET_IFNAME"] = "lo"
    env["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "0"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=127.0.0.1:0",
        f"--rdzv-id={uuid4()}",
        "--local-addr=127.0.0.1",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(_RUNNER),
        "--scenario",
        scenario,
        "--run-dir",
        str(run_dir),
        "--project-dir",
        str(project_dir),
        "--in-order",
        "true" if in_order else "false",
        "--workers",
        str(workers),
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))

    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        # The process group belongs exclusively to this launch. Reap ranks and
        # their workers so a deadlock regression cannot leak test processes.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=10)
        pytest.fail(
            f"two-rank {scenario} timed out\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    assert process.returncode == 0, (
        f"two-rank {scenario} failed to finish: {process.returncode}\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
    results = []
    for rank in (0, 1):
        path = run_dir / f"result_rank{rank}.json"
        assert path.is_file(), (
            f"rank {rank} did not report in {scenario}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["rank"] == rank
        assert result.get("world_size") == 2, result
        assert result.get("backend") == "gloo", result
        results.append(result)
    return results


def _assert_ok(results: list[dict[str, Any]]) -> None:
    assert all(result["status"] == "ok" for result in results), results
    assert all(not result["active_children"] for result in results), results


def _checkpoint_dir(project_dir: Path) -> Path:
    """Find this test's single successful save with both rank RNG files."""
    rank0_name = f"{RNG_STATE_NAME}_0.pkl"
    rank1_name = f"{RNG_STATE_NAME}_1.pkl"
    candidates = sorted(
        path.parent
        for path in project_dir.rglob(rank0_name)
        if (path.parent / rank1_name).is_file()
    )
    assert len(candidates) == 1, (
        f"Expected one saved checkpoint with both rank RNG files: {candidates}"
    )
    return candidates[0]


def _worker_reads(run_dir: Path, rank: int) -> list[tuple[int, int]]:
    return [
        tuple(json.loads(line))
        for path in sorted(run_dir.glob(f"reads_rank{rank}_worker*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize(
    "in_order", [True, False], ids=["ordered", "unordered"]
)
def test_two_rank_fresh_process_resume_preserves_delivery_frontier(
    PROJECT_ROOT: str, tmp_path: Path, in_order: bool
) -> None:
    project = tmp_path / "project"
    save_dir = tmp_path / "save"
    saved = _launch(PROJECT_ROOT, save_dir, project, "save", in_order=in_order)
    _assert_ok(saved)
    checkpoint = _checkpoint_dir(project)

    resume_dir = tmp_path / "resume"
    resumed = _launch(
        PROJECT_ROOT,
        resume_dir,
        project,
        "resume",
        in_order=in_order,
        checkpoint=checkpoint,
    )
    _assert_ok(resumed)
    for rank in (0, 1):
        saved_rank = saved[rank]
        resumed_rank = resumed[rank]
        assert saved_rank["end_progress"]["epoch"] == 0
        assert saved_rank["end_progress"]["global_step"] == 2
        assert saved_rank["end_progress"]["micro_epoch"] == 2
        assert resumed_rank["start_progress"] == saved_rank["end_progress"]
        assert resumed_rank["start_parameters"] == pytest.approx(
            saved_rank["end_parameters"], rel=0, abs=1e-7
        )
        assert resumed_rank["end_progress"]["epoch"] == 1
        assert resumed_rank["end_progress"]["global_step"] > 2

        state = resumed_rank["start_loader_state"]
        assert state["schema"] == 1
        assert state["epoch"] == 0
        assert state["delivered_batches"] == 2
        assert state["layout"]["num_workers"] == 2
        assert state["layout"]["in_order"] is in_order
        consumed = [tuple(item) for item in saved_rank["processor_calls"]]
        assert len(consumed) == 2
        per_worker = Counter(worker for _, worker in consumed)
        assert state["worker_items"] == [per_worker[0], per_worker[1]]

        # Reads ahead of the committed frontier must not be credited as
        # processed. In the worker-local mode they must remain recoverable.
        prefetched = set(_worker_reads(save_dir, rank)) - set(consumed)
        assert prefetched, f"rank {rank} did not exercise real prefetch"
        later = {tuple(item) for item in resumed_rank["processor_calls"]}
        assert later
        recovery = state["recovery"]
        if recovery["accuracy"] == "worker_local":
            assert prefetched <= later
            assert set(consumed).isdisjoint(later)
        else:
            # Best-effort replay may change the suffix; it must say so.
            assert recovery["accuracy"] == "unknown"
            assert recovery["reason"]


@pytest.mark.parametrize("stale_snapshot", [False, True])
def test_two_rank_missing_data_state_replays_and_can_save_again(
    PROJECT_ROOT: str, tmp_path: Path, stale_snapshot: bool
) -> None:
    project = tmp_path / "project"
    saved = _launch(PROJECT_ROOT, tmp_path / "save", project, "save")
    _assert_ok(saved)
    checkpoint = _checkpoint_dir(project)
    progress_path = checkpoint / "custom_checkpoint_0.pkl"
    progress = torch.load(progress_path, weights_only=False)
    if stale_snapshot:
        for rank_state in progress["rank_states"]:
            rank_state["loader_state"]["delivered_batches"] -= 1
    else:
        progress["rank_states"] = []
    torch.save(progress, progress_path)

    continued = _launch(
        PROJECT_ROOT,
        tmp_path / "continue",
        project,
        "resume_save",
        checkpoint=checkpoint,
    )
    _assert_ok(continued)
    next_checkpoint = project / "checkpoints" / "checkpoint_1"
    reloaded = _launch(
        PROJECT_ROOT,
        tmp_path / "reload",
        project,
        "resume",
        checkpoint=next_checkpoint,
    )
    _assert_ok(reloaded)
    for rank in (0, 1):
        assert continued[rank]["start_progress"] == saved[rank]["end_progress"]
        assert (
            continued[rank]["start_loader_state"]["recovery"]["accuracy"]
            == "unknown"
        )
        assert (
            reloaded[rank]["start_progress"] == continued[rank]["end_progress"]
        )
        assert reloaded[rank]["start_loader_state"]["delivered_batches"] == 4
        assert reloaded[rank]["processor_calls"]


def test_two_rank_resumed_training_saves_new_checkpoint_in_same_project(
    PROJECT_ROOT: str, tmp_path: Path
) -> None:
    """A fresh process can continue and save without overwriting its source."""
    project = tmp_path / "project"
    saved = _launch(
        PROJECT_ROOT, tmp_path / "save", project, "save", workers=0
    )
    _assert_ok(saved)
    checkpoint = _checkpoint_dir(project)
    assert checkpoint == project / "checkpoints" / "checkpoint_0"
    saved_contents = {
        path.relative_to(checkpoint): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    assert saved_contents

    continued = _launch(
        PROJECT_ROOT,
        tmp_path / "continue",
        project,
        "resume_save",
        workers=0,
        checkpoint=checkpoint,
    )
    _assert_ok(continued)
    new_checkpoint = project / "checkpoints" / "checkpoint_1"
    assert new_checkpoint.is_dir()
    assert all(
        (new_checkpoint / f"{RNG_STATE_NAME}_{rank}.pkl").is_file()
        for rank in (0, 1)
    )
    assert {
        path.relative_to(checkpoint)
        for path in checkpoint.rglob("*")
        if path.is_file()
    } == set(saved_contents)
    assert all(
        (checkpoint / name).read_bytes() == content
        for name, content in saved_contents.items()
    )

    reloaded = _launch(
        PROJECT_ROOT,
        tmp_path / "reload",
        project,
        "resume",
        workers=0,
        checkpoint=new_checkpoint,
    )
    _assert_ok(reloaded)
    for rank in (0, 1):
        assert continued[rank]["start_progress"] == saved[rank]["end_progress"]
        assert continued[rank]["end_progress"]["global_step"] == 4
        assert (
            reloaded[rank]["start_progress"] == continued[rank]["end_progress"]
        )
        assert reloaded[rank]["start_parameters"] == pytest.approx(
            continued[rank]["end_parameters"], rel=0, abs=1e-7
        )
        assert reloaded[rank]["start_loader_state"]["delivered_batches"] == 4
        assert reloaded[rank]["processor_calls"]


def test_two_rank_uneven_real_iterable_stops_at_peer_eof(
    PROJECT_ROOT: str, tmp_path: Path
) -> None:
    run_dir = tmp_path / "uneven_g1"
    results = _launch(
        PROJECT_ROOT,
        run_dir,
        tmp_path / "project",
        "uneven_g1",
        workers=0,
    )
    # These markers observe the real IterableWithLenDataset.shard result.
    shards = [
        json.loads((run_dir / f"shard_rank{rank}.json").read_text())
        for rank in (0, 1)
    ]
    assert [shard["rows"] for shard in shards] == [2, 1]
    _assert_ok(results)
    assert all(
        (run_dir / f"iterator_rank{rank}.txt").is_file() for rank in (0, 1)
    )
    assert [len(result["processor_calls"]) for result in results] == [1, 1]
    assert [result["end_progress"]["global_step"] for result in results] == [
        1,
        1,
    ]


def test_two_rank_custom_batch_sampler_completes_accumulation_window(
    PROJECT_ROOT: str, tmp_path: Path
) -> None:
    run_dir = tmp_path / "custom_batch_g2"
    results = _launch(
        PROJECT_ROOT,
        run_dir,
        tmp_path / "project",
        "custom_batch_g2",
        workers=0,
    )
    _assert_ok(results)
    for rank, result in enumerate(results):
        assert result["trainer_started"], result
        assert (run_dir / f"custom_batch_iter_rank{rank}.txt").is_file()
        assert len(result["processor_calls"]) == 2, result
        assert result["start_progress"]["global_step"] == 0, result
        assert result["end_progress"]["global_step"] == 1, result
        assert result["end_progress"]["epoch_step"] == 1, result
        assert result["end_progress"]["micro_epoch"] == 2, result
    assert sorted(
        call[0] for result in results for call in result["processor_calls"]
    ) == [0, 1, 2, 3]


def test_two_rank_standard_map_sampler_skips_consumed_prefix(
    PROJECT_ROOT: str, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    full = _launch(
        PROJECT_ROOT, tmp_path / "map_full", project, "map_full", workers=0
    )
    _assert_ok(full)
    saved = _launch(
        PROJECT_ROOT, tmp_path / "map_save", project, "map_save", workers=0
    )
    _assert_ok(saved)
    checkpoint = _checkpoint_dir(project)
    resume_dir = tmp_path / "map_resume"
    resumed = _launch(
        PROJECT_ROOT,
        resume_dir,
        project,
        "map_resume",
        workers=0,
        checkpoint=checkpoint,
    )
    _assert_ok(resumed)

    full_ids = [
        sample[0] for result in full for sample in result["processor_calls"]
    ]
    assert sorted(full_ids) == list(range(32))
    for rank in (0, 1):
        reference = full[rank]
        saved_rank = saved[rank]
        resumed_rank = resumed[rank]
        assert reference["start_loader_state"] is None
        assert saved_rank["start_loader_state"] is None
        assert resumed_rank["start_loader_state"] is None
        assert saved_rank["end_dataloader_seed"] == 42
        assert resumed_rank["start_dataloader_seed"] == 42
        assert resumed_rank["start_progress"] == saved_rank["end_progress"]
        assert resumed_rank["start_parameters"] == pytest.approx(
            saved_rank["end_parameters"], rel=0, abs=1e-7
        )
        consumed = saved_rank["processor_calls"]
        continued = resumed_rank["processor_calls"]
        assert len(consumed) == 2
        assert consumed + continued == reference["processor_calls"]
        # The private standard sampler advances indices without old reads.
        assert {tuple(item) for item in consumed}.isdisjoint(
            _worker_reads(resume_dir, rank)
        )
