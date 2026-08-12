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
import hashlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pytest
import torch
from robo_orchard_core.datatypes import BatchJointsState
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.math import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)

import robo_orchard_lab.envs.robotwin.env as robotwin_env_module
from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchFrameTransformGraph,
)
from robo_orchard_lab.envs.robotwin import RoboTwinEnv, RoboTwinEnvCfg
from robo_orchard_lab.envs.robotwin.curobo_base_patch import (
    RoboTwinCuroboPatchUnsupportedError,
)
from robo_orchard_lab.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_lab.policy.evaluator.benchmark.robotwin import (
    RoboTwinBenchmarkEvaluator,
    RoboTwinBenchmarkEvaluatorCfg,
)

pytestmark = pytest.mark.sim_env

_RESULT_MARKER = "ROBOTWIN_MATRIX_RESULT="
_WORKER_TIMEOUT_SECONDS = 600
_EVIDENCE_ENV = "ROBOTWIN_EMBODIMENT_EVIDENCE_DIR"
_BASE_EMBODIMENT_LINE = b"embodiment: [aloha-agilex]"
_EEF_POSITION_ERROR_LIMIT_M = 1e-3
_EEF_ORIENTATION_ERROR_LIMIT_RAD = 1e-3
_FK_DISTRIBUTED_SAMPLE_COUNT = 9
_FK_JOINT_LIMIT_MARGIN_RATIO = 0.1


@dataclass(frozen=True, slots=True)
class _EmbodimentCase:
    case_id: str
    embodiment_line: str
    topology: Literal["combined_articulation", "split_articulations"]
    qpos_width: int
    planner_worker_count: int = 0


class _ViewerHandle(Protocol):
    window: object | None


class _ConnectionHandle(Protocol):
    closed: bool


class _ProcessHandle(Protocol):
    pid: int | None

    def is_alive(self) -> bool: ...


class _SapienPoseHandle(Protocol):
    def get_p(self) -> np.ndarray: ...

    def get_q(self) -> np.ndarray: ...


class _RuntimeLinkHandle(Protocol):
    entity_pose: _SapienPoseHandle

    def get_name(self) -> str: ...


class _RuntimeJointHandle(Protocol):
    child_link: _RuntimeLinkHandle
    pose_in_child: _SapienPoseHandle

    def get_limits(self) -> np.ndarray: ...

    def get_name(self) -> str: ...


class _ArticulationHandle(Protocol):
    def get_active_joints(self) -> list[_RuntimeJointHandle]: ...

    def get_qpos(self) -> np.ndarray: ...

    def set_qpos(self, qpos: np.ndarray) -> None: ...


class _PatchedPlannerHandle(Protocol):
    frame_bias: list[float]
    _robo_orchard_entity_T_curobo_base: object | None  # noqa: N815

    def plan_path(self, *args: object, **kwargs: object) -> object: ...


class _PlannerRobotHandle(Protocol):
    communication_flag: bool
    left_delta_matrix: np.ndarray
    left_global_trans_matrix: np.ndarray
    left_gripper_bias: float
    right_delta_matrix: np.ndarray
    right_global_trans_matrix: np.ndarray
    right_gripper_bias: float
    left_ee: _RuntimeJointHandle
    right_ee: _RuntimeJointHandle
    left_arm_joints: list[_RuntimeJointHandle]
    left_entity: _ArticulationHandle
    right_arm_joints: list[_RuntimeJointHandle]
    right_entity: _ArticulationHandle
    left_conn: _ConnectionHandle | None
    right_conn: _ConnectionHandle | None
    left_proc: _ProcessHandle | None
    right_proc: _ProcessHandle | None
    left_planner: _PatchedPlannerHandle
    right_planner: _PatchedPlannerHandle

    def get_left_arm_real_jointState(self) -> list[float]: ...  # noqa: N802

    def get_right_arm_real_jointState(self) -> list[float]: ...  # noqa: N802

    def get_left_ee_pose(self) -> list[float]: ...

    def get_right_ee_pose(self) -> list[float]: ...


class _TaskRuntimeHandle(Protocol):
    robot: _PlannerRobotHandle
    viewer: _ViewerHandle | None


@dataclass(frozen=True, slots=True)
class _RuntimeHandles:
    task: _TaskRuntimeHandle
    viewer: _ViewerHandle | None
    connections: tuple[_ConnectionHandle, ...]
    processes: tuple[_ProcessHandle, ...]
    worker_pids: tuple[int, ...]


@dataclass(slots=True)
class _LifecycleCapture:
    runtime_handles: list[_RuntimeHandles]
    closed_envs: list[RoboTwinEnv]


_CASES = (
    _EmbodimentCase(
        "aloha",
        "embodiment: [aloha-agilex]",
        "combined_articulation",
        14,
    ),
    _EmbodimentCase(
        "ur5_wsg",
        "embodiment: [ur5-wsg, ur5-wsg, 0.8]",
        "split_articulations",
        14,
    ),
    _EmbodimentCase(
        "arx_x5",
        "embodiment: [ARX-X5, ARX-X5, 0.8]",
        "split_articulations",
        14,
    ),
    _EmbodimentCase(
        "franka_panda",
        "embodiment: [franka-panda, franka-panda, 0.8]",
        "split_articulations",
        16,
    ),
    _EmbodimentCase(
        "piper",
        "embodiment: [piper, piper, 0.8]",
        "split_articulations",
        14,
    ),
    _EmbodimentCase(
        "piper_franka",
        "embodiment: [piper, franka-panda, 0.8]",
        "split_articulations",
        15,
        2,
    ),
)
_CASES_BY_ID = {case.case_id: case for case in _CASES}
# Keep CI runtime bounded while covering the combined, heterogeneous, qpos,
# and EE execution contracts. Asset preflight still validates every known case.
_DIRECT_RUNTIME_CASES: tuple[tuple[str, Literal["qpos", "ee"]], ...] = (
    ("aloha", "qpos"),
    ("piper_franka", "ee"),
)


@pytest.fixture(scope="session", autouse=True)
def _configure_evidence_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Use worker-local pytest storage unless explicitly overridden."""
    if _EVIDENCE_ENV in os.environ:
        yield
        return
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(
            _EVIDENCE_ENV,
            str(tmp_path_factory.mktemp("robotwin-embodiment-evidence")),
        )
        yield


@pytest.fixture(scope="session")
def _runtime_capability() -> dict[str, Any]:
    completed, result = _invoke_worker("capability", evidence_id="capability")
    assert completed.returncode == 0, _format_worker_failure(completed)
    assert result["asset_preflight_ok"], result
    return result


@pytest.mark.parametrize(
    ("case_id", "action_type"),
    _DIRECT_RUNTIME_CASES,
)
def test_direct_env_embodiment_matrix(
    case_id: str,
    action_type: Literal["qpos", "ee"],
    tmp_path: Path,
    _runtime_capability: dict[str, Any],
) -> None:
    """Exercise combined qpos and heterogeneous EE runtime cells."""
    _require_runtime_capability(_runtime_capability)
    case = _CASES_BY_ID[case_id]
    config_path, digest = _write_task_config(tmp_path, case)
    completed, result = _invoke_worker(
        "direct",
        case=case,
        action_type=action_type,
        config_path=config_path,
        evidence_id=f"direct-{case.case_id}-{action_type}",
        expected_digest=digest,
    )
    assert completed.returncode == 0, _format_worker_failure(completed)
    assert result["status"] == "passed"
    assert result["config_sha256"] == digest


def test_fk_and_state_round_trip_reuse_piper_franka_runtime(
    tmp_path: Path,
    _runtime_capability: dict[str, Any],
) -> None:
    """Reuse one qpos runtime for FK samples before State recreation."""
    _require_runtime_capability(_runtime_capability)
    case = _CASES_BY_ID["piper_franka"]
    config_path, digest = _write_task_config(tmp_path, case)
    completed, result = _invoke_worker(
        "fk_state",
        case=case,
        config_path=config_path,
        evidence_id=f"fk-state-{case.case_id}",
        expected_digest=digest,
    )
    assert completed.returncode == 0, _format_worker_failure(completed)
    assert result["status"] == "passed"
    assert result["fk"]["sample_count"] == _FK_DISTRIBUTED_SAMPLE_COUNT
    assert result["fk"]["minimum_normalized_joint_span"] >= 0.8
    assert result["config_sha256"] == digest
    assert result["state"]["restored_config_sha256"] == digest


def test_ee_curobo_base_patch_supported_aloha(
    tmp_path: Path,
    _runtime_capability: dict[str, Any],
) -> None:
    """Exercise supported EE patch mode for the combined Aloha runtime."""
    _require_runtime_capability(_runtime_capability)
    case = _CASES_BY_ID["aloha"]
    config_path, digest = _write_task_config(tmp_path, case)
    completed, result = _invoke_worker(
        "direct",
        case=case,
        action_type="ee",
        patch=True,
        config_path=config_path,
        evidence_id=f"patch-{case.case_id}-ee",
        expected_digest=digest,
    )
    assert completed.returncode == 0, _format_worker_failure(completed)
    assert result["status"] == "passed"


def test_heterogeneous_ee_patch_rejects_without_worker_leak(
    tmp_path: Path,
    _runtime_capability: dict[str, Any],
) -> None:
    """Reject heterogeneous EE patch mode and reap both planner workers."""
    _require_runtime_capability(_runtime_capability)
    case = _CASES_BY_ID["piper_franka"]
    config_path, digest = _write_task_config(tmp_path, case)
    completed, result = _invoke_worker(
        "reject_patch",
        case=case,
        action_type="ee",
        patch=True,
        config_path=config_path,
        evidence_id="patch-piper-franka-rejection",
        expected_digest=digest,
    )
    assert completed.returncode == 0, _format_worker_failure(completed)
    assert result["rejected_as_expected"] is True
    assert result["cleanup"]["active_child_pids_after_close"] == []


def test_local_benchmark_evaluator_smoke(
    tmp_path: Path,
    _runtime_capability: dict[str, Any],
) -> None:
    """Run one heterogeneous EE benchmark attempt without task success."""
    _require_runtime_capability(_runtime_capability)
    case = _CASES_BY_ID["piper_franka"]
    action_type: Literal["qpos", "ee"] = "ee"
    config_path, digest = _write_task_config(tmp_path, case)
    completed, result = _invoke_worker(
        "evaluator",
        case=case,
        action_type=action_type,
        config_path=config_path,
        evidence_id=f"evaluator-{case.case_id}-{action_type}",
        expected_digest=digest,
    )
    assert completed.returncode == 0, _format_worker_failure(completed)
    assert result["status"] == "passed"
    assert result["episode_error_type"] is None


def _require_runtime_capability(capability: dict[str, Any]) -> None:
    if not capability["cuda_available"]:
        pytest.fail("RoboTwin embodiment matrix requires CUDA.", pytrace=False)
    if not capability["renderer_available"]:
        pytest.fail(
            "RoboTwin embodiment matrix requires a usable SAPIEN Vulkan "
            f"renderer: {capability['renderer_error']}",
            pytrace=False,
        )


def _robotwin_root() -> Path:
    value = os.environ.get("RoboTwin_PATH")
    if not value:
        raise RuntimeError("RoboTwin_PATH is required for the real matrix.")
    root = Path(value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboTwin_PATH does not exist: {root}")
    return root


def _write_task_config(
    output_dir: Path,
    case: _EmbodimentCase,
) -> tuple[Path, str]:
    base_content = (
        _robotwin_root() / "task_config/demo_clean.yml"
    ).read_bytes()
    if base_content.count(_BASE_EMBODIMENT_LINE) != 1:
        raise AssertionError(
            "demo_clean.yml must contain exactly one canonical embodiment line"
        )
    content = base_content.replace(
        _BASE_EMBODIMENT_LINE,
        case.embodiment_line.encode("utf-8"),
    )
    output_path = output_dir / f"{case.case_id}.yml"
    output_path.write_bytes(content)
    return output_path, hashlib.sha256(content).hexdigest()


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "COV_CORE_CONFIG",
        "COV_CORE_DATAFILE",
        "COV_CORE_SOURCE",
        "COVERAGE_PROCESS_START",
    ):
        env.pop(key, None)
    return env


def _invoke_worker(
    worker: Literal[
        "capability",
        "direct",
        "fk_state",
        "reject_patch",
        "evaluator",
    ],
    *,
    evidence_id: str,
    case: _EmbodimentCase | None = None,
    action_type: Literal["qpos", "ee"] | None = None,
    patch: bool = False,
    config_path: Path | None = None,
    expected_digest: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
    ]
    if case is not None:
        command.extend(["--case", case.case_id])
    if action_type is not None:
        command.extend(["--action-type", action_type])
    if patch:
        command.append("--patch")
    if config_path is not None:
        command.extend(["--config-path", str(config_path)])

    started_at = time.monotonic()
    completed = _run_worker_process(command)
    duration_s = time.monotonic() - started_at
    result = _parse_result(completed.stdout)
    evidence = {
        "evidence_id": evidence_id,
        "worker": worker,
        "case_id": None if case is None else case.case_id,
        "action_type": action_type,
        "patch": patch,
        "expected_config_sha256": expected_digest,
        "duration_s": duration_s,
        "returncode": completed.returncode,
        "result": result,
        "stdout": "\n".join(
            line
            for line in completed.stdout.splitlines()
            if not line.startswith(_RESULT_MARKER)
        ),
        "stderr": completed.stderr,
    }
    _write_evidence(evidence_id, evidence)
    return completed, result


def _run_worker_process(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run one isolated cell and reap its full process group on timeout."""
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=_subprocess_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_WORKER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_worker_process_group(process)
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=stdout,
            stderr=stderr + "\nworker process group timed out and was reaped",
        )
    except BaseException:
        _terminate_worker_process_group(process)
        raise
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _terminate_worker_process_group(
    process: subprocess.Popen[str],
) -> tuple[str, str]:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def _parse_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            value = json.loads(line.removeprefix(_RESULT_MARKER))
            if not isinstance(value, dict):
                raise AssertionError("Worker result must be a JSON object.")
            return value
    return {"status": "missing_result"}


def _write_evidence(evidence_id: str, evidence: dict[str, Any]) -> None:
    try:
        output_dir = Path(os.environ[_EVIDENCE_ENV])
    except KeyError as exc:
        raise RuntimeError(
            "RoboTwin evidence output must be configured by pytest or "
            f"{_EVIDENCE_ENV}."
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{evidence_id}.json"
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    assert output_path.is_file()


def _format_worker_failure(
    completed: subprocess.CompletedProcess[str],
) -> str:
    return "\n".join(
        [
            f"worker returncode={completed.returncode}",
            "stdout:",
            completed.stdout,
            "stderr:",
            completed.stderr,
        ]
    )


def _capability_result() -> dict[str, Any]:
    root = _robotwin_root()
    asset_results: dict[str, Any] = {}
    asset_error: str | None = None
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for case in _CASES:
                config_path, digest = _write_task_config(output_dir, case)
                cfg = RoboTwinEnvCfg(
                    task_name="place_empty_cup",
                    task_config_path=str(config_path),
                    check_expert=False,
                    check_task_init=False,
                )
                task_config = cfg.get_task_config_for_seed(0)
                asset_results[case.case_id] = _asset_preflight(
                    task_config,
                    digest=digest,
                )
    except BaseException as exc:
        asset_error = f"{type(exc).__name__}: {exc}"

    display, display_error = _display_preflight()
    renderer_summary: str | None = None
    renderer_error: str | None = None
    try:
        import sapien

        sapien.SapienRenderer()
        renderer_summary = str(sapien.render.get_device_summary())
    except BaseException as exc:
        renderer_error = f"{type(exc).__name__}: {exc}"

    render_nodes = sorted(Path("/dev/dri").glob("renderD*"))
    return {
        "status": "passed",
        "robotwin_root": str(root),
        "robotwin_snapshot_id": root.name,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "display": display,
        "display_available": display_error is None,
        "display_error": display_error,
        "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
        "render_nodes": [
            {
                "path": str(path),
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
            }
            for path in render_nodes
        ],
        "renderer_available": renderer_error is None,
        "renderer_summary": renderer_summary,
        "renderer_error": renderer_error,
        "asset_preflight_ok": asset_error is None,
        "asset_preflight_error": asset_error,
        "assets": asset_results,
    }


def _display_preflight() -> tuple[str | None, str | None]:
    display = os.environ.get("DISPLAY")
    if not display:
        return None, "DISPLAY is not set."
    try:
        completed = subprocess.run(
            ["xdpyinfo", "-display", display],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except FileNotFoundError:
        return display, "xdpyinfo is not available."
    except subprocess.TimeoutExpired:
        return display, f"xdpyinfo timed out for {display!r}."
    if completed.returncode == 0:
        return display, None
    detail = completed.stderr.strip() or completed.stdout.strip()
    return display, (
        f"xdpyinfo failed for {display!r} with exit code "
        f"{completed.returncode}: {detail}"
    )


def _asset_preflight(
    task_config: dict[str, Any],
    *,
    digest: str,
) -> dict[str, Any]:
    checked_paths: list[str] = []
    for side in ("left", "right"):
        robot_dir = Path(task_config[f"{side}_robot_file"])
        config = task_config[f"{side}_embodiment_config"]
        urdf_path = robot_dir / config["urdf_path"]
        paths = [robot_dir / "config.yml", urdf_path]
        srdf_path = config.get("srdf_path")
        if srdf_path:
            paths.append(robot_dir / srdf_path)
        curobo_paths = [robot_dir / "curobo.yml"]
        if task_config["dual_arm_embodied"]:
            curobo_paths = [
                robot_dir / "curobo_left.yml",
                robot_dir / "curobo_right.yml",
            ]
        paths.extend(curobo_paths)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing RoboTwin asset: {path}")
            checked_paths.append(str(path.resolve()))
    return {
        "config_sha256": digest,
        "embodiment_name": task_config["embodiment_name"],
        "dual_arm_embodied": task_config["dual_arm_embodied"],
        "checked_paths": sorted(set(checked_paths)),
    }


def _make_env(
    *,
    config_path: Path,
    action_type: Literal["qpos", "ee"],
    patch: bool,
) -> RoboTwinEnv:
    return RoboTwinEnv(
        RoboTwinEnvCfg(
            task_name="place_empty_cup",
            seed=0,
            episode_id=0,
            eval_mode=False,
            check_expert=False,
            check_task_init=False,
            format_datatypes=True,
            action_type=action_type,
            task_config_path=str(config_path),
            patch_curobo_base_transform=patch,
        )
    )


def _validate_observation(
    env: RoboTwinEnv,
    obs: dict[str, Any],
    case: _EmbodimentCase,
    *,
    expected_step_index: int,
) -> dict[str, Any]:
    assert obs["step_index"] == expected_step_index
    layout = env._require_runtime_layout()
    assert layout.topology == case.topology

    robots = obs["robots"]
    expected_robot_keys = (
        ["left"]
        if case.topology == "combined_articulation"
        else ["left", "right"]
    )
    assert list(robots) == expected_robot_keys
    urdf_map = env.get_robot_urdf()
    assert list(urdf_map) == expected_robot_keys
    for index, robot_key in enumerate(expected_robot_keys):
        robot = robots[robot_key]
        assert robot.name == robot_key
        assert robot.index == index
        assert robot.content == urdf_map[robot_key].decode("utf-8")

    joints = obs["joints"]
    assert isinstance(joints, BatchJointsState)
    assert joints.position is not None
    assert joints.names is not None
    assert joints.joint_num == case.qpos_width
    assert len(joints.names) == len(set(joints.names)) == case.qpos_width
    left_width = layout.left_arm_joint_count + 1
    right_width = layout.right_arm_joint_count + 1
    assert left_width + right_width == case.qpos_width
    if case.topology == "split_articulations":
        assert all(
            name.startswith("left/") for name in joints.names[:left_width]
        )
        assert all(
            name.startswith("right/") for name in joints.names[left_width:]
        )

    tf_graph = obs["tf"]
    assert isinstance(tf_graph, BatchFrameTransformGraph)
    assert len(tf_graph.nodes) == len(set(tf_graph.nodes))
    base_frames = (
        ["robot_base"]
        if case.topology == "combined_articulation"
        else ["left_robot_base", "right_robot_base"]
    )
    direct_tf_edges = {
        (transform.parent_frame_id, transform.child_frame_id)
        for transform in tf_graph.as_state().tf_list
    }
    base_xyz: dict[str, list[float]] = {}
    for frame_id in base_frames:
        assert ("world", frame_id) in direct_tf_edges
        transform = tf_graph.get_tf("world", frame_id)
        assert transform is not None
        base_xyz[frame_id] = transform.xyz[0].tolist()
    base_x_separation: float | None = None
    if case.topology == "split_articulations":
        base_x_separation = abs(
            base_xyz["right_robot_base"][0] - base_xyz["left_robot_base"][0]
        )
        assert base_x_separation == pytest.approx(0.8)
    assert (
        layout.left_control_eef_frame_id != layout.right_control_eef_frame_id
    )
    if case.topology == "split_articulations":
        assert layout.left_control_eef_frame_id.startswith("left/")
        assert layout.right_control_eef_frame_id.startswith("right/")
    robot = cast(_TaskRuntimeHandle, env.unwrapped_env()).robot
    actual_joint_eef = env._joints2ee_pose(
        np.asarray(
            robot.get_left_arm_real_jointState()
            + robot.get_right_arm_real_jointState(),
            dtype=np.float32,
        )
    )
    eef_pose_errors = {
        "left": _validate_eef_pose_consistency(
            tf_graph,
            target_joint_frame_id="left_eef_from_joint",
            actual_joint_eef=actual_joint_eef.left_eef,
            end_effector_joint=robot.left_ee,
            robotwin_frame_id=layout.left_control_eef_frame_id,
            global_trans_matrix=robot.left_global_trans_matrix,
            delta_matrix=robot.left_delta_matrix,
            gripper_bias=robot.left_gripper_bias,
        ),
        "right": _validate_eef_pose_consistency(
            tf_graph,
            target_joint_frame_id="right_eef_from_joint",
            actual_joint_eef=actual_joint_eef.right_eef,
            end_effector_joint=robot.right_ee,
            robotwin_frame_id=layout.right_control_eef_frame_id,
            global_trans_matrix=robot.right_global_trans_matrix,
            delta_matrix=robot.right_delta_matrix,
            gripper_bias=robot.right_gripper_bias,
        ),
    }
    return {
        "topology": layout.topology,
        "robot_keys": list(robots),
        "joint_width": joints.joint_num,
        "joint_names": joints.names,
        "tf_frame_ids": sorted(tf_graph.nodes),
        "base_xyz": base_xyz,
        "base_x_separation": base_x_separation,
        "eef_pose_errors": eef_pose_errors,
        "direct_tf_edges": sorted(direct_tf_edges),
    }


def _validate_eef_pose_consistency(
    tf_graph: BatchFrameTransformGraph,
    *,
    target_joint_frame_id: str,
    actual_joint_eef: BatchFrameTransform,
    end_effector_joint: _RuntimeJointHandle,
    robotwin_frame_id: str,
    global_trans_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    gripper_bias: float,
) -> dict[str, object]:
    """Compare target and actual-qpos FK with the live runtime poses."""
    world_to_target_joint_eef = tf_graph.get_tf("world", target_joint_frame_id)
    world_to_robotwin_eef = tf_graph.get_tf("world", robotwin_frame_id)
    assert isinstance(world_to_target_joint_eef, BatchFrameTransform)
    assert isinstance(world_to_robotwin_eef, BatchFrameTransform)

    world_to_target_control = _fk_eef_to_robotwin_control_pose(
        world_to_target_joint_eef,
        joint_pose_in_child=end_effector_joint.pose_in_child,
        robotwin_frame_id=robotwin_frame_id,
        global_trans_matrix=global_trans_matrix,
        delta_matrix=delta_matrix,
        gripper_bias=gripper_bias,
    )
    world_to_actual_control = _fk_eef_to_robotwin_control_pose(
        actual_joint_eef,
        joint_pose_in_child=end_effector_joint.pose_in_child,
        robotwin_frame_id=robotwin_frame_id,
        global_trans_matrix=global_trans_matrix,
        delta_matrix=delta_matrix,
        gripper_bias=gripper_bias,
    )
    target_error = _eef_pose_error(
        world_to_target_control,
        world_to_robotwin_eef,
        source="drive_target_fk",
    )
    actual_control_error = _eef_pose_error(
        world_to_actual_control,
        world_to_robotwin_eef,
        source="actual_qpos_fk_to_robotwin_control_eef",
    )
    world_to_actual_link = actual_joint_eef.model_copy(
        update={"child_frame_id": f"{robotwin_frame_id}_fk_link"}
    )
    world_to_runtime_link = _sapien_pose_to_frame_transform(
        end_effector_joint.child_link.entity_pose,
        parent_frame_id="world",
        child_frame_id=f"{robotwin_frame_id}_runtime_link",
    )
    actual_link_error = _eef_pose_error(
        world_to_actual_link,
        world_to_runtime_link,
        source="actual_qpos_fk_to_sapien_child_link",
    )
    assert actual_control_error["within_limit"], actual_control_error
    assert actual_link_error["within_limit"], actual_link_error
    return {
        "target_joint_frame_id": target_joint_frame_id,
        "robotwin_frame_id": robotwin_frame_id,
        "drive_target_fk": target_error,
        "actual_qpos_fk_to_sapien_child_link": actual_link_error,
        "actual_qpos_fk_to_robotwin_control_eef": actual_control_error,
    }


def _fk_eef_to_robotwin_control_pose(
    world_to_eef_link: BatchFrameTransform,
    *,
    joint_pose_in_child: _SapienPoseHandle,
    robotwin_frame_id: str,
    global_trans_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    gripper_bias: float,
) -> BatchFrameTransform:
    """Map a URDF child-link FK pose to RoboTwin's control EEF frame.

    RoboTwin starts from the SAPIEN joint's global pose, while URDF FK ends
    at that joint's child link. SAPIEN ``pose_in_child`` is therefore part of
    the frame mapping; omitting it creates embodiment-specific fixed offsets.
    """
    eef_link_frame_id = f"{robotwin_frame_id}_fk_link"
    joint_frame_id = f"{robotwin_frame_id}_joint"
    world_to_eef_link = world_to_eef_link.model_copy(
        update={"child_frame_id": eef_link_frame_id}
    )
    eef_link_to_joint = _sapien_pose_to_frame_transform(
        joint_pose_in_child,
        parent_frame_id=eef_link_frame_id,
        child_frame_id=joint_frame_id,
    )
    world_to_joint = eef_link_to_joint.compose(world_to_eef_link)
    joint_rotation = quaternion_to_matrix(world_to_joint.quat)
    control_rotation = (
        joint_rotation
        @ torch.as_tensor(
            global_trans_matrix,
            dtype=joint_rotation.dtype,
            device=joint_rotation.device,
        )
        @ torch.as_tensor(
            delta_matrix,
            dtype=joint_rotation.dtype,
            device=joint_rotation.device,
        )
    )
    control_offset = torch.tensor(
        [gripper_bias - 0.12, 0.0, 0.0],
        dtype=joint_rotation.dtype,
        device=joint_rotation.device,
    )
    control_xyz = world_to_joint.xyz + torch.matmul(
        control_rotation,
        control_offset,
    )
    return BatchFrameTransform(
        xyz=control_xyz,
        quat=matrix_to_quaternion(control_rotation),
        parent_frame_id="world",
        child_frame_id=robotwin_frame_id,
    )


def _sapien_pose_to_frame_transform(
    pose: _SapienPoseHandle,
    *,
    parent_frame_id: str,
    child_frame_id: str,
) -> BatchFrameTransform:
    """Convert a SAPIEN ``(w, x, y, z)`` pose at a typed frame boundary."""
    return BatchFrameTransform(
        xyz=torch.from_numpy(np.asarray(pose.get_p())).unsqueeze(0),
        quat=torch.from_numpy(np.asarray(pose.get_q())).unsqueeze(0),
        parent_frame_id=parent_frame_id,
        child_frame_id=child_frame_id,
    )


def _eef_pose_error(
    world_to_first_pose: BatchFrameTransform,
    world_to_second_pose: BatchFrameTransform,
    *,
    source: str,
) -> dict[str, float | str | bool]:
    """Measure position and orientation error between two world poses."""

    position_error_m = torch.linalg.vector_norm(
        world_to_first_pose.xyz.to(torch.float64)
        - world_to_second_pose.xyz.to(torch.float64),
        dim=-1,
    )
    first_quat = torch.nn.functional.normalize(
        world_to_first_pose.quat.to(torch.float64),
        dim=-1,
    )
    second_quat = torch.nn.functional.normalize(
        world_to_second_pose.quat.to(torch.float64),
        dim=-1,
    )
    quat_alignment = torch.sum(first_quat * second_quat, dim=-1).abs()
    orientation_error_rad = 2.0 * torch.acos(
        quat_alignment.clamp(min=0.0, max=1.0)
    )
    max_position_error_m = float(position_error_m.max().item())
    max_orientation_error_rad = float(orientation_error_rad.max().item())
    within_limit = (
        max_position_error_m <= _EEF_POSITION_ERROR_LIMIT_M
        and max_orientation_error_rad <= _EEF_ORIENTATION_ERROR_LIMIT_RAD
    )
    return {
        "source": source,
        "position_error_m": max_position_error_m,
        "orientation_error_rad": max_orientation_error_rad,
        "within_limit": within_limit,
    }


def _action_from_observation(
    obs: dict[str, Any],
    action_type: Literal["qpos", "ee"],
) -> np.ndarray:
    if action_type == "qpos":
        joints = obs["joints"]
        assert isinstance(joints, BatchJointsState)
        assert joints.position is not None
        return joints.position[0].detach().cpu().numpy().astype(np.float32)
    endpose = obs["endpose"]
    return np.concatenate(
        [
            np.asarray(endpose["left_endpose"], dtype=np.float32),
            np.asarray([endpose["left_gripper"]], dtype=np.float32),
            np.asarray(endpose["right_endpose"], dtype=np.float32),
            np.asarray([endpose["right_gripper"]], dtype=np.float32),
        ]
    )


def _patch_planners(task: object) -> tuple[_PatchedPlannerHandle, ...]:
    robot = cast(_TaskRuntimeHandle, task).robot
    return (robot.left_planner, robot.right_planner)


def _validate_patch_active(task: object) -> dict[str, Any]:
    planner_evidence = []
    for planner in _patch_planners(task):
        class_patched = bool(
            getattr(
                type(planner),
                "_robo_orchard_base_transform_patched",
                False,
            )
        )
        transform_configured = (
            planner._robo_orchard_entity_T_curobo_base is not None
        )
        frame_bias = list(planner.frame_bias)
        assert class_patched
        assert transform_configured
        assert frame_bias == pytest.approx([0.0, 0.0, 0.0])
        planner_evidence.append(
            {
                "class_patched": class_patched,
                "transform_configured": transform_configured,
                "frame_bias": frame_bias,
            }
        )
    return {"planners": planner_evidence}


def _install_patch_plan_capture(
    task: object,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    statuses: list[str] = []
    for planner in _patch_planners(task):
        original_plan_path = planner.plan_path

        def captured_plan_path(
            *args: object,
            _original_plan_path=original_plan_path,
            **kwargs: object,
        ) -> object:
            result = _original_plan_path(*args, **kwargs)
            assert isinstance(result, dict)
            status = result.get("status")
            assert isinstance(status, str)
            statuses.append(status)
            return result

        monkeypatch.setattr(planner, "plan_path", captured_plan_path)
    return statuses


def _capture_runtime_handles(task: object) -> _RuntimeHandles:
    runtime_task = cast(_TaskRuntimeHandle, task)
    robot = runtime_task.robot
    try:
        viewer = runtime_task.viewer
    except AttributeError:
        viewer = None
    if robot.communication_flag:
        connections = tuple(
            connection
            for connection in (robot.left_conn, robot.right_conn)
            if connection is not None
        )
        processes = tuple(
            process
            for process in (robot.left_proc, robot.right_proc)
            if process is not None
        )
    else:
        connections = ()
        processes = ()
    worker_pids = tuple(
        process.pid for process in processes if process.pid is not None
    )
    return _RuntimeHandles(
        task=runtime_task,
        viewer=viewer,
        connections=connections,
        processes=processes,
        worker_pids=worker_pids,
    )


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_cleanup_status(process: _ProcessHandle) -> dict[str, bool]:
    try:
        alive = process.is_alive()
    except ValueError:
        return {"object_closed": True, "alive": False}
    return {"object_closed": False, "alive": alive}


def _install_lifecycle_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> _LifecycleCapture:
    capture = _LifecycleCapture(runtime_handles=[], closed_envs=[])
    original_dispose = robotwin_env_module.dispose_task_runtime
    original_close = RoboTwinEnv.close

    def captured_dispose(task: object, *, clear_cache: bool) -> bool:
        capture.runtime_handles.append(_capture_runtime_handles(task))
        return original_dispose(task, clear_cache=clear_cache)

    def captured_close(env: RoboTwinEnv, clear_cache: bool = True) -> None:
        try:
            original_close(env, clear_cache=clear_cache)
        finally:
            if all(candidate is not env for candidate in capture.closed_envs):
                capture.closed_envs.append(env)

    monkeypatch.setattr(
        robotwin_env_module,
        "dispose_task_runtime",
        captured_dispose,
    )
    monkeypatch.setattr(RoboTwinEnv, "close", captured_close)
    return capture


def _cleanup_evidence(
    envs: list[RoboTwinEnv],
    runtime_handles: list[_RuntimeHandles],
    *,
    expected_workers_per_runtime: int,
) -> dict[str, Any]:
    assert envs
    assert runtime_handles
    assert all(
        len(handles.connections) == expected_workers_per_runtime
        and len(handles.processes) == expected_workers_per_runtime
        for handles in runtime_handles
    )
    worker_pids = sorted(
        pid for handles in runtime_handles for pid in handles.worker_pids
    )
    active_child_pids = sorted(
        process.pid
        for process in multiprocessing.active_children()
        if process.pid is not None
    )
    connection_closed = [
        connection.closed
        for handles in runtime_handles
        for connection in handles.connections
    ]
    process_status = [
        _process_cleanup_status(process)
        for handles in runtime_handles
        for process in handles.processes
    ]
    viewer_window_cleared = [
        handles.viewer is None or handles.viewer.window is None
        for handles in runtime_handles
    ]
    viewer_present = [
        handles.viewer is not None for handles in runtime_handles
    ]
    robot_handles_cleared = [
        not handles.task.robot.communication_flag
        or (
            handles.task.robot.left_conn is None
            and handles.task.robot.right_conn is None
            and handles.task.robot.left_proc is None
            and handles.task.robot.right_proc is None
        )
        for handles in runtime_handles
    ]
    env_cleanup = [
        {
            "task_cleared": env._task is None,
            "pending_disposals": len(env._pending_disposal_tasks),
            "video_writer_closed": env._video_writer.is_closed,
        }
        for env in envs
    ]
    cleanup = {
        "worker_pids_before_close": worker_pids,
        "worker_pids_alive_after_close": [
            pid for pid in worker_pids if _pid_is_alive(pid)
        ],
        "active_child_pids_after_close": active_child_pids,
        "expected_workers_per_runtime": expected_workers_per_runtime,
        "runtime_count": len(runtime_handles),
        "connection_closed": connection_closed,
        "process_status": process_status,
        "viewer_present": viewer_present,
        "viewer_window_cleared": viewer_window_cleared,
        "robot_handles_cleared": robot_handles_cleared,
        "env_cleanup": env_cleanup,
    }
    assert cleanup["worker_pids_alive_after_close"] == []
    assert cleanup["active_child_pids_after_close"] == []
    assert all(connection_closed)
    assert all(status["object_closed"] for status in process_status)
    assert not any(status["alive"] for status in process_status)
    assert all(viewer_window_cleared)
    assert all(robot_handles_cleared)
    assert all(item["task_cleared"] for item in env_cleanup)
    assert not any(item["pending_disposals"] for item in env_cleanup)
    assert all(item["video_writer_closed"] for item in env_cleanup)
    return cleanup


def _run_direct_worker(
    case: _EmbodimentCase,
    config_path: Path,
    action_type: Literal["qpos", "ee"],
    patch: bool,
) -> dict[str, Any]:
    env = _make_env(
        config_path=config_path,
        action_type=action_type,
        patch=patch,
    )
    runtime_handles: _RuntimeHandles | None = None
    try:
        obs, info = env.reset(clear_cache=True, episode_id=0)
        assert obs is not None
        task = env.unwrapped_env()
        runtime_handles = _capture_runtime_handles(task)
        assert env._video_writer.is_closed
        reset_observation = _validate_observation(
            env,
            obs,
            case,
            expected_step_index=0,
        )
        action = _action_from_observation(obs, action_type)
        assert action.shape == (
            case.qpos_width if action_type == "qpos" else 16,
        )
        patch_evidence = _validate_patch_active(task) if patch else None
        if patch:
            with pytest.MonkeyPatch.context() as monkeypatch:
                plan_statuses = _install_patch_plan_capture(
                    task,
                    monkeypatch,
                )
                step_return = env.step(action)
            assert plan_statuses == ["Success", "Success"]
        else:
            plan_statuses = []
            step_return = env.step(action)
        assert step_return.observations is not None
        step_observation = _validate_observation(
            env,
            step_return.observations,
            case,
            expected_step_index=1,
        )
    except BaseException:
        env.close(clear_cache=True)
        raise
    env.close(clear_cache=True)
    assert runtime_handles is not None
    return {
        "status": "passed",
        "case_id": case.case_id,
        "action_type": action_type,
        "patch": patch,
        "config_sha256": env.cfg._task_config_content_sha256,
        "reset_info": info,
        "reset_observation": reset_observation,
        "step_observation": step_observation,
        "action_width": int(action.shape[0]),
        "patch_evidence": patch_evidence,
        "plan_statuses": plan_statuses,
        "cleanup": _cleanup_evidence(
            [env],
            [runtime_handles],
            expected_workers_per_runtime=case.planner_worker_count,
        ),
    }


def _collect_fk_distribution_evidence(
    env: RoboTwinEnv,
    case: _EmbodimentCase,
) -> dict[str, Any]:
    """Run the FK samples on an already initialized qpos runtime."""
    task = cast(_TaskRuntimeHandle, env.unwrapped_env())
    robot = task.robot
    saved_articulation_qpos = _capture_articulation_qpos(robot)
    try:
        left_samples, right_samples, joint_domains = (
            _distributed_arm_joint_samples(robot, case.case_id)
        )
        sample_evidence = []
        for sample_index, (left_joints, right_joints) in enumerate(
            zip(left_samples, right_samples, strict=True)
        ):
            _set_arm_joint_positions(robot, left_joints, right_joints)
            left_readback = np.asarray(
                robot.get_left_arm_real_jointState()[:-1],
                dtype=np.float64,
            )
            right_readback = np.asarray(
                robot.get_right_arm_real_jointState()[:-1],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                left_readback,
                left_joints,
                atol=1e-6,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                right_readback,
                right_joints,
                atol=1e-6,
                rtol=0.0,
            )

            joint_eef = env._joints2ee_pose(
                np.concatenate([left_joints, right_joints]).astype(np.float32)
            )
            left_errors = _sampled_fk_eef_errors(
                joint_eef.left_eef,
                robot.left_ee,
                robot.get_left_ee_pose(),
                robotwin_frame_id="left_robotwin_control_eef",
                global_trans_matrix=robot.left_global_trans_matrix,
                delta_matrix=robot.left_delta_matrix,
                gripper_bias=robot.left_gripper_bias,
            )
            right_errors = _sampled_fk_eef_errors(
                joint_eef.right_eef,
                robot.right_ee,
                robot.get_right_ee_pose(),
                robotwin_frame_id="right_robotwin_control_eef",
                global_trans_matrix=robot.right_global_trans_matrix,
                delta_matrix=robot.right_delta_matrix,
                gripper_bias=robot.right_gripper_bias,
            )
            result = {
                "sample_index": sample_index,
                "left_joints": left_joints.tolist(),
                "right_joints": right_joints.tolist(),
                "left_errors": left_errors,
                "right_errors": right_errors,
            }
            assert left_errors["child_link"]["within_limit"], result
            assert left_errors["control_eef"]["within_limit"], result
            assert right_errors["child_link"]["within_limit"], result
            assert right_errors["control_eef"]["within_limit"], result
            sample_evidence.append(result)
    finally:
        _restore_articulation_qpos(saved_articulation_qpos)

    return {
        "sample_strategy": "deterministic_latin_hypercube",
        "sample_count": _FK_DISTRIBUTED_SAMPLE_COUNT,
        "joint_limit_margin_ratio": _FK_JOINT_LIMIT_MARGIN_RATIO,
        "joint_domains": joint_domains,
        "minimum_normalized_joint_span": min(
            float(domain["normalized_sample_span"]) for domain in joint_domains
        ),
        "frame_contract": {
            "child_link": "URDF FK compared with SAPIEN child-link pose",
            "control_eef": (
                "FK child link mapped through SAPIEN pose_in_child and "
                "RoboTwin's control transform before comparison"
            ),
        },
        "samples": sample_evidence,
        "max_child_link_position_error_m": max(
            float(arm_error["position_error_m"])
            for sample in sample_evidence
            for arm_error in (
                sample["left_errors"]["child_link"],
                sample["right_errors"]["child_link"],
            )
        ),
        "max_child_link_orientation_error_rad": max(
            float(arm_error["orientation_error_rad"])
            for sample in sample_evidence
            for arm_error in (
                sample["left_errors"]["child_link"],
                sample["right_errors"]["child_link"],
            )
        ),
        "max_control_eef_position_error_m": max(
            float(arm_error["position_error_m"])
            for sample in sample_evidence
            for arm_error in (
                sample["left_errors"]["control_eef"],
                sample["right_errors"]["control_eef"],
            )
        ),
        "max_control_eef_orientation_error_rad": max(
            float(arm_error["orientation_error_rad"])
            for sample in sample_evidence
            for arm_error in (
                sample["left_errors"]["control_eef"],
                sample["right_errors"]["control_eef"],
            )
        ),
        "max_unmapped_link_to_control_position_error_m": max(
            float(arm_error["position_error_m"])
            for sample in sample_evidence
            for arm_error in (
                sample["left_errors"]["unmapped_link_to_control"],
                sample["right_errors"]["unmapped_link_to_control"],
            )
        ),
        "max_unmapped_link_to_control_orientation_error_rad": max(
            float(arm_error["orientation_error_rad"])
            for sample in sample_evidence
            for arm_error in (
                sample["left_errors"]["unmapped_link_to_control"],
                sample["right_errors"]["unmapped_link_to_control"],
            )
        ),
    }


def _run_fk_state_worker(
    case: _EmbodimentCase,
    config_path: Path,
) -> dict[str, Any]:
    """Run FK sampling and State restoration on one initial runtime."""
    env = _make_env(
        config_path=config_path,
        action_type="qpos",
        patch=False,
    )
    runtime_handles: list[_RuntimeHandles] = []
    try:
        obs, _ = env.reset(clear_cache=True, episode_id=0)
        assert obs is not None
        old_task = env.unwrapped_env()
        old_runtime_handles = _capture_runtime_handles(old_task)
        runtime_handles.append(old_runtime_handles)
        state = env.get_state()

        fk_evidence = _collect_fk_distribution_evidence(env, case)

        restored_obs, restored_info = env.reset_from_state(state)
        assert restored_obs is not None
        assert env.unwrapped_env() is not old_task
        restored_observation = _validate_observation(
            env,
            restored_obs,
            case,
            expected_step_index=0,
        )
        assert [
            pid
            for pid in old_runtime_handles.worker_pids
            if _pid_is_alive(pid)
        ] == []
        final_task = env.unwrapped_env()
        runtime_handles.append(_capture_runtime_handles(final_task))
    except BaseException:
        env.close(clear_cache=True)
        raise

    env.close(clear_cache=True)
    assert isinstance(state.config, RoboTwinEnvCfg)
    return {
        "status": "passed",
        "case_id": case.case_id,
        "config_sha256": env.cfg._task_config_content_sha256,
        "fk": fk_evidence,
        "state": {
            "restored_config_sha256": state.config._task_config_content_sha256,
            "restored_info": restored_info,
            "restored_observation": restored_observation,
            "old_worker_pids": old_runtime_handles.worker_pids,
        },
        "cleanup": _cleanup_evidence(
            [env],
            runtime_handles,
            expected_workers_per_runtime=case.planner_worker_count,
        ),
    }


def _distributed_arm_joint_samples(
    robot: _PlannerRobotHandle,
    case_id: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    tagged_joints = [
        *(("left", joint) for joint in robot.left_arm_joints),
        *(("right", joint) for joint in robot.right_arm_joints),
    ]
    domains: list[dict[str, float | int | str]] = []
    lower_bounds = []
    upper_bounds = []
    for arm, joint in tagged_joints:
        lower, upper = _safe_joint_sampling_bounds(joint)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        domains.append(
            {
                "joint": f"{arm}/{joint.get_name()}",
                "sample_lower": lower,
                "sample_upper": upper,
            }
        )

    sample_count = _FK_DISTRIBUTED_SAMPLE_COUNT
    fractions = (np.arange(sample_count, dtype=np.float64) + 0.5) / (
        sample_count
    )
    rng = np.random.default_rng(
        sum((index + 1) * ord(char) for index, char in enumerate(case_id))
    )
    samples = np.empty((sample_count, len(tagged_joints)), dtype=np.float64)
    lower_array = np.asarray(lower_bounds, dtype=np.float64)
    upper_array = np.asarray(upper_bounds, dtype=np.float64)
    for joint_index in range(len(tagged_joints)):
        joint_fractions = rng.permutation(fractions)
        samples[:, joint_index] = (
            lower_array[joint_index]
            + (upper_array[joint_index] - lower_array[joint_index])
            * joint_fractions
        )

    for joint_index, domain in enumerate(domains):
        joint_samples = samples[:, joint_index]
        normalized_span = float(
            np.ptp(joint_samples)
            / (upper_array[joint_index] - lower_array[joint_index])
        )
        normalized_samples = (joint_samples - lower_array[joint_index]) / (
            upper_array[joint_index] - lower_array[joint_index]
        )
        covered_strata_count = int(
            np.unique(
                np.floor(normalized_samples * sample_count).astype(np.int64)
            ).size
        )
        domain.update(
            {
                "sample_min": float(joint_samples.min()),
                "sample_max": float(joint_samples.max()),
                "normalized_sample_span": normalized_span,
                "distinct_sample_count": int(np.unique(joint_samples).size),
                "covered_strata_count": covered_strata_count,
            }
        )
        assert normalized_span >= 0.8, domain
        assert domain["distinct_sample_count"] == sample_count, domain
        assert covered_strata_count == sample_count, domain

    left_width = len(robot.left_arm_joints)
    return samples[:, :left_width], samples[:, left_width:], domains


def _safe_joint_sampling_bounds(
    joint: _RuntimeJointHandle,
) -> tuple[float, float]:
    limits = np.asarray(joint.get_limits(), dtype=np.float64).reshape(-1, 2)
    if limits.shape[0] != 1:
        raise ValueError(
            f"Expected one limit interval for joint {joint.get_name()!r}, "
            f"got shape {tuple(limits.shape)}."
        )
    lower, upper = (float(value) for value in limits[0])
    if not np.isfinite(lower):
        lower = -float(np.pi)
    if not np.isfinite(upper):
        upper = float(np.pi)
    if lower >= upper:
        raise ValueError(
            f"Joint {joint.get_name()!r} has invalid limits "
            f"[{lower}, {upper}]."
        )
    margin = (upper - lower) * _FK_JOINT_LIMIT_MARGIN_RATIO
    return lower + margin, upper - margin


def _capture_articulation_qpos(
    robot: _PlannerRobotHandle,
) -> list[tuple[_ArticulationHandle, np.ndarray]]:
    articulations = [robot.left_entity]
    if robot.right_entity is not robot.left_entity:
        articulations.append(robot.right_entity)
    return [
        (articulation, np.asarray(articulation.get_qpos()).copy())
        for articulation in articulations
    ]


def _restore_articulation_qpos(
    saved_qpos: list[tuple[_ArticulationHandle, np.ndarray]],
) -> None:
    for articulation, qpos in saved_qpos:
        articulation.set_qpos(qpos)


def _set_arm_joint_positions(
    robot: _PlannerRobotHandle,
    left_joints: np.ndarray,
    right_joints: np.ndarray,
) -> None:
    if robot.left_entity is robot.right_entity:
        _set_articulation_joint_positions(
            robot.left_entity,
            [
                *zip(robot.left_arm_joints, left_joints, strict=True),
                *zip(robot.right_arm_joints, right_joints, strict=True),
            ],
        )
        return
    _set_articulation_joint_positions(
        robot.left_entity,
        zip(robot.left_arm_joints, left_joints, strict=True),
    )
    _set_articulation_joint_positions(
        robot.right_entity,
        zip(robot.right_arm_joints, right_joints, strict=True),
    )


def _set_articulation_joint_positions(
    articulation: _ArticulationHandle,
    joint_values: Iterable[tuple[_RuntimeJointHandle, float]],
) -> None:
    qpos = np.asarray(articulation.get_qpos()).copy()
    active_joint_indices = {
        joint.get_name(): index
        for index, joint in enumerate(articulation.get_active_joints())
    }
    for joint, value in joint_values:
        qpos[active_joint_indices[joint.get_name()]] = value
    articulation.set_qpos(qpos)


def _sampled_fk_eef_errors(
    world_to_fk_eef: BatchFrameTransform,
    end_effector_joint: _RuntimeJointHandle,
    robotwin_pose: list[float],
    *,
    robotwin_frame_id: str,
    global_trans_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    gripper_bias: float,
) -> dict[str, dict[str, float | str | bool]]:
    fk_link_frame_id = f"{robotwin_frame_id}_fk_link"
    world_to_fk_link = world_to_fk_eef.model_copy(
        update={"child_frame_id": fk_link_frame_id}
    )
    world_to_sapien_link = _sapien_pose_to_frame_transform(
        end_effector_joint.child_link.entity_pose,
        parent_frame_id="world",
        child_frame_id=f"{robotwin_frame_id}_sapien_link",
    )
    world_to_fk_control = _fk_eef_to_robotwin_control_pose(
        world_to_fk_link,
        joint_pose_in_child=end_effector_joint.pose_in_child,
        robotwin_frame_id=robotwin_frame_id,
        global_trans_matrix=global_trans_matrix,
        delta_matrix=delta_matrix,
        gripper_bias=gripper_bias,
    )
    world_to_robotwin_eef = _robotwin_pose_vector_to_tf(
        robotwin_pose,
        child_frame_id=robotwin_frame_id,
    )
    return {
        "child_link": _eef_pose_error(
            world_to_fk_link,
            world_to_sapien_link,
            source="sampled_fk_to_sapien_child_link",
        ),
        "control_eef": _eef_pose_error(
            world_to_fk_control,
            world_to_robotwin_eef,
            source="sampled_fk_to_robotwin_control_eef",
        ),
        "unmapped_link_to_control": _eef_pose_error(
            world_to_fk_link,
            world_to_robotwin_eef,
            source="different_frames_non_gating",
        ),
    }


def _robotwin_pose_vector_to_tf(
    pose_vector: list[float],
    *,
    child_frame_id: str,
) -> BatchFrameTransform:
    pose = np.asarray(pose_vector, dtype=np.float32)
    assert pose.shape == (7,)
    return BatchFrameTransform(
        xyz=torch.from_numpy(pose[:3]).unsqueeze(0),
        quat=torch.from_numpy(pose[3:]).unsqueeze(0),
        parent_frame_id="world",
        child_frame_id=child_frame_id,
    )


def _run_patch_rejection_worker(
    case: _EmbodimentCase,
    config_path: Path,
) -> dict[str, Any]:
    env = _make_env(
        config_path=config_path,
        action_type="ee",
        patch=True,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        capture = _install_lifecycle_capture(monkeypatch)
        try:
            with pytest.raises(RoboTwinCuroboPatchUnsupportedError):
                env.reset(clear_cache=True, episode_id=0)
        finally:
            env.close(clear_cache=True)
    return {
        "status": "passed",
        "case_id": case.case_id,
        "rejected_as_expected": True,
        "config_sha256": env.cfg._task_config_content_sha256,
        "cleanup": _cleanup_evidence(
            capture.closed_envs,
            capture.runtime_handles,
            expected_workers_per_runtime=case.planner_worker_count,
        ),
    }


class _CurrentObservationPolicy(PolicyMixin[dict[str, Any], np.ndarray]):
    cfg: _CurrentObservationPolicyCfg

    def __init__(self, cfg: _CurrentObservationPolicyCfg) -> None:
        self.cfg = cfg
        self.action_widths: list[int] = []

    def reset(self, **kwargs: Any) -> None:
        del kwargs

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        action = _action_from_observation(obs, self.cfg.action_type)
        self.action_widths.append(int(action.shape[0]))
        return action

    def to(self, device: torch.device | str) -> _CurrentObservationPolicy:
        del device
        return self


class _CurrentObservationPolicyCfg(PolicyConfig[_CurrentObservationPolicy]):
    class_type: ClassType[_CurrentObservationPolicy] = (
        _CurrentObservationPolicy
    )
    action_type: Literal["qpos", "ee"] = "qpos"


def _run_evaluator_worker(
    case: _EmbodimentCase,
    config_path: Path,
    action_type: Literal["qpos", "ee"],
) -> dict[str, Any]:
    policy = _CurrentObservationPolicyCfg(action_type=action_type)()
    cfg = RoboTwinBenchmarkEvaluatorCfg(
        task_names=["place_empty_cup"],
        episode_num=1,
        max_retries=0,
        max_steps=1,
        config_type=str(config_path),
        start_seed=0,
        format_datatypes=True,
        action_type=action_type,
        log_progress=False,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        capture = _install_lifecycle_capture(monkeypatch)
        result = RoboTwinBenchmarkEvaluator(cfg).evaluate(policy, device="cpu")
    assert len(result.episodes) == 1
    record = result.episodes[0]
    assert record.attempts == 1
    assert record.error_type is None
    expected_width = case.qpos_width if action_type == "qpos" else 16
    assert policy.action_widths == [expected_width]
    cleanup = _cleanup_evidence(
        capture.closed_envs,
        capture.runtime_handles,
        expected_workers_per_runtime=case.planner_worker_count,
    )
    return {
        "status": "passed",
        "case_id": case.case_id,
        "action_type": action_type,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "episode_succeeded": record.succeeded,
        "episode_error_type": record.error_type,
        "episode_error_message": record.error_message,
        "episode_attempts": record.attempts,
        "action_widths": policy.action_widths,
        "result_metadata": result.metadata,
        "metrics": result.metrics,
        "cleanup": cleanup,
    }


def _worker_result(args: argparse.Namespace) -> dict[str, Any]:
    if args.worker == "capability":
        return _capability_result()
    if args.case is None or args.config_path is None:
        raise ValueError("case and config path are required for this worker")
    case = _CASES_BY_ID[args.case]
    config_path = Path(args.config_path)
    if args.worker == "direct":
        return _run_direct_worker(
            case,
            config_path,
            args.action_type,
            args.patch,
        )
    if args.worker == "fk_state":
        return _run_fk_state_worker(case, config_path)
    if args.worker == "reject_patch":
        return _run_patch_rejection_worker(case, config_path)
    if args.worker == "evaluator":
        return _run_evaluator_worker(case, config_path, args.action_type)
    raise AssertionError(f"Unhandled worker: {args.worker}")


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        choices=(
            "capability",
            "direct",
            "fk_state",
            "reject_patch",
            "evaluator",
        ),
        required=True,
    )
    parser.add_argument("--case", choices=tuple(_CASES_BY_ID))
    parser.add_argument(
        "--action-type", choices=("qpos", "ee"), default="qpos"
    )
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--config-path")
    args = parser.parse_args()
    try:
        result = _worker_result(args)
    except BaseException as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(_RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
        return 1
    print(_RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
