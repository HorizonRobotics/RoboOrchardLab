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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence, cast

import numpy as np
import torch
from robo_orchard_core.utils.logging import LoggerManager
from typing_extensions import Literal

from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchFrameTransformGraph,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Robot,
    RobotDescriptionFormat,
)
from robo_orchard_lab.envs.robotwin.kinematics import (
    RoboTwinEEF,
    RoboTwinJointsToEEF,
)
from robo_orchard_lab.envs.robotwin.workspace import in_robotwin_workspace
from robo_orchard_lab.envs.sapien import sapien_pose_to_orchard

if TYPE_CHECKING:
    from envs._base_task import (  # pyright: ignore[reportMissingImports]
        Base_Task,
    )

_COMBINED_DUAL_ARM_OBS_ROBOT_KEY = "left"
_ROBOTWIN_WORKER_JOIN_TIMEOUT_SECONDS = 5.0
logger = LoggerManager().get_child(__name__)


class _NamedLink(Protocol):
    def get_name(self) -> str: ...


class _EndEffectorJoint(Protocol):
    child_link: _NamedLink


class _RoboTwinRobotRuntime(Protocol):
    """Robot fields populated by RoboTwin setup and planner creation."""

    is_dual_arm: bool
    left_arm_joints_name: Sequence[str]
    right_arm_joints_name: Sequence[str]
    left_gripper_name: dict[str, str]
    right_gripper_name: dict[str, str]
    left_ee: _EndEffectorJoint
    right_ee: _EndEffectorJoint
    left_entity_origion_pose: Any
    right_entity_origion_pose: Any
    left_entity: object | None
    right_entity: object | None
    left_urdf_path: str
    right_urdf_path: str
    left_conn: _WorkerConnection | None
    right_conn: _WorkerConnection | None
    left_proc: _WorkerProcess | None
    right_proc: _WorkerProcess | None


class _WorkerConnection(Protocol):
    closed: bool

    def send(self, message: object) -> None: ...

    def close(self) -> None: ...


class _WorkerProcess(Protocol):
    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def close(self) -> None: ...


class _RoboTwinViewer(Protocol):
    def close(self) -> None: ...


class _RoboTwinTaskRuntime(Protocol):
    robot: _RoboTwinRobotRuntime
    viewer: _RoboTwinViewer

    def close_env(self, *, clear_cache: bool) -> None: ...


def _runtime_robot(task: Base_Task) -> _RoboTwinRobotRuntime:
    """Return the narrow typed view expected after RoboTwin setup."""
    try:
        return cast(_RoboTwinTaskRuntime, task).robot
    except AttributeError as exc:
        raise RuntimeError(
            "RoboTwin task does not expose an initialized robot."
        ) from exc


@dataclass(frozen=True, slots=True)
class _RoboTwinRuntimeLayout:
    """Minimal runtime-only facts for one initialized RoboTwin robot."""

    topology: Literal["combined_articulation", "split_articulations"]
    left_arm_joint_count: int
    right_arm_joint_count: int
    left_control_eef_frame_id: str
    right_control_eef_frame_id: str


def _transform_joint_vector_to_eef(
    transform: RoboTwinJointsToEEF,
    joints: np.ndarray,
    layout: _RoboTwinRuntimeLayout,
) -> RoboTwinEEF:
    """Adapt one flat RoboTwin joint vector to the two-arm FK contract."""
    joints_np = np.asarray(joints, dtype=np.float32)
    if joints_np.ndim == 1:
        joints_np = joints_np[None, :]
    if joints_np.ndim != 2 or joints_np.shape[0] != 1:
        raise ValueError(
            "Expected joints to have shape (D,) or (1, D), got "
            f"{tuple(joints_np.shape)}."
        )

    arm_joint_count = (
        layout.left_arm_joint_count + layout.right_arm_joint_count
    )
    if joints_np.shape[-1] == arm_joint_count + 2:
        right_start = layout.left_arm_joint_count + 1
    elif joints_np.shape[-1] == arm_joint_count:
        right_start = layout.left_arm_joint_count
    else:
        raise ValueError(
            "Expected RoboTwin joints to contain left/right arm joints "
            "with optional gripper values, got shape "
            f"{tuple(joints_np.shape)}."
        )

    return transform.transform(
        left_arm_joints=torch.from_numpy(
            joints_np[:, : layout.left_arm_joint_count]
        ),
        right_arm_joints=torch.from_numpy(
            joints_np[
                :,
                right_start : right_start + layout.right_arm_joint_count,
            ]
        ),
    )


def derive_runtime_layout(task: Base_Task) -> _RoboTwinRuntimeLayout:
    """Validate and derive observation layout from an initialized task."""
    robot = _runtime_robot(task)
    try:
        left_joint_count = len(robot.left_arm_joints_name)
        right_joint_count = len(robot.right_arm_joints_name)
        left_runtime_eef = robot.left_ee.child_link.get_name()
        right_runtime_eef = robot.right_ee.child_link.get_name()
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "RoboTwin runtime robot does not expose the expected left/right "
            "arm joints and end-effector links."
        ) from exc
    if left_joint_count <= 0 or right_joint_count <= 0:
        raise ValueError(
            "RoboTwin runtime robot must expose at least one arm joint for "
            "each side."
        )
    if not left_runtime_eef or not right_runtime_eef:
        raise ValueError(
            "RoboTwin runtime end-effector frame names must be non-empty."
        )

    left_base_tf = sapien_pose_to_orchard(robot.left_entity_origion_pose)
    right_base_tf = sapien_pose_to_orchard(robot.right_entity_origion_pose)
    left_entity = robot.left_entity
    right_entity = robot.right_entity
    if left_entity is None or right_entity is None:
        raise RuntimeError(
            "RoboTwin runtime robot must expose initialized left/right "
            "articulations."
        )
    is_dual_arm = robot.is_dual_arm
    if is_dual_arm is True:
        if left_entity is not right_entity:
            raise RuntimeError(
                "Combined RoboTwin layout must share one articulation "
                "between the left and right sides."
            )
        if left_base_tf != right_base_tf:
            raise RuntimeError(
                "Combined RoboTwin articulation must expose one shared "
                "left/right robot base pose."
            )
        topology = "combined_articulation"
        left_control_eef_frame_id = left_runtime_eef
        right_control_eef_frame_id = right_runtime_eef
    elif is_dual_arm is False:
        if left_entity is right_entity:
            raise RuntimeError(
                "Split RoboTwin layout must expose separate left/right "
                "articulations."
            )
        topology = "split_articulations"
        left_control_eef_frame_id = f"left/{left_runtime_eef}_from_obs"
        right_control_eef_frame_id = f"right/{right_runtime_eef}_from_obs"
    else:
        raise RuntimeError(
            "RoboTwin runtime robot must expose boolean is_dual_arm."
        )

    required_urdf_paths = [("left", robot.left_urdf_path)]
    if topology == "split_articulations":
        required_urdf_paths.append(("right", robot.right_urdf_path))
    for side, urdf_path in required_urdf_paths:
        if not isinstance(urdf_path, str) or not urdf_path:
            raise RuntimeError(
                f"RoboTwin runtime {side} URDF path is unavailable."
            )
        with in_robotwin_workspace():
            if not Path(urdf_path).is_file():
                raise FileNotFoundError(
                    f"RoboTwin runtime {side} URDF does not exist: "
                    f"{urdf_path}."
                )

    return _RoboTwinRuntimeLayout(
        topology=topology,
        left_arm_joint_count=left_joint_count,
        right_arm_joint_count=right_joint_count,
        left_control_eef_frame_id=left_control_eef_frame_id,
        right_control_eef_frame_id=right_control_eef_frame_id,
    )


def get_joint_state_names(
    task: Base_Task,
    layout: _RoboTwinRuntimeLayout,
) -> list[str]:
    """Return flat observation joint names for the derived layout."""
    robot = _runtime_robot(task)
    left_names = [
        *robot.left_arm_joints_name,
        robot.left_gripper_name["base"],
    ]
    right_names = [
        *robot.right_arm_joints_name,
        robot.right_gripper_name["base"],
    ]
    if layout.topology == "split_articulations":
        left_names = [f"left/{name}" for name in left_names]
        right_names = [f"right/{name}" for name in right_names]
    return [*left_names, *right_names]


def get_robot_base_tf_graph(
    task: Base_Task,
    layout: _RoboTwinRuntimeLayout,
) -> BatchFrameTransformGraph:
    """Build absolute world-frame robot-base edges for the runtime layout."""
    robot = _runtime_robot(task)
    left_base_tf = sapien_pose_to_orchard(robot.left_entity_origion_pose)
    right_base_tf = sapien_pose_to_orchard(robot.right_entity_origion_pose)
    if layout.topology == "combined_articulation":
        base_tfs = [
            BatchFrameTransform(
                xyz=left_base_tf.xyz,
                quat=left_base_tf.quat,
                timestamps=left_base_tf.timestamps,
                parent_frame_id="world",
                child_frame_id="robot_base",
            )
        ]
    else:
        base_tfs = [
            BatchFrameTransform(
                xyz=left_base_tf.xyz,
                quat=left_base_tf.quat,
                timestamps=left_base_tf.timestamps,
                parent_frame_id="world",
                child_frame_id="left_robot_base",
            ),
            BatchFrameTransform(
                xyz=right_base_tf.xyz,
                quat=right_base_tf.quat,
                timestamps=right_base_tf.timestamps,
                parent_frame_id="world",
                child_frame_id="right_robot_base",
            ),
        ]
    return BatchFrameTransformGraph(
        tf_list=base_tfs,
        static_tf=[True] * len(base_tfs),
    )


def read_robot_urdfs(
    task: Base_Task,
    layout: _RoboTwinRuntimeLayout,
) -> dict[str, bytes]:
    """Read combined or truthful split URDF bytes from the runtime robot."""
    robot = _runtime_robot(task)
    urdf_paths = {"left": robot.left_urdf_path}
    if layout.topology == "split_articulations":
        urdf_paths["right"] = robot.right_urdf_path

    ret: dict[str, bytes] = {}
    with in_robotwin_workspace():
        for side, urdf_path in urdf_paths.items():
            if not isinstance(urdf_path, str) or not urdf_path:
                raise RuntimeError(
                    f"RoboTwin runtime {side} URDF path is unavailable."
                )
            with open(urdf_path, "rb") as file:
                ret[side] = file.read()
    return ret


def build_obs_robots(
    urdf_map: dict[str, bytes],
    layout: _RoboTwinRuntimeLayout,
) -> dict[str, Robot]:
    """Build observation-facing Robot records for one runtime layout."""
    robot_keys = (
        (_COMBINED_DUAL_ARM_OBS_ROBOT_KEY,)
        if layout.topology == "combined_articulation"
        else ("left", "right")
    )
    robots: dict[str, Robot] = {}
    for index, robot_key in enumerate(robot_keys):
        urdf_content = urdf_map.get(robot_key)
        if not isinstance(urdf_content, bytes):
            raise RuntimeError(
                f"Expected RoboTwin layout to expose {robot_key!r} URDF."
            )
        robot = Robot(
            index=index,
            name=robot_key,
            content=urdf_content.decode("utf-8"),
            content_format=RobotDescriptionFormat.URDF,
        )
        robot.update_md5()
        robots[robot_key] = robot
    return robots


def build_joints_to_eef_transform(
    task: Base_Task,
    layout: _RoboTwinRuntimeLayout,
) -> RoboTwinJointsToEEF:
    """Build the FK adapter from an initialized RoboTwin runtime."""
    robot = _runtime_robot(task)
    urdf_map = read_robot_urdfs(task, layout)
    left_robot_base_tf = sapien_pose_to_orchard(robot.left_entity_origion_pose)
    right_robot_base_tf = sapien_pose_to_orchard(
        robot.right_entity_origion_pose
    )

    return RoboTwinJointsToEEF(
        urdf_content=urdf_map["left"],
        right_urdf_content=(
            urdf_map["right"]
            if layout.topology == "split_articulations"
            else None
        ),
        left_eef_name=robot.left_ee.child_link.get_name(),
        right_eef_name=robot.right_ee.child_link.get_name(),
        robot_base_xyz=left_robot_base_tf.xyz[0].tolist(),
        robot_base_quat=left_robot_base_tf.quat[0].tolist(),
        right_robot_base_xyz=right_robot_base_tf.xyz[0].tolist(),
        right_robot_base_quat=right_robot_base_tf.quat[0].tolist(),
    )


def dispose_task_runtime(
    task: Base_Task,
    *,
    clear_cache: bool,
) -> bool:
    """Best-effort close one RoboTwin task and its planner workers.

    This function owns only upstream shutdown mechanics. The caller remains
    responsible for retaining tasks whose workers cannot be confirmed stopped
    and for clearing Env-owned caches and active-task references.

    Args:
        task (Base_Task): Initialized or partially initialized RoboTwin task.
        clear_cache (bool): Value forwarded to RoboTwin ``close_env``.

    Returns:
        bool: True when every discovered planner worker is confirmed stopped.
    """
    cleanup_errors: list[tuple[str, BaseException]] = []
    workers_stopped = True
    runtime_task = cast(_RoboTwinTaskRuntime, task)
    try:
        robot = runtime_task.robot
    except AttributeError:
        robot = None
    try:
        viewer = runtime_task.viewer
    except AttributeError:
        viewer = None

    try:
        runtime_task.close_env(clear_cache=clear_cache)
    except BaseException as exc:
        cleanup_errors.append(("task.close_env", exc))

    if robot is not None:
        for side in ("left", "right"):
            try:
                conn = robot.left_conn if side == "left" else robot.right_conn
            except AttributeError:
                conn = None
            if conn is None:
                continue
            try:
                if not conn.closed:
                    conn.send({"cmd": "exit"})
            except BaseException as exc:
                cleanup_errors.append((f"robot.{side}_conn.send", exc))
            try:
                conn.close()
            except BaseException as exc:
                cleanup_errors.append((f"robot.{side}_conn.close", exc))
            try:
                if side == "left":
                    robot.left_conn = None
                else:
                    robot.right_conn = None
            except BaseException:
                pass

        for side in ("left", "right"):
            try:
                proc = robot.left_proc if side == "left" else robot.right_proc
            except AttributeError:
                proc = None
            if proc is None:
                continue
            try:
                if proc.is_alive():
                    proc.join(timeout=_ROBOTWIN_WORKER_JOIN_TIMEOUT_SECONDS)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=_ROBOTWIN_WORKER_JOIN_TIMEOUT_SECONDS)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=_ROBOTWIN_WORKER_JOIN_TIMEOUT_SECONDS)
                if proc.is_alive():
                    raise RuntimeError(
                        f"RoboTwin worker {side}_proc remained alive after "
                        "exit, terminate, and kill attempts."
                    )
                proc.close()
                if side == "left":
                    robot.left_proc = None
                else:
                    robot.right_proc = None
            except BaseException as exc:
                workers_stopped = False
                cleanup_errors.append((f"robot.{side}_proc", exc))

    if viewer is not None:
        try:
            viewer.close()
        except BaseException as exc:
            cleanup_errors.append(("task.viewer.close", exc))

    for operation, exc in cleanup_errors:
        logger.warning(
            "Failed while disposing RoboTwin task operation %s.",
            operation,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    return workers_stopped
