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
import importlib
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch
import yaml
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.math import (
    Transform3D_M,
    euler_angles_to_matrix,
    normalize,
    quaternion_to_matrix,
)

from robo_orchard_lab.envs.robotwin.workspace import in_robotwin_workspace

if TYPE_CHECKING:
    from envs._base_task import (  # pyright: ignore[reportMissingImports]
        Base_Task,
    )

__all__ = [
    "RoboTwinCuroboPatchUnsupportedError",
    "prepare_robotwin_runtime_for_cfg",
    "setup_robotwin_demo_with_runtime_guards",
]

logger = LoggerManager().get_child(__name__)
_PATCH_FLAG = "_robo_orchard_base_transform_patched"
_PATCH_TRANSFORM_ATTR = "_robo_orchard_entity_T_curobo_base"
_PATCH_ORIGINAL_FRAME_BIAS_ATTR = "_robo_orchard_original_frame_bias"
_FRAME_BIAS_ATOL = 1e-6
_QUAT_NORM_EPS = 1e-9
_WORKER_JOIN_TIMEOUT_SECONDS = 2.0
_PATCH_INSTALLED_IN_PROCESS = False


class RoboTwinCuroboPatchUnsupportedError(RuntimeError):
    """Raised when the current RoboTwin planner mode cannot be patched."""


class _PlannerWithFrameBias(Protocol):
    frame_bias: list[float]


@dataclass(frozen=True, slots=True)
class _CuroboBaseTransform:
    """Fixed transform from RoboTwin entity frame to Curobo base link."""

    entity_to_base: Transform3D_M

    @property
    def entity_to_base_xyz(self) -> np.ndarray:
        """Translation from RoboTwin entity frame to Curobo base link."""
        return _tensor_to_numpy(
            self.entity_to_base.get_translation()[0],
        )

    @property
    def entity_to_base_rotation_mat(self) -> np.ndarray:
        """Rotation from RoboTwin entity frame to Curobo base link."""
        return _tensor_to_numpy(
            self.entity_to_base.get_matrix()[0, :3, :3],
        )

    def transform_entity_target_to_base(
        self,
        entity_target_xyz: Sequence[float],
        entity_target_quat: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert an entity-frame target pose to Curobo base-link frame.

        Args:
            entity_target_xyz (Sequence[float]): Target position in the
                RoboTwin entity frame.
            entity_target_quat (Sequence[float]): Target orientation in the
                RoboTwin entity frame, using ``[qw, qx, qy, qz]``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Position and quaternion in Curobo
            ``base_link`` frame, with quaternion ordered as
            ``[qw, qx, qy, qz]``.
        """
        entity_target_xyz_tensor = _float_tensor(
            entity_target_xyz,
            expected_size=3,
            context="entity target xyz",
        )
        entity_target_quat_tensor = _float_tensor(
            entity_target_quat,
            expected_size=4,
            context="entity target quaternion",
        )
        if (
            float(torch.linalg.vector_norm(entity_target_quat_tensor))
            <= _QUAT_NORM_EPS
        ):
            raise ValueError("Quaternion must not be zero.")
        entity_target_quat_tensor = normalize(
            entity_target_quat_tensor,
            dim=-1,
        )
        entity_to_target = Transform3D_M.from_rot_trans(
            R=quaternion_to_matrix(entity_target_quat_tensor),
            T=entity_target_xyz_tensor,
        )
        base_to_target = self.entity_to_base.inverse() @ entity_to_target
        return (
            _tensor_to_numpy(base_to_target.get_translation()[0]),
            _tensor_to_numpy(
                base_to_target.get_rotation_quaternion(normalize=True)[0],
            ),
        )


def prepare_robotwin_runtime_for_cfg(cfg: object) -> None:
    """Install configured RoboTwin runtime patches before task setup."""
    if getattr(cfg, "action_type", None) != "ee":
        return
    if not bool(getattr(cfg, "patch_curobo_base_transform", False)):
        if _PATCH_INSTALLED_IN_PROCESS:
            raise RuntimeError(
                "patch_curobo_base_transform=False cannot restore the "
                "original RoboTwin CuroboPlanner after the process-wide "
                "runtime patch has already been installed. Use a fresh "
                "Python process to compare against the original RoboTwin "
                "behavior."
            )
        return

    with in_robotwin_workspace():
        _install_robotwin_curobo_base_transform_patch()


def setup_robotwin_demo_with_runtime_guards(
    cfg: object,
    task: "Base_Task",
    task_config: dict[str, object],
) -> None:
    """Run RoboTwin setup, validate patched runtime, and cleanup on failure."""
    try:
        task.setup_demo(**task_config)  # type: ignore[attr-defined]
        if _cfg_needs_curobo_base_transform_patch(cfg):
            _validate_robotwin_curobo_patch_runtime(task)
    except Exception:
        _cleanup_robotwin_task_after_failed_setup(task)
        raise


def _install_robotwin_curobo_base_transform_patch() -> None:
    """Patch RoboTwin CuroboPlanner to use full base-link transforms.

    The patch is process-wide and idempotent. It mutates the original
    RoboTwin ``CuroboPlanner`` class object so already-bound class references
    inside RoboTwin modules see the patched methods.
    """
    global _PATCH_INSTALLED_IN_PROCESS

    planner_module = importlib.import_module("envs.robot.planner")
    planner_cls = getattr(planner_module, "CuroboPlanner", None)
    if planner_cls is None:
        raise RuntimeError(
            "RoboTwin CuroboPlanner is unavailable. Disable "
            "patch_curobo_base_transform only if you intentionally want the "
            "original RoboTwin behavior."
        )
    if getattr(planner_cls, _PATCH_FLAG, False):
        _PATCH_INSTALLED_IN_PROCESS = True
        return

    original_init = planner_cls.__init__
    original_world_to_base = planner_cls._trans_from_world_to_base

    def patched_init(self: object, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        yml_path = getattr(self, "yml_path", None)
        if not isinstance(yml_path, str):
            raise RuntimeError(
                "Patched RoboTwin CuroboPlanner expected string yml_path."
            )
        entity_to_base = _parse_entity_to_base_from_curobo_yml(yml_path)
        frame_bias = _float_tensor(
            getattr(self, "frame_bias", None),
            expected_size=3,
            context="planner.frame_bias",
        )
        _validate_frame_bias(entity_to_base, frame_bias)
        setattr(
            self,
            _PATCH_ORIGINAL_FRAME_BIAS_ATTR,
            [float(value) for value in frame_bias],
        )
        setattr(self, _PATCH_TRANSFORM_ATTR, entity_to_base)
        cast(_PlannerWithFrameBias, self).frame_bias = [0.0, 0.0, 0.0]

    def patched_world_to_base(
        self: object,
        base_pose: object,
        target_pose: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        entity_target_xyz, entity_target_quat = original_world_to_base(
            self,
            base_pose,
            target_pose,
        )
        entity_to_base = getattr(self, _PATCH_TRANSFORM_ATTR, None)
        if entity_to_base is None:
            return entity_target_xyz, entity_target_quat
        if not isinstance(entity_to_base, _CuroboBaseTransform):
            raise RuntimeError(
                "Patched RoboTwin CuroboPlanner has invalid base transform."
            )
        return entity_to_base.transform_entity_target_to_base(
            entity_target_xyz,
            entity_target_quat,
        )

    planner_cls.__init__ = patched_init
    planner_cls._trans_from_world_to_base = patched_world_to_base
    setattr(planner_cls, _PATCH_FLAG, True)
    _PATCH_INSTALLED_IN_PROCESS = True


def _parse_entity_to_base_from_curobo_yml(
    yml_path: str,
) -> _CuroboBaseTransform:
    """Parse the direct ``footprint -> base_link`` fixed joint from Curobo yml.

    Args:
        yml_path (str): Path to a RoboTwin Curobo yml file.

    Returns:
        _CuroboBaseTransform: The transform from RoboTwin entity frame
        ``footprint`` to the configured Curobo ``base_link``.

    Raises:
        ValueError: If the yml/URDF shape is unsupported or inconsistent with
            ``planner.frame_bias``.
    """
    yml_file = Path(yml_path)
    with yml_file.open("r", encoding="utf-8") as f:
        yml_data = _expect_mapping(
            yaml.safe_load(f),
            context=f"Curobo yml {yml_path}",
        )

    robot_cfg = _mapping_field(yml_data, "robot_cfg", "Curobo yml")
    kinematics_cfg = _mapping_field(
        robot_cfg,
        "kinematics",
        "robot_cfg",
    )
    urdf_path_value = _string_field(
        kinematics_cfg,
        "urdf_path",
        "robot_cfg.kinematics",
    )
    base_link = _string_field(
        kinematics_cfg,
        "base_link",
        "robot_cfg.kinematics",
    )
    planner_cfg = _mapping_field(yml_data, "planner", "Curobo yml")
    frame_bias = _float_tensor(
        planner_cfg.get("frame_bias"),
        expected_size=3,
        context="planner.frame_bias",
    )

    urdf_path = _resolve_urdf_path(yml_file, urdf_path_value)
    entity_to_base = _parse_entity_to_base_from_urdf(
        urdf_path=urdf_path,
        base_link=base_link,
    )
    _validate_frame_bias(entity_to_base, frame_bias)
    return entity_to_base


def _validate_robotwin_curobo_patch_runtime(task: "Base_Task") -> None:
    """Reject RoboTwin planner runtime modes not covered by this patch."""
    robot = getattr(task, "robot", None)
    if robot is None:
        return
    if getattr(robot, "communication_flag", False) is True:
        raise RoboTwinCuroboPatchUnsupportedError(
            "patch_curobo_base_transform does not support RoboTwin "
            "communication_flag=True yet because Curobo planners are created "
            "inside worker subprocesses."
        )


def _cleanup_robotwin_task_after_failed_setup(task: object) -> None:
    """Best-effort cleanup for failed RoboTwin setup or post-setup guard."""
    cleanup_errors: list[BaseException] = []

    close_env = getattr(task, "close_env", None)
    if callable(close_env):
        try:
            close_env(clear_cache=True)
        except BaseException as exc:
            cleanup_errors.append(exc)

    robot = getattr(task, "robot", None)
    if robot is not None:
        _close_worker_connections(robot, cleanup_errors)
        _stop_worker_processes(robot, cleanup_errors)

    for exc in cleanup_errors:
        logger.warning(
            "Failed while cleaning up RoboTwin task after setup failure.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _cfg_needs_curobo_base_transform_patch(cfg: object) -> bool:
    return getattr(cfg, "action_type", None) == "ee" and bool(
        getattr(cfg, "patch_curobo_base_transform", False)
    )


def _parse_entity_to_base_from_urdf(
    *,
    urdf_path: Path,
    base_link: str,
) -> _CuroboBaseTransform:
    root = ET.parse(urdf_path).getroot()
    matching_joint = None
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.attrib.get("link") == base_link:
            matching_joint = joint
            break

    if matching_joint is None:
        raise ValueError(
            f"URDF {urdf_path} has no joint whose child is {base_link!r}."
        )
    if matching_joint.attrib.get("type") != "fixed":
        raise ValueError(
            f"URDF joint for {base_link!r} must be fixed, got "
            f"{matching_joint.attrib.get('type')!r}."
        )
    parent = matching_joint.find("parent")
    parent_link = None if parent is None else parent.attrib.get("link")
    if parent_link != "footprint":
        raise ValueError(
            f"URDF joint for {base_link!r} must have parent 'footprint', "
            f"got {parent_link!r}."
        )

    origin = matching_joint.find("origin")
    xyz = _parse_urdf_vector(
        None if origin is None else origin.attrib.get("xyz"),
        context=f"{base_link} origin xyz",
    )
    rpy = _parse_urdf_vector(
        None if origin is None else origin.attrib.get("rpy"),
        context=f"{base_link} origin rpy",
    )
    return _CuroboBaseTransform(
        entity_to_base=Transform3D_M.from_rot_trans(
            R=_urdf_fixed_axis_rpy_to_matrix(rpy),
            T=xyz,
        ),
    )


def _urdf_fixed_axis_rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert URDF fixed-axis roll/pitch/yaw to a rotation matrix."""
    # URDF origin rpy is fixed-axis Rz(yaw) @ Ry(pitch) @ Rx(roll).
    # The core Euler helper interprets the convention as intrinsic rotations,
    # so reverse the angles and use ZYX instead of passing rpy through XYZ.
    return euler_angles_to_matrix(rpy.flip(dims=(-1,)), "ZYX")


def _validate_frame_bias(
    entity_to_base: _CuroboBaseTransform,
    frame_bias: Sequence[float],
) -> None:
    expected_xyz = -_float_tensor(
        frame_bias,
        expected_size=3,
        context="planner.frame_bias",
    )
    entity_to_base_xyz = entity_to_base.entity_to_base.get_translation()[0]
    if not torch.allclose(
        expected_xyz,
        entity_to_base_xyz,
        atol=_FRAME_BIAS_ATOL,
    ):
        raise ValueError(
            "Curobo planner.frame_bias is inconsistent with URDF "
            "footprint->base_link translation: "
            f"expected {list(_tensor_to_numpy(-entity_to_base_xyz))}, "
            f"got {list(_tensor_to_numpy(frame_bias))}."
        )


def _resolve_urdf_path(yml_file: Path, urdf_path_value: str) -> Path:
    urdf_path = Path(urdf_path_value)
    if urdf_path.is_absolute():
        resolved_path = urdf_path
    else:
        yml_relative_path = yml_file.parent / urdf_path
        resolved_path = (
            yml_relative_path if yml_relative_path.exists() else urdf_path
        )
    if not resolved_path.exists():
        raise ValueError(f"Curobo URDF path does not exist: {resolved_path}")
    return resolved_path


def _expect_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return value


def _mapping_field(
    mapping: Mapping[str, object],
    key: str,
    context: str,
) -> Mapping[str, object]:
    return _expect_mapping(
        mapping.get(key),
        context=f"{context}.{key}",
    )


def _string_field(
    mapping: Mapping[str, object],
    key: str,
    context: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string.")
    return value


def _parse_urdf_vector(value: str | None, *, context: str) -> torch.Tensor:
    if value is None:
        return torch.zeros(3, dtype=torch.float64)
    try:
        return _float_tensor(
            [float(part) for part in value.split()],
            expected_size=3,
            context=context,
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid URDF vector for {context!r}: {value!r}"
        ) from exc


def _float_tensor(
    value: object,
    *,
    expected_size: int,
    context: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    elif isinstance(value, np.ndarray):
        tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    else:
        raise ValueError(f"{context} must be a numeric sequence.")
    if tensor.shape != (expected_size,):
        raise ValueError(
            f"{context} must contain {expected_size} values, got "
            f"{tensor.shape[0]}."
        )
    return tensor


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _close_worker_connections(
    robot: object,
    cleanup_errors: list[BaseException],
) -> None:
    for conn_name in ("left_conn", "right_conn"):
        conn = getattr(robot, conn_name, None)
        if conn is None:
            continue
        close = getattr(conn, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            cleanup_errors.append(exc)


def _stop_worker_processes(
    robot: object,
    cleanup_errors: list[BaseException],
) -> None:
    for proc_name in ("left_proc", "right_proc"):
        proc = getattr(robot, proc_name, None)
        if proc is None:
            continue
        try:
            is_alive = getattr(proc, "is_alive", None)
            terminate = getattr(proc, "terminate", None)
            join = getattr(proc, "join", None)
            if callable(is_alive) and is_alive() and callable(terminate):
                terminate()
            if callable(join):
                join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
        except BaseException as exc:
            cleanup_errors.append(exc)
