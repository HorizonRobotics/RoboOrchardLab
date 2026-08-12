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
import base64
import hashlib
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

_ROBOTWIN_DEFAULT_TASK_CONFIG_PRESET = "demo_clean"
_ROBOTWIN_TASK_CONFIG_PRESETS = frozenset(
    {_ROBOTWIN_DEFAULT_TASK_CONFIG_PRESET, "demo_randomized"}
)
_ROBOTWIN_TASK_CONFIG_SNAPSHOT_KEY = "__robotwin_task_config_snapshot__"


@dataclass(frozen=True, slots=True)
class _RoboTwinTaskConfigSnapshot:
    """Immutable task-YAML payload pinned when an EnvCfg is constructed."""

    yaml_bytes: bytes

    @property
    def content_sha256(self) -> str:
        """Return the digest derived from the immutable YAML bytes."""
        return hashlib.sha256(self.yaml_bytes).hexdigest()


_ROBOTWIN_TASK_CONFIG_SNAPSHOT_CONTEXT: ContextVar[
    _RoboTwinTaskConfigSnapshot | None
] = ContextVar("robotwin_task_config_snapshot", default=None)


def inject_serialized_task_config_snapshot(
    serialized: dict[str, Any],
    snapshot: _RoboTwinTaskConfigSnapshot,
) -> dict[str, Any]:
    """Add the verified private snapshot envelope to serialized config."""
    serialized[_ROBOTWIN_TASK_CONFIG_SNAPSHOT_KEY] = {
        "yaml_base64": base64.b64encode(snapshot.yaml_bytes).decode("ascii"),
        "content_sha256": snapshot.content_sha256,
    }
    return serialized


def extract_serialized_task_config_snapshot(
    data: Any,
) -> tuple[Any, _RoboTwinTaskConfigSnapshot | None]:
    """Remove and validate a serialized snapshot envelope when present."""
    snapshot: _RoboTwinTaskConfigSnapshot | None = None
    if isinstance(data, dict) and _ROBOTWIN_TASK_CONFIG_SNAPSHOT_KEY in data:
        data = data.copy()
        envelope = data.pop(_ROBOTWIN_TASK_CONFIG_SNAPSHOT_KEY)
        if not isinstance(envelope, dict):
            raise ValueError(
                "Serialized RoboTwin task-config snapshot must be a mapping."
            )
        yaml_base64 = envelope.get("yaml_base64")
        content_sha256 = envelope.get("content_sha256")
        if not isinstance(yaml_base64, str) or not isinstance(
            content_sha256, str
        ):
            raise ValueError(
                "Serialized RoboTwin task-config snapshot must contain "
                "string yaml_base64 and content_sha256 values."
            )
        try:
            yaml_bytes = base64.b64decode(yaml_base64, validate=True)
        except ValueError as exc:
            raise ValueError(
                "Serialized RoboTwin task-config snapshot contains invalid "
                "base64 YAML bytes."
            ) from exc
        actual_sha256 = hashlib.sha256(yaml_bytes).hexdigest()
        if actual_sha256 != content_sha256:
            raise ValueError(
                "Serialized RoboTwin task-config snapshot digest does not "
                "match its YAML bytes."
            )
        snapshot = _RoboTwinTaskConfigSnapshot(yaml_bytes=yaml_bytes)
    return data, snapshot


@contextmanager
def task_config_snapshot_restore_context(
    snapshot: _RoboTwinTaskConfigSnapshot | None,
) -> Iterator[None]:
    """Expose a serialized snapshot only during Pydantic reconstruction."""
    token = _ROBOTWIN_TASK_CONFIG_SNAPSHOT_CONTEXT.set(snapshot)
    try:
        yield
    finally:
        _ROBOTWIN_TASK_CONFIG_SNAPSHOT_CONTEXT.reset(token)


def resolve_task_config_source(
    task_config_path: str | None,
    *,
    robotwin_root: str,
) -> tuple[str, _RoboTwinTaskConfigSnapshot]:
    """Resolve preset/path syntax and pin the task YAML exactly once."""
    if task_config_path is None:
        task_config_path = _ROBOTWIN_DEFAULT_TASK_CONFIG_PRESET
    if task_config_path in _ROBOTWIN_TASK_CONFIG_PRESETS:
        task_config_path = str(
            Path(robotwin_root) / "task_config" / f"{task_config_path}.yml"
        )
    canonical_path = Path(task_config_path).resolve()

    serialized_snapshot = _ROBOTWIN_TASK_CONFIG_SNAPSHOT_CONTEXT.get()
    if serialized_snapshot is not None:
        return str(canonical_path), serialized_snapshot
    if not canonical_path.is_file():
        raise FileNotFoundError(
            f"Task configuration file {task_config_path} does not exist."
        )
    yaml_bytes = canonical_path.read_bytes()
    return str(canonical_path), _RoboTwinTaskConfigSnapshot(
        yaml_bytes=yaml_bytes
    )


def build_task_config(
    *,
    snapshot: _RoboTwinTaskConfigSnapshot,
    source: str,
    task_name: str,
    runtime_seed: int,
    episode_id: int,
    eval_mode: bool,
    embodiment_config_path: str,
    camera_config_path: str,
    robotwin_root: str,
    task_config_overrides: list[tuple[str, Any]] | None,
) -> dict[str, Any]:
    """Parse pinned YAML and lower official RoboTwin runtime configuration."""
    task_config = _load_yaml_mapping(snapshot.yaml_bytes, source=source)
    _lower_task_config(
        task_config,
        runtime_seed=runtime_seed,
        episode_id=episode_id,
        eval_mode=eval_mode,
        embodiment_config_path=embodiment_config_path,
        camera_config_path=camera_config_path,
        robotwin_root=robotwin_root,
    )
    _apply_task_config_overrides(task_config, task_config_overrides)
    task_config["task_name"] = task_name
    return task_config


def _load_yaml_mapping(
    content: str | bytes,
    *,
    source: str,
) -> dict[str, Any]:
    value = yaml.load(content, Loader=yaml.FullLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected {source} to contain a YAML mapping.")
    return value


def _apply_task_config_overrides(
    task_config: dict[str, Any],
    task_config_overrides: list[tuple[str, Any]] | None,
) -> None:
    """Apply validated final patches to a fully lowered task config."""
    reserved_paths = {
        "task_name",
        "seed",
        "now_ep_num",
        "eval_mode",
        "is_test",
        "camera/head_camera_type",
        "embodiment",
    }
    reserved_lowering_roots = {
        "left_robot_file",
        "right_robot_file",
        "left_embodiment_config",
        "right_embodiment_config",
        "dual_arm_embodied",
        "embodiment_dis",
        "embodiment_name",
    }
    for path, value in task_config_overrides or []:
        path_root = path.split("/", 1)[0]
        if path in reserved_paths or path_root in reserved_lowering_roots:
            raise ValueError(
                f"Task config override path {path!r} is not supported "
                "because it affects env-managed or derived fields."
            )

        keys = path.split("/")
        if not path or any(key == "" for key in keys):
            raise ValueError(f"Invalid task config override path {path!r}.")

        target: Any = task_config
        for key in keys[:-1]:
            if not isinstance(target, dict):
                raise KeyError(
                    f"Task config override path {path!r} does not resolve "
                    f"to a nested dict at {key!r}."
                )
            if key not in target:
                raise KeyError(
                    f"Task config override path {path!r} is missing segment "
                    f"{key!r}."
                )
            target = target[key]

        if not isinstance(target, dict):
            raise KeyError(
                f"Task config override path {path!r} does not resolve to a "
                "dict parent."
            )

        leaf_key = keys[-1]
        if leaf_key not in target:
            raise KeyError(
                f"Task config override path {path!r} is missing leaf key "
                f"{leaf_key!r}."
            )
        target[leaf_key] = value


def _lower_task_config(
    task_args: dict[str, Any],
    *,
    runtime_seed: int,
    episode_id: int,
    eval_mode: bool,
    embodiment_config_path: str,
    camera_config_path: str,
    robotwin_root: str,
) -> None:
    embodiment = task_args.get("embodiment")
    if not isinstance(embodiment, list) or len(embodiment) not in (1, 3):
        raise ValueError(
            "RoboTwin task config 'embodiment' must contain either one "
            "combined embodiment name or left/right names plus distance."
        )

    registry_path = Path(embodiment_config_path)
    embodiment_types = _load_yaml_mapping(
        registry_path.read_bytes(),
        source=str(registry_path),
    )
    resolved_robotwin_root = Path(robotwin_root).resolve()

    camera_path = Path(camera_config_path)
    camera_config = _load_yaml_mapping(
        camera_path.read_bytes(),
        source=str(camera_path),
    )
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment) == 1:
        embodiment_name = embodiment[0]
        robot_file, robot_config = _resolve_embodiment_asset(
            embodiment_name,
            embodiment_types=embodiment_types,
            registry_path=registry_path,
            robotwin_root=resolved_robotwin_root,
        )
        if robot_config["dual_arm"] is not True:
            raise ValueError(
                f"RoboTwin embodiment {embodiment_name!r} is a single-arm "
                "asset and cannot use the one-item combined syntax. Use "
                "[name, name, distance] instead."
            )
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
        task_args.pop("embodiment_dis", None)
        left_config = robot_config
        right_config = robot_config
    else:
        left_name, right_name, distance_value = embodiment
        if (
            isinstance(distance_value, bool)
            or not isinstance(distance_value, (int, float))
            or not math.isfinite(float(distance_value))
            or float(distance_value) <= 0
        ):
            raise ValueError(
                "RoboTwin split embodiment distance must be a finite number "
                "greater than 0."
            )
        left_file, left_config = _resolve_embodiment_asset(
            left_name,
            embodiment_types=embodiment_types,
            registry_path=registry_path,
            robotwin_root=resolved_robotwin_root,
        )
        if right_name == left_name:
            right_file, right_config = left_file, left_config
        else:
            right_file, right_config = _resolve_embodiment_asset(
                right_name,
                embodiment_types=embodiment_types,
                registry_path=registry_path,
                robotwin_root=resolved_robotwin_root,
            )
        for side, name, config in (
            ("left", left_name, left_config),
            ("right", right_name, right_config),
        ):
            if config["dual_arm"] is not False:
                raise ValueError(
                    f"RoboTwin {side} embodiment {name!r} is a combined "
                    "dual-arm asset and cannot be nested in split syntax."
                )
        task_args["left_robot_file"] = left_file
        task_args["right_robot_file"] = right_file
        task_args["embodiment_dis"] = float(distance_value)
        task_args["dual_arm_embodied"] = False
        embodiment_name = f"{left_name}+{right_name}"

    task_args["left_embodiment_config"] = left_config
    task_args["right_embodiment_config"] = right_config
    task_args["embodiment_name"] = str(embodiment_name)
    task_args["seed"] = runtime_seed
    task_args["now_ep_num"] = episode_id
    task_args["eval_mode"] = eval_mode
    task_args["is_test"] = eval_mode


def _resolve_embodiment_asset(
    embodiment_name: object,
    *,
    embodiment_types: dict[str, Any],
    registry_path: Path,
    robotwin_root: Path,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(embodiment_name, str) or not embodiment_name:
        raise ValueError(
            "RoboTwin embodiment names must be non-empty strings."
        )
    if embodiment_name not in embodiment_types:
        raise KeyError(
            f"RoboTwin embodiment {embodiment_name!r} is not registered in "
            f"{registry_path}."
        )
    registry_entry = embodiment_types[embodiment_name]
    if not isinstance(registry_entry, dict):
        raise ValueError(
            f"Registry entry for embodiment {embodiment_name!r} must be a "
            "mapping."
        )
    file_path = registry_entry.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        raise ValueError(
            f"No asset file_path is configured for embodiment "
            f"{embodiment_name!r}."
        )
    robot_path = Path(file_path)
    if not robot_path.is_absolute():
        robot_path = robotwin_root / robot_path
    robot_path = robot_path.resolve()
    if not robot_path.is_dir():
        raise FileNotFoundError(
            f"RoboTwin embodiment asset directory does not exist: "
            f"{robot_path}."
        )

    robot_config_path = robot_path / "config.yml"
    if not robot_config_path.is_file():
        raise FileNotFoundError(
            f"RoboTwin embodiment config does not exist: {robot_config_path}."
        )
    robot_config = _load_yaml_mapping(
        robot_config_path.read_bytes(),
        source=str(robot_config_path),
    )
    _validate_embodiment_asset_config(
        embodiment_name=embodiment_name,
        robot_path=robot_path,
        robot_config=robot_config,
    )
    return str(robot_path), robot_config


def _validate_embodiment_asset_config(
    *,
    embodiment_name: str,
    robot_path: Path,
    robot_config: dict[str, Any],
) -> None:
    """Validate only asset fields consumed by this adapter/upstream."""
    if type(robot_config.get("dual_arm")) is not bool:
        raise ValueError(
            f"RoboTwin embodiment {embodiment_name!r} must declare a "
            "boolean config.yml 'dual_arm' field."
        )

    gripper_bias = robot_config.get("gripper_bias")
    if (
        isinstance(gripper_bias, bool)
        or not isinstance(gripper_bias, (int, float))
        or not math.isfinite(float(gripper_bias))
    ):
        raise ValueError(
            f"RoboTwin embodiment {embodiment_name!r} config.yml "
            "gripper_bias must be a finite number."
        )
    gripper_scale = robot_config.get("gripper_scale")
    if (
        not isinstance(gripper_scale, list)
        or len(gripper_scale) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in gripper_scale
        )
        or float(gripper_scale[0]) == float(gripper_scale[1])
    ):
        raise ValueError(
            f"RoboTwin embodiment {embodiment_name!r} config.yml "
            "gripper_scale must contain two distinct finite numbers."
        )

    for key in ("urdf_path", "srdf_path"):
        referenced_path = robot_config.get(key)
        if key == "srdf_path" and referenced_path is None:
            continue
        if not isinstance(referenced_path, str) or not referenced_path:
            raise KeyError(
                f"RoboTwin embodiment {embodiment_name!r} must define a "
                f"non-empty config.yml {key!r}."
            )
        asset_path = Path(referenced_path)
        if not asset_path.is_absolute():
            asset_path = robot_path / asset_path
        if not asset_path.resolve().is_file():
            raise FileNotFoundError(
                f"RoboTwin embodiment {embodiment_name!r} references "
                f"missing {key}: {asset_path.resolve()}."
            )

    for key in ("move_group", "ee_joints", "arm_joints_name", "gripper_name"):
        value = robot_config.get(key)
        if (
            not isinstance(value, list)
            or len(value) < 2
            or value[0] is None
            or value[1] is None
        ):
            raise KeyError(
                f"RoboTwin embodiment {embodiment_name!r} config.yml field "
                f"{key!r} must provide left and right entries."
            )

    for side_index, side in enumerate(("left", "right")):
        arm_joints = robot_config["arm_joints_name"][side_index]
        if (
            not isinstance(arm_joints, list)
            or not arm_joints
            or any(
                not isinstance(joint, str) or not joint for joint in arm_joints
            )
        ):
            raise ValueError(
                f"RoboTwin embodiment {embodiment_name!r} {side} arm "
                "joint names must be a non-empty string list."
            )
        gripper = robot_config["gripper_name"][side_index]
        if (
            not isinstance(gripper, dict)
            or not isinstance(gripper.get("base"), str)
            or not gripper["base"]
        ):
            raise ValueError(
                f"RoboTwin embodiment {embodiment_name!r} {side} gripper "
                "entry must define a non-empty base joint."
            )
        for key in ("move_group", "ee_joints"):
            frame_name = robot_config[key][side_index]
            if not isinstance(frame_name, str) or not frame_name:
                raise ValueError(
                    f"RoboTwin embodiment {embodiment_name!r} {side} "
                    f"{key} entry must be a non-empty string."
                )

    curobo_names = (
        ("curobo_left.yml", "curobo_right.yml")
        if robot_config["dual_arm"]
        else ("curobo.yml",)
    )
    for curobo_name in curobo_names:
        curobo_path = robot_path / curobo_name
        if not curobo_path.is_file():
            raise FileNotFoundError(
                f"RoboTwin embodiment {embodiment_name!r} is missing "
                f"planner config: {curobo_path}."
            )
        curobo_config = _load_yaml_mapping(
            curobo_path.read_bytes(),
            source=str(curobo_path),
        )
        robot_cfg = curobo_config.get("robot_cfg")
        kinematics = (
            robot_cfg.get("kinematics")
            if isinstance(robot_cfg, dict)
            else None
        )
        if not isinstance(kinematics, dict):
            raise KeyError(
                f"RoboTwin planner config {curobo_path} must define "
                "robot_cfg.kinematics."
            )
        for key in ("urdf_path", "collision_spheres"):
            referenced_path = kinematics.get(key)
            if not isinstance(referenced_path, str) or not referenced_path:
                raise KeyError(
                    f"RoboTwin planner config {curobo_path} must define a "
                    f"non-empty robot_cfg.kinematics.{key}."
                )
            asset_path = Path(referenced_path)
            if not asset_path.is_absolute():
                asset_path = robot_path / asset_path
            if not asset_path.resolve().is_file():
                raise FileNotFoundError(
                    f"RoboTwin planner config {curobo_path} references "
                    f"missing {key}: {asset_path.resolve()}."
                )
