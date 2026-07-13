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

"""Typed loader for the per-embodiment URDF alignment manifests.

Each embodiment ships one ``alignment.yaml`` under
``projects/holobrain_internal/common/urdf_tools/manifests/<dataset>/<embodiment>/``
(git-tracked, authoritative validation contract). The aligned URDF asset it
contracts against lives under a parallel tree
``projects/holobrain_internal/common/urdf_align/<dataset>/<embodiment>/`` —
in practice a symlink to a shared bucket; not git-tracked.

The manifest declares:

* ``adapter`` — packer family (``interna1``, ``robotwin``);
* ``config`` — dataset-config module + getter + one-or-more keys
  (base + aligned) identifying the packer entries this embodiment
  backs;
* ``arms`` — per-arm kinematic chain and EE spec
  (``arm_link_keys``, ``ee.{parent, rotate_z_deg, gripper_forward}``);
* ``camera_references`` — bare list of URDF link names whose FK pose must be
  reachable in the aligned URDF (unchanged, or via a
  ``*_camera_mount_compat`` compat child);
* ``mesh_search_roots`` (optional) — extra dirs the mesh resolver may search.

Fields the pipeline never reads (and that were previously in the JSON) are
gone: ``link_renames`` (no rename happens), ``joints`` (both URDFs share the
same actuated-joint name list), ``ee_frames[].child`` (always
``<parent>_ee``), ``camera_references[].stream`` (metadata only).

Shared embodiments — ``config.key`` accepts a list — expand to one
``UrdfAlignmentCase`` per (manifest, config_key) pair so downstream callers
still see one case per packer key.

The origin URDF is not in the manifest: it is resolved from the packer config
entry through the ``DatasetAdapter.resolve_urdf_path`` contract, keeping the
packer config as the single source of truth.
"""

from __future__ import annotations
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from projects.holobrain_internal.common.urdf_tools.fk_baseline import (
    LinkNamePair,
)

DEFAULT_URDF_ALIGN_ROOT = Path(__file__).resolve().parents[1] / "urdf_align"
DEFAULT_MANIFEST_ROOT = Path(__file__).resolve().parent / "manifests"

# Packer config paths in ``config[key]["urdf"]`` are expressed relative to
# ``projects/holobrain_internal/common``, so that is the base we resolve them
# against when computing the absolute origin URDF path. The default packer
# config module tree lives at the same root.
_PACKER_CONFIG_COMMON_SUBDIR = Path("projects/holobrain_internal/common")
_PACKER_CONFIG_ROOT_SUBDIR = _PACKER_CONFIG_COMMON_SUBDIR / "configs"

_ROTATION_CHOICES: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
_DEFAULT_GRIPPER_FORWARD = 0.20


# ---------------------------------------------------------------------------
# Manifest-derived typed containers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigWiring:
    """Where a case's dataset-config entries live.

    ``keys`` / ``aligned_keys`` carry one or more packer keys for the same
    physical robot (shared embodiments like RoboTwin aloha_v1 / aloha_v2).
    The singular ``key`` / ``aligned_key`` properties return the first entry
    for legacy call sites that assume one key.
    """

    module: str
    getter: str
    keys: tuple[str, ...]
    aligned_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.aligned_keys):
            raise ValueError(
                "config.key and config.aligned_key must have the same length; "
                f"got {len(self.keys)} keys, {len(self.aligned_keys)} "
                "aligned keys"
            )
        if not self.keys:
            raise ValueError("config.key must not be empty")

    @property
    def key(self) -> str:
        return self.keys[0]

    @property
    def aligned_key(self) -> str:
        return self.aligned_keys[0]


@dataclass(frozen=True)
class GripperEndOverride:
    """Per-arm override placing ``<parent>_gripper_end`` on a real link.

    Instead of the default ``xyz="0 0 gripper_forward"`` (identity rpy)
    attached to the EE frame, the override attaches the child to a real link
    (e.g. the gripper link) with a full ``xyz`` + ``rpy`` origin. Most
    embodiments hang their gripper along the last arm joint's +Z axis, so the
    default offset is correct and this override is left ``None``. Robots
    whose gripper pokes perpendicular to the actuated axis (e.g. behavior R1
    Pro, whose last joint rotates about X) cannot reach the gripper with any
    pure-+Z offset: they set ``attach_link`` to the real gripper link and
    supply a full ``xyz`` + ``rpy`` origin. The emitted child link name is
    unchanged (``f"{ee.parent}_gripper_end"``), so runtime ``finger_keys`` keep
    working regardless of which path a case takes.

    ``xyz`` is in meters. The manifest declares ``rpy`` in **degrees**
    (matching ``rotate_z_deg``); the stored ``rpy`` is converted to **radians**
    at parse time, since the URDF emit and FK consumers expect radians.
    """

    attach_link: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class EeFrameSpec:
    """Semantic EE frame spec for one arm.

    ``rotate_z_deg == 0`` means no ``<parent>_ee`` child is inserted; the
    parent link is the semantic EE frame directly. ``gripper_forward`` is the
    +Z offset (in meters) of the emitted ``<parent>_gripper_end`` fixed child
    from the EE frame origin, used only on the default path. ``gripper_end``
    optionally overrides both the attach link and the full origin of that
    child (see :class:`GripperEndOverride`).
    """

    parent: str
    rotate_z_deg: int
    gripper_forward: float = _DEFAULT_GRIPPER_FORWARD
    gripper_end: GripperEndOverride | None = None

    @property
    def child(self) -> str:
        """Convention: the ``*_ee`` child link name is ``<parent>_ee``."""

        return f"{self.parent}_ee"


@dataclass(frozen=True)
class ArmSpec:
    """One arm's kinematic chain and EE spec."""

    arm_link_keys: tuple[str, ...]
    ee: EeFrameSpec

    def __post_init__(self) -> None:
        if not self.arm_link_keys:
            raise ValueError("arm_link_keys must not be empty")
        if self.arm_link_keys[-1] != self.ee.parent:
            raise ValueError(
                f"ee.parent '{self.ee.parent}' must equal arm_link_keys[-1] "
                f"'{self.arm_link_keys[-1]}'"
            )


@dataclass(frozen=True)
class GripperEndSpec:
    """Derived per-arm spec for the ``<parent>_gripper_end`` fixed child.

    Computed from :class:`ArmSpec` at load time.

    - ``attach_link`` is where the fixed joint's ``<parent>`` points: on
      the default path, when the arm's EE rotates (``rotate_z_deg != 0``),
      it is ``<parent>_ee``; when the arm's last link is already
      convention-compliant, it is ``<parent>``. When an override is
      present, it is the override's ``attach_link`` (a real link, e.g.
      the gripper link).
    - ``child`` is the emitted ``<link name>``. It is uniformly named
      ``f"{ee.parent}_gripper_end"`` regardless of which path sets the attach
      link, so the runtime ``finger_keys`` slug stays short and identical
      across rotated / non-rotated / overridden arms.
    - ``forward`` is the default-path +Z offset (in meters) from
      ``attach_link`` to the emitted child link. Retained for the
      visual-verify default-path marker; ignored when ``has_override``.
    - ``xyz`` / ``rpy`` are the full resolved joint origin (radians for
      ``rpy``). On the default path they are ``(0, 0, forward)`` /
      ``(0, 0, 0)``; under an override they come from
      :class:`GripperEndOverride` (manifest degrees converted to radians).
    """

    attach_link: str
    child: str
    forward: float
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    has_override: bool = False


@dataclass(frozen=True)
class AlignmentSpec:
    """Transformer-facing alignment intent extracted from the manifest."""

    arms: tuple[ArmSpec, ...] = ()
    camera_references: tuple[str, ...] = ()

    @property
    def ee_frames(self) -> tuple[EeFrameSpec, ...]:
        """Return per-arm EE specs.

        ``rotate_z_deg == 0`` entries are still returned; the stage 2
        transformer skips them internally so callers do not need to
        filter.
        """

        return tuple(arm.ee for arm in self.arms)

    @property
    def gripper_ends(self) -> tuple[GripperEndSpec, ...]:
        """Derived attach-link + child + origin per arm.

        Without an override, the attach link is ``<parent>_ee`` when the arm
        rotates else ``<parent>``, and the origin is
        ``xyz=(0, 0, gripper_forward)`` with identity rpy. With an override,
        the attach link and full ``xyz`` / ``rpy`` origin come from the
        override. The emitted child link is always
        ``f"{arm.ee.parent}_gripper_end"``.
        """

        specs: list[GripperEndSpec] = []
        for arm in self.arms:
            child = f"{arm.ee.parent}_gripper_end"
            override = arm.ee.gripper_end
            if override is None:
                attach = (
                    arm.ee.child if arm.ee.rotate_z_deg != 0 else arm.ee.parent
                )
                forward = arm.ee.gripper_forward
                specs.append(
                    GripperEndSpec(
                        attach_link=attach,
                        child=child,
                        forward=forward,
                        xyz=(0.0, 0.0, forward),
                        rpy=(0.0, 0.0, 0.0),
                        has_override=False,
                    )
                )
            else:
                specs.append(
                    GripperEndSpec(
                        attach_link=override.attach_link,
                        child=child,
                        forward=arm.ee.gripper_forward,
                        xyz=override.xyz,
                        rpy=override.rpy,
                        has_override=True,
                    )
                )
        return tuple(specs)


# ---------------------------------------------------------------------------
# Case.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UrdfAlignmentCase:
    """Reusable metadata for one origin-to-aligned URDF pair.

    A single ``alignment.yaml`` produces one case per packer key; shared
    embodiments (aloha_v1 / aloha_v2) yield two cases pointing at the same
    aligned URDF asset.
    """

    name: str
    origin_urdf: Path
    aligned_urdf: Path
    config: ConfigWiring
    adapter: str = ""
    alignment: AlignmentSpec = field(default_factory=AlignmentSpec)
    mesh_search_roots: tuple[Path, ...] = ()
    """Extra directories the mesh resolver may search when copying assets."""

    manifest_path: Path | None = None
    """Absolute path of the ``alignment.yaml`` this case was loaded from.

    Populated on load; useful for debug output and for CLI callers that want
    to round-trip back to the manifest file. Not used for equality.
    """

    resolved: bool = True
    """Whether ``aligned_urdf`` points at a real file on disk.

    ``False`` means the aligned URDF asset is missing (typically because the
    ``urdf_align/`` bucket symlink is dangling or the specific embodiment has
    not been re-emitted into the bucket yet). Callers that need the aligned
    URDF — the pytest suite, ``align.py``, ``visual_verify`` — should check
    ``resolved`` and either skip or emit an actionable error. Manifest fields
    (``config``, ``alignment``, ``adapter``, …) are still fully populated so
    library callers can inspect the case without touching the URDF file.
    """

    # ------------------------------------------------------------------ #
    # Backwards-compatible accessors used by the pytest contract and by the
    # config-wiring / discover / cli helpers.
    # ------------------------------------------------------------------ #

    @property
    def config_module(self) -> str:
        return self.config.module

    @property
    def config_getter(self) -> str:
        return self.config.getter

    @property
    def config_key(self) -> str:
        return self.config.key

    @property
    def aligned_config_key(self) -> str:
        return self.config.aligned_key

    @property
    def joints(self) -> tuple[str, ...]:
        """Return the actuated joint name list read from the origin URDF.

        The alignment pipeline never renames actuated joints (only inserts
        new fixed ``*_ee`` / ``*_gripper_end`` / ``*_camera_mount_compat``
        joints), so this same list applies verbatim to the aligned URDF.
        """

        return tuple(_read_urdf_actuated_joint_names(self.origin_urdf))

    @property
    def motion_links(self) -> tuple[LinkNamePair, ...]:
        """Derive motion-consistency link pairs by walking the origin URDF.

        Every link reachable from a kinematic root is exposed as a motion
        link. Origin and aligned URDFs share the same link names (the pipeline
        adds sibling ``*_ee`` / ``*_gripper_end`` / ``*_camera_mount_compat``
        children rather than renaming existing links), so all three name
        fields are identical.

        Orphaned links (declared but disconnected from the root) are filtered
        out; pytorch_kinematics's FK output only contains reachable links, so
        including orphans would produce spurious ``KeyError``s downstream.
        """

        origin_links = _read_urdf_link_names(self.origin_urdf)
        reachable = _reachable_link_names(self.origin_urdf)
        return tuple(
            LinkNamePair(
                semantic_name=_semantic_link_name(link),
                origin_name=link,
                aligned_name=link,
            )
            for link in origin_links
            if link in reachable
        )

    @property
    def camera_mount_links(self) -> tuple[LinkNamePair, ...]:
        """Compute camera compat aliases from the URDF pair.

        A ``<link>_camera_mount_compat`` link is emitted by the transformer
        only when the reference link's frame moved during alignment. Detect
        that by scanning the aligned URDF for links whose names end in
        ``_camera_mount_compat`` and whose base name matches a declared
        ``camera_references`` entry.
        """

        declared = set(self.alignment.camera_references)
        if not self.aligned_urdf.exists():
            return ()
        aligned_links = _read_urdf_link_names(self.aligned_urdf)
        pairs: list[LinkNamePair] = []
        for aligned_link in aligned_links:
            base = _strip_camera_compat_suffix(aligned_link)
            if base is None or base not in declared:
                continue
            pairs.append(
                LinkNamePair(
                    semantic_name=_semantic_link_name(base) + "_camera_mount",
                    origin_name=base,
                    aligned_name=aligned_link,
                )
            )
        return tuple(pairs)

    @property
    def link_renames(self) -> Mapping[str, str]:
        """Empty mapping preserved for legacy callers.

        The alignment pipeline never renames existing links; this property is
        retained so third-party tooling that reads ``case.link_renames`` keeps
        working through the migration.
        """

        return {}

    @property
    def joint_samples(self) -> tuple[dict[str, float], ...]:
        """Deterministic FK samples: zero pose + swept nonzero pose.

        Uses the actuated joint names from the origin URDF as the semantic
        joint identifiers.
        """

        zero_pose: dict[str, float] = {}
        sweep_pose: dict[str, float] = {}
        for idx, joint in enumerate(self.joints):
            value = 0.08 * (idx + 1) * (1 if idx % 2 == 0 else -1)
            sweep_pose[joint] = float(value)
        return (zero_pose, sweep_pose)


# ---------------------------------------------------------------------------
# Manifest loading.
# ---------------------------------------------------------------------------


def default_alignment_cases(
    repo_root: Path,
    config_root: Path | None = None,
) -> list[UrdfAlignmentCase]:
    """Return every URDF alignment case discovered under the default roots."""

    return load_alignment_cases(repo_root, config_root=config_root)


def load_alignment_cases(
    repo_root: Path,
    manifest_root: Path | None = None,
    align_root: Path | None = None,
    config_root: Path | None = None,
) -> list[UrdfAlignmentCase]:
    """Walk ``urdf_tools/manifests/<dataset>/<embodiment>/alignment.yaml``.

    Args:
        repo_root: Repository root used to resolve relative URDF paths.
        manifest_root: Root of the git-tracked manifest tree. Defaults to
            ``projects/holobrain_internal/common/urdf_tools/manifests``.
        align_root: Root of the (bucket-mounted, not git-tracked) aligned
            URDF tree. Defaults to
            ``projects/holobrain_internal/common/urdf_align``. Each manifest
            at ``<manifest_root>/<dataset>/<embodiment>/alignment.yaml`` maps
            to an aligned URDF under
            ``<align_root>/<dataset>/<embodiment>/``.
        config_root: Directory containing packer config modules. Default is
            ``<repo_root>/projects/holobrain_internal/common/configs``.

    Returns:
        list[UrdfAlignmentCase]: One case per (manifest, config_key) pair,
        sorted by manifest path so iteration order is deterministic. Cases
        whose aligned URDF is missing on disk are still returned with
        ``resolved=False``; the pytest layer handles per-case skips.
    """

    resolved_manifest_root = manifest_root or DEFAULT_MANIFEST_ROOT
    resolved_align_root = align_root or DEFAULT_URDF_ALIGN_ROOT
    resolved_config_root = (
        config_root or repo_root / _PACKER_CONFIG_ROOT_SUBDIR
    )
    cases: list[UrdfAlignmentCase] = []
    if not resolved_manifest_root.exists():
        return cases
    for manifest_path in sorted(
        resolved_manifest_root.rglob("alignment.yaml")
    ):
        cases.extend(
            _cases_from_manifest_path(
                manifest_path=manifest_path,
                repo_root=repo_root,
                config_root=resolved_config_root,
                manifest_root=resolved_manifest_root,
                align_root=resolved_align_root,
            )
        )
    return cases


def load_alignment_manifest(
    manifest_path: Path,
    repo_root: Path,
    config_root: Path | None = None,
    manifest_root: Path | None = None,
    align_root: Path | None = None,
) -> list[UrdfAlignmentCase]:
    """Load one ``alignment.yaml`` and expand it into per-key cases.

    Used by the CLI ``align`` / ``discover`` subcommands that operate on a
    single manifest file rather than the whole tree.
    """

    resolved_config_root = (
        config_root or repo_root / _PACKER_CONFIG_ROOT_SUBDIR
    )
    resolved_manifest_root = manifest_root or DEFAULT_MANIFEST_ROOT
    resolved_align_root = align_root or DEFAULT_URDF_ALIGN_ROOT
    return _cases_from_manifest_path(
        manifest_path=manifest_path,
        repo_root=repo_root,
        config_root=resolved_config_root,
        manifest_root=resolved_manifest_root,
        align_root=resolved_align_root,
    )


def _cases_from_manifest_path(
    manifest_path: Path,
    repo_root: Path,
    config_root: Path,
    manifest_root: Path,
    align_root: Path,
) -> list[UrdfAlignmentCase]:
    context = f"{manifest_path}"
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{context}: root must be a mapping")
    adapter = _required_str(document, "adapter", context)
    config_block = _required_object(document, "config", context)
    config = _config_from_manifest(config_block, f"{context}.config")

    arms_block = document.get("arms", [])
    if not isinstance(arms_block, list) or not arms_block:
        raise ValueError(f"{context}.arms must be a non-empty list")
    arms = tuple(
        _arm_from_manifest(entry, f"{context}.arms[{idx}]")
        for idx, entry in enumerate(arms_block)
    )
    camera_references = _camera_references_from_manifest(
        document.get("camera_references"),
        f"{context}.camera_references",
    )
    alignment = AlignmentSpec(
        arms=arms,
        camera_references=camera_references,
    )
    mesh_search_roots = _mesh_search_roots_from_manifest(
        document.get("mesh_search_roots"),
        f"{context}.mesh_search_roots",
    )

    # The aligned URDF lives in the parallel ``urdf_align`` tree (bucket
    # symlink), at the same <dataset>/<embodiment> subpath as the manifest.
    aligned_urdf, resolved = _resolve_aligned_urdf_path(
        manifest_path=manifest_path,
        manifest_root=manifest_root,
        align_root=align_root,
    )

    cases: list[UrdfAlignmentCase] = []
    for key, aligned_key in zip(config.keys, config.aligned_keys, strict=True):
        per_key_config = ConfigWiring(
            module=config.module,
            getter=config.getter,
            keys=(key,),
            aligned_keys=(aligned_key,),
        )
        origin_urdf = _resolve_origin_urdf_from_config(
            repo_root=repo_root,
            config_root=config_root,
            adapter_key=adapter,
            config=per_key_config,
            context=f"{context}({key})",
        )
        cases.append(
            UrdfAlignmentCase(
                name=key,
                adapter=adapter,
                origin_urdf=origin_urdf,
                aligned_urdf=aligned_urdf,
                config=per_key_config,
                alignment=alignment,
                mesh_search_roots=mesh_search_roots,
                manifest_path=manifest_path,
                resolved=resolved,
            )
        )
    return cases


def _config_from_manifest(
    data: dict[str, Any],
    context: str,
) -> ConfigWiring:
    keys = _keys_from_manifest(
        _required_value(data, "key", context),
        f"{context}.key",
    )
    aligned_keys = _keys_from_manifest(
        _required_value(data, "aligned_key", context),
        f"{context}.aligned_key",
    )
    return ConfigWiring(
        module=_required_str(data, "module", context),
        getter=_required_str(data, "getter", context),
        keys=keys,
        aligned_keys=aligned_keys,
    )


def _keys_from_manifest(value: Any, context: str) -> tuple[str, ...]:
    """Normalize ``str | list[str]`` config keys to a tuple of strings."""

    if isinstance(value, str):
        if not value:
            raise ValueError(f"{context} must be a non-empty string")
        return (value,)
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"{context}[{idx}] must be a non-empty string"
                )
        if not value:
            raise ValueError(f"{context} must not be empty")
        return tuple(value)
    raise ValueError(f"{context} must be a string or a list of strings")


def _arm_from_manifest(entry: Any, context: str) -> ArmSpec:
    if not isinstance(entry, dict):
        raise ValueError(f"{context} must be a mapping")
    arm_link_keys_raw = _required_value(entry, "arm_link_keys", context)
    if not isinstance(arm_link_keys_raw, list) or not arm_link_keys_raw:
        raise ValueError(
            f"{context}.arm_link_keys must be a non-empty list of strings"
        )
    arm_link_keys: list[str] = []
    for idx, item in enumerate(arm_link_keys_raw):
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"{context}.arm_link_keys[{idx}] must be a non-empty string"
            )
        arm_link_keys.append(item)
    ee = _ee_from_manifest(
        _required_object(entry, "ee", context),
        f"{context}.ee",
    )
    return ArmSpec(arm_link_keys=tuple(arm_link_keys), ee=ee)


def _ee_from_manifest(data: dict[str, Any], context: str) -> EeFrameSpec:
    parent = _required_str(data, "parent", context)
    rotate = int(data.get("rotate_z_deg", 0))
    if rotate not in _ROTATION_CHOICES:
        raise ValueError(
            f"{context}.rotate_z_deg must be one of {_ROTATION_CHOICES}, "
            f"got {rotate}"
        )
    gripper_forward = data.get("gripper_forward", _DEFAULT_GRIPPER_FORWARD)
    if not isinstance(gripper_forward, (int, float)) or isinstance(
        gripper_forward, bool
    ):
        raise ValueError(
            f"{context}.gripper_forward must be a number, "
            f"got {type(gripper_forward).__name__}"
        )
    gripper_end = _gripper_end_override_from_manifest(
        data.get("gripper_end"),
        f"{context}.gripper_end",
    )
    return EeFrameSpec(
        parent=parent,
        rotate_z_deg=rotate,
        gripper_forward=float(gripper_forward),
        gripper_end=gripper_end,
    )


def _vec3_from_manifest(
    data: dict[str, Any],
    key: str,
    context: str,
) -> tuple[float, float, float]:
    """Parse a required length-3 numeric list as a float tuple.

    Booleans are rejected so a stray ``true`` does not coerce to ``1.0``.
    """

    value = _required_value(data, key, context)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            f"{context}.{key} must be a list of 3 numbers, got {value!r}"
        )
    parsed: list[float] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(
                f"{context}.{key}[{idx}] must be a number, "
                f"got {type(item).__name__}"
            )
        parsed.append(float(item))
    return (parsed[0], parsed[1], parsed[2])


def _gripper_end_override_from_manifest(
    data: Any,
    context: str,
) -> GripperEndOverride | None:
    """Parse the optional ``ee.gripper_end`` override block.

    ``None`` / absent means the default ``xyz="0 0 gripper_forward"`` path is
    used. When present, ``attach_link`` (a real URDF link) plus a full ``xyz``
    and ``rpy`` origin are all required — a partial override is almost
    certainly a mistake. Unknown keys inside the block are rejected (stricter
    than the ee-level "silently ignore" rule) so a typo like ``rpy_rad`` fails
    loudly.

    ``xyz`` is in meters; ``rpy`` is declared in **degrees** (matching
    ``rotate_z_deg``) and converted to radians here, since the URDF emit and
    FK consumers expect radians.
    """

    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping")
    allowed = {"attach_link", "xyz", "rpy"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"{context} has unknown key(s): {sorted(unknown)} "
            f"(allowed: {sorted(allowed)})"
        )
    attach_link = _required_str(data, "attach_link", context)
    xyz = _vec3_from_manifest(data, "xyz", context)
    rpy_deg = _vec3_from_manifest(data, "rpy", context)
    rpy = tuple(math.radians(v) for v in rpy_deg)
    return GripperEndOverride(attach_link=attach_link, xyz=xyz, rpy=rpy)


def _camera_references_from_manifest(
    value: Any,
    context: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of URDF link names")
    entries: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"{context}[{idx}] must be a non-empty string link name"
            )
        entries.append(item)
    return tuple(entries)


def _mesh_search_roots_from_manifest(
    value: Any,
    context: str,
) -> tuple[Path, ...]:
    """Parse the optional ``mesh_search_roots`` list from a manifest.

    Missing / empty is fine (returns an empty tuple). Entries are kept as
    ``Path`` objects and resolved against the origin URDF's parent at copy
    time — see :func:`sync_assets._search_bases`.
    """

    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")
    roots: list[Path] = []
    for idx, entry in enumerate(value):
        if not isinstance(entry, str) or not entry:
            raise ValueError(
                f"{context}[{idx}] must be a non-empty string path"
            )
        roots.append(Path(entry))
    return tuple(roots)


def _resolve_origin_urdf_from_config(
    repo_root: Path,
    config_root: Path,
    adapter_key: str,
    config: ConfigWiring,
    context: str,
) -> Path:
    """Read the URDF path from the packer config and absolutize it."""

    from projects.holobrain_internal.common.urdf_tools.adapters import (
        get_adapter,
    )

    dataset_config = _import_packer_config(config, config_root, context)
    entry = dataset_config.get(config.key)
    if not isinstance(entry, dict):
        raise ValueError(
            f"{context}.config.key '{config.key}' is not a dict in "
            f"{config.module}.{config.getter}(); cannot derive origin_urdf"
        )
    adapter = get_adapter(adapter_key)
    urdf_value = adapter.resolve_urdf_path(entry)
    if urdf_value is None:
        raise ValueError(
            f"{context}.config.key '{config.key}' has no URDF path "
            f"resolvable via adapter '{adapter_key}' in "
            f"{config.module}.{config.getter}(); cannot derive origin_urdf"
        )
    common_root = repo_root / _PACKER_CONFIG_COMMON_SUBDIR
    return _resolve_repo_path(common_root, urdf_value)


def _import_packer_config(
    config: ConfigWiring,
    config_root: Path,
    context: str,
) -> Mapping[str, Any]:
    """Import the packer config module and call its dataset getter."""

    import importlib

    config_root_str = str(config_root)
    sys.path.insert(0, config_root_str)
    try:
        module = importlib.import_module(config.module)
    except ImportError as exc:
        raise ValueError(
            f"{context}.config.module '{config.module}' failed to import "
            f"from {config_root}: {exc}"
        ) from exc
    finally:
        try:
            sys.path.remove(config_root_str)
        except ValueError:
            pass
    getter = getattr(module, config.getter, None)
    if not callable(getter):
        raise ValueError(
            f"{context}.config.getter '{config.getter}' is not callable on "
            f"{config.module}"
        )
    result = getter()
    if not isinstance(result, Mapping):
        raise ValueError(
            f"{context}.config.getter '{config.getter}' on {config.module} "
            f"must return a mapping, got {type(result).__name__}"
        )
    return result


def _resolve_aligned_urdf_path(
    manifest_path: Path,
    manifest_root: Path,
    align_root: Path,
) -> tuple[Path, bool]:
    """Locate the aligned URDF for a manifest under the parallel align tree.

    Manifests live at
    ``<manifest_root>/<dataset>/<embodiment>/alignment.yaml`` and their
    aligned URDFs live at
    ``<align_root>/<dataset>/<embodiment>/*.urdf`` (the ``urdf_align/``
    tree is a symlink to a shared bucket, not git-tracked).

    Returns ``(aligned_urdf_path, resolved)`` where ``resolved`` is ``True``
    when the aligned URDF file exists on disk. When no URDF is found (bucket
    unmounted, symlink dangling, or embodiment not re-emitted yet), a
    placeholder ``<folder>/<embodiment>.urdf`` path is returned with
    ``resolved=False`` so callers still see a stable Path object.
    """

    try:
        rel_folder = manifest_path.parent.relative_to(manifest_root)
    except ValueError:
        # Off-tree manifests (rare; e.g. CLI called on an arbitrary path)
        # fall back to a sibling folder-scan behavior.
        rel_folder = Path(manifest_path.parent.name)
        aligned_folder = manifest_path.parent
    else:
        aligned_folder = align_root / rel_folder
    if aligned_folder.exists():
        urdfs = sorted(aligned_folder.glob("*.urdf"))
        if urdfs:
            return urdfs[0], True
    placeholder = aligned_folder / f"{aligned_folder.name}.urdf"
    return placeholder, placeholder.exists()


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _required_object(
    data: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    value = _required_value(data, key, context)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a mapping")
    return value


def _required_str(
    data: dict[str, Any],
    key: str,
    context: str,
) -> str:
    value = _required_value(data, key, context)
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be a string")
    return value


def _required_value(data: dict[str, Any], key: str, context: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping")
    if key not in data:
        raise ValueError(f"{context}.{key} is required")
    return data[key]


# ---------------------------------------------------------------------------
# URDF helpers (small; do not import the transformer here to avoid cycles).
# ---------------------------------------------------------------------------


def _read_urdf_link_names(urdf_path: Path) -> tuple[str, ...]:
    """Return the ordered list of ``<link name>`` values in a URDF file."""

    tree = ET.parse(urdf_path)
    return tuple(
        link.get("name", "")
        for link in tree.getroot().findall("link")
        if link.get("name")
    )


def _read_urdf_actuated_joint_names(urdf_path: Path) -> tuple[str, ...]:
    """Return actuated joint names from a URDF file in declaration order.

    Actuated types are ``revolute``, ``continuous``, and ``prismatic``.
    Fixed / mimic joints do not contribute to the FK sample vector, so
    they are excluded. Mirrors
    ``pytorch_kinematics.Chain.get_joint_parameter_names``.
    """

    actuated_types = {"revolute", "continuous", "prismatic"}
    tree = ET.parse(urdf_path)
    names: list[str] = []
    for joint in tree.getroot().findall("joint"):
        joint_type = joint.get("type", "")
        if joint_type not in actuated_types:
            continue
        mimic = joint.find("mimic")
        if mimic is not None:
            continue
        name = joint.get("name", "")
        if name:
            names.append(name)
    return tuple(names)


def _reachable_link_names(urdf_path: Path) -> frozenset[str]:
    """Return links reachable from the primary kinematic root.

    Mirrors what pytorch_kinematics considers part of the tree: the URDF is
    treated as a single-rooted graph whose root is the first link (in
    declaration order) that never appears as any joint's ``<child>``.
    """

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    declared_order: list[str] = [
        link.get("name", "")
        for link in root.findall("link")
        if link.get("name")
    ]
    declared_links = set(declared_order)
    edges: dict[str, list[str]] = {name: [] for name in declared_links}
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent_name = parent_el.get("link", "")
        child_name = child_el.get("link", "")
        if parent_name not in declared_links:
            continue
        if child_name not in declared_links:
            continue
        edges[parent_name].append(child_name)
        child_links.add(child_name)
    primary_root: str | None = next(
        (link for link in declared_order if link not in child_links),
        None,
    )
    if primary_root is None:
        return frozenset()
    reachable: set[str] = set()
    stack: list[str] = [primary_root]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(edges.get(node, ()))
    return frozenset(reachable)


_CAMERA_COMPAT_SUFFIX = "_camera_mount_compat"


def _strip_camera_compat_suffix(link_name: str) -> str | None:
    if not link_name.endswith(_CAMERA_COMPAT_SUFFIX):
        return None
    return link_name[: -len(_CAMERA_COMPAT_SUFFIX)] or None


def _semantic_link_name(link: str) -> str:
    """Convert a URDF link name into a stable semantic slug."""

    return link.replace("/", "_").replace(".", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Public utility exports.
# ---------------------------------------------------------------------------


__all__ = [
    "AlignmentSpec",
    "ArmSpec",
    "ConfigWiring",
    "DEFAULT_MANIFEST_ROOT",
    "DEFAULT_URDF_ALIGN_ROOT",
    "EeFrameSpec",
    "GripperEndSpec",
    "GripperEndOverride",
    "UrdfAlignmentCase",
    "default_alignment_cases",
    "load_alignment_cases",
    "load_alignment_manifest",
]

_ = Sequence  # keep type-only import referenced for linters
