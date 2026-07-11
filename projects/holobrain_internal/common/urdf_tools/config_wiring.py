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

"""Verify that a case's aligned-config entry is wired correctly.

The align pipeline stops short of editing the packer config module — that is
still an explicit human step (small, easy to review, and lets the author add
domain-specific fields like `task_names`). But we can cheaply verify that the
declared `aligned_key` exists in the config, points at the transformer output
path, and preserves the `robot_type`/`cam_names` from the base key. When any
of those checks fails, the CLI surfaces the problem so the user can update
the packer config.
"""

from __future__ import annotations
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projects.holobrain_internal.common.urdf_tools.adapters import (
    get_adapter,
)
from projects.holobrain_internal.common.urdf_tools.cases import (
    UrdfAlignmentCase,
)


@dataclass(frozen=True)
class ConfigWiringReport:
    """Result of validating one case's aligned config wiring.

    ``notes`` carries non-fatal observations (e.g. "fell back to the base
    config key because no aligned entry was declared"). It is populated even
    when ``ok=True`` so the CLI can surface which key satisfied the check —
    silent success would hide base-cutover wiring from the user.
    """

    ok: bool
    issues: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def format(self) -> str:
        parts: list[str] = []
        if self.notes:
            parts.extend(f"- (note) {note}" for note in self.notes)
        if self.ok and not parts:
            return "config wiring OK"
        if not self.ok:
            parts.extend(f"- {issue}" for issue in self.issues)
        return "\n".join(parts) if parts else "config wiring OK"


def verify_case_config_wiring(
    case: UrdfAlignmentCase,
    repo_root: Path,
    config_root: Path | None = None,
) -> ConfigWiringReport:
    """Validate that the case's aligned URDF is wired into the packer config.

    Two wiring shapes are accepted:

    1. **Companion entry** (original design): the config declares a distinct
       ``aligned_config_key`` (typically ``<base>_urdf_v2``) whose ``urdf``
       points at the aligned tree, alongside the untouched base entry. All
       three checks fire: aligned key exists, its ``urdf`` matches the
       emitted path, and its ``robot_type`` / ``cam_names`` mirror the base
       entry.
    2. **Base cutover**: no companion entry exists; the base entry itself
       already points at the aligned URDF. In that case the aligned URDF the
       tool just wrote *is* what the packer will load, and adding a redundant
       companion row would just be another place to drift out of sync. We
       fall back to verifying the base entry's ``urdf`` equals the emitted
       path and record a note so the operator can see which shape validated.

    When neither shape validates, the failure message lists both keys we
    tried, the ``urdf`` we found under each (if any), and the aligned path we
    just emitted, so the fix is obvious from the CLI output alone.

    Args:
        case: The parsed manifest entry.
        repo_root: Repository root; used to resolve the aligned URDF path
            relative to ``projects/holobrain_internal/common``.
        config_root: Directory containing packer config modules. Defaults to
            ``<repo_root>/projects/holobrain_internal/common/configs`` when
            not provided; the caller may override for tests.

    Returns:
        ConfigWiringReport: ``ok=True`` when either wiring shape validated.
        ``notes`` names which key satisfied the check (useful for shared
        embodiments where different keys may take different shapes).
    """

    resolved_config_root = (
        config_root or repo_root / "projects/holobrain_internal/common/configs"
    )
    config = _load_dataset_config(
        case,
        config_root=resolved_config_root,
    )
    adapter = get_adapter(case.adapter)
    expected = _expected_urdf_path(case, repo_root)

    aligned_key = case.aligned_config_key
    base_key = case.config_key
    aligned_entry = config.get(aligned_key)
    base_entry = config.get(base_key)

    # Prefer the companion-entry shape when the aligned key is declared. The
    # companion path can still fail (wrong urdf, non-dict entry, robot_type
    # drift) and each failure produces its own actionable issue.
    if isinstance(aligned_entry, dict):
        return _verify_companion_entry(
            case=case,
            adapter=adapter,
            config=config,
            aligned_entry=aligned_entry,
            expected=expected,
        )

    # Aligned key is absent or malformed. Fall back to the base cutover shape
    # only when the base entry itself points at the aligned URDF. When the
    # base entry still points at the origin tree, neither shape is wired up
    # and the aligned URDF we just wrote has no consumer — that is the
    # failure mode Stage 4 is designed to catch.
    if aligned_entry is not None:
        # Present but the wrong type — surface that specifically rather than
        # letting the fallback claim success on the base key.
        return ConfigWiringReport(
            ok=False,
            issues=(
                f"aligned key '{aligned_key}' must be a dict, "
                f"got {type(aligned_entry).__name__}",
            ),
        )
    if not isinstance(base_entry, dict):
        return ConfigWiringReport(
            ok=False,
            issues=(
                f"neither aligned key '{aligned_key}' nor base key "
                f"'{base_key}' is a dict in "
                f"{case.config_module}.{case.config_getter}(); expected one "
                f"of them to hold the aligned URDF entry at "
                f"'{expected}'",
            ),
        )
    base_urdf = adapter.resolve_urdf_path(base_entry)
    if base_urdf == expected:
        # Base cutover — base entry already points at the aligned URDF.
        # ``robot_type`` / ``cam_names`` cross-check does not apply here
        # (nothing to compare against), so the check ends after the URDF
        # match.
        return ConfigWiringReport(
            ok=True,
            notes=(
                f"aligned key '{aligned_key}' not declared; base key "
                f"'{base_key}' already points at the aligned URDF "
                f"('{expected}'), treating as a base-cutover wiring",
            ),
        )
    # Neither key is wired up. Emit one message covering both attempts so the
    # operator can pick which shape to fix without re-running the CLI.
    return ConfigWiringReport(
        ok=False,
        issues=(
            f"aligned URDF '{expected}' has no consumer in "
            f"{case.config_module}.{case.config_getter}():\n"
            f"    - aligned key '{aligned_key}' is not declared\n"
            f"    - base key '{base_key}' has urdf='{base_urdf}', "
            f"not '{expected}'\n"
            f"  Fix one of:\n"
            f"    (a) add a companion entry '{aligned_key}' with "
            f"urdf='{expected}' (matching '{base_key}'.robot_type / "
            f"cam_names), or\n"
            f"    (b) repoint '{base_key}'.urdf to '{expected}' "
            f"(base cutover; drops the '{aligned_key}' expectation)",
        ),
    )


def _verify_companion_entry(
    case: UrdfAlignmentCase,
    adapter,
    config: dict[str, Any],
    aligned_entry: dict[str, Any],
    expected: str,
) -> ConfigWiringReport:
    """Companion-shape check: aligned key declared, urdf + metadata match."""

    issues: list[str] = []
    urdf_value = adapter.resolve_urdf_path(aligned_entry)
    if urdf_value != expected:
        issues.append(
            f"'{case.aligned_config_key}'.urdf must be '{expected}', "
            f"got '{urdf_value}'"
        )
    if case.config_key in config:
        base = config[case.config_key]
        # Only compare fields that actually exist on the base entry. Some
        # packer schemas record fields like ``robot_type`` at the top level
        # while others push them under ``kinematics_config`` (or omit them
        # entirely). The adapter tells us where to look for URDF paths; for
        # other fields we do a permissive presence check so the wiring
        # assertion is uniform across schemas without duplicating adapter
        # logic here.
        for field_name in ("robot_type", "cam_names"):
            if field_name not in base:
                continue
            base_value = base.get(field_name)
            aligned_value = aligned_entry.get(field_name)
            if base_value != aligned_value:
                issues.append(
                    f"'{case.aligned_config_key}'.{field_name} must equal "
                    f"'{case.config_key}'.{field_name} "
                    f"({base_value!r}); got {aligned_value!r}"
                )
    return ConfigWiringReport(ok=not issues, issues=tuple(issues))


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _load_dataset_config(
    case: UrdfAlignmentCase,
    config_root: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(config_root))
    try:
        module = importlib.import_module(case.config_module)
    finally:
        sys.path.remove(str(config_root))
    getter = getattr(module, case.config_getter)
    return getter()


def _expected_urdf_path(case: UrdfAlignmentCase, repo_root: Path) -> str:
    """Format the aligned URDF path the way the packer configs use.

    Packer configs use paths relative to
    ``projects/holobrain_internal/common``, prefixed with ``./``. Keeping the
    formatter here means the wiring check does not depend on how any given
    config module resolves paths.
    """

    common_root = repo_root / "projects/holobrain_internal/common"
    aligned = case.aligned_urdf.resolve()
    try:
        relative = aligned.relative_to(common_root.resolve())
    except ValueError:
        return case.aligned_urdf.as_posix()
    return f"./{relative.as_posix()}"


__all__ = [
    "ConfigWiringReport",
    "verify_case_config_wiring",
]
