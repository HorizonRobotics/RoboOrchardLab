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

"""Deterministic visual verification reports for URDF alignment cases."""

from __future__ import annotations
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from projects.holobrain_internal.common.urdf_tools.cases import (
    UrdfAlignmentCase,
    load_alignment_cases,
)
from projects.holobrain_internal.common.urdf_tools.transform._linalg import (
    OriginTransform,
)

DEFAULT_VISUAL_VERIFY_DIR = Path(
    "projects/holobrain_internal/common/workspace_test/"
    "urdf_align_logs/visual_verify"
)
FRAMES_PER_DATASET_ROW = 3
AXIS_LENGTH = 0.12


@dataclass(frozen=True)
class VisualVerifyRequest:
    """Inputs for generating URDF alignment visual verification artifacts."""

    repo_root: Path
    align_root: Path
    config_root: Path | None = None
    output_dir: Path | None = None
    case_filters: tuple[str, ...] = ()
    final_review_status: str = "pending_manual_review"
    # Manifest tree scanned for ``alignment.yaml`` files. When ``None`` the
    # loader falls back to ``DEFAULT_MANIFEST_ROOT`` (the git-tracked
    # in-repo manifests) — the standard production case. Tests override
    # this to point at a hermetic tmp tree; the CLI plumbs it through
    # ``--manifest-root``.
    manifest_root: Path | None = None


def run_visual_verify(request: VisualVerifyRequest) -> dict[str, Any]:
    """Generate PNG contact sheets, per-case JSON, and a checklist.

    The probe is dataset-I/O-free by design: each aligned config row gets one
    deterministic FK "episode" with three frames (zero, sweep, reverse sweep)
    so visual verification can run wherever the URDF assets are present.
    """

    repo_root = request.repo_root.resolve()
    align_root = _resolve(repo_root, request.align_root)
    config_root = (
        _resolve(repo_root, request.config_root)
        if request.config_root is not None
        else None
    )
    output_dir = _resolve(
        repo_root,
        request.output_dir or DEFAULT_VISUAL_VERIFY_DIR,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_root = (
        _resolve(repo_root, request.manifest_root)
        if request.manifest_root is not None
        else None
    )
    cases = load_alignment_cases(
        repo_root=repo_root,
        manifest_root=manifest_root,
        align_root=align_root,
        config_root=config_root,
    )
    cases = _filter_cases(cases, request.case_filters)
    if not cases:
        filters = ", ".join(request.case_filters) or "<none>"
        raise ValueError(
            f"no visual verification cases matched filter(s): {filters}"
        )

    created_files: list[Path] = []
    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        report = _case_visual_report(case, output_dir)
        case_summaries.append(report["summary"])
        created_files.extend(
            [Path(report["json_path"]), Path(report["png_path"])]
        )

    index_path = output_dir / "index.json"
    checklist_path = output_dir / "CHECKLIST.md"
    index = {
        "kind": "urdf_alignment_visual_verify",
        "output_dir": str(output_dir),
        "episode": {
            "name": "deterministic_fk_episode_000",
            "frames_per_dataset_row": FRAMES_PER_DATASET_ROW,
            "source": "synthetic_fk_probe",
        },
        "cases": case_summaries,
        "created_files": [str(path) for path in created_files],
        "checklist": str(checklist_path),
    }
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    created_files.append(index_path)
    _write_checklist(
        checklist_path,
        cases=case_summaries,
        created_files=created_files,
        final_review_status=request.final_review_status,
    )
    index["created_files"].append(str(index_path))
    index["created_files"].append(str(checklist_path))
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def _case_visual_report(
    case: UrdfAlignmentCase,
    output_dir: Path,
) -> dict[str, Any]:
    slug = _case_slug(case)
    png_path = output_dir / f"{slug}.png"
    json_path = output_dir / f"{slug}.json"
    fk_joint_names = _fk_joint_names(case)
    joint_samples = _frame_joint_samples(fk_joint_names)
    origin_edges = _joint_edges(case.origin_urdf)
    aligned_edges = _joint_edges(case.aligned_urdf)
    origin_link_names = tuple(link.origin_name for link in case.motion_links)
    aligned_link_names = tuple(link.aligned_name for link in case.motion_links)
    origin_extra_links, aligned_extra_links = _visual_reference_links(case)
    origin_fk_link_names = tuple(
        dict.fromkeys((*origin_link_names, *origin_extra_links))
    )
    aligned_fk_link_names = tuple(
        dict.fromkeys((*aligned_link_names, *aligned_extra_links))
    )
    frames = []
    render_frames = []
    for frame_index, joint_values in enumerate(joint_samples):
        origin_matrices = _fk_matrices(
            case.origin_urdf,
            joint_values,
            origin_fk_link_names,
        )
        aligned_matrices = _fk_matrices(
            case.aligned_urdf,
            joint_values,
            aligned_fk_link_names,
        )
        origin_positions = _positions_from_matrices(
            origin_matrices, origin_link_names
        )
        aligned_positions = _positions_from_matrices(
            aligned_matrices, aligned_link_names
        )
        max_delta = _max_position_delta(
            origin_positions,
            aligned_positions,
            case.motion_links,
        )
        gripper_end_axis_markers = _frame_gripper_end_axis_markers(
            case,
            origin_matrices=origin_matrices,
            aligned_matrices=aligned_matrices,
        )
        camera_reference_markers = _frame_camera_reference_markers(
            case,
            origin_matrices=origin_matrices,
            aligned_matrices=aligned_matrices,
        )
        frames.append(
            {
                "frame_index": frame_index,
                "joint_positions": {
                    joint: _round_float(value)
                    for joint, value in joint_values.items()
                },
                "max_motion_link_position_delta": _round_float(max_delta),
                "origin_link_count": len(origin_positions),
                "aligned_link_count": len(aligned_positions),
                "gripper_end_axes": _json_gripper_end_axes(
                    gripper_end_axis_markers
                ),
                "camera_references": _json_camera_references(
                    camera_reference_markers
                ),
            }
        )
        render_frames.append(
            {
                "frame_index": frame_index,
                "origin_positions": origin_positions,
                "aligned_positions": aligned_positions,
                "origin_edges": origin_edges,
                "aligned_edges": aligned_edges,
                "gripper_end_axes": gripper_end_axis_markers,
                "camera_references": camera_reference_markers,
            }
        )

    _render_contact_sheet(
        case=case,
        png_path=png_path,
        frames=render_frames,
    )
    dataset_rows = _dataset_rows_for_case(case)
    report = {
        "kind": "urdf_alignment_visual_verify_case",
        "case": case.name,
        "adapter": case.adapter,
        "config_key": case.config_key,
        "aligned_config_key": case.aligned_config_key,
        "dataset_row": case.aligned_config_key,
        "dataset_rows": dataset_rows,
        "origin_urdf": str(case.origin_urdf),
        "aligned_urdf": str(case.aligned_urdf),
        "case_joint_names": list(case.joints),
        "fk_joint_names": list(fk_joint_names),
        "contact_sheet": str(png_path),
        "episode": {
            "name": "deterministic_fk_episode_000",
            "frames_per_dataset_row": FRAMES_PER_DATASET_ROW,
        },
        "frames": frames,
    }
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "json_path": str(json_path),
        "png_path": str(png_path),
        "summary": {
            "case": case.name,
            "adapter": case.adapter,
            "config_key": case.config_key,
            "aligned_config_key": case.aligned_config_key,
            "dataset_rows": dataset_rows,
            "json": str(json_path),
            "png": str(png_path),
            "max_motion_link_position_delta": max(
                frame["max_motion_link_position_delta"] for frame in frames
            ),
        },
    }


def _render_contact_sheet(
    case: UrdfAlignmentCase,
    png_path: Path,
    frames: list[dict[str, Any]],
) -> None:
    _prepare_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    fig.suptitle(
        f"{case.config_key} -> {case.aligned_config_key}",
        fontsize=12,
    )
    for col, frame in enumerate(frames):
        bounds = _frame_bounds(
            frame["origin_positions"],
            frame["aligned_positions"],
            frame["gripper_end_axes"],
            frame["camera_references"],
        )
        for row, (label, positions_key, edges_key) in enumerate(
            (
                ("origin", "origin_positions", "origin_edges"),
                ("aligned", "aligned_positions", "aligned_edges"),
            )
        ):
            ax = fig.add_subplot(
                2,
                len(frames),
                row * len(frames) + col + 1,
                projection="3d",
            )
            _plot_robot(
                ax,
                positions=frame[positions_key],
                edges=frame[edges_key],
            )
            _plot_gripper_end_axes(
                ax,
                markers=frame["gripper_end_axes"],
                side=label,
            )
            _plot_camera_references(
                ax,
                markers=frame["camera_references"],
                side=label,
            )
            ax.set_title(f"{label} frame {frame['frame_index']}", fontsize=9)
            _set_equal_bounds(ax, bounds)
            ax.view_init(elev=24, azim=-58)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _plot_robot(
    ax: Any,
    positions: Mapping[str, np.ndarray],
    edges: Iterable[tuple[str, str]],
) -> None:
    for parent, child in edges:
        if parent not in positions or child not in positions:
            continue
        points = np.stack([positions[parent], positions[child]])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#3b5b92")
    if not positions:
        return
    all_points = np.stack(list(positions.values()))
    ax.scatter(
        all_points[:, 0],
        all_points[:, 1],
        all_points[:, 2],
        s=8,
        color="#c43b3b",
    )


def _plot_gripper_end_axes(
    ax: Any,
    markers: list[dict[str, Any]],
    side: str,
) -> None:
    colors = {
        "x": "#d62728",
        "y": "#2ca02c",
        "z": "#1f77b4",
    }
    for marker in markers:
        axis_frame = marker[side]
        origin = axis_frame["position"]
        for axis_name, color in colors.items():
            endpoint = axis_frame[f"{axis_name}_endpoint"]
            vector = endpoint - origin
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                vector[0],
                vector[1],
                vector[2],
                color=color,
                linewidth=1.4,
                arrow_length_ratio=0.25,
                normalize=False,
            )


def _plot_camera_references(
    ax: Any,
    markers: list[dict[str, Any]],
    side: str,
) -> None:
    for marker in markers:
        link_key = f"{side}_link"
        position = marker[side]["position"]
        ax.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            s=46,
            color="#c218c2",
            marker="^",
            depthshade=False,
        )
        ax.text(
            position[0],
            position[1],
            position[2],
            marker[link_key],
            color="#8a008a",
            fontsize=6,
        )


def _prepare_matplotlib() -> None:
    mpl_config = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/robo_mpl"))
    mpl_config.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)


def _fk_matrices(
    urdf: Path,
    joint_values: Mapping[str, float],
    link_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    import pytorch_kinematics as pk

    chain = pk.build_chain_from_urdf(urdf.read_bytes())
    chain_joint_names = tuple(chain.get_joint_parameter_names())
    ordered_values = [joint_values[joint] for joint in chain_joint_names]
    fk = chain.forward_kinematics(
        torch.tensor([ordered_values], dtype=torch.float32)
    )
    missing_links = [link for link in link_names if link not in fk]
    if missing_links:
        raise KeyError(
            f"{urdf}: FK chain link(s) missing: " + ", ".join(missing_links)
        )
    return {
        link: fk[link].get_matrix()[0].detach().cpu().numpy()
        for link in link_names
    }


def _positions_from_matrices(
    matrices: Mapping[str, np.ndarray],
    link_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return {link: matrices[link][:3, 3] for link in link_names}


def _visual_reference_links(
    case: UrdfAlignmentCase,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    origin_links: list[str] = []
    aligned_links: list[str] = []
    for arm, gripper_end in zip(
        case.alignment.arms,
        case.alignment.gripper_ends,
        strict=True,
    ):
        origin_links.append(arm.ee.parent)
        # On the override path the gripper_end attaches to a real link (e.g.
        # the gripper link) rather than ``<parent>_ee``; FK that real link on
        # the origin side too so the override origin can be composed against
        # it below.
        if gripper_end.has_override:
            origin_links.append(gripper_end.attach_link)
        aligned_links.append(gripper_end.child)
    compat_by_origin = {
        mount.origin_name: mount.aligned_name
        for mount in case.camera_mount_links
    }
    for reference in case.alignment.camera_references:
        origin_links.append(reference)
        aligned_links.append(compat_by_origin.get(reference, reference))
    return tuple(origin_links), tuple(aligned_links)


def _frame_gripper_end_axis_markers(
    case: UrdfAlignmentCase,
    origin_matrices: Mapping[str, np.ndarray],
    aligned_matrices: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    markers = []
    for arm_index, (arm, gripper_end) in enumerate(
        zip(
            case.alignment.arms,
            case.alignment.gripper_ends,
            strict=True,
        )
    ):
        if gripper_end.has_override:
            # Override path: the gripper_end attaches to a real link (e.g. the
            # gripper link) at a full ``xyz`` / ``rpy`` origin. Compose that
            # origin against the real link's origin-side FK pose to get the
            # expected gripper_end pose.
            origin_reference_link = gripper_end.attach_link
            origin_matrix = _apply_origin(
                origin_matrices[origin_reference_link],
                gripper_end.xyz,
                gripper_end.rpy,
            )
        else:
            # Default path: the gripper_end sits at ``forward`` along the EE
            # frame's +Z. The origin URDF has no ``<parent>_ee`` link, but a
            # pure-Rz EE rotation preserves the +Z axis, so translating
            # ``ee.parent`` along its own +Z by ``forward`` is the consistent
            # origin-side expectation.
            origin_reference_link = arm.ee.parent
            origin_matrix = _translated_along_local_z(
                origin_matrices[origin_reference_link],
                gripper_end.forward,
            )
        origin_axes = _axis_marker(origin_matrix)
        aligned_axes = _axis_marker(aligned_matrices[gripper_end.child])
        markers.append(
            {
                "arm_index": arm_index,
                "origin_link": gripper_end.child,
                "origin_reference_link": origin_reference_link,
                "aligned_link": gripper_end.child,
                "origin": origin_axes,
                "aligned": aligned_axes,
                "axis_dot_products": {
                    axis: _round_float(
                        float(
                            np.dot(
                                origin_axes[f"{axis}_vector"],
                                aligned_axes[f"{axis}_vector"],
                            )
                        )
                    )
                    for axis in ("x", "y", "z")
                },
            }
        )
    return markers


def _translated_along_local_z(
    matrix: np.ndarray,
    distance: float,
) -> np.ndarray:
    translated = matrix.copy()
    translated[:3, 3] = matrix[:3, 3] + float(distance) * matrix[:3, 2]
    return translated


def _apply_origin(
    matrix: np.ndarray,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    """Compose a URDF ``<origin xyz rpy>`` against a base 4x4 pose.

    Used on the override path to build the origin-side expected gripper_end
    pose from the real attach link's FK pose and the override origin.
    """

    origin = OriginTransform.from_rpy_xyz(rpy, xyz).as_matrix()
    return matrix @ origin


def _axis_marker(matrix: np.ndarray) -> dict[str, np.ndarray]:
    position = matrix[:3, 3]
    x_vector = matrix[:3, 0]
    y_vector = matrix[:3, 1]
    z_vector = matrix[:3, 2]
    return {
        "position": position,
        "x_vector": x_vector,
        "y_vector": y_vector,
        "z_vector": z_vector,
        "x_endpoint": position + AXIS_LENGTH * x_vector,
        "y_endpoint": position + AXIS_LENGTH * y_vector,
        "z_endpoint": position + AXIS_LENGTH * z_vector,
    }


def _frame_camera_reference_markers(
    case: UrdfAlignmentCase,
    origin_matrices: Mapping[str, np.ndarray],
    aligned_matrices: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    compat_by_origin = {
        mount.origin_name: mount.aligned_name
        for mount in case.camera_mount_links
    }
    markers = []
    for origin_link in case.alignment.camera_references:
        aligned_link = compat_by_origin.get(origin_link, origin_link)
        markers.append(
            {
                "origin_link": origin_link,
                "aligned_link": aligned_link,
                "uses_compat_link": aligned_link != origin_link,
                "origin": {
                    "position": origin_matrices[origin_link][:3, 3],
                },
                "aligned": {
                    "position": aligned_matrices[aligned_link][:3, 3],
                },
            }
        )
    return markers


def _json_gripper_end_axes(
    markers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "arm_index": marker["arm_index"],
            "origin_link": marker["origin_link"],
            "origin_reference_link": marker["origin_reference_link"],
            "aligned_link": marker["aligned_link"],
            "origin": _json_axis_marker(marker["origin"]),
            "aligned": _json_axis_marker(marker["aligned"]),
            "axis_dot_products": marker["axis_dot_products"],
        }
        for marker in markers
    ]


def _json_axis_marker(
    marker: Mapping[str, np.ndarray],
) -> dict[str, list[float]]:
    return {
        "position": _round_vector(marker["position"]),
        "x_endpoint": _round_vector(marker["x_endpoint"]),
        "y_endpoint": _round_vector(marker["y_endpoint"]),
        "z_endpoint": _round_vector(marker["z_endpoint"]),
    }


def _json_camera_references(
    markers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "origin_link": marker["origin_link"],
            "aligned_link": marker["aligned_link"],
            "uses_compat_link": marker["uses_compat_link"],
        }
        for marker in markers
    ]


def _fk_joint_names(case: UrdfAlignmentCase) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *case.joints,
                *_chain_joint_names(case.origin_urdf),
                *_chain_joint_names(case.aligned_urdf),
            )
        )
    )


def _chain_joint_names(urdf: Path) -> tuple[str, ...]:
    import pytorch_kinematics as pk

    chain = pk.build_chain_from_urdf(urdf.read_bytes())
    return tuple(chain.get_joint_parameter_names())


def _frame_joint_samples(
    joint_names: tuple[str, ...],
) -> tuple[dict[str, float], ...]:
    sweep = {
        joint: value
        for joint, value in zip(
            joint_names,
            (
                0.08 * (idx + 1) * (1 if idx % 2 == 0 else -1)
                for idx, _joint in enumerate(joint_names)
            ),
            strict=True,
        )
    }
    return (
        {joint: 0.0 for joint in joint_names},
        sweep,
        {joint: -value for joint, value in sweep.items()},
    )


def _joint_edges(urdf: Path) -> tuple[tuple[str, str], ...]:
    root = ET.parse(urdf).getroot()
    edges: list[tuple[str, str]] = []
    for joint in root.iter("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.attrib.get("link")
        child_link = child.attrib.get("link")
        if parent_link and child_link:
            edges.append((parent_link, child_link))
    return tuple(edges)


def _max_position_delta(
    origin_positions: Mapping[str, np.ndarray],
    aligned_positions: Mapping[str, np.ndarray],
    motion_links: tuple[Any, ...],
) -> float:
    deltas = []
    for link in motion_links:
        origin = origin_positions[link.origin_name]
        aligned = aligned_positions[link.aligned_name]
        deltas.append(float(np.linalg.norm(origin - aligned)))
    return max(deltas, default=0.0)


def _frame_bounds(
    origin_positions: Mapping[str, np.ndarray],
    aligned_positions: Mapping[str, np.ndarray],
    gripper_end_axes: list[dict[str, Any]] | None = None,
    camera_references: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = list(origin_positions.values()) + list(aligned_positions.values())
    for marker in gripper_end_axes or []:
        for side in ("origin", "aligned"):
            frame = marker[side]
            points.extend(
                [
                    frame["position"],
                    frame["x_endpoint"],
                    frame["y_endpoint"],
                    frame["z_endpoint"],
                ]
            )
    for marker in camera_references or []:
        points.append(marker["origin"]["position"])
        points.append(marker["aligned"]["position"])
    if not points:
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])
    stacked = np.stack(points)
    minimum = stacked.min(axis=0)
    maximum = stacked.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = float(max((maximum - minimum).max(), 0.1))
    half = span / 2.0
    return center - half, center + half


def _set_equal_bounds(ax: Any, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    minimum, maximum = bounds
    ax.set_xlim(float(minimum[0]), float(maximum[0]))
    ax.set_ylim(float(minimum[1]), float(maximum[1]))
    ax.set_zlim(float(minimum[2]), float(maximum[2]))


def _filter_cases(
    cases: list[UrdfAlignmentCase],
    filters: tuple[str, ...],
) -> list[UrdfAlignmentCase]:
    if not filters:
        return cases
    wanted = set(filters)
    return [
        case
        for case in cases
        if wanted
        & {
            case.name,
            case.config_key,
            case.aligned_config_key,
            f"{case.adapter}/{case.name}",
        }
    ]


def _dataset_rows_for_case(case: UrdfAlignmentCase) -> list[str]:
    try:
        from projects.holobrain_internal.common.configs import dataset_specs
    except Exception:
        return [case.aligned_config_key]

    rows = []
    for entry in dataset_specs.TRAINING_DATASETS:
        if entry.get("dataset_type") != case.adapter:
            continue
        if entry.get("setting_type") == case.aligned_config_key:
            rows.append(entry["dataset_name"])
        elif entry.get("profile_key") == case.aligned_config_key:
            rows.append(entry["dataset_name"])
    return rows or [case.aligned_config_key]


def _write_checklist(
    path: Path,
    cases: list[dict[str, Any]],
    created_files: list[Path],
    final_review_status: str,
) -> None:
    lines = [
        "# URDF Alignment Visual Verification Checklist",
        "",
        "## Agent precheck",
        "",
        "- [x] Generated base vs aligned contact sheets.",
        "- [x] Drew world-view gripper-end axes.",
        "- [x] Highlighted manifest-declared camera reference links.",
        "- [x] Wrote per-case JSON reports.",
        (
            "- [x] Covered one deterministic synthetic FK episode with "
            f"{FRAMES_PER_DATASET_ROW} frames per dataset row."
        ),
        "",
        "## Final review",
        "",
        f"- [ ] Status: {final_review_status}",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        rows = ", ".join(case["dataset_rows"])
        lines.append(
            f"- [ ] `{case['config_key']}` -> "
            f"`{case['aligned_config_key']}` ({rows})"
        )
    lines.extend(["", "## Artifacts", ""])
    for created in created_files:
        lines.append(f"- `{created}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _case_slug(case: UrdfAlignmentCase) -> str:
    return _safe_slug(
        f"{case.adapter}__{case.config_key}__{case.aligned_config_key}"
    )


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _round_float(value: float) -> float:
    return round(float(value), 8)


def _round_vector(values: np.ndarray) -> list[float]:
    return [_round_float(value) for value in values]


def _resolve(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


__all__ = [
    "AXIS_LENGTH",
    "DEFAULT_VISUAL_VERIFY_DIR",
    "FRAMES_PER_DATASET_ROW",
    "VisualVerifyRequest",
    "run_visual_verify",
]
