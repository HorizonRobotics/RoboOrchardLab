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
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytorch_kinematics as pk
import torch
from pytorch3d.transforms import matrix_to_quaternion


@dataclass(frozen=True)
class FkBaselineSpec:
    name: str
    urdf: Path
    link_keys: list[str]
    joint_positions: list[list[float]]
    output_path: Path | None = None
    precision: int = 6


@dataclass(frozen=True)
class FkEquivalenceSpec:
    name: str
    origin_urdf: Path
    aligned_urdf: Path
    link_keys: list[str]
    joint_positions: list[list[float]]
    precision: int = 6


@dataclass(frozen=True)
class LinkNamePair:
    """Semantic link mapping between origin and aligned URDFs.

    Tests compare link pairs through `semantic_name` so failure messages stay
    meaningful even when the raw URDF link names differ substantially.
    """

    semantic_name: str
    origin_name: str
    aligned_name: str


@dataclass(frozen=True)
class NamedFkConsistencySpec:
    """Named-joint FK consistency check for one aligned URDF.

    This spec validates control/motion semantics: for the same semantic joint
    values, selected links must stay at the same global positions. Orientation
    is deliberately excluded because axis/EE-frame alignment can rotate link
    frames while preserving physical motion.

    The alignment pipeline never renames actuated joints (only inserts fixed
    ``*_ee`` / ``*_gripper_end`` / ``*_camera_mount_compat`` joints), so origin
    and aligned URDFs share the same actuated joint name list. ``joint_names``
    is therefore a plain ordered list of names shared by both chains; each
    sample maps names → position and is applied verbatim to both.
    """

    name: str
    origin_urdf: Path
    aligned_urdf: Path
    joint_names: Sequence[str]
    links: Sequence[LinkNamePair]
    joint_samples: Sequence[Mapping[str, float]]
    atol: float = 1e-5


@dataclass(frozen=True)
class CameraMountConsistencySpec:
    """Compare origin camera/EE frame with aligned compatibility frame.

    Camera extrinsics calibrated against the origin URDF should continue to
    work by targeting a `*_camera_mount_compat` link in the aligned URDF.
    Unlike motion consistency, this check compares the full 4x4 pose.

    See :class:`NamedFkConsistencySpec` for why ``joint_names`` is a plain
    list.
    """

    name: str
    origin_urdf: Path
    aligned_urdf: Path
    joint_names: Sequence[str]
    mounts: Sequence[LinkNamePair]
    joint_samples: Sequence[Mapping[str, float]]
    atol: float = 1e-5


def compute_fk_baseline(spec: FkBaselineSpec) -> dict:
    """Compute deterministic FK poses for selected links and joint samples."""

    # This function is intentionally generic: it does not know whether the URDF
    # is origin or aligned. The caller chooses link keys and dense joint
    # vectors that are meaningful for the specific manual FK snapshot being
    # generated.
    chain = pk.build_chain_from_urdf(spec.urdf.read_bytes())
    joint_positions = torch.tensor(spec.joint_positions, dtype=torch.float32)
    link_poses = chain.forward_kinematics(joint_positions)

    samples = []
    for sample_idx, joint_position in enumerate(spec.joint_positions):
        sample_links = {}
        for link_key in spec.link_keys:
            matrix = link_poses[link_key].get_matrix()[sample_idx]
            quat = matrix_to_quaternion(matrix[:3, :3])
            # Store position plus quaternion rather than a raw matrix to keep
            # generated logs compact and easy to diff during manual inspection.
            sample_links[link_key] = {
                "position": _round_list(matrix[:3, 3], spec.precision),
                "quaternion_wxyz": _round_list(quat, spec.precision),
            }
        samples.append(
            {
                "joint_position": [
                    round(float(value), spec.precision)
                    for value in joint_position
                ],
                "links": sample_links,
            }
        )

    baseline = {
        "name": spec.name,
        "urdf": str(spec.urdf),
        "link_keys": list(spec.link_keys),
        "samples": samples,
    }
    if spec.output_path is not None:
        # JSON baselines are debugging artifacts only. Regression tests below
        # recompute FK directly from the URDFs so ignored logs cannot become a
        # stale source of truth.
        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        spec.output_path.write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return baseline


def assert_fk_equivalent(spec: FkEquivalenceSpec) -> None:
    """Assert origin and aligned URDFs produce identical FK samples."""

    # Strict sample equality is useful for unchanged link frames. Once an
    # alignment intentionally rotates frames, prefer the named consistency
    # checks below, which compare the exact contract that must be preserved.
    origin = compute_fk_baseline(
        FkBaselineSpec(
            name=f"{spec.name}_origin",
            urdf=spec.origin_urdf,
            link_keys=spec.link_keys,
            joint_positions=spec.joint_positions,
            precision=spec.precision,
        )
    )
    aligned = compute_fk_baseline(
        FkBaselineSpec(
            name=f"{spec.name}_aligned",
            urdf=spec.aligned_urdf,
            link_keys=spec.link_keys,
            joint_positions=spec.joint_positions,
            precision=spec.precision,
        )
    )
    assert origin["samples"] == aligned["samples"]


def assert_named_motion_consistent(spec: NamedFkConsistencySpec) -> None:
    """Assert mapped links keep global positions for named joint samples.

    This intentionally compares positions only. URDF alignment may change
    link frame orientation while preserving the physical position observed
    from the base/global frame for the same control values.
    """

    origin_fk, aligned_fk = _paired_fk_for_named_samples(
        origin_urdf=spec.origin_urdf,
        aligned_urdf=spec.aligned_urdf,
        joint_names=spec.joint_names,
        joint_samples=spec.joint_samples,
    )

    for sample_idx, _sample in enumerate(spec.joint_samples):
        for link_pair in spec.links:
            # Motion preservation is measured in the global/base frame. The
            # aligned URDF may rename links and rotate their local axes, but
            # the same semantic action should put each physical link at the
            # same xyz.
            origin_xyz = origin_fk[link_pair.origin_name].get_matrix()[
                sample_idx, :3, 3
            ]
            aligned_xyz = aligned_fk[link_pair.aligned_name].get_matrix()[
                sample_idx, :3, 3
            ]
            _assert_close(
                origin_xyz,
                aligned_xyz,
                spec.atol,
                (
                    f"{spec.name}:{link_pair.semantic_name} sample "
                    f"{sample_idx} position mismatch"
                ),
            )


def assert_camera_mount_consistent(
    spec: CameraMountConsistencySpec,
) -> None:
    """Assert aligned compatibility links match origin camera/EE frames."""

    origin_fk, aligned_fk = _paired_fk_for_named_samples(
        origin_urdf=spec.origin_urdf,
        aligned_urdf=spec.aligned_urdf,
        joint_names=spec.joint_names,
        joint_samples=spec.joint_samples,
    )

    for sample_idx, _sample in enumerate(spec.joint_samples):
        for mount in spec.mounts:
            # Camera extrinsics depend on orientation as well as position, so
            # compatibility links must reproduce the full origin EE/camera
            # pose.
            origin_matrix = origin_fk[mount.origin_name].get_matrix()[
                sample_idx
            ]
            aligned_matrix = aligned_fk[mount.aligned_name].get_matrix()[
                sample_idx
            ]
            _assert_close(
                origin_matrix,
                aligned_matrix,
                spec.atol,
                (
                    f"{spec.name}:{mount.semantic_name} sample "
                    f"{sample_idx} camera mount mismatch"
                ),
            )


def _round_list(values: torch.Tensor, precision: int) -> list[float]:
    return [round(float(value), precision) for value in values]


def _paired_fk_for_named_samples(
    origin_urdf: Path,
    aligned_urdf: Path,
    joint_names: Sequence[str],
    joint_samples: Sequence[Mapping[str, float]],
):
    """Build both FK chains and evaluate the same sample against each.

    Origin and aligned URDFs share the same actuated joint name list (the
    pipeline only inserts new fixed joints, never renames actuated ones), so
    a single dense joint tensor is applied verbatim to both chains.
    """

    origin_chain = pk.build_chain_from_urdf(origin_urdf.read_bytes())
    aligned_chain = pk.build_chain_from_urdf(aligned_urdf.read_bytes())
    origin_positions = _named_joint_tensor(
        origin_chain, joint_names, joint_samples
    )
    aligned_positions = _named_joint_tensor(
        aligned_chain, joint_names, joint_samples
    )
    return (
        origin_chain.forward_kinematics(origin_positions),
        aligned_chain.forward_kinematics(aligned_positions),
    )


def _named_joint_tensor(
    chain,
    joint_names: Sequence[str],
    joint_samples: Sequence[Mapping[str, float]],
) -> torch.Tensor:
    """Convert semantic joint samples into a dense tensor for one FK chain.

    Each URDF joint listed in ``joint_names`` must exist in the chain's
    actuated joint list. Missing entries in a sample default to zero — this
    keeps sample dictionaries focused on the joints that are intentionally
    moved.
    """

    chain_joint_names = chain.get_joint_parameter_names()
    joint_index = {name: idx for idx, name in enumerate(chain_joint_names)}
    positions = torch.zeros(
        (len(joint_samples), len(chain_joint_names)),
        dtype=torch.float32,
    )
    for sample_idx, sample in enumerate(joint_samples):
        for joint_name in joint_names:
            if joint_name not in joint_index:
                raise KeyError(f"Joint {joint_name!r} is missing from URDF")
            positions[sample_idx, joint_index[joint_name]] = float(
                sample.get(joint_name, 0.0)
            )
    return positions


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    message: str,
) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        diff = (actual - expected).abs().max()
        raise AssertionError(
            f"{message}; max_abs_diff={float(diff):.8g}; "
            f"actual={actual.tolist()}; expected={expected.tolist()}"
        )
