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

"""Minimal linear-algebra helpers for URDF alignment.

Kept dependency-light on purpose — imports numpy only. All rotation math is
expressed as 3x3 matrices; URDF `<origin>` elements are round-tripped through
`(rpy, xyz)` triples so the transformer never depends on a full URDF parser.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class OriginTransform:
    """A URDF-style origin element: `xyz` translation + `rpy` fixed-axis rot.

    Stored as a translation vector plus a 3x3 rotation matrix so composition
    stays numerically clean. Use :meth:`to_rpy_xyz` to serialize back to
    strings suitable for URDF attributes.
    """

    xyz: np.ndarray  # shape (3,)
    rot: np.ndarray  # shape (3, 3)

    @classmethod
    def identity(cls) -> "OriginTransform":
        return cls(xyz=np.zeros(3), rot=np.eye(3))

    @classmethod
    def from_rpy_xyz(
        cls,
        rpy: Iterable[float] | None,
        xyz: Iterable[float] | None,
    ) -> "OriginTransform":
        rpy_arr = np.zeros(3) if rpy is None else np.asarray(rpy, dtype=float)
        xyz_arr = np.zeros(3) if xyz is None else np.asarray(xyz, dtype=float)
        return cls(xyz=xyz_arr, rot=_rpy_to_matrix(rpy_arr))

    def to_rpy_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        return _matrix_to_rpy(self.rot), self.xyz.copy()

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.rot
        mat[:3, 3] = self.xyz
        return mat

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> "OriginTransform":
        return cls(xyz=mat[:3, 3].copy(), rot=mat[:3, :3].copy())


# ---------------------------------------------------------------------------
# Vector / rotation utilities.
# ---------------------------------------------------------------------------


def unit(vec: Iterable[float]) -> np.ndarray:
    """Return the unit-length version of a 3-vector; zero-safe."""

    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return v.copy()
    return v / n


def is_unit(vec: Iterable[float], atol: float = 1e-6) -> bool:
    v = np.asarray(vec, dtype=float)
    return abs(float(np.linalg.norm(v)) - 1.0) <= atol


def rotation_from_ez_to(target: Iterable[float]) -> np.ndarray:
    """Compute the minimum-rotation matrix mapping `ẑ` onto `target`.

    Uses Rodrigues' formula around the axis `ẑ × target`. Degenerate cases:
        * `target ≈ ẑ` → identity.
        * `target ≈ -ẑ` → half-turn around any axis in the xy-plane (uses x̂).
    """

    z = np.array([0.0, 0.0, 1.0])
    t = unit(target)
    dot = float(np.clip(np.dot(z, t), -1.0, 1.0))
    if dot > 1.0 - _EPS:
        return np.eye(3)
    if dot < -1.0 + _EPS:
        # Half-turn around x-axis (arbitrary axis orthogonal to ẑ).
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, t)
    axis = unit(axis)
    theta = float(np.arccos(dot))
    return rotation_from_axis_angle(axis, theta)


def rotation_from_axis_angle(
    axis: Iterable[float], theta: float
) -> np.ndarray:
    """Rodrigues' rotation formula for `axis` (unit) and `theta` radians."""

    a = unit(axis)
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    C = 1.0 - c  # noqa: N806
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rotation_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


# ---------------------------------------------------------------------------
# RPY <-> matrix.
# ---------------------------------------------------------------------------


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF-style extrinsic XYZ euler → rotation matrix (Rz * Ry * Rx)."""

    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _matrix_to_rpy(mat: np.ndarray) -> np.ndarray:
    """Inverse of `_rpy_to_matrix` for the URDF convention (Rz * Ry * Rx)."""

    sy = float(np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        r = float(np.arctan2(mat[2, 1], mat[2, 2]))
        p = float(np.arctan2(-mat[2, 0], sy))
        y = float(np.arctan2(mat[1, 0], mat[0, 0]))
    else:
        r = float(np.arctan2(-mat[1, 2], mat[1, 1]))
        p = float(np.arctan2(-mat[2, 0], sy))
        y = 0.0
    return np.array([r, p, y])


# ---------------------------------------------------------------------------
# Inertia rotation (3x3 symmetric matrix under change of frame).
# ---------------------------------------------------------------------------


def rotate_inertia(inertia_3x3: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Rotate a 3x3 inertia tensor into a new frame: `R I R^T`."""

    return rot @ inertia_3x3 @ rot.T


def inertia_from_components(
    ixx: float,
    ixy: float,
    ixz: float,
    iyy: float,
    iyz: float,
    izz: float,
) -> np.ndarray:
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
    )


def inertia_to_components(inertia_3x3: np.ndarray) -> dict[str, float]:
    return dict(
        ixx=float(inertia_3x3[0, 0]),
        ixy=float((inertia_3x3[0, 1] + inertia_3x3[1, 0]) * 0.5),
        ixz=float((inertia_3x3[0, 2] + inertia_3x3[2, 0]) * 0.5),
        iyy=float(inertia_3x3[1, 1]),
        iyz=float((inertia_3x3[1, 2] + inertia_3x3[2, 1]) * 0.5),
        izz=float(inertia_3x3[2, 2]),
    )


# ---------------------------------------------------------------------------
# Formatting helpers (URDF attribute serialization).
# ---------------------------------------------------------------------------


def format_vec(vec: Iterable[float]) -> str:
    """Serialize a float vector as a URDF-style space-separated string.

    Uses Python's shortest-round-trip float `repr` so values like
    `math.pi` serialize as `3.141592653589793` instead of a truncated
    `3.14159`. Preserving full precision keeps the transformer output
    diff-clean against the hand-tuned reference URDFs, which were emitted
    from the same round-trip formatter. Values within a small epsilon of
    zero are snapped to `0.0` so tiny numerical residue from S/S⁻¹
    composition does not print as, e.g., `1.2246467991473532e-16`.
    """

    parts = []
    for value in vec:
        f = float(value)
        if abs(f) < _ZERO_SNAP_EPS:
            f = 0.0
        parts.append(repr(f))
    return " ".join(parts)


# Snap threshold for `format_vec`. Chosen small enough that any real
# translation or rotation component in a URDF stays intact, and large
# enough to absorb double-precision round-off from R @ R⁻¹ products
# (about 1e-16 in magnitude).
_ZERO_SNAP_EPS = 1e-12
