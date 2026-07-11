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

"""Copy an origin URDF (and its mesh assets) into the aligned tree.

Mesh references land in a URDF in a few shapes and the resolver here tries
each in a fixed, explicit order — no dataset-name heuristics. Whatever the
upstream layout is, the manifest opts a case into extra search roots when
the URDF-relative default doesn't cover them:

1. ``filename="rel/path/mesh.stl"`` — resolved against the URDF's parent
   directory first (the traditional URDF convention), then against every
   ``extra_search_roots`` entry supplied by the case.
2. ``filename="package://<pkgname>/<rel>"`` — a ROS-style package URI that
   many upstream URDFs use even when there is no live ROS workspace on
   disk. Resolved against ``<base>/<pkgname>/<rel>`` where ``<base>`` is
   the URDF's parent first, then each ``extra_search_roots`` entry. When
   the URI resolves to a local file, the aligned URDF is rewritten so the
   reference becomes ``<pkgname>/<rel>`` (relative to the aligned URDF's
   parent, matching the copied on-disk layout) — this keeps the aligned
   tree self-contained without depending on a live ROS workspace.
3. Anything the above didn't resolve is preserved verbatim in the
   :attr:`CopiedUrdfAsset.skipped_meshes` list so callers can surface an
   actionable checklist.

Unresolved ``package://`` URIs are kept as-is in the aligned URDF so a
ROS environment can still resolve them at runtime.

The resolver does not consult the packer adapter — mesh layout is a
property of the upstream URDF, not the packer schema, so sharing one
resolver across every case keeps behavior uniform.
"""

from __future__ import annotations
import os
import re
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

# XML attribute matcher for `<mesh filename="..."/>`. We rewrite package
# URIs textually (rather than parsing/re-serializing) so the aligned URDF's
# formatting stays byte-stable everywhere the transformer didn't touch it.
_MESH_FILENAME_ATTR_RE = re.compile(
    r'(<mesh\b[^>]*?\bfilename\s*=\s*")([^"]+)(")',
)


@dataclass(frozen=True)
class CopiedUrdfAsset:
    """Result of a URDF+mesh copy pass.

    Attributes:
        urdf: The copied URDF path in the aligned tree.
        copied_files: URDF plus every mesh/sidecar file copied under the
            aligned tree.
        local_mesh_files: ``copied_files`` minus the URDF itself.
        skipped_meshes: Verbatim ``filename`` strings whose reference no
            resolver candidate could find. Useful for the WARN report.
        package_uri_rewrites: Mapping from ``package://<pkg>/<rel>``
            references that DID resolve to their new aligned-tree-relative
            path (typically ``<pkg>/<rel>``). Callers apply this to the
            aligned URDF bytes via :func:`rewrite_resolved_package_uris`.
    """

    urdf: Path
    copied_files: tuple[Path, ...]
    local_mesh_files: tuple[Path, ...]
    skipped_meshes: tuple[str, ...]
    package_uri_rewrites: dict[str, str] = field(default_factory=dict)


def copy_urdf_asset_tree(
    source_urdf: Path,
    target_urdf: Path,
    extra_search_roots: tuple[Path, ...] = (),
) -> CopiedUrdfAsset:
    """Copy a URDF and its resolvable mesh files to the aligned target tree.

    The URDF itself is copied byte-for-byte so alignment edits happen in the
    target tree (preserving an inspectable origin→aligned history). Meshes
    are resolved through :func:`_resolve_mesh_reference`; every reference
    that resolves to an existing file is copied, and every reference that
    does not is returned in :attr:`CopiedUrdfAsset.skipped_meshes` verbatim
    so the caller can dump an actionable report.

    Args:
        source_urdf: Origin URDF path.
        target_urdf: Aligned URDF destination.
        extra_search_roots: Optional extra directories to try (in order,
            after the URDF's parent) when a mesh reference doesn't resolve
            against the default location. Entries are treated as relative
            to ``source_urdf.parent`` unless absolute.
    """

    target_urdf.parent.mkdir(parents=True, exist_ok=True)
    _overwrite_copy(source_urdf, target_urdf)
    # `shutil.copy2` preserves the source mode bits; some upstream URDF assets
    # are shipped read-only (0555), which would then prevent the transformer
    # from overwriting the aligned URDF in-place. Add owner-write so the align
    # pipeline can rewrite the file it just staged, without changing readability.
    _ensure_owner_writable(target_urdf)
    copied_files: list[Path] = [target_urdf]
    skipped_meshes: list[str] = []
    package_uri_rewrites: dict[str, str] = {}
    search_bases = _search_bases(source_urdf, extra_search_roots)

    for mesh_filename in _iter_mesh_filenames(source_urdf):
        resolved = _resolve_mesh_reference(mesh_filename, search_bases)
        if resolved is None:
            skipped_meshes.append(mesh_filename)
            continue
        source_mesh, aligned_relative = resolved
        target_mesh = target_urdf.parent / aligned_relative
        copied_files.extend(_copy_mesh_with_sidecars(source_mesh, target_mesh))
        if mesh_filename.startswith("package://"):
            # Aligned URDF should no longer point at package://; use the local
            # aligned-tree-relative path instead.
            package_uri_rewrites[mesh_filename] = aligned_relative.as_posix()

    return CopiedUrdfAsset(
        urdf=target_urdf,
        copied_files=tuple(copied_files),
        local_mesh_files=tuple(copied_files[1:]),
        skipped_meshes=tuple(skipped_meshes),
        package_uri_rewrites=package_uri_rewrites,
    )


def rewrite_resolved_package_uris(
    urdf_bytes: bytes,
    package_uri_rewrites: dict[str, str],
) -> bytes:
    """Textually rewrite `<mesh filename="package://...">` refs in URDF bytes.

    Only refs listed in ``package_uri_rewrites`` (the URIs that actually
    resolved to a local copy) are rewritten; every other filename stays as-is.
    Kept as a textual pass (rather than XML parse + re-serialize) so aligned
    URDFs stay byte-stable in the places the transformer did not touch,
    which keeps diffs reviewable.
    """

    if not package_uri_rewrites:
        return urdf_bytes
    text = urdf_bytes.decode("utf-8")

    def _sub(match: re.Match[str]) -> str:
        prefix, filename, suffix = match.group(1), match.group(2), match.group(3)
        replacement = package_uri_rewrites.get(filename)
        if replacement is None:
            return match.group(0)
        return f"{prefix}{replacement}{suffix}"

    return _MESH_FILENAME_ATTR_RE.sub(_sub, text).encode("utf-8")


# ---------------------------------------------------------------------------
# Mesh reference resolution.
# ---------------------------------------------------------------------------


def _search_bases(
    source_urdf: Path,
    extra_search_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Resolve ``extra_search_roots`` against the URDF's parent directory.

    Absolute entries are kept as-is — cases whose canonical upstream asset
    tree lives outside the repo (for example, on a shared bucket mount) can
    declare that absolute path directly and the resolver will read meshes
    from there without needing an in-repo mirror. Relative entries are
    resolved against the origin URDF's parent so cases can declare paths
    like ``"piper"`` without knowing the URDF's absolute location.
    """

    urdf_parent = source_urdf.parent
    bases: list[Path] = [urdf_parent]
    for root in extra_search_roots:
        base = root if root.is_absolute() else (urdf_parent / root)
        if base not in bases:
            bases.append(base)
    return tuple(bases)


def _resolve_mesh_reference(
    filename: str,
    search_bases: tuple[Path, ...],
) -> tuple[Path, Path] | None:
    """Return ``(source_mesh, aligned_relative)`` or ``None`` if unresolved.

    ``aligned_relative`` is the path relative to the aligned URDF's parent
    at which the mesh should be copied. For URDF-relative references the
    aligned copy keeps the URDF's textual ``filename`` unchanged, so it is
    just the original relative path. For ``package://<pkg>/<rel>`` URIs
    the aligned copy uses ``<pkg>/<rel>`` — the caller separately rewrites
    the aligned URDF's ``filename`` attribute to match.
    """

    if filename.startswith("package://"):
        return _resolve_package_uri(filename, search_bases)
    mesh_path = Path(filename)
    if mesh_path.is_absolute():
        # Absolute mesh paths are environment-specific and should not be
        # copied into the repository without an explicit migration choice.
        return None
    for base in search_bases:
        candidate = base / mesh_path
        if candidate.exists():
            return candidate, mesh_path
    return None


def _resolve_package_uri(
    filename: str,
    search_bases: tuple[Path, ...],
) -> tuple[Path, Path] | None:
    """Resolve ``package://<pkgname>/<rel>`` against every search base.

    Many upstream URDFs use ROS package URIs even when there is no live ROS
    workspace on disk. When the package directory sits alongside the URDF
    (e.g. ``<urdf_parent>/aloha_maniskill_sim/meshes/...``) the default
    search base resolves it; otherwise the case declares an extra root
    pointing at the package's actual parent.

    Unresolved URIs are left as ``package://`` in the aligned URDF so a ROS
    environment can still resolve them.
    """

    without_scheme = filename[len("package://") :]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2:
        return None
    package_name, package_rel = parts[0], parts[1]
    if not package_name or not package_rel:
        return None
    package_rel_path = Path(package_rel)
    aligned_relative = Path(package_name) / package_rel_path
    for base in search_bases:
        candidate = base / package_name / package_rel_path
        if candidate.exists():
            return candidate, aligned_relative
    return None


# ---------------------------------------------------------------------------
# Copy helpers.
# ---------------------------------------------------------------------------


def _copy_mesh_with_sidecars(
    source_mesh: Path,
    target_mesh: Path,
) -> list[Path]:
    copied_files: list[Path] = []
    for source_path in [source_mesh, *_iter_sidecars(source_mesh)]:
        # Sidecars such as .mtl files usually sit beside the mesh and are
        # referenced by relative name inside the mesh file, so preserve that
        # local layout beside the copied mesh.
        relative_path = source_path.relative_to(source_mesh.parent)
        target_path = target_mesh.parent / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _overwrite_copy(source_path, target_path)
        copied_files.append(target_path)
    return copied_files


def _iter_sidecars(source_mesh: Path) -> list[Path]:
    sidecars: list[Path] = []
    for suffix in (".mtl", ".png", ".jpg", ".jpeg"):
        candidate = source_mesh.with_suffix(suffix)
        if candidate.exists():
            sidecars.append(candidate)
    return sidecars


def _iter_mesh_filenames(urdf_path: Path) -> list[str]:
    root = ElementTree.parse(urdf_path).getroot()
    filenames: list[str] = []
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename and filename not in filenames:
            filenames.append(filename)
    return filenames


def _ensure_owner_writable(path: Path) -> None:
    """Ensure ``path`` has S_IWUSR set, tolerating missing chmod support."""

    try:
        current = path.stat().st_mode
        os.chmod(path, current | stat.S_IWUSR)
    except (OSError, NotImplementedError):
        # Filesystem may not support chmod (some network mounts); leave the
        # mode as-is and let a subsequent write surface a clearer error.
        return


def _overwrite_copy(source: Path, target: Path) -> None:
    """`shutil.copy2` that overwrites read-only destinations in place.

    Prior align runs can leave the aligned tree populated with read-only
    mesh copies (upstream mesh mode 0444 is preserved by ``copy2``). Re-running
    the aligner then hits ``PermissionError`` when opening the destination
    for write. Unlink first (best-effort chmod the parent if needed) so
    each run is a clean overwrite.
    """

    if target.exists() or target.is_symlink():
        try:
            target.unlink()
        except PermissionError:
            # Parent directory itself may be read-only; make it writable and
            # retry once before giving up. Very rare in practice.
            _ensure_owner_writable(target.parent)
            target.unlink()
    shutil.copy2(source, target)
