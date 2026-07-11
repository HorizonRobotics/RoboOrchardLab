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

"""URDF alignment CLI (yaml-first flow).

The ``align`` subcommand consumes a manifest identifier — either a
``<dataset>/<embodiment>`` slug (e.g. ``robotwin/aloha``) resolved against
the default ``urdf_align/`` root, or an absolute path to an
``alignment.yaml`` file — and drives the alignment pipeline end-to-end for
every case declared by that manifest. Shared embodiments (one yaml listing
multiple ``config.key`` entries) copy + transform + write the aligned URDF
once, then verify each config key's wiring against the shared URDF.

The other subcommands (``origin``, ``fk-baseline``, ``aligned-from-origin``,
``copy-asset``) are unchanged and continue to work off the same packer
config surface.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from projects.holobrain_internal.common.urdf_tools.aligned import (
    UrdfAlignManifest,
)
from projects.holobrain_internal.common.urdf_tools.cases import (
    DEFAULT_MANIFEST_ROOT,
    DEFAULT_URDF_ALIGN_ROOT,
    UrdfAlignmentCase,
    load_alignment_manifest,
)
from projects.holobrain_internal.common.urdf_tools.config_wiring import (
    verify_case_config_wiring,
)
from projects.holobrain_internal.common.urdf_tools.fk_baseline import (
    FkBaselineSpec,
    compute_fk_baseline,
)
from projects.holobrain_internal.common.urdf_tools.origin import (
    collect_urdf_inventory,
)
from projects.holobrain_internal.common.urdf_tools.sync_assets import (
    CopiedUrdfAsset,
    copy_urdf_asset_tree,
    rewrite_resolved_package_uris,
)
from projects.holobrain_internal.common.urdf_tools.transform import (
    apply_alignment,
)


def main() -> None:
    # Keep the CLI vocabulary aligned with the workflow docs:
    # `origin` scans configs, `aligned-from-origin` proposes target paths,
    # and `align` runs the transformer end-to-end from a per-embodiment yaml.
    parser = argparse.ArgumentParser(
        description="URDF origin scan and alignment validation utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    origin_parser = subparsers.add_parser("origin")
    origin_parser.add_argument("--repo-root", type=Path, default=Path("."))
    origin_parser.add_argument(
        "--config-root",
        type=Path,
        action="append",
        default=[],
        help="Config root to scan. Can be passed multiple times.",
    )
    origin_parser.add_argument(
        "--urdf-root",
        type=Path,
        action="append",
        default=[],
        help="URDF root to scan. Can be passed multiple times.",
    )
    origin_parser.add_argument("--output", type=Path, required=True)

    baseline_parser = subparsers.add_parser("fk-baseline")
    baseline_parser.add_argument("--name", required=True)
    baseline_parser.add_argument("--urdf", type=Path, required=True)
    baseline_parser.add_argument("--link-key", action="append", required=True)
    baseline_parser.add_argument(
        "--joint-position",
        action="append",
        required=True,
        help="Comma-separated joint position sample, e.g. 0,0,0,0,0,0.",
    )
    baseline_parser.add_argument("--output", type=Path, required=True)

    manifest_parser = subparsers.add_parser("aligned-from-origin")
    manifest_parser.add_argument("--origin", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument(
        "--root",
        type=Path,
        default=Path("projects/holobrain_internal/common/urdf_align"),
    )

    copy_parser = subparsers.add_parser("copy-asset")
    copy_parser.add_argument("--source-urdf", type=Path, required=True)
    copy_parser.add_argument("--target-urdf", type=Path, required=True)
    copy_parser.add_argument("--report", type=Path, default=None)

    align_parser = subparsers.add_parser("align")
    align_parser.add_argument(
        "manifest",
        help=(
            "Either a '<dataset>/<embodiment>' slug resolved under the "
            "default urdf_align/ root (e.g. 'robotwin/aloha') or an absolute "
            "path to an alignment.yaml file."
        ),
    )
    align_parser.add_argument("--repo-root", type=Path, default=Path("."))
    align_parser.add_argument(
        "--align-root",
        type=Path,
        default=DEFAULT_URDF_ALIGN_ROOT,
        help="Root of the urdf_align/ tree used to resolve manifest slugs.",
    )
    align_parser.add_argument(
        "--manifest-root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="Root of the manifests/ tree used to resolve manifest slugs.",
    )
    align_parser.add_argument("--dry-run", action="store_true")
    align_parser.add_argument(
        "--skip-config-wiring",
        action="store_true",
        help="Do not verify the aligned config key wiring for this case.",
    )
    align_parser.add_argument(
        "--copy-report",
        type=Path,
        default=None,
        help=(
            "Optional path to write the full copy report (copied files + "
            "skipped meshes) as JSON. Handy when the WARN line reports "
            "missing meshes and you want the exact list to fix."
        ),
    )

    visual_parser = subparsers.add_parser("visual-verify")
    visual_parser.add_argument("--repo-root", type=Path, default=Path("."))
    visual_parser.add_argument(
        "--align-root",
        type=Path,
        default=Path("projects/holobrain_internal/common/urdf_align"),
        help="Root of the urdf_align/ tree to scan.",
    )
    visual_parser.add_argument(
        "--manifest-root",
        type=Path,
        default=None,
        help=(
            "Root of the manifests/ tree scanned for alignment.yaml files. "
            "Defaults to DEFAULT_MANIFEST_ROOT (the in-repo manifests). Set "
            "this when running against a hermetic fixture tree."
        ),
    )
    visual_parser.add_argument(
        "--config-root",
        type=Path,
        default=None,
        help=(
            "Config root used to resolve origin URDFs. Defaults to "
            "projects/holobrain_internal/common/configs under repo root."
        ),
    )
    visual_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for PNG/JSON/checklist output. Defaults to the "
            "ignored workspace_test/urdf_align_logs/visual_verify folder."
        ),
    )
    visual_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Limit output to a case name/config key/aligned config key. "
            "Can be passed multiple times."
        ),
    )
    visual_parser.add_argument(
        "--final-review-status",
        default="pending_manual_review",
        help="Status text recorded in CHECKLIST.md.",
    )

    args = parser.parse_args()
    if args.command == "origin":
        # Scan config files for URDF references and scan URDF folders for mesh
        # references. The JSON output is an inventory of the current/origin
        # state; it does not create or modify aligned assets.
        repo_root = args.repo_root.resolve()
        inventory = collect_urdf_inventory(
            repo_root=repo_root,
            config_roots=[
                _resolve(repo_root, path) for path in args.config_root
            ],
            urdf_roots=[_resolve(repo_root, path) for path in args.urdf_root],
        )
        inventory.write_json(args.output)
    elif args.command == "fk-baseline":
        # Produce a deterministic pose log for selected links and joint
        # samples. Tests compute FK directly, so this JSON is only a readable
        # diagnostic artifact for comparing experiments by hand.
        compute_fk_baseline(
            FkBaselineSpec(
                name=args.name,
                urdf=args.urdf,
                link_keys=args.link_key,
                joint_positions=[
                    [float(value) for value in sample.split(",")]
                    for sample in args.joint_position
                ],
                output_path=args.output,
            )
        )
    elif args.command == "aligned-from-origin":
        # Convert the origin inventory into a path plan under urdf_align.
        # This is intentionally a proposal step: copying/editing assets remains
        # explicit so reviewers can inspect each robot migration.
        manifest = UrdfAlignManifest.from_inventory_json(
            args.origin,
            root=args.root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    elif args.command == "copy-asset":
        # Copy the chosen origin URDF plus resolvable meshes into the
        # aligned tree. Meshes that no resolver candidate could find are
        # reported as skipped; MISSING_MESHES.md next to the URDF holds the
        # human-readable checklist.
        copied = copy_urdf_asset_tree(args.source_urdf, args.target_urdf)
        _write_missing_meshes_md(copied)
        report = {
            "urdf": str(copied.urdf),
            "copied_files": [str(path) for path in copied.copied_files],
            "local_mesh_files": [
                str(path) for path in copied.local_mesh_files
            ],
            "skipped_meshes": list(copied.skipped_meshes),
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "align":
        _run_align(args)
    elif args.command == "visual-verify":
        _run_visual_verify(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def _run_align(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    manifest_path = _resolve_manifest_path(
        args.manifest,
        align_root=args.align_root.resolve(),
        manifest_root=args.manifest_root.resolve(),
    )
    cases = load_alignment_manifest(
        manifest_path,
        repo_root=repo_root,
        manifest_root=args.manifest_root.resolve(),
        align_root=args.align_root.resolve(),
    )
    if not cases:
        raise ValueError(f"manifest '{manifest_path}' yielded no cases")

    # For shared embodiments the yaml lists multiple config keys pointing at
    # the same aligned URDF. We copy assets + run the transform + write the
    # aligned URDF exactly once (using the first case as the driver), then
    # verify config wiring for every case so a stale key surfaces here rather
    # than at dataset-load time.
    driver = cases[0]
    print(
        f"[align] manifest: {manifest_path} "
        f"({len(cases)} config key(s): "
        f"{', '.join(c.config_key for c in cases)})",
        file=sys.stderr,
    )
    print(
        f"[align] source_urdf (from config {driver.config_module}"
        f"[{driver.config_key}].urdf): {driver.origin_urdf}",
        file=sys.stderr,
    )
    print(f"[align] aligned_urdf: {driver.aligned_urdf}", file=sys.stderr)
    if not driver.origin_urdf.exists():
        raise FileNotFoundError(f"origin URDF missing: {driver.origin_urdf}")

    # Stage 1: mirror origin URDF + meshes into the aligned output tree.
    copied: CopiedUrdfAsset | None = None
    if not args.dry_run:
        copied = copy_urdf_asset_tree(
            driver.origin_urdf,
            driver.aligned_urdf,
            extra_search_roots=driver.mesh_search_roots,
        )
        print(
            f"[align] copied assets → {driver.aligned_urdf}", file=sys.stderr
        )
        _report_copy_status(copied, args.copy_report)
    else:
        print(
            f"[align] (dry-run) would copy assets → {driver.aligned_urdf}",
            file=sys.stderr,
        )

    # Stage 2: transform.
    result = apply_alignment(driver.origin_urdf, driver)
    print(
        "[align] normalized_joints: "
        + json.dumps(list(result.report.normalized_joints)),
        file=sys.stderr,
    )
    print(
        "[align] inserted_ee_joints: "
        + json.dumps(list(result.report.inserted_ee_joints)),
        file=sys.stderr,
    )
    print(
        "[align] inserted_gripper_end_joints: "
        + json.dumps(list(result.report.inserted_gripper_end_joints)),
        file=sys.stderr,
    )
    print(
        "[align] inserted_camera_compat_links: "
        + json.dumps(list(result.report.inserted_camera_compat_links)),
        file=sys.stderr,
    )

    # Stage 3: write aligned URDF (unless dry-run).
    if not args.dry_run:
        driver.aligned_urdf.parent.mkdir(parents=True, exist_ok=True)
        aligned_bytes = result.aligned_urdf_bytes
        if copied is not None and copied.package_uri_rewrites:
            # Point resolved package:// refs at their local aligned-tree copies
            # so the aligned URDF is self-contained. Unresolved URIs stay
            # as-is so a live ROS environment can still find them.
            aligned_bytes = rewrite_resolved_package_uris(
                aligned_bytes,
                copied.package_uri_rewrites,
            )
        driver.aligned_urdf.write_bytes(aligned_bytes)
        print(f"[align] wrote {driver.aligned_urdf}", file=sys.stderr)
    else:
        print(
            f"[align] (dry-run) would write {driver.aligned_urdf}",
            file=sys.stderr,
        )

    # Stage 4: verify config wiring for every case (shared embodiments yield
    # multiple).
    if not args.skip_config_wiring:
        for case in cases:
            wiring = verify_case_config_wiring(case, repo_root=repo_root)
            if wiring.notes:
                # Base-cutover fallback and similar non-fatal observations are
                # worth surfacing even on success, so the operator can see
                # which wiring shape validated. Kept on one grouped line so
                # they do not drown out the align output.
                print(
                    f"[align] config wiring notes for {case.config_key}:",
                    file=sys.stderr,
                )
                for note in wiring.notes:
                    print(f"  - {note}", file=sys.stderr)
            if not wiring.ok:
                print(
                    f"[align] config wiring issues for {case.config_key}:",
                    file=sys.stderr,
                )
                print(wiring.format(), file=sys.stderr)
                sys.exit(2)


def _run_visual_verify(args: argparse.Namespace) -> None:
    from projects.holobrain_internal.common.urdf_tools.visual_verify import (
        VisualVerifyRequest,
        run_visual_verify,
    )

    repo_root = args.repo_root.resolve()
    index = run_visual_verify(
        VisualVerifyRequest(
            repo_root=repo_root,
            align_root=args.align_root,
            manifest_root=args.manifest_root,
            config_root=args.config_root,
            output_dir=args.output_dir,
            case_filters=tuple(args.case),
            final_review_status=args.final_review_status,
        )
    )
    print(json.dumps(index, indent=2, sort_keys=True))


def _resolve_manifest_path(
    spec: str, align_root: Path, manifest_root: Path | None = None
) -> Path:
    """Turn a slug or path string into an absolute manifest file path.

    Accepts these shapes:

    - An absolute or existing relative path ending in ``alignment.yaml``.
    - A ``<dataset>/<embodiment>`` slug, resolved first under
      ``<manifest_root>/<slug>/alignment.yaml`` (git-tracked) then under
      ``<align_root>/<slug>/alignment.yaml`` (bucket-side fallback).
    - An absolute or relative path to an embodiment directory containing an
      ``alignment.yaml`` file.
    """

    candidate = Path(spec)
    if candidate.suffix == ".yaml" and candidate.exists():
        return candidate.resolve()
    if candidate.is_dir() and (candidate / "alignment.yaml").exists():
        return (candidate / "alignment.yaml").resolve()
    tried: list[Path] = [candidate]
    if manifest_root is not None:
        manifest_slug_path = manifest_root / spec / "alignment.yaml"
        tried.append(manifest_slug_path)
        if manifest_slug_path.exists():
            return manifest_slug_path.resolve()
    slug_path = align_root / spec / "alignment.yaml"
    tried.append(slug_path)
    if slug_path.exists():
        return slug_path.resolve()
    tried_str = ", ".join(f"'{p}'" for p in tried)
    raise FileNotFoundError(
        f"cannot resolve alignment manifest '{spec}'. Tried: {tried_str}."
    )


def _report_copy_status(
    copied: CopiedUrdfAsset,
    report_path: Path | None,
) -> None:
    """Warn on missing meshes; always drop a companion MISSING_MESHES.md.

    The stderr warning stays a single grouped line so it does not drown out
    the rest of the align output. Alongside the aligned URDF we always write
    a ``MISSING_MESHES.md`` file so:

      * a human inspecting the aligned tree has a checklist of assets to
        chase upstream without having to re-run the CLI, and
      * an empty (or removed) file is the "all resolved" signal — the next
        align run rewrites it, so it never drifts stale.

    Users who want the full copy manifest (copied files, sidecars, etc.) as
    JSON still pass ``--copy-report <path>``.
    """

    skipped = copied.skipped_meshes
    _write_missing_meshes_md(copied)
    if skipped:
        hint = (
            "pass --copy-report <path> to inspect list"
            if report_path is None
            else f"list written to {report_path}"
        )
        print(
            f"[align] WARN: {len(skipped)} mesh reference(s) not found at "
            f"source; {hint}",
            file=sys.stderr,
        )
    if report_path is not None:
        report = {
            "urdf": str(copied.urdf),
            "copied_files": [str(path) for path in copied.copied_files],
            "local_mesh_files": [
                str(path) for path in copied.local_mesh_files
            ],
            "skipped_meshes": list(skipped),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_missing_meshes_md(copied: CopiedUrdfAsset) -> None:
    """Write a ``MISSING_MESHES.md`` checklist next to the aligned URDF.

    Written on every align run (including when the skipped list is empty)
    so the file's presence, size, and contents can be tracked over time.
    Kept human-readable rather than JSON so it renders cleanly in a diff or
    a code review.
    """

    md_path = copied.urdf.parent / "MISSING_MESHES.md"
    skipped = copied.skipped_meshes
    if not skipped:
        md_path.write_text(
            f"# Missing meshes for `{copied.urdf.name}`\n\n"
            "All `<mesh filename=...>` references resolved at align time. "
            "This file is rewritten on every `urdf_tools.cli align` run.\n",
            encoding="utf-8",
        )
        return
    lines = [
        f"# Missing meshes for `{copied.urdf.name}`",
        "",
        (
            f"{len(skipped)} `<mesh filename=...>` reference(s) did not "
            "resolve during the last `urdf_tools.cli align` run and were "
            "left uncopied. This file is rewritten on every align run; "
            "the aligned URDF still opens without them, but visual "
            "inspection tools will show the affected links as untextured "
            "wireframes."
        ),
        "",
        "## References",
        "",
    ]
    for ref in skipped:
        lines.append(f"- `{ref}`")
    lines.append("")
    lines.append("## What to do")
    lines.append("")
    lines.append(
        "1. Confirm whether the upstream URDF ships these meshes at all."
    )
    lines.append(
        "2. If it does but under a nonstandard path, extend the resolver "
        "in `urdf_tools/sync_assets.py` — do not edit the aligned URDF."
    )
    lines.append(
        "3. If it does not, either patch the upstream URDF (remove the "
        "reference or repoint it) or accept the visual gap and leave a "
        "note here explaining why."
    )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


# Silence unused-import warnings when the module is imported for its side
# effects but the align entrypoint is not invoked.
_ = UrdfAlignmentCase


if __name__ == "__main__":
    main()
