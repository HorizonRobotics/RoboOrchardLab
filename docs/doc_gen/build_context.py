# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

"""Shared docs build context for full builds and debug-only docs flows.

The normal docs build renders the repository's full docs tree. The debug
helpers reuse the same ``conf.py`` but narrow different inputs depending on
mode:

1. AutoAPI only scans the requested Python files or package directories.
2. Sphinx Gallery only scans the requested tutorial files or directories.
3. Sphinx only discovers generated sources under an isolated debug tree.

Keeping this logic in one helper module avoids scattering environment parsing
and path normalization across ``conf.py``.
"""

from __future__ import annotations
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BUILD_MODE = "default"
DEBUG_API_BUILD_MODE = "debug-api"
DEBUG_TUTORIAL_BUILD_MODE = "debug-tutorial"
DEBUG_OVERVIEW_BUILD_MODE = "debug-overview"
_PYTHON_SUFFIXES = (".py", ".pyi")
_TUTORIAL_SUFFIXES = (".py",)
_DEFAULT_GALLERY_NAMES = (
    "dataset_tutorial",
    "trainer_tutorial",
    "model_zoo_tutorial",
)


@dataclass(frozen=True)
class DocsBuildContext:
    """Normalized docs build settings derived from environment variables.

    The context keeps all build-mode-dependent paths in one place so the main
    Sphinx config can switch between the full build and the debug-only API
    build without duplicating path calculations.
    """

    mode: str
    docs_root: str
    repo_root: str
    package_root: str
    debug_root: str
    debug_autoapi_root: str
    debug_index_file: str
    api_targets: tuple[str, ...]
    tutorial_targets: tuple[str, ...]
    no_tutorials: bool

    @property
    def is_debug_api(self) -> bool:
        return self.mode == DEBUG_API_BUILD_MODE

    @property
    def is_debug_tutorial(self) -> bool:
        return self.mode == DEBUG_TUTORIAL_BUILD_MODE

    @property
    def is_debug_overview(self) -> bool:
        return self.mode == DEBUG_OVERVIEW_BUILD_MODE

    @property
    def is_debug_build(self) -> bool:
        return (
            self.is_debug_api
            or self.is_debug_tutorial
            or self.is_debug_overview
        )

    @property
    def autoapi_root(self) -> str:
        if self.is_debug_build:
            return os.path.relpath(self.debug_autoapi_root, self.docs_root)
        return "autoapi"

    @property
    def master_doc(self) -> str:
        if self.is_debug_overview:
            return "overview/index"
        if self.is_debug_build:
            return os.path.splitext(
                os.path.relpath(self.debug_index_file, self.docs_root)
            )[0]
        return "index"


def _split_env_list(env_name: str) -> list[str]:
    """Parse a comma-separated env var into stripped values.

    Args:
        env_name (str): Environment variable name to read.

    Returns:
        list[str]: Non-empty, whitespace-trimmed entries.
    """

    raw_value = os.environ.get(env_name, "")
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_target_path(
    raw_target: str, repo_root: Path, package_root: Path
) -> Path:
    """Resolve a debug target and keep it inside the package tree.

    Args:
        raw_target (str): User-provided target from
            ``ROBO_ORCHARD_API_TARGETS``.
        repo_root (Path): Repository root used for repo-relative resolution.
        package_root (Path): Package root used for package-relative resolution.

    Returns:
        Path: Existing absolute target path inside ``package_root``.

    Raises:
        ValueError: If the target does not exist or resolves outside the main
            package tree.
    """

    candidate_paths = []
    if os.path.isabs(raw_target):
        candidate_paths.append(Path(raw_target).resolve())
    else:
        # Accept both repo-relative paths and paths relative to the Python
        # package root so the CLI stays short during targeted debugging.
        candidate_paths.extend(
            [
                (repo_root / raw_target).resolve(),
                (package_root / raw_target).resolve(),
            ]
        )

    for candidate in candidate_paths:
        if candidate.exists():
            try:
                candidate.relative_to(package_root)
            except ValueError as exc:
                raise ValueError(
                    "ROBO_ORCHARD_API_TARGETS only supports paths under "
                    f"{package_root}: {raw_target}"
                ) from exc
            return candidate

    raise ValueError(
        f"ROBO_ORCHARD_API_TARGETS contains a missing path: {raw_target}"
    )


def _expand_target_path(target_path: Path) -> set[Path]:
    """Expand a file or directory target into Python source files.

    Args:
        target_path (Path): Existing file or directory selected for debugging.

    Returns:
        set[Path]: Concrete Python source files that AutoAPI should consider.

    Raises:
        ValueError: If the target is neither a Python file nor a directory.
    """

    if target_path.is_dir():
        target_files = set()
        for suffix in _PYTHON_SUFFIXES:
            target_files.update(
                path.resolve() for path in target_path.rglob(f"*{suffix}")
            )
        return target_files

    if target_path.suffix in _PYTHON_SUFFIXES:
        return {target_path.resolve()}

    raise ValueError(
        "ROBO_ORCHARD_API_TARGETS only supports Python files or directories: "
        f"{target_path}"
    )


def _resolve_tutorial_target_path(
    raw_target: str,
    repo_root: Path,
    docs_root: Path,
    tutorials_root: Path,
) -> Path:
    """Resolve a tutorial debug target and keep it inside docs/tutorials.

    Args:
        raw_target (str): User-provided target from
            ``ROBO_ORCHARD_TUTORIAL_TARGETS``.
        repo_root (Path): Repository root used for repo-relative resolution.
        docs_root (Path): Docs root used for docs-relative resolution.
        tutorials_root (Path): Allowed root for tutorial targets.

    Returns:
        Path: Existing absolute target path inside ``tutorials_root``.

    Raises:
        ValueError: If the target does not exist or resolves outside
            ``docs/tutorials``.
    """

    candidate_paths = []
    if os.path.isabs(raw_target):
        candidate_paths.append(Path(raw_target).resolve())
    else:
        candidate_paths.extend(
            [
                (repo_root / raw_target).resolve(),
                (docs_root / raw_target).resolve(),
                (tutorials_root / raw_target).resolve(),
            ]
        )

    for candidate in candidate_paths:
        if candidate.exists():
            try:
                candidate.relative_to(tutorials_root)
            except ValueError as exc:
                raise ValueError(
                    "ROBO_ORCHARD_TUTORIAL_TARGETS only supports paths under "
                    f"{tutorials_root}: {raw_target}"
                ) from exc
            return candidate

    raise ValueError(
        f"ROBO_ORCHARD_TUTORIAL_TARGETS contains a missing path: {raw_target}"
    )


def _expand_tutorial_target_path(target_path: Path) -> set[Path]:
    """Expand a tutorial file or directory target into concrete tutorial files.

    Args:
        target_path (Path): Existing tutorial file or directory selected for
            debugging.

    Returns:
        set[Path]: Concrete tutorial source files that Sphinx Gallery should
            consider.

    Raises:
        ValueError: If the target is neither a tutorial Python file nor a
            directory.
    """

    if target_path.is_dir():
        target_files = set()
        for suffix in _TUTORIAL_SUFFIXES:
            target_files.update(
                path.resolve() for path in target_path.rglob(f"*{suffix}")
            )
        return target_files

    if target_path.suffix in _TUTORIAL_SUFFIXES:
        return {target_path.resolve()}

    raise ValueError(
        "ROBO_ORCHARD_TUTORIAL_TARGETS only supports tutorial Python files or "
        f"directories: {target_path}"
    )


def _collect_parent_package_files(
    target_files: set[Path], package_root: Path
) -> set[Path]:
    """Add parent package ``__init__`` files so AutoAPI can render hierarchy.

    Args:
        target_files (set[Path]): Concrete Python sources selected for the
            debug build.
        package_root (Path): Root package directory.

    Returns:
        set[Path]: Target files plus ancestor package ``__init__`` modules.
    """

    expanded_targets = set(target_files)
    for target_file in target_files:
        current_dir = target_file.parent
        while True:
            # Parent package pages are needed so AutoAPI can still render the
            # package chain above a nested module such as pipeline/foo/bar.py.
            init_file = current_dir / "__init__.py"
            if init_file.exists():
                expanded_targets.add(init_file.resolve())

            init_stub = current_dir / "__init__.pyi"
            if init_stub.exists():
                expanded_targets.add(init_stub.resolve())

            if current_dir == package_root:
                break
            current_dir = current_dir.parent

    return expanded_targets


def resolve_api_targets(
    raw_targets: list[str], repo_root: str, package_root: str
) -> tuple[str, ...]:
    """Resolve user-facing debug targets into a sorted AutoAPI file list.

    Args:
        raw_targets (list[str]): Raw CLI targets supplied by the user.
        repo_root (str): Repository root path.
        package_root (str): Python package root path.

    Returns:
        tuple[str, ...]: Sorted absolute file paths consumed by AutoAPI
            filtering logic.
    """

    repo_root_path = Path(repo_root).resolve()
    package_root_path = Path(package_root).resolve()
    target_files: set[Path] = set()
    for raw_target in raw_targets:
        resolved_target = _resolve_target_path(
            raw_target, repo_root_path, package_root_path
        )
        target_files.update(_expand_target_path(resolved_target))

    target_files = _collect_parent_package_files(
        target_files, package_root_path
    )
    return tuple(sorted(str(path) for path in target_files))


def resolve_tutorial_targets(
    raw_targets: list[str], repo_root: str, docs_root: str
) -> tuple[str, ...]:
    """Resolve user-facing tutorial targets into a sorted file list.

    Args:
        raw_targets (list[str]): Raw CLI tutorial targets supplied by the user.
        repo_root (str): Repository root path.
        docs_root (str): Docs root path.

    Returns:
        tuple[str, ...]: Sorted absolute tutorial file paths.
    """

    repo_root_path = Path(repo_root).resolve()
    docs_root_path = Path(docs_root).resolve()
    tutorials_root = (docs_root_path / "tutorials").resolve()
    target_files: set[Path] = set()
    for raw_target in raw_targets:
        resolved_target = _resolve_tutorial_target_path(
            raw_target,
            repo_root_path,
            docs_root_path,
            tutorials_root,
        )
        target_files.update(_expand_tutorial_target_path(resolved_target))

    return tuple(sorted(str(path) for path in target_files))


def build_context_from_env(docs_root: str) -> DocsBuildContext:
    """Build a docs context for the normal build or a debug docs mode.

    Args:
        docs_root (str): Absolute or relative path to the docs source root.

    Returns:
        DocsBuildContext: Fully resolved build settings for ``conf.py``.

    Raises:
        ValueError: If the requested build mode is unknown or a debug build is
            missing its required target environment variable.
    """

    docs_root_path = Path(docs_root).resolve()
    repo_root = docs_root_path.parent.resolve()
    package_root = (repo_root / "robo_orchard_lab").resolve()

    mode = os.environ.get("ROBO_ORCHARD_DOC_BUILD_MODE", DEFAULT_BUILD_MODE)
    if mode not in {
        DEFAULT_BUILD_MODE,
        DEBUG_API_BUILD_MODE,
        DEBUG_TUTORIAL_BUILD_MODE,
        DEBUG_OVERVIEW_BUILD_MODE,
    }:
        raise ValueError(
            "ROBO_ORCHARD_DOC_BUILD_MODE must be one of: "
            f"{DEFAULT_BUILD_MODE}, {DEBUG_API_BUILD_MODE}, "
            f"{DEBUG_TUTORIAL_BUILD_MODE}, {DEBUG_OVERVIEW_BUILD_MODE}"
        )

    # The debug flow should be fast by default, so tutorials stay disabled
    # unless the caller explicitly opts back in.
    no_tutorials_default = (
        "1"
        if mode in {DEBUG_API_BUILD_MODE, DEBUG_OVERVIEW_BUILD_MODE}
        else "0"
    )
    no_tutorials = (
        os.environ.get("ROBO_ORCHARD_NO_TUTORIALS", no_tutorials_default)
        == "1"
    )

    api_targets: tuple[str, ...] = ()
    tutorial_targets: tuple[str, ...] = ()
    if mode == DEBUG_API_BUILD_MODE:
        raw_targets = _split_env_list("ROBO_ORCHARD_API_TARGETS")
        if not raw_targets:
            raise ValueError(
                "ROBO_ORCHARD_API_TARGETS is required for debug-api builds"
            )
        api_targets = resolve_api_targets(
            raw_targets,
            str(repo_root),
            str(package_root),
        )
    elif mode == DEBUG_TUTORIAL_BUILD_MODE:
        raw_targets = _split_env_list("ROBO_ORCHARD_TUTORIAL_TARGETS")
        if not raw_targets:
            raise ValueError(
                "ROBO_ORCHARD_TUTORIAL_TARGETS is required for "
                "debug-tutorial builds"
            )
        tutorial_targets = resolve_tutorial_targets(
            raw_targets,
            str(repo_root),
            str(docs_root_path),
        )

    # Generated debug-only sources live under docs/build so Sphinx can still
    # discover them from the same source tree while staying isolated from the
    # normal docs/autoapi output.
    debug_root_name = (
        "debug_api"
        if mode == DEBUG_API_BUILD_MODE
        else "debug_tutorial"
        if mode == DEBUG_TUTORIAL_BUILD_MODE
        else "debug_overview"
        if mode == DEBUG_OVERVIEW_BUILD_MODE
        else "debug_api"
    )
    debug_root = docs_root_path / "build" / debug_root_name
    return DocsBuildContext(
        mode=mode,
        docs_root=str(docs_root_path),
        repo_root=str(repo_root),
        package_root=str(package_root),
        debug_root=str(debug_root),
        debug_autoapi_root=str(debug_root / "autoapi"),
        debug_index_file=str(debug_root / "index.rst"),
        api_targets=api_targets,
        tutorial_targets=tutorial_targets,
        no_tutorials=no_tutorials,
    )


def build_gallery_entries(
    context: DocsBuildContext,
) -> OrderedDict[str, list[dict[str, str]]]:
    """Build Sphinx Gallery directory entries for the active build mode.

    Args:
        context (DocsBuildContext): Current docs build settings.

    Returns:
        OrderedDict[str, list[dict[str, str]]]: Gallery groups consumed by
            ``conf.py``.
    """

    gallery_dict: OrderedDict[str, list[dict[str, str]]] = OrderedDict()

    if context.is_debug_api or context.no_tutorials:
        return gallery_dict

    if not context.is_debug_tutorial:
        for gallery_name in _DEFAULT_GALLERY_NAMES:
            gallery_dict[gallery_name] = [
                {"path": f"tutorials/{gallery_name}/"}
            ]
        return gallery_dict

    docs_root_path = Path(context.docs_root)
    debug_root_rel = Path(
        os.path.relpath(context.debug_root, context.docs_root)
    )
    rendered_dirs: set[Path] = set()
    for target in context.tutorial_targets:
        target_dir = Path(target).resolve().parent
        rel_target_dir = target_dir.relative_to(docs_root_path)
        if rel_target_dir in rendered_dirs:
            continue
        rendered_dirs.add(rel_target_dir)

        gallery_name = rel_target_dir.parts[-1]
        gallery_dict.setdefault(gallery_name, []).append(
            {
                "path": rel_target_dir.as_posix(),
                "output_path": (debug_root_rel / rel_target_dir).as_posix(),
            }
        )

    return gallery_dict


def build_gallery_filename_pattern(context: DocsBuildContext) -> str:
    """Build a Sphinx Gallery filename regex for the active build mode.

    Args:
        context (DocsBuildContext): Current docs build settings.

    Returns:
        str: Regex passed to ``sphinx_gallery_conf[\"filename_pattern\"]``.
    """

    if not context.is_debug_tutorial:
        return ".py"

    docs_root_path = Path(context.docs_root)
    patterns = []
    for target in context.tutorial_targets:
        rel_target = Path(target).resolve().relative_to(docs_root_path)
        patterns.append(re.escape(rel_target.as_posix()) + r"$")

    return "(" + "|".join(patterns) + ")"


def build_targeted_autoapi_ignore(context: DocsBuildContext) -> list[str]:
    """Ignore every Python source outside the requested debug target set.

    Args:
        context (DocsBuildContext): Current docs build settings.

    Returns:
        list[str]: Ignore patterns passed to AutoAPI. Empty for the normal
            full-doc build.
    """

    if not context.is_debug_api:
        return []

    package_root = Path(context.package_root)
    target_files = {Path(target) for target in context.api_targets}
    keep_dirs = {package_root}
    for target_file in target_files:
        current_dir = target_file.parent
        while True:
            # Keep the directory chain that leads to each target so AutoAPI can
            # still walk from the package root down to the requested module.
            keep_dirs.add(current_dir)
            if current_dir == package_root:
                break
            current_dir = current_dir.parent

    ignore_patterns = []
    for root, subdirectories, filenames in os.walk(package_root):
        root_path = Path(root)
        for sub_dir in subdirectories:
            sub_dir_path = root_path / sub_dir
            if sub_dir_path not in keep_dirs:
                ignore_patterns.append(str(sub_dir_path))

        for filename in filenames:
            file_path = (root_path / filename).resolve()
            if file_path.suffix not in _PYTHON_SUFFIXES:
                continue
            if file_path not in target_files:
                ignore_patterns.append(str(file_path))

    return sorted(set(ignore_patterns))


def build_exclude_patterns(context: DocsBuildContext) -> list[str]:
    """Shrink Sphinx source discovery to the debug source tree when needed.

    Args:
        context (DocsBuildContext): Current docs build settings.

    Returns:
        list[str]: Sphinx exclude patterns for the active build mode.
    """

    base_patterns = ["**/nonb**.ipynb", "tutorials/**/*.rst"]
    if not context.is_debug_build:
        return base_patterns + [
            "build/debug_api/**",
            "build/debug_tutorial/**",
            "build/debug_overview/**",
        ]

    if context.is_debug_overview:
        return base_patterns + [
            "autoapi/**",
            "build/**",
            "_templates/**",
            "tutorials/**",
            "getting_started/**",
            "index.rst",
            "readme.md",
            "sg_execution_times.rst",
        ]

    debug_patterns = []
    docs_root = Path(context.docs_root)
    visible_debug_dir = Path(context.debug_root).name
    for child in docs_root.iterdir():
        if child.name == "build":
            if child.exists():
                # Keep build/debug_api visible while excluding other generated
                # sources from the debug-only Sphinx pass.
                for build_child in child.iterdir():
                    if build_child.name != visible_debug_dir:
                        debug_patterns.append(f"build/{build_child.name}/**")
            continue

        if child.is_dir():
            # Hide the normal docs tree so the debug build only sees the
            # generated debug index plus the targeted AutoAPI pages.
            debug_patterns.append(f"{child.name}/**")
        elif child.suffix in {".rst", ".md"}:
            debug_patterns.append(child.name)

    return base_patterns + sorted(set(debug_patterns))
