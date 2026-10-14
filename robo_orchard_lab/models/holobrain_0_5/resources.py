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

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

__all__ = [
    "export_urdf_files",
    "pretrained_search_paths",
    "pretrained_source",
    "pretrained_export_path",
    "export_pretrained_directory",
    "link_pretrained_directory",
]

CKPT_BASE_ENV = "HOLOBRAIN_CKPT_BASE"
DEFAULT_CKPT_BASE = "./ckpt"


def export_urdf_files(
    transforms: list[dict] | None, directory: str, urdf_dir: str
) -> None:
    """Copy URDFs and update paths in a private copy of transform configs.

    Preserve existing filenames where possible. Different URDFs with the
    same basename receive a content suffix so exporting another processor
    cannot overwrite a file referenced by a previously saved config.

    Args:
        transforms (list[dict] | None): Copied transform configs to update.
        directory (str): Directory containing the exported config.
        urdf_dir (str): Destination for URDFs, relative to the exported config
            unless an absolute directory is explicitly supplied.
    """
    for transform in transforms or []:
        if "urdf" not in transform:
            continue
        source = Path(transform["urdf"])
        content = source.read_bytes()
        reference = Path(urdf_dir) / source.name
        target = Path(directory) / reference
        if target.exists() and target.read_bytes() != content:
            digest = hashlib.sha256(content).hexdigest()
            reference = (
                Path(urdf_dir) / f"{source.stem}-{digest}{source.suffix}"
            )
            target = Path(directory) / reference
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        elif target.read_bytes() != content:
            raise FileExistsError(f"Conflicting exported URDF: {target}")
        transform["urdf"] = str(reference)


def _is_explicit_path(source: str) -> bool:
    # Inspect the raw string: Path("./name").parts drops the "./", which would
    # reclassify an explicit cwd reference as a bare name.
    return (
        source in (".", "..")
        or source.startswith("~")
        or "/" in source
        or os.sep in source
        or Path(source).is_absolute()
    )


def pretrained_search_paths(source: str) -> list[Path]:
    """Return locations tried for a pretrained reference, in priority order.

    An explicit path (absolute, ``~``-prefixed, or containing a separator) is
    honored verbatim. A bare name is looked up under the directory named by
    ``HOLOBRAIN_CKPT_BASE``, defaulting to ``./ckpt``.

    Args:
        source (str): The configured pretrained reference.

    Returns:
        list[Path]: Candidate directories, most preferred first.
    """
    if _is_explicit_path(source):
        return [Path(source).expanduser()]
    base = os.environ.get(CKPT_BASE_ENV, DEFAULT_CKPT_BASE)
    return [Path(base).expanduser() / source]


def pretrained_source(source: str) -> str:
    """Resolve an existing local pretrained directory to an absolute path."""
    candidates = pretrained_search_paths(source)
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.absolute())
    configured = os.environ.get(CKPT_BASE_ENV)
    base = (
        f"{CKPT_BASE_ENV}={configured}"
        if configured is not None
        else f"{CKPT_BASE_ENV} unset, default {DEFAULT_CKPT_BASE!r}"
    )
    tried = "\n".join(
        f"  - {candidate.absolute()}" for candidate in candidates
    )
    raise FileNotFoundError(
        f"Pretrained directory not found for {source!r}. Tried:\n{tried}\n"
        f"({base}) is prepended to bare names; use a path containing a "
        f"separator, such as './{source}', to bypass it."
    )


def pretrained_export_path(source: str, model_prefix: str = "model") -> str:
    """Return a portable resource reference within a model export."""
    name = Path(source).expanduser().name
    if name in ("", ".", ".."):
        name = Path(source).expanduser().absolute().resolve().name
    prefix = Path() if model_prefix == "model" else Path(model_prefix)
    return str(prefix / "ckpt" / name)


def _export_target(directory: str, reference: str) -> Path:
    root = Path(directory).absolute()
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid pretrained export reference: {reference}")
    target = root / relative
    # Check every parent: a custom model prefix can also be a symlink.
    for parent in target.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise FileExistsError(
                f"Cannot export pretrained files through symlink: {parent}"
            )
    return target


@contextmanager
def export_pretrained_directory(
    directory: str, reference: str
) -> Iterator[str]:
    """Stage an export and replace old links without writing through them.

    Only the dedicated resource directory is replaced. A linked parent is
    rejected because replacing its children would change external resources.
    """
    target = _export_target(directory, reference)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pretrained-", dir=target.parent) as tmp:
        staged = Path(tmp) / "resource"
        staged.mkdir()
        yield str(staged)
        previous = Path(tmp) / "previous"
        had_previous = os.path.lexists(target)
        if had_previous:
            os.replace(target, previous)
        try:
            os.replace(staged, target)
        except OSError:
            if had_previous:
                os.replace(previous, target)
            raise


def link_pretrained_directory(
    directory: str, reference: str, source: str
) -> None:
    """Link a local resource without silently keeping a wrong link."""
    target = _export_target(directory, reference)
    source_path = Path(source).expanduser().absolute()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Missing pretrained resource: {source_path}")
    if target.is_symlink():
        if target.resolve() == source_path.resolve():
            return
        target.unlink()
    elif target.exists():
        if target.resolve() == source_path.resolve():
            return
        raise FileExistsError(f"Pretrained export already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source_path.resolve(), target_is_directory=True)
