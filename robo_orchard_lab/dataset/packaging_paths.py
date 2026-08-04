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

"""Local output-path ownership for RODataset packaging."""

from __future__ import annotations
import errno
import hashlib
import os
import stat
from dataclasses import dataclass

import filelock

if os.name == "posix":
    import fcntl

__all__ = ["DatasetPackagingPaths", "normalize_local_dataset_path"]

_PACKAGING_WORKSPACE_SUFFIX = ".__robo_orchard_packaging__"


if os.name == "posix":

    class _NoFollowFileLock(filelock.FileLock):
        """Acquire a Unix advisory lock without following its final path."""

        def _acquire(self) -> None:
            """Open and lock a regular file without truncating symlinks."""

            parent_dir = os.path.dirname(self.lock_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            open_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            open_flags |= getattr(os, "O_NOFOLLOW", 0)
            open_mode_getter = getattr(self, "_open_mode", None)
            open_mode = (
                open_mode_getter()
                if callable(open_mode_getter)
                else self._context.mode
            )
            try:
                fd = os.open(self.lock_file, open_flags, open_mode)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(
                        "Dataset packaging coordination lock must not be a "
                        f"symbolic link: {self.lock_file!r}."
                    ) from exc
                raise

            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise ValueError(
                    "Dataset packaging coordination lock must be a "
                    f"regular file: {self.lock_file!r}."
                )
            if getattr(self, "has_explicit_mode", False):
                try:
                    os.fchmod(fd, self._context.mode)
                except PermissionError:
                    pass
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno in {
                    errno.EACCES,
                    errno.EAGAIN,
                    errno.EWOULDBLOCK,
                }:
                    # Let BaseFileLock retry until its configured timeout.
                    return
                if exc.errno == errno.ENOSYS:
                    raise NotImplementedError(
                        "FileSystem does not appear to support flock; use "
                        "SoftFileLock instead."
                    ) from exc
                raise
            except BaseException:
                os.close(fd)
                raise
            if os.fstat(fd).st_nlink == 0:
                os.close(fd)
                return
            self._context.lock_file_fd = fd

        def _release(self) -> None:
            """Unlock without unlinking the shared coordination file."""

            fd = self._context.lock_file_fd
            self._context.lock_file_fd = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

else:

    class _NoFollowFileLock(filelock.FileLock):
        """Reject an existing lock symlink before platform lock handling."""

        def _acquire(self) -> None:
            """Reject a final symlink before delegating to filelock."""

            if os.path.islink(self.lock_file):
                raise ValueError(
                    "Dataset packaging coordination lock must not be a "
                    f"symbolic link: {self.lock_file!r}."
                )
            super()._acquire()


def _create_coordination_lock(lock_path: str) -> filelock.BaseFileLock:
    """Create the shared safe lock used by direct and staged packaging."""

    return _NoFollowFileLock(lock_path)


def normalize_local_dataset_path(
    dataset_path: str | os.PathLike[str],
) -> str:
    """Normalize a local RODataset output path without touching storage.

    Args:
        dataset_path (str | os.PathLike[str]): Local output path. URI-style
            paths are not supported.

    Returns:
        str: Expanded absolute local path.

    Raises:
        TypeError: If ``dataset_path`` is not a text local path.
        ValueError: If ``dataset_path`` is URI-style.
    """

    dataset_path_str = os.fspath(dataset_path)
    if not isinstance(dataset_path_str, str):
        raise TypeError("dataset_path must be a string or os.PathLike[str].")
    if "://" in dataset_path_str:
        raise ValueError(
            "DatasetPackaging only supports local filesystem dataset_path. "
            f"URI paths are not supported: {dataset_path_str!r}."
        )
    return os.path.abspath(os.path.expanduser(dataset_path_str))


@dataclass(frozen=True, slots=True)
class DatasetPackagingPaths:
    """Name the local paths owned by one direct packaging target.

    The final dataset and deterministic sibling workspace remain separate so
    Hugging Face build state never appears in the published target. The
    workspace suffix is reserved; a dataset target at or below that namespace
    is rejected, which makes a forced cleanup unambiguous. A cache-resident
    file lock serializes cooperative writers for the same normalized target.
    The caller owns cross-host coordination by configuring a shared
    ``XDG_CACHE_HOME`` on filesystems that support advisory locks.
    """

    dataset_dir: str
    """Canonical final local RODataset directory."""

    requested_dataset_dir: str
    """Caller path before resolving existing parent-directory aliases."""

    workspace_dir: str
    """Reserved disposable sibling workspace for direct packaging only."""

    builder_output_dir: str
    """Completed Hugging Face dataset root staged inside ``workspace_dir``."""

    hf_cache_dir: str
    """Disposable Hugging Face cache scoped to ``workspace_dir``."""

    coordination_lock_path: str
    """Stable advisory-lock path outside the target's output tree."""

    @classmethod
    def resolve(
        cls,
        dataset_path: str | os.PathLike[str],
    ) -> DatasetPackagingPaths:
        """Resolve paths without creating or deleting filesystem entries."""

        requested_dataset_dir = normalize_local_dataset_path(dataset_path)
        dataset_dir = os.path.realpath(requested_dataset_dir)
        _validate_not_workspace_target(requested_dataset_dir)
        _validate_not_workspace_target(dataset_dir)
        workspace_dir = f"{dataset_dir}{_PACKAGING_WORKSPACE_SUFFIX}"
        return cls(
            dataset_dir=dataset_dir,
            requested_dataset_dir=requested_dataset_dir,
            workspace_dir=workspace_dir,
            builder_output_dir=os.path.join(workspace_dir, "dataset"),
            hf_cache_dir=os.path.join(workspace_dir, "hf_cache"),
            coordination_lock_path=_resolve_coordination_lock_path(
                dataset_dir=dataset_dir,
                requested_dataset_dir=requested_dataset_dir,
            ),
        )

    @property
    def output_roots(self) -> tuple[str, str]:
        """Return direct-writer roots that share explicit overwrite policy."""

        return self.dataset_dir, self.workspace_dir

    @property
    def write_roots(self) -> tuple[str, ...]:
        """Return direct-writer roots plus its cooperative lock location."""

        return *self.output_roots, self.coordination_lock_path

    def validate_preconditions(
        self,
        *,
        force_overwrite: bool,
        include_workspace: bool = True,
    ) -> None:
        """Validate ordinary writer preconditions without mutation.

        Args:
            force_overwrite (bool): Allows the direct writer to remove an
                existing final target and its reserved stale workspace.
            include_workspace (bool): Whether this caller owns the direct
                workspace. Repack staging validates only the final target.

        Raises:
            ValueError: If an owned output root is a symbolic link.
            FileExistsError: If a required absent root already exists.
        """

        output_roots = (
            self.output_roots if include_workspace else (self.dataset_dir,)
        )
        for path in (self.requested_dataset_dir, *output_roots):
            if os.path.islink(path):
                raise ValueError(
                    "Dataset packaging write targets must not be symbolic "
                    f"links: {path!r}."
                )
        if force_overwrite:
            return
        for path in output_roots:
            if os.path.lexists(path):
                raise FileExistsError(
                    f"The packaging path {path!r} already exists. Remove it "
                    "or set force_overwrite=True to replace it."
                )

    def validate_locked_target_identity(self) -> None:
        """Recheck a target name after acquiring its cooperative lock.

        This compatibility preflight supports callers that resolve paths,
        wait on ``coordination_lock_path``, then perform their own authorized
        cleanup. It detects a changed canonical target or lock-cache setting
        while waiting, but does not provide a long-lived filesystem identity
        guarantee against external path replacement.

        Raises:
            RuntimeError: If the requested path now resolves to a different
                canonical target or coordination lock.
        """

        current = type(self).resolve(self.requested_dataset_dir)
        if (
            current.dataset_dir != self.dataset_dir
            or current.coordination_lock_path != self.coordination_lock_path
        ):
            raise RuntimeError(
                "Dataset packaging target identity changed while waiting for "
                f"its coordination lock: {self.requested_dataset_dir!r}."
            )


def _resolve_coordination_lock_path(
    *,
    dataset_dir: str,
    requested_dataset_dir: str,
) -> str:
    """Return the target-stable cooperative lock outside output parents."""

    target_digest = hashlib.sha256(dataset_dir.encode("utf-8")).hexdigest()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if not cache_home:
        cache_home = os.path.join(os.path.expanduser("~"), ".cache")
    elif not os.path.isabs(os.path.expanduser(cache_home)):
        raise ValueError(
            "XDG_CACHE_HOME must be an absolute path for dataset packaging."
        )
    cache_root = os.path.realpath(os.path.expanduser(cache_home))
    lock_path = os.path.join(
        cache_root,
        "robo_orchard",
        "dataset_packaging_locks",
        f"{target_digest}.lock",
    )
    workspace_dir = f"{dataset_dir}{_PACKAGING_WORKSPACE_SUFFIX}"
    for output_root in (dataset_dir, workspace_dir):
        if _is_same_or_descendant(lock_path, output_root):
            raise ValueError(
                "The dataset packaging coordination lock must be outside "
                f"the dataset output roots: {requested_dataset_dir!r}."
            )
    return lock_path


def _validate_not_workspace_target(dataset_dir: str) -> None:
    """Reject a dataset nested below a reserved direct-writer workspace."""

    current_path = dataset_dir
    while True:
        if os.path.basename(current_path).endswith(
            _PACKAGING_WORKSPACE_SUFFIX
        ):
            raise ValueError(
                "Dataset packaging target paths cannot be inside a directory "
                "with the reserved workspace suffix "
                f"{_PACKAGING_WORKSPACE_SUFFIX!r}: {dataset_dir!r}."
            )
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            return
        current_path = parent_path


def _is_same_or_descendant(path: str, parent: str) -> bool:
    """Return whether ``path`` is contained by ``parent``."""

    try:
        return os.path.commonpath((path, parent)) == parent
    except ValueError:
        return False
