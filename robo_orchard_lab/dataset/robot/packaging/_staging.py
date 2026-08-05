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

"""Narrow staged publication lifecycle for complete local dataset roots."""

from __future__ import annotations
import ctypes
import os
import stat
import tempfile
from dataclasses import dataclass, field
from types import TracebackType

import filelock

from robo_orchard_lab.dataset.packaging_paths import (
    DatasetPackagingPaths,
    _create_coordination_lock,
)
from robo_orchard_lab.utils.filesystem import (
    remove_path,
    rename_noreplace,
)

__all__ = ["StagedDatasetWriteSession"]


@dataclass(slots=True)
class StagedDatasetWriteSession:
    """Build one complete local dataset root beside its final target.

    The session owns a unique same-parent stage and holds the final target's
    coordination lock while a direct dataset writer builds the complete root
    at :attr:`path`. ``commit()`` replaces only the target generation observed
    at entry; a failed publication restores that generation when possible.
    Exiting without a successful commit removes the stage, direct writer's
    workspace, and the private coordination lock that the direct writer
    creates for that unique stage.

    This is deliberately a narrow local-filesystem contract. It coordinates
    one complete dataset-root publication; it does not provide a general
    transaction for arbitrary side effects, remote filesystems, or concurrent
    external replacement of the target, lock, stage, or backup.

    Args:
        target_path: Final local dataset directory.
        force_overwrite: Whether ``commit()`` may replace the target
            generation observed when the session begins.
    """

    target_path: str | os.PathLike[str]
    force_overwrite: bool
    _staging_path: str | None = field(default=None, init=False, repr=False)
    _staging_paths: DatasetPackagingPaths | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """Paths owned by the direct writer inside the active unique stage."""
    _target_paths: DatasetPackagingPaths | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _coordination_lock: filelock.BaseFileLock | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _target_identity: tuple[int, int, int] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _target_identity_fd: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    """Open handle that keeps the initial target identity from being reused."""

    def __enter__(self) -> StagedDatasetWriteSession:
        target_paths = DatasetPackagingPaths.resolve(self.target_path)
        os.makedirs(
            os.path.dirname(target_paths.coordination_lock_path),
            exist_ok=True,
        )
        coordination_lock = _create_coordination_lock(
            target_paths.coordination_lock_path
        )
        coordination_lock.acquire()
        try:
            target_paths.validate_locked_target_identity()
            target_paths.validate_preconditions(
                force_overwrite=self.force_overwrite,
                include_workspace=False,
            )
            self.target_path = target_paths.dataset_dir
            self._target_paths = target_paths
            self._coordination_lock = coordination_lock
            (
                self._target_identity,
                self._target_identity_fd,
            ) = _hold_target_identity(target_paths.dataset_dir)
            self._staging_path = self._make_staging_path()
            self._staging_paths = DatasetPackagingPaths.resolve(
                self._staging_path
            )
            return self
        except BaseException as setup_error:
            try:
                self._close_target_identity_fd()
            except BaseException as close_error:
                _add_exception_note(
                    setup_error,
                    "Failed to close the staged dataset target identity "
                    f"handle during setup rollback: {close_error!r}",
                )
            try:
                coordination_lock.release()
            except BaseException as release_error:
                _add_exception_note(
                    setup_error,
                    "Failed to release the dataset target coordination lock "
                    f"during setup rollback: {release_error!r}",
                )
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        primary_error = exc_value
        try:
            if self._staging_path is not None:
                self._cleanup_staging_paths()
        except BaseException as cleanup_error:
            primary_error = _preserve_primary_error(
                primary_error,
                cleanup_error,
                f"Failed to clean staged dataset paths: {cleanup_error!r}",
            )
        finally:
            try:
                if self._coordination_lock is not None:
                    self._coordination_lock.release()
            except BaseException as release_error:
                primary_error = _preserve_primary_error(
                    primary_error,
                    release_error,
                    "Failed to release the dataset target coordination lock: "
                    f"{release_error!r}",
                )
            finally:
                try:
                    self._close_target_identity_fd()
                except BaseException as close_error:
                    primary_error = _preserve_primary_error(
                        primary_error,
                        close_error,
                        "Failed to close the staged dataset target identity "
                        f"handle: {close_error!r}",
                    )
                finally:
                    self._staging_path = None
                    self._staging_paths = None
                    self._target_paths = None
                    self._coordination_lock = None
                    self._target_identity = None
        if exc_value is None and primary_error is not None:
            raise primary_error

    @property
    def path(self) -> str:
        """Return the absent dataset root owned by this active session."""

        if self._staging_path is None:
            raise RuntimeError("Staged dataset session is not active.")
        return self._staging_path

    def commit(self) -> None:
        """Publish the completed stage and clean its prior generation.

        Raises:
            FileExistsError: If an unforced replacement finds a target.
            OSError: If publication, restoration, or required cleanup fails.
            RuntimeError: If the session is not active.
        """

        staging_path = self.path
        target_path = os.fspath(self.target_path)
        target_paths = self._target_paths
        staging_paths = self._staging_paths
        if (
            target_paths is None
            or staging_paths is None
            or self._coordination_lock is None
        ):
            raise RuntimeError("Staged dataset session is not active.")
        target_paths.validate_locked_target_identity()
        current_target_identity = _read_target_identity(target_path)
        if current_target_identity != self._target_identity:
            raise RuntimeError(
                "The dataset target changed during the active session; "
                "refusing to replace or remove it."
            )
        backup_path: str | None = None
        try:
            if os.path.lexists(target_path):
                if not self.force_overwrite:
                    raise FileExistsError(
                        f"The dataset path {target_path!r} already exists. "
                        "Set force_overwrite=True to replace it."
                    )
                backup_path = tempfile.mkdtemp(
                    prefix=f".{os.path.basename(target_path)}.backup-",
                    dir=os.path.dirname(target_path),
                )
                remove_path(backup_path)
                os.rename(target_path, backup_path)
            if (
                backup_path is not None
                and _read_target_identity(backup_path) != self._target_identity
            ):
                raise RuntimeError(
                    "The dataset target changed before replacement; "
                    "refusing to publish the staged dataset."
                )
            rename_noreplace(staging_path, target_path)
        except BaseException as publish_error:
            if (
                backup_path is not None
                and os.path.lexists(backup_path)
                and not os.path.lexists(target_path)
            ):
                try:
                    rename_noreplace(backup_path, target_path)
                except BaseException as restore_error:
                    _add_exception_note(
                        publish_error,
                        "Failed to restore the prior dataset target from "
                        f"{backup_path!r}: {restore_error!r}",
                    )
            raise
        if backup_path is not None:
            remove_path(backup_path)
        remove_path(staging_paths.coordination_lock_path)
        self._staging_path = None

    def _make_staging_path(self) -> str:
        target_path = os.fspath(self.target_path)
        target_parent = os.path.dirname(target_path)
        os.makedirs(target_parent, exist_ok=True)
        staging_path = tempfile.mkdtemp(
            prefix=f".{os.path.basename(target_path)}.tmp-",
            dir=target_parent,
        )
        remove_path(staging_path)
        return staging_path

    def _cleanup_staging_paths(self) -> None:
        """Remove the stage, direct-writer workspace, and private lock."""

        staging_path = self.path
        staging_paths = self._staging_paths
        if staging_paths is None:
            raise RuntimeError("Staged dataset session is not active.")
        cleanup_error: BaseException | None = None
        for path, description in (
            (staging_paths.workspace_dir, "direct-writer workspace"),
            (staging_paths.coordination_lock_path, "private stage lock"),
            (staging_path, "staging path"),
        ):
            try:
                remove_path(path)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    _add_exception_note(
                        cleanup_error,
                        f"Failed to remove repack {description} {path!r}: "
                        f"{error!r}",
                    )
        if cleanup_error is not None:
            raise cleanup_error

    def _close_target_identity_fd(self) -> None:
        """Close the handle retained for the initial target generation."""

        target_identity_fd = self._target_identity_fd
        self._target_identity_fd = None
        if target_identity_fd is not None:
            os.close(target_identity_fd)


def _add_exception_note(error: BaseException, message: str) -> None:
    """Attach cleanup context without replacing the primary failure."""

    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(message)
        return
    notes = getattr(error, "__notes__", None)
    if notes is None:
        notes = []
        error.__notes__ = notes  # type: ignore[attr-defined]
    notes.append(message)


def _preserve_primary_error(
    primary_error: BaseException | None,
    later_error: BaseException,
    message: str,
) -> BaseException:
    """Keep the first lifecycle error and attach later teardown failures."""

    if primary_error is not None:
        _add_exception_note(primary_error, message)
        return primary_error
    return later_error


def _read_target_identity(path: str) -> tuple[int, int, int] | None:
    """Read the target's stable local identity without following symlinks."""

    try:
        target_stat = os.lstat(path)
    except FileNotFoundError:
        return None
    return (target_stat.st_dev, target_stat.st_ino, target_stat.st_mode)


def _hold_target_identity(
    path: str,
) -> tuple[tuple[int, int, int] | None, int | None]:
    """Return an initial target identity and retain its inode until exit.

    Holding an fd prevents a removed target inode from being immediately
    reused for an externally recreated directory. That makes the normal
    path-based identity comparison reliable for the session lifetime.
    """

    if os.name == "nt":
        return _hold_windows_target_identity(path)

    flags = getattr(os, "O_PATH", os.O_RDONLY)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        target_identity_fd = os.open(path, flags)
    except FileNotFoundError:
        return None, None
    try:
        target_stat = os.fstat(target_identity_fd)
        if stat.S_ISLNK(target_stat.st_mode):
            raise ValueError(
                "Dataset packaging write targets must not be symbolic links: "
                f"{path!r}."
            )
        return (
            (target_stat.st_dev, target_stat.st_ino, target_stat.st_mode),
            target_identity_fd,
        )
    except BaseException:
        os.close(target_identity_fd)
        raise


def _hold_windows_target_identity(
    path: str,
) -> tuple[tuple[int, int, int] | None, int | None]:
    """Retain a Windows directory handle through the session lifetime.

    ``os.open`` cannot open a directory on Windows. A handle opened with
    backup semantics and delete sharing keeps the observed file identity from
    being reused while still allowing the session to rename its own target.
    """

    try:
        path_stat = os.lstat(path)
    except FileNotFoundError:
        return None, None
    if stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(
            "Dataset packaging write targets must not be symbolic links: "
            f"{path!r}."
        )

    target_identity_fd = _open_windows_path_fd(path)
    try:
        target_stat = os.fstat(target_identity_fd)
        target_identity = (
            target_stat.st_dev,
            target_stat.st_ino,
            target_stat.st_mode,
        )
        path_identity = (
            path_stat.st_dev,
            path_stat.st_ino,
            path_stat.st_mode,
        )
        if target_identity != path_identity:
            raise RuntimeError(
                "The dataset target changed while acquiring its identity "
                "handle; refusing to stage a replacement."
            )
        return target_identity, target_identity_fd
    except BaseException:
        os.close(target_identity_fd)
        raise


def _open_windows_path_fd(path: str) -> int:
    """Open a local Windows path with directory-capable identity semantics."""

    import msvcrt
    from ctypes import wintypes

    file_share_read = 0x00000001
    file_share_write = 0x00000002
    file_share_delete = 0x00000004
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    invalid_handle_value = ctypes.c_void_p(-1).value

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        path,
        0,
        file_share_read | file_share_write | file_share_delete,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    if handle == invalid_handle_value:
        error_number = ctypes.get_last_error()
        raise OSError(error_number, os.strerror(error_number), path)
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except BaseException:
        close_handle(handle)
        raise
