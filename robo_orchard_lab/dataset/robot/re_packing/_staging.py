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

"""Private staged replacement lifecycle for dataset repacking."""

from __future__ import annotations
import os
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

__all__: list[str] = []


@dataclass(slots=True)
class _StagedDatasetWriteSession:
    """Build a repacked dataset beside its target before replacement.

    The session owns its unique same-parent staging path and holds the target
    lock while the caller builds there. On ``commit()``, an authorized old
    target is moved into the session's sibling backup, then the complete
    staged root is published without replacement. A failed publication
    attempts to restore that backup. This is a cooperative lifecycle: callers
    must not externally replace the target, lock, stage, or backup while the
    session is active.
    """

    target_path: str | os.PathLike[str]
    force_overwrite: bool
    _staging_path: str | None = field(default=None, init=False, repr=False)
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

    def __enter__(self) -> _StagedDatasetWriteSession:
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
            self._target_identity = _read_target_identity(
                target_paths.dataset_dir
            )
            self._staging_path = self._make_staging_path()
            return self
        except BaseException:
            coordination_lock.release()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del traceback
        active_error = exc_value
        try:
            if self._staging_path is not None:
                self._cleanup_staging_paths()
        except BaseException as cleanup_error:
            if active_error is None:
                raise
            _add_exception_note(
                active_error,
                f"Failed to clean repack staging paths: {cleanup_error!r}",
            )
        finally:
            try:
                if self._coordination_lock is not None:
                    self._coordination_lock.release()
            except BaseException as release_error:
                if active_error is None:
                    raise
                _add_exception_note(
                    active_error,
                    "Failed to release the repack target coordination lock: "
                    f"{release_error!r}",
                )
            finally:
                self._staging_path = None
                self._target_paths = None
                self._coordination_lock = None
                self._target_identity = None

    @property
    def path(self) -> str:
        """Return the absent dataset root owned by this active session."""

        if self._staging_path is None:
            raise RuntimeError("Repack staging session is not active.")
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
        if target_paths is None or self._coordination_lock is None:
            raise RuntimeError("Repack staging session is not active.")
        target_paths.validate_locked_target_identity()
        current_target_identity = _read_target_identity(target_path)
        if current_target_identity != self._target_identity:
            raise RuntimeError(
                "The repack target changed during the active session; "
                "refusing to replace or remove it."
            )
        backup_path: str | None = None
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
            if _read_target_identity(backup_path) != self._target_identity:
                try:
                    rename_noreplace(backup_path, target_path)
                except BaseException as restore_error:
                    identity_error = RuntimeError(
                        "The repack target changed before replacement; "
                        "refusing to publish the staged dataset."
                    )
                    _add_exception_note(
                        identity_error,
                        "The changed target backup was preserved at "
                        f"{backup_path!r}: {restore_error!r}",
                    )
                    raise identity_error from restore_error
                raise RuntimeError(
                    "The repack target changed before replacement; "
                    "refusing to publish the staged dataset."
                )
        try:
            rename_noreplace(staging_path, target_path)
        except BaseException as publish_error:
            if backup_path is not None and not os.path.lexists(target_path):
                try:
                    rename_noreplace(backup_path, target_path)
                except BaseException as restore_error:
                    _add_exception_note(
                        publish_error,
                        "Failed to restore the prior dataset target from "
                        f"{backup_path!r}: {restore_error!r}",
                    )
            raise
        self._staging_path = None
        if backup_path is not None:
            remove_path(backup_path)

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
        """Remove the session-created stage and its direct-writer workspace."""

        staging_path = self.path
        staging_paths = DatasetPackagingPaths.resolve(staging_path)
        remove_path(staging_paths.workspace_dir)
        remove_path(staging_path)


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


def _read_target_identity(path: str) -> tuple[int, int, int] | None:
    """Read the target's stable local identity without following symlinks."""

    try:
        target_stat = os.lstat(path)
    except FileNotFoundError:
        return None
    return (target_stat.st_dev, target_stat.st_ino, target_stat.st_mode)
