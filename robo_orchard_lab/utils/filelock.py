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

"""Small filelock extension for cooperative local lock-file reclamation."""

from __future__ import annotations
import errno
import os
import stat

import filelock

if os.name == "posix":
    import fcntl

__all__ = ["ReclaimingFileLock"]


class ReclaimingFileLock(filelock.FileLock):
    """Serialize cooperative local writers without persistent POSIX lock files.

    Keep the ``filelock.FileLock`` acquire/release, timeout, polling and
    reentrant-counting API. Create the lock's parent directory on acquisition.
    On POSIX, additionally open without truncation or
    following a final symlink, reject an unlinked descriptor after flock, and
    unlink the lock name before final unlock/close. Only these lock boundaries
    change; no retry scheduler or general exception framework is added.

    All users of a lock path must use this protocol. Old or plain FileLock
    participants must exit before activation. The lock directory must be
    managed by cooperating programs using the same path: external replacement,
    renaming, deletion of active locks and extra hardlinks are unsupported.
    There is no pathname/inode replacement protection or cross-host/distributed
    coordination guarantee. All protected work must finish before release.

    A killed process can leave a file; the next normal acquire/release reclaims
    it. Neither immediate crash cleanup nor historical-cache scanning is
    provided. Cleanup follows filelock/Python exception chaining, without a
    custom ordering or aggregation policy. Outside POSIX, reject an existing
    final symlink and delegate to filelock; POSIX reclamation is not promised.

    Example::

        with ReclaimingFileLock("cache/task.lock", timeout=10):
            write_owned_output()
    """

    if os.name == "posix":

        def _acquire(self) -> None:
            """Open safely and let filelock retry an unlinked stale handle."""
            parent = os.path.dirname(self.lock_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            mode_getter = getattr(self, "_open_mode", None)
            mode = (
                mode_getter() if callable(mode_getter) else self._context.mode
            )
            try:
                fd = os.open(self.lock_file, flags, mode)
            except OSError as error:
                if error.errno == errno.ELOOP:
                    raise ValueError(
                        "Lock file must not be a symbolic link: "
                        f"{self.lock_file!r}."
                    ) from error
                raise
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError(
                        f"Lock file must be regular: {self.lock_file!r}."
                    )
                if getattr(self, "has_explicit_mode", False):
                    try:
                        os.fchmod(fd, self._context.mode)
                    except PermissionError:
                        pass
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    if error.errno in {
                        errno.EACCES,
                        errno.EAGAIN,
                        errno.EWOULDBLOCK,
                    }:
                        return
                    if error.errno == errno.ENOSYS:
                        raise NotImplementedError(
                            "FileSystem does not support flock."
                        ) from error
                    raise
                # A predecessor may have removed this name after we opened it.
                if os.fstat(fd).st_nlink == 0:
                    return
                self._context.lock_file_fd = fd
            finally:
                if self._context.lock_file_fd != fd:
                    os.close(fd)

        def _release(self) -> None:
            """Retire the name after work, attempting unlock and close once."""
            fd = self._context.lock_file_fd
            self._context.lock_file_fd = None
            self._context.lock_counter = 0
            try:
                try:
                    os.unlink(self.lock_file)
                except FileNotFoundError:
                    pass
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    else:

        def _acquire(self) -> None:
            """Retain the existing platform lock behavior and symlink guard."""
            if os.path.islink(self.lock_file):
                raise ValueError(
                    "Lock file must not be a symbolic link: "
                    f"{self.lock_file!r}."
                )
            super()._acquire()
