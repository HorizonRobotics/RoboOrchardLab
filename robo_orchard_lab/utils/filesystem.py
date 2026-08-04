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

"""Filesystem primitives shared by RoboOrchard Lab dependents."""

from __future__ import annotations
import ctypes
import os
import shutil
import sys
from typing import Final

__all__ = ["remove_path", "rename_noreplace"]

_AT_FDCWD: Final = -100
_RENAME_NOREPLACE: Final = 1


def remove_path(
    path: str | os.PathLike[str],
    *,
    missing_ok: bool = True,
) -> None:
    """Remove a local file, link, or directory without following dir links.

    Args:
        path (str | os.PathLike[str]): Local filesystem path to remove.
        missing_ok (bool, optional): Whether a missing root path is accepted.
            Defaults to True.

    Raises:
        FileNotFoundError: If ``path`` is missing and ``missing_ok`` is False.
        OSError: If the path cannot otherwise be removed.
    """

    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError:
        if not missing_ok or os.path.lexists(path):
            raise


def rename_noreplace(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
) -> None:
    """Atomically rename a local path only when ``target`` is absent.

    Linux uses ``renameat2(RENAME_NOREPLACE)`` and Windows uses its ordinary
    no-replace ``os.rename`` behavior. Other platforms fail closed rather
    than falling back to a clobbering rename.

    Args:
        source (str | os.PathLike[str]): Owned local source path.
        target (str | os.PathLike[str]): Destination that must not exist.

    Raises:
        ValueError: If either path contains a null byte.
        FileExistsError: If ``target`` already exists.
        NotImplementedError: If no supported atomic no-replace operation is
            available.
        OSError: If the underlying rename fails for another reason.
    """

    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if b"\0" in source_bytes or b"\0" in target_bytes:
        raise ValueError("Filesystem paths cannot contain null bytes.")
    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform != "linux":
        raise NotImplementedError(
            "Atomic no-replace rename requires Linux renameat2 or Windows "
            "os.rename semantics."
        )

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NotImplementedError(
            "Atomic no-replace rename requires renameat2(RENAME_NOREPLACE)."
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            target_bytes,
            _RENAME_NOREPLACE,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(target),
        )
