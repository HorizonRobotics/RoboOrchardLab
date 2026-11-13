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

"""Cooperative POSIX reclamation without importing dataset machinery."""

from __future__ import annotations
import multiprocessing
import os
import stat
from pathlib import Path
from typing import Any

import filelock
import pytest

from robo_orchard_lab.utils import filelock as lock_module
from robo_orchard_lab.utils.filelock import ReclaimingFileLock

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX reclamation")


def _paused_waiter(
    path: str,
    opened: Any,
    proceed: Any,
    retried: Any,
    entered: Any,
    finish: Any,
) -> None:
    """Hold a real stale descriptor across its predecessor's release."""
    flock = lock_module.fcntl.flock

    def gate(fd: int, operation: int) -> None:
        if operation == lock_module.fcntl.LOCK_EX | lock_module.fcntl.LOCK_NB:
            if not opened.is_set():
                opened.set()
                assert proceed.wait(20)
            else:
                retried.set()
        flock(fd, operation)

    lock_module.fcntl.flock = gate
    with ReclaimingFileLock(path):
        entered.set()
        assert finish.wait(20)


def _holder(path: str, phase: str, ready: Any) -> None:
    """Expose actual ownership before or during release for a SIGKILL."""
    if phase == "unlinked":
        unlink = os.unlink

        def pause_after_unlink(name: Any) -> None:
            unlink(name)
            ready.set()
            multiprocessing.Event().wait(30)

        lock_module.os.unlink = pause_after_unlink
    with ReclaimingFileLock(path):
        if phase == "held":
            ready.set()
            multiprocessing.Event().wait(30)


@pytest.mark.parametrize("form", ["context", "acquire-proxy", "manual"])
@pytest.mark.parametrize("body_error", [False, True])
def test_filelock_api_reclaims_after_work(
    tmp_path: Path, form: str, body_error: bool
) -> None:
    path = tmp_path / "nested" / "work.lock"
    lock = ReclaimingFileLock(path, timeout=0.05)

    def work() -> None:
        assert path.is_file() and lock.is_locked
        assert stat.S_IMODE(path.stat().st_mode) & 0o111 == 0
        with pytest.raises(filelock.Timeout):
            ReclaimingFileLock(path).acquire(timeout=0)
        if body_error:
            raise ValueError("body")

    def run() -> None:
        if form == "context":
            with lock:
                work()
        elif form == "acquire-proxy":
            with lock.acquire():
                work()
        else:
            lock.acquire()
            try:
                work()
            finally:
                lock.release()

    if body_error:
        with pytest.raises(ValueError, match="body"):
            run()
    else:
        run()
    assert not path.exists() and not lock.is_locked
    with lock:
        with lock:
            assert lock.lock_counter == 2
        assert path.exists() and lock.lock_counter == 1
    lock.release()
    assert not path.exists()


def test_stale_open_waiter_retries_after_unlink(tmp_path: Path) -> None:
    path = str(tmp_path / "work.lock")
    owner, replacement = ReclaimingFileLock(path), ReclaimingFileLock(path)
    owner.acquire()
    context = multiprocessing.get_context("spawn")
    opened, proceed, retried, entered, finish = [
        context.Event() for _ in range(5)
    ]
    waiter = context.Process(
        target=_paused_waiter,
        args=(path, opened, proceed, retried, entered, finish),
    )
    waiter.start()
    try:
        assert opened.wait(20)
        owner.release()
        replacement.acquire(timeout=0)
        proceed.set()
        assert retried.wait(20), "waiter accepted the deleted lock file"
        assert not entered.is_set(), "waiter overlapped the new holder"
        replacement.release()
        assert entered.wait(20)
        finish.set()
        waiter.join(20)
        assert waiter.exitcode == 0
        assert not Path(path).exists()
    finally:
        proceed.set()
        finish.set()
        owner.release()
        replacement.release()
        if waiter.is_alive():
            waiter.kill()
        waiter.join(5)


@pytest.mark.parametrize("phase", ["held", "unlinked"])
def test_killed_process_allows_next_normal_reclamation(
    tmp_path: Path, phase: str
) -> None:
    path = tmp_path / "work.lock"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    holder = context.Process(target=_holder, args=(str(path), phase, ready))
    holder.start()
    try:
        assert ready.wait(20)
        holder.kill()
        holder.join(5)
        assert holder.exitcode is not None and holder.exitcode != 0
        assert path.exists() == (phase == "held")
        with ReclaimingFileLock(path, timeout=1):
            assert path.exists()
        assert not path.exists()
    finally:
        if holder.is_alive():
            holder.kill()
        holder.join(5)


@pytest.mark.parametrize("stage", ["unlink", "unlock", "close"])
def test_release_attempts_remaining_resources_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    path = tmp_path / "work.lock"
    lock = ReclaimingFileLock(path)
    lock.acquire()
    lock.acquire()
    fd = lock._context.lock_file_fd
    unlink, flock, close = os.unlink, lock_module.fcntl.flock, os.close
    attempted = []

    def checked_unlink(name: Any) -> None:
        attempted.append("unlink")
        if stage == "unlink":
            raise OSError("unlink failed")
        unlink(name)

    def checked_unlock(handle: int, operation: int) -> None:
        attempted.append("unlock")
        if stage == "unlock":
            raise OSError("unlock failed")
        flock(handle, operation)

    def checked_close(handle: int) -> None:
        attempted.append("close")
        close(handle)
        if stage == "close":
            raise OSError("close failed after syscall")

    with monkeypatch.context() as patch:
        patch.setattr(lock_module.os, "unlink", checked_unlink)
        patch.setattr(lock_module.fcntl, "flock", checked_unlock)
        patch.setattr(lock_module.os, "close", checked_close)
        with pytest.raises(OSError, match=stage):
            lock.release(force=True)
    assert attempted == ["unlink", "unlock", "close"]
    assert not lock.is_locked and lock.lock_counter == 0
    with pytest.raises(OSError):
        os.fstat(fd)
    with lock:
        assert lock.lock_counter == 1
    assert not path.exists()


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_rejects_nonregular_or_symlink_without_truncating(
    tmp_path: Path, kind: str
) -> None:
    path = tmp_path / "work.lock"
    victim = tmp_path / "victim"
    victim.write_text("retained")
    if kind == "symlink":
        path.symlink_to(victim)
        error = ValueError
    else:
        path.mkdir()
        error = IsADirectoryError
    with pytest.raises(error):
        ReclaimingFileLock(path).acquire(timeout=0)
    assert victim.read_text() == "retained"
