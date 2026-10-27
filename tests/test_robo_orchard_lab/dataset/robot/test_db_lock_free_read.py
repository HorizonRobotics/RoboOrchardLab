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

import os
import pickle
import sqlite3
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from selectors import DefaultSelector

import duckdb
import pytest
from datasets import Dataset
from sqlalchemy import URL, event, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool, QueuePool, SingletonThreadPool

from robo_orchard_lab.dataset.robot import (
    dataset_db_engine as metadata_db,
    engine as db_engine,
)
from robo_orchard_lab.dataset.robot.dataset import (
    RODataset,
    ROMultiRowDataset,
    _RODatasetDBResource,
)
from robo_orchard_lab.dataset.robot.db_orm import Episode, Task
from robo_orchard_lab.dataset.robot.db_orm.table_info import TableInfo
from robo_orchard_lab.dataset.robot.db_orm.task import TaskDeprecatedVersion1
from robo_orchard_lab.dataset.robot.row_sampler import (
    ColumnIndexOffsetSamplerConfig,
)


@pytest.fixture(autouse=True)
def _local_policy_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBO_ORCHARD_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", raising=False)


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, False), ("", False), ("0", False), ("1", True)],
)
def test_lock_free_read_environment(monkeypatch, value, enabled):
    """Only the documented 1/0 switch values select the reader policy."""
    if value is None:
        monkeypatch.delenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", raising=False)
    else:
        monkeypatch.setenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", value)
    assert RODataset._get_db_lock_free_read_enabled() is enabled


@pytest.mark.parametrize("value", ["true", "false", "2", " 1", "1 "])
def test_lock_free_read_invalid_environment(monkeypatch, value):
    """Invalid nonempty settings fail instead of silently changing policy."""
    monkeypatch.setenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", value)
    with pytest.raises(ValueError, match="ROBO_ORCHARD_DB_LOCK_FREE_READ"):
        RODataset._get_db_lock_free_read_enabled()


def _make_duckdb(path: Path) -> None:
    connection = duckdb.connect(str(path), config={"threads": 1})
    try:
        connection.execute("CREATE TABLE sample AS SELECT 7 AS value")
    finally:
        connection.close()


def test_duckdb_copy_helper_cold_hot_and_owned_paths(tmp_path, monkeypatch):
    """Cold copies are complete and hot/owned URLs need no source read."""
    source = tmp_path / "source.duckdb"
    _make_duckdb(source)
    source.chmod(0o644)
    os.utime(source, ns=(1_000_000_000, 2_000_000_000))
    source_stat = source.stat()
    temporary_stats = []
    real_mkstemp = db_engine.tempfile.mkstemp

    def record_temporary_metadata(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        temporary_stats.append(os.fstat(fd))
        return fd, path

    monkeypatch.setattr(
        db_engine.tempfile, "mkstemp", record_temporary_metadata
    )
    url = URL.create("duckdb", database=str(source))
    cached_url = db_engine._get_duckdb_read_cache_url(url)
    cached = Path(cached_url.database)
    assert cached != source
    (temporary_stat,) = temporary_stats
    cached_stat = cached.stat()
    assert cached_stat.st_mtime_ns == source_stat.st_mtime_ns
    assert cached_stat.st_atime_ns == temporary_stat.st_atime_ns
    assert cached_stat.st_atime_ns != source_stat.st_atime_ns
    assert stat.S_IMODE(cached_stat.st_mode) == stat.S_IMODE(
        temporary_stat.st_mode
    )
    assert stat.S_IMODE(cached_stat.st_mode) != stat.S_IMODE(
        source_stat.st_mode
    )
    assert (cached_stat.st_uid, cached_stat.st_gid) == (
        temporary_stat.st_uid,
        temporary_stat.st_gid,
    )
    assert cached.read_bytes() == source.read_bytes()
    connection = duckdb.connect(
        str(cached), read_only=True, config={"threads": 1}
    )
    try:
        assert connection.execute("SELECT value FROM sample").fetchone() == (
            7,
        )
    finally:
        connection.close()

    original_open = Path.open

    def no_source_read(path, *args, **kwargs):
        if path == source:
            pytest.fail("hot cache must not read source contents")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_source_read)
    assert db_engine._get_duckdb_read_cache_url(url) == cached_url
    assert db_engine._get_duckdb_read_cache_url(cached_url) == cached_url
    cached.unlink()
    with pytest.raises(FileNotFoundError):
        db_engine._get_duckdb_read_cache_url(cached_url)


@pytest.mark.parametrize("failure", ["exception", "short", "source-change"])
def test_duckdb_copy_failure_keeps_previous_copy(
    tmp_path, monkeypatch, failure
):
    """Pre-publish failures retain the committed copy and remove owned temp."""
    source = tmp_path / "source.duckdb"
    _make_duckdb(source)
    url = URL.create("duckdb", database=str(source))
    cached = Path(db_engine._get_duckdb_read_cache_url(url).database)
    previous = cached.read_bytes()
    stamp = source.stat()
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    real_copy = db_engine.shutil.copyfileobj

    def fail_copy(original, target):
        if failure == "exception":
            target.write(b"partial")
            raise OSError("injected interrupted copy")
        if failure == "short":
            target.write(b"short")
            return
        real_copy(original, target)
        changed = source.stat()
        os.utime(source, ns=(changed.st_atime_ns, changed.st_mtime_ns + 1))

    monkeypatch.setattr(db_engine.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError):
        db_engine._get_duckdb_read_cache_url(url)
    assert cached.read_bytes() == previous
    assert not list(cached.parent.glob(".copy-*.tmp"))


def _make_metadata(path: Path, driver: str, *, legacy: bool = False) -> None:
    writer = db_engine.create_engine(
        URL.create(driver, database=str(path)), readonly=False
    )
    try:
        if legacy:
            TaskDeprecatedVersion1.__table__.create(writer)
        metadata_db.create_tables(writer)
        with Session(writer) as session:
            if legacy:
                task = TaskDeprecatedVersion1(index=0, name="legacy")
                record = session.get(TableInfo, "task")
                record.table_version = "0.0.1"
            else:
                task = Task(index=0, name="current")
            task.update_md5()
            session.add(task)
            session.commit()
            session.add(Episode(index=0, frame_num=1, dataset_begin_index=0))
            session.commit()
    finally:
        writer.dispose()


def test_invalid_duckdb_connect_options_fail_before_copy(tmp_path):
    """An invalid readonly request must not materialize a source replica."""
    source = tmp_path / "source.duckdb"
    _make_duckdb(source)
    with pytest.raises(ValueError, match="read_only"):
        db_engine.create_engine(
            URL.create("duckdb", database=str(source)),
            readonly=True,
            _lock_free_read=True,
            connect_args={"read_only": False},
        )
    assert not (tmp_path / "home" / "datasets").exists()


def _make_reader_root(tmp_path: Path, driver: str) -> Path:
    root = tmp_path / "dataset"
    Dataset.from_dict(
        {
            key: [0]
            for key in (
                "index",
                "episode_index",
                "frame_index",
                "task_index",
                "robot_index",
                "instruction_index",
                "timestamp_min",
                "timestamp_max",
            )
        }
    ).save_to_disk(root)
    _make_metadata(root / f"meta_db.{driver}", driver)
    return root


@contextmanager
def _exclusive_source_lock(path: Path):
    code = (
        "import fcntl,sys; "
        "f=open(sys.argv[1],'rb'); "
        "fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB); "
        "print('LOCKED',flush=True); sys.stdin.read(1)"
    )
    # A write lock needs a writable descriptor, but never writes DB bytes.
    code = code.replace("'rb'", "'r+b'")
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with DefaultSelector() as selector:
            selector.register(process.stdout, 1)
            assert selector.select(timeout=15), "source lock process readiness"
        assert process.stdout.readline().strip() == "LOCKED"
        yield
    finally:
        process.communicate("x", timeout=15)
        assert process.returncode == 0


@pytest.mark.parametrize("driver", ["duckdb", "sqlite"])
def test_native_source_lock_control(tmp_path, monkeypatch, driver):
    """Real source contention blocks ordinary SQL but not selected reads."""
    path = tmp_path / f"source.{driver}"
    _make_metadata(path, driver)
    url = URL.create(driver, database=str(path))
    monkeypatch.setenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", "1")
    with _exclusive_source_lock(path):
        ordinary = db_engine.create_engine(url, readonly=True)
        try:
            with pytest.raises(Exception, match="lock|Lock"):
                with ordinary.connect() as connection:
                    connection.execute(text("SELECT name FROM task"))
        finally:
            ordinary.dispose()
        selected = metadata_db.create_engine(
            url, readonly=True, _lock_free_read=True
        )
        try:
            with Session(selected) as session:
                assert session.scalar(select(Task.name)) == "current"
            effective = Path(selected.url.database)
            assert (effective != path) is (driver == "duckdb")
        finally:
            selected.dispose()


@pytest.mark.parametrize("driver", ["duckdb", "sqlite"])
def test_direct_factories_ignore_env_and_writer_bypasses_policy(
    tmp_path, monkeypatch, driver
):
    """The env alone does not activate direct factories or maintenance."""
    path = tmp_path / f"source.{driver}"
    _make_metadata(path, driver)
    monkeypatch.setenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", "1")
    url = URL.create(driver, database=str(path))
    assert metadata_db.need_upgrade(url) is False
    assert metadata_db.try_upgrade_database(url).database == str(path)
    for factory in (db_engine.create_engine, metadata_db.create_engine):
        engine = factory(url, readonly=True)
        try:
            with Session(engine) as session:
                assert session.scalar(select(Task.name)) == "current"
            assert engine.url.database in (None, "", str(path))
        finally:
            engine.dispose()
    writer = db_engine.create_engine(url, _lock_free_read=True)
    try:
        with writer.begin() as connection:
            connection.execute(text("UPDATE task SET name='written'"))
    finally:
        writer.dispose()
    assert not (tmp_path / "home" / "datasets").exists()


@pytest.mark.parametrize("driver", ["duckdb", "sqlite"])
def test_metadata_upgrade_keeps_original_and_uses_existing_cache(
    tmp_path, driver
):
    """Real legacy rows upgrade locally without changing the raw database."""
    path = tmp_path / f"legacy.{driver}"
    _make_metadata(path, driver, legacy=True)
    original = path.read_bytes()
    engine = metadata_db.create_engine(
        URL.create(driver, database=str(path)),
        readonly=True,
        _lock_free_read=True,
    )
    try:
        assert "upgrade_cache" in Path(engine.url.database).parts
        with Session(engine) as session:
            task = session.scalar(select(Task))
            assert task.name == "legacy" and task.info is None
        assert metadata_db.need_upgrade(engine) is False
        if driver == "duckdb":
            assert (
                db_engine._get_duckdb_read_cache_url(engine.url) == engine.url
            )
    finally:
        engine.dispose()
    assert path.read_bytes() == original


def test_sqlite_uri_and_live_thread_pool_contract(tmp_path):
    """Immutable reads reuse pooled connections across threads and reopen."""
    path = tmp_path / "literal ?#%.sqlite"
    _make_metadata(path, "sqlite")
    engine = db_engine.create_engine(
        URL.create("sqlite", database=str(path)),
        readonly=True,
        _lock_free_read=True,
    )
    assert isinstance(engine.pool, QueuePool)
    assert engine.url.database == str(path)
    start = threading.Barrier(2)
    first_done = threading.Event()
    connections = []
    checkouts = []

    @event.listens_for(engine, "connect")
    def record_connection(connection, _record):
        connections.append((id(connection), threading.get_ident()))

    @event.listens_for(engine, "checkout")
    def record_checkout(connection, _record, _proxy):
        checkouts.append((id(connection), threading.get_ident()))

    def read(index):
        start.wait(timeout=15)
        try:
            if index == 1:
                assert first_done.wait(timeout=15)
            with Session(engine) as session:
                assert session.scalar(select(Task.name)) == "current"
            if index == 0:
                first_done.set()
        finally:
            first_done.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(read, index) for index in range(2)]
            for future in futures:
                future.result(timeout=45)
        assert len({thread for _, thread in checkouts}) == 2
        assert len({connection for connection, _ in checkouts}) == 1
        assert len(connections) == 1
        # Dispose from the caller thread, then verify fresh native handles
        # still use the escaped immutable URI.
        engine.dispose()
        for _ in range(2):
            with Session(engine) as session:
                assert session.scalar(select(Task.name)) == "current"
                session.connection().invalidate()
        assert len(connections) == 3
    finally:
        engine.dispose()
    explicit = db_engine.create_engine(
        URL.create("sqlite", database=str(path)),
        readonly=True,
        _lock_free_read=True,
        poolclass=NullPool,
    )
    try:
        assert isinstance(explicit.pool, NullPool)
        with Session(explicit) as session:
            assert session.scalar(select(Task.name)) == "current"
    finally:
        explicit.dispose()


def test_sqlite_default_read_keeps_thread_check(tmp_path):
    """Disabled mode retains the old pool and native thread restriction."""
    path = tmp_path / "ordinary.sqlite"
    _make_metadata(path, "sqlite")
    engine = db_engine.create_engine(
        URL.create("sqlite", database=str(path)), readonly=True
    )
    try:
        assert isinstance(engine.pool, SingletonThreadPool)
        with engine.connect() as connection:
            native = connection.connection.driver_connection
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(
                    sqlite3.ProgrammingError, match="same thread"
                ):
                    pool.submit(native.execute, "SELECT 1").result(timeout=15)
    finally:
        engine.dispose()


@pytest.mark.parametrize("driver", ["duckdb", "sqlite"])
def test_rodataset_graph_and_pickle_keep_policy(tmp_path, monkeypatch, driver):
    """Path readers select once; views and pickle retain the resource bool."""
    root = _make_reader_root(tmp_path, driver)
    monkeypatch.setenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", "1")
    original = RODataset(str(root))
    selected = original.select([0])
    multi = ROMultiRowDataset.from_dataset(
        original, ColumnIndexOffsetSamplerConfig(column_offsets={"index": [0]})
    )
    assert original._db_resource is selected._db_resource is multi._db_resource
    assert original._db_resource._lock_free_read is True
    payload = pickle.dumps((original, selected, multi))
    transport = tmp_path / "reader.pkl"
    transport.write_bytes(payload)
    child_env = os.environ.copy()
    child_env["ROBO_ORCHARD_DB_LOCK_FREE_READ"] = "0"
    for key in list(child_env):
        if key.startswith("COV_CORE_") or key == "COVERAGE_PROCESS_START":
            child_env.pop(key)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pickle,sys; "
                "r=pickle.load(open(sys.argv[1],'rb')); "
                "assert r[0]._db_resource._lock_free_read is True; "
                "assert r[0]._db_resource is r[1]._db_resource; "
                "assert r[0].episode_num==1; "
                "r[0].close(); print('PICKLE_PASS')"
            ),
            str(transport),
        ],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    assert "PICKLE_PASS" in child.stdout
    monkeypatch.setenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", "0")
    restored = pickle.loads(payload)
    try:
        assert restored[0]._db_resource is restored[1]._db_resource
        assert restored[0]._db_resource._lock_free_read is True
        assert restored[0].db_engine.url == original.db_engine.url
        assert restored[0].episode_num == 1
    finally:
        restored[0].close()
        original.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = selected.db_engine


def test_external_engine_and_old_pickle_do_not_activate(tmp_path, monkeypatch):
    """Wrapping existing engines and old payloads never rereads ambient env."""
    root = _make_reader_root(tmp_path, "duckdb")
    url = URL.create("duckdb", database=str(root / "meta_db.duckdb"))
    external = metadata_db.create_engine(url, readonly=True)
    monkeypatch.setenv("ROBO_ORCHARD_DB_LOCK_FREE_READ", "1")
    reader = RODataset.from_dataset(Dataset.load_from_disk(root), external)
    assert reader.db_engine is external
    assert reader._db_resource._lock_free_read is False
    old = _RODatasetDBResource.__new__(_RODatasetDBResource)
    old.__setstate__({"db_engine_url": url})
    try:
        assert old._lock_free_read is False
        assert old.engine.url.database == str(root / "meta_db.duckdb")
    finally:
        old.close()
        reader.close()


@pytest.mark.parametrize("failure", ["replace", "directory-fsync"])
def test_duckdb_publication_commit_boundary(tmp_path, monkeypatch, failure):
    """Failures retain whole old or new bytes across the replace boundary."""
    source = tmp_path / "source.duckdb"
    _make_duckdb(source)
    url = URL.create("duckdb", database=str(source))
    cached = Path(db_engine._get_duckdb_read_cache_url(url).database)
    old = cached.read_bytes()
    with duckdb.connect(str(source), config={"threads": 1}) as writer:
        writer.execute("CREATE TABLE another AS SELECT 19 AS value")
    new = source.read_bytes()
    assert old != new
    if failure == "replace":

        def reject_replace(*args):
            raise OSError("injected replace failure")

        monkeypatch.setattr(db_engine.os, "replace", reject_replace)
    else:
        fsync = db_engine.os.fsync

        def reject_directory_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("injected directory fsync failure")
            fsync(fd)

        monkeypatch.setattr(db_engine.os, "fsync", reject_directory_fsync)
    with pytest.raises(OSError, match="injected"):
        db_engine._get_duckdb_read_cache_url(url)
    assert cached.read_bytes() == (old if failure == "replace" else new)
    assert not list(cached.parent.glob(".copy-*.tmp"))


def test_parallel_publishers_and_fresh_source_reopen(tmp_path):
    """Independent publishers converge on one complete, refreshable copy."""
    source = tmp_path / "source.duckdb"
    _make_duckdb(source)
    code = (
        "from sqlalchemy import URL; "
        "from robo_orchard_lab.dataset.robot.engine import "
        "_get_duckdb_read_cache_url; import sys; "
        "print(_get_duckdb_read_cache_url("
        "URL.create('duckdb',database=sys.argv[1])).database)"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(source)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=60)
            assert process.returncode == 0, stderr
            results.append(stdout.strip().splitlines()[-1])
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=15)
    assert results[0] == results[1]
    cached = Path(results[0])
    assert cached.read_bytes() == source.read_bytes()
    assert not list(cached.parent.glob(".copy-*.tmp"))
    before = source.stat()
    with duckdb.connect(str(source), config={"threads": 1}) as writer:
        writer.execute("INSERT INTO sample VALUES (8)")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    refreshed = db_engine._get_duckdb_read_cache_url(
        URL.create("duckdb", database=str(source))
    )
    assert refreshed.database == str(cached)
    assert cached.read_bytes() == source.read_bytes()
