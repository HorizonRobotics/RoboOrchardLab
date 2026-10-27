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

"""The database engine utilities that wrap SQLAlchemy.

For database engine which is used specifically for RoboOrchard dataset,
see :py:mod:`robo_orchard_lab.dataset.robot.dataset_db_engine`.

"""

import hashlib
import json
import os
import shutil
import stat
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Sequence, Type

from filelock import FileLock
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.patches import patch_class_method
from sqlalchemy import (
    URL,
    Connection,
    Engine,
    Table,
    create_engine as _create_engine,
    event,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase

from robo_orchard_lab.utils.env import (
    get_robo_orchard_home,
)

__all__ = [
    "create_engine",
    "create_tables",
    "drop_tables",
    "create_temp_engine",
]

logger = LoggerManager().get_child(__name__)


def create_tables(
    engine: Engine,
    base: Type[DeclarativeBase],
    tables: Sequence[Table] | None = None,
    checkfirst: bool = True,
):
    return base.metadata.create_all(
        engine, tables=tables, checkfirst=checkfirst
    )


def drop_tables(
    engine: Engine,
    base: Type[DeclarativeBase],
    tables: Sequence[Table] | None = None,
    checkfirst: bool = True,
):
    """Drop tables in the database.

    Args:
        engine (Engine): The database engine.
        base (Type[DeclarativeBase]): The base class of the ORM models.
        tables (Sequence[Table], optional): The tables to drop. If None,
            all tables in the base are dropped. Defaults to None.
        checkfirst (bool, optional): Whether to check if the tables exist
            before dropping them. Defaults to True.
    """
    base.metadata.drop_all(engine, tables=tables, checkfirst=checkfirst)


@patch_class_method(Connection, "_commit_impl", check_method_name_exists=True)
def _commit_impl(self) -> None:
    """Adhoc patch to support readonly mode for SQLAlchemy Connection."""
    if getattr(self, "_orchard_readonly", False):
        raise OperationalError(
            statement=None,
            params=None,
            orig=RuntimeError("Cannot commit in readonly mode."),
        )
    self.__old__commit_impl()


def _readonly_begin(connection: Connection):
    name = connection.engine.url.get_backend_name()
    if name in ["mysql"]:
        connection.execute(text("START TRANSACTION READ ONLY"))
    elif name in ["duckdb"]:
        connection._orchard_readonly = True  # type: ignore
    else:
        warnings.warn(
            "`START TRANSACTION READ ONLY` is not supported "
            f"for backend {name}. "
            "Use ad-hoc `Connection` path instead."
        )
        connection._orchard_readonly = True  # type: ignore


def _make_readonly_engine(engine: Engine) -> Engine:
    be_name = engine.url.get_backend_name()
    # optimize isolation level
    if be_name in ["mysql"]:
        engine = engine.execution_options(isolation_level="AUTOCOMMIT")
    event.listen(engine, "begin", _readonly_begin)
    return engine


def create_engine(
    url: URL,
    readonly: bool = False,
    json_serializer: Callable[[Any], str] = lambda obj: json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
    ),
    *,
    _lock_free_read: bool = False,
    **kwargs,
) -> Engine:
    """Create a database engine.

    This method is a wrapper of :py:func:`sqlalchemy.create_engine` with
    additional support for readonly mode, and default JSON serializer.

    RODataset may pass the internal ``_lock_free_read`` policy for completed
    files that remain unchanged during reader lifetimes. This function does
    not read the environment. DuckDB reads a shared atomic
    local copy under ``ROBO_ORCHARD_HOME/datasets/duckdb_read_cache``;
    SQLite reads the original using ``mode=ro&immutable=1``. The caller must
    place the home/cache on a writable local filesystem. The returned URL
    names the actual file, also on serialization/reopen; reopening that
    effective URL does not recheck the original DuckDB source. Construct a
    new engine from the source URL to check its size and mtime again.
    SQLite immutable reads use the default file-database pool with native
    cross-thread reuse enabled; callers still own a Session per thread.
    Writable engines and the default disabled policy retain their behavior.

    See :py:func:`sqlalchemy.create_engine` for more details.

    Args:
        url (URL): The database URL.
        readonly (bool, optional): Whether to create a readonly engine.
            Defaults to False.
        json_serializer (Callable[[Any], str], optional): The JSON serializer
            for the engine. Defaults to
            `lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=True).`
        _lock_free_read (bool, optional): Internal RODataset read policy.
            Defaults to False; ignored for writable engines. Direct callers
            do not inherit the environment switch.
        **kwargs: Additional keyword arguments for
            :py:func:`sqlalchemy.create_engine`.

    Returns:
        Engine: The database engine.

    """
    if readonly:
        name = url.get_backend_name()
        lock_free = _lock_free_read
        if name == "sqlite":
            # the following does not work!
            # new_url = url.set(query={"mode": "rodd"})
            assert url.database is not None
            if not os.path.exists(url.database):
                raise IOError(f"Database `{url.database}` does not exist.")
            import sqlite3

            sqlite_uri = "file:" + url.database + "?mode=ro"
            if lock_free and url.database != ":memory:":
                database_path = os.path.abspath(url.database)
                url = url.set(database=database_path)
                sqlite_uri = (
                    Path(database_path).as_uri() + "?mode=ro&immutable=1"
                )

            def creator():
                return sqlite3.connect(
                    sqlite_uri, uri=True, check_same_thread=not lock_free
                )

            # sqlite does not support pool_use_lifo
            if "pool_use_lifo" in kwargs:
                kwargs.pop("pool_use_lifo")
            return _create_engine(
                url if lock_free else "sqlite:///",
                creator=creator,
                json_serializer=json_serializer,
                **kwargs,
            )
        if name == "duckdb":
            if "connect_args" not in kwargs:
                kwargs["connect_args"] = {}
            connect_args = kwargs["connect_args"]
            if connect_args.get("read_only", readonly) is not readonly:
                raise ValueError(
                    "connect_args['read_only'] must be consistent with "
                    "`readonly` argument."
                )
            connect_args["read_only"] = readonly
            if lock_free and url.database not in (None, "", ":memory:"):
                url = _get_duckdb_read_cache_url(url)

        engine = _create_engine(url, json_serializer=json_serializer, **kwargs)
        return _make_readonly_engine(engine)
    else:
        return _create_engine(url, json_serializer=json_serializer, **kwargs)


@contextmanager
def create_temp_engine(
    dir: str = os.path.abspath("./"),
    prefix: str = "duckdb_temp",
    drivername: str = "duckdb",
    **kwargs,
) -> Generator[Engine, Any, None]:
    """Create one temporary database engine in a caller-owned directory.

    The generated file changes only its final ``.tmp`` suffix to the selected
    backend suffix. This preserves a caller directory whose own name contains
    ``.tmp``, such as the standard linked-worktree location.

    Args:
        dir: Existing local directory that owns the temporary database files.
        prefix: Prefix for the generated database filename.
        drivername: SQLAlchemy backend name used for the temporary database.
        **kwargs: Options forwarded to :func:`create_engine`.

    Yields:
        A writable SQLAlchemy engine that is disposed and removed on exit.
    """
    db_fd, db_path = tempfile.mkstemp(dir=dir, prefix=prefix, suffix=".tmp")

    db_root, _ = os.path.splitext(db_path)
    db_file = f"{db_root}.{drivername}"
    tmp_db_url = URL.create(drivername=drivername, database=db_file)
    local_engine = create_engine(tmp_db_url, readonly=False, **kwargs)
    try:
        yield local_engine
    finally:
        local_engine.dispose()
        os.close(db_fd)
        os.unlink(db_path)
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(db_file):
            os.remove(db_file)


def _get_duckdb_read_cache_url(url: URL) -> URL:
    """Publish or reuse one complete local copy of a finished DuckDB file.

    The caller selects the readonly policy. Source access here is ordinary
    stat/read only. A short local FileLock owns freshness checks and atomic
    publication; SQL connections do not hold it. Existing replica/upgrade
    files are used directly. Only source mtime is preserved; other file
    metadata remains local. No source URL or reader lifecycle state is kept.

    Args:
        url (URL): A file-backed DuckDB URL whose input remains immutable
            while readers use it. The configured home must be local/writable.

    Returns:
        URL: The effective local file URL, retaining other URL components.

    Raises:
        OSError: If stat, copy or publication fails. A failure before replace
            preserves the previous copy; a later directory fsync failure may
            leave the complete replacement published but is still surfaced.
        ValueError: If the source is not a regular file-backed database.
    """
    if not url.database or url.database == ":memory:":
        raise ValueError("A file-backed DuckDB URL is required.")
    source = Path(os.path.abspath(url.database))
    datasets_home = Path(get_robo_orchard_home()).absolute() / "datasets"
    cache_root = datasets_home / "duckdb_read_cache"
    resolved_source = source.resolve()
    if any(
        resolved_source.is_relative_to(root.resolve())
        for root in (cache_root, datasets_home / "upgrade_cache")
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        return url.set(database=str(source))

    source_stat = source.stat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(f"Database source is not a regular file: {source}")
    path_hash = hashlib.sha256(os.fsencode(source)).hexdigest()
    cache_folder = cache_root / path_hash
    cache_folder.mkdir(parents=True, exist_ok=True)
    cached = cache_folder / source.name
    with FileLock(cache_folder / "cache.lock"):
        source_stat = source.stat()
        signature = (source_stat.st_size, source_stat.st_mtime_ns)
        if cached.is_file():
            cached_stat = cached.stat()
            if signature == (cached_stat.st_size, cached_stat.st_mtime_ns):
                return url.set(database=str(cached))
        fd, temp_path = tempfile.mkstemp(
            prefix=".copy-", suffix=".tmp", dir=cache_folder
        )
        temporary = Path(temp_path)
        try:
            with os.fdopen(fd, "wb") as target:
                with source.open("rb") as original:
                    shutil.copyfileobj(original, target)
                target.flush()
                after = source.stat()
                target_stat = os.fstat(target.fileno())
                if (
                    signature != (after.st_size, after.st_mtime_ns)
                    or target_stat.st_size != source_stat.st_size
                ):
                    raise OSError(
                        f"Database source changed during copy: {source}"
                    )
                os.utime(
                    temporary,
                    ns=(target_stat.st_atime_ns, source_stat.st_mtime_ns),
                )
                os.fsync(target.fileno())
            os.replace(temporary, cached)
            directory_fd = os.open(cache_folder, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Failed to remove owned cache temp %s", temporary
                )
            raise
    return url.set(database=str(cached))
