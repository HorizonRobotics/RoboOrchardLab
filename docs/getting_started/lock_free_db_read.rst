Lock-free DB Read
=================

RODataset can read completed databases without acquiring advisory locks on
their source filesystem. The feature is disabled by default. Enable it before
constructing path-based readers or starting workers::

    export ROBO_ORCHARD_HOME=/local-disk/robo-orchard
    export ROBO_ORCHARD_DB_LOCK_FREE_READ=1

``ROBO_ORCHARD_HOME`` must be writable and reside on a local filesystem.
Unset, empty, and ``0`` disable the switch; other nonempty values raise
``ValueError``. Keep the setting fixed during reader lifetimes. Inputs must
be completed standalone databases and remain unchanged while readers use
them; this feature does not coordinate live writers or recover WAL/journals.

Reading paths
-------------

The policy is selected once by a path-based RODataset constructor, including
the integrated Arrow/Lance readers. Dataset views and multirow readers share
the selection. Pickle restoration preserves it rather than reading the target
process's environment again.

Direct engine, metadata factory, store, maintenance, packaging and writer
calls do not read this switch. Wrapping an already opened external engine
with ``RODataset.from_dataset`` does not reopen or migrate that engine. Code
that explicitly constructs a path-based RODataset, including conversion code,
uses the same reading entrypoint.

* DuckDB uses one current copy per source path under
  ``$ROBO_ORCHARD_HOME/datasets/duckdb_read_cache``. A fresh open from the
  original source compares size and nanosecond mtime; an unchanged copy is
  reused without reading or hashing source contents. Copies preserve only
  source mtime and publish atomically; access time, ownership and permissions
  come from the new local file. Existing engines and their local URLs do not
  refresh proactively. Close the reader and reconstruct from the original
  dataset path to check again. Changes preserving both size and mtime are
  outside this quick check's detection boundary.
* SQLite opens the original file with ``mode=ro&immutable=1`` and does not
  copy it. This skips SQLite locking and change detection; changing an input
  during the connection lifetime can yield incorrect results. Enabled reads
  use SQLAlchemy's file-database default ``QueuePool`` with native
  ``check_same_thread=False``. Each thread checks out its own Session or
  Connection; native connections may be reused sequentially across threads.
  Explicit caller pool choices are respected. Disabled mode retains its
  previous pool and thread checks.

Frame tables, video payloads, memory/server databases and write interfaces
keep their existing storage paths. Existing schema upgrades still write derived
databases under ``datasets/upgrade_cache``; raw databases are not upgraded in
place by reading. Upgrade-key algorithms are unchanged, though the effective
local URL can produce a different URL digest from a previous source URL.

DuckDB thread budget
--------------------

Set the DuckDB execution-thread budget before starting Python::

    export DUCKDB_THREADS=1

This is a Lab startup setting, not a native DuckDB environment variable. It
must be a positive integer; invalid values raise ``ValueError`` when Lab's
database engine module is imported. Unset, it leaves DuckDB's native defaults
unchanged. The value is read once at module import, so changing the environment
afterward does not update the budget.

The setting configures DuckDB's module-level default connection at import
and supplies the default for new DuckDB engines created by Lab's database
factory, both readonly and writable. It also covers temporary metadata schema
checks and default engine reconstruction in spawned workers, which inherit
the environment. It is independent of the lock-free read policy. Each reader
keeps its existing engine ownership and close lifecycle; DuckDB may share the
underlying instance for the same database.

Direct engine callers can override the returned engine's budget at creation::

    from robo_orchard_lab.dataset.robot.engine import create_engine

    engine = create_engine(
        url,
        readonly=True,
        connect_args={"config": {"threads": 2}},
    )

The ``worker_threads`` alias is also accepted. Supplied option mappings are
not modified. An explicit budget only overrides the returned engine, not the
module-level default connection or a temporary metadata schema check. DuckDB's
existing requirement for compatible configuration when sharing an already-open
database instance still applies.

The budget is per database instance, not a total process-thread limit. It does
not resize other existing database instances, configure connections created
outside Lab's factory, or control other libraries' thread pools. In DuckDB
versions that create the default connection during import, applying the budget
shrinks that pool afterward; it cannot prevent the transient import-time peak.
The existing reader pickle format restores engines from their URL and read
policy, using the receiving process's thread budget rather than persisting
arbitrary caller-provided engine options.

Cache ownership
---------------

Local cache publication locks and local DuckDB connection locks remain.
Closing a reader releases its database resources but retains the shared
cache. There is no automatic cache expiry or deletion. Failed copies never
publish partial files. Failure after an atomic replacement, such as directory
fsync failure, is reported even though the complete new copy may be visible;
the opener does not fall back to connecting to the source database.

See the `SQLite immutable URI contract <https://www.sqlite.org/uri.html>`_
for its underlying read-only semantics.
