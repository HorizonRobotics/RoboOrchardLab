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
keep their existing paths. Existing schema upgrades still write derived
databases under ``datasets/upgrade_cache``; raw databases are not upgraded in
place by reading. Upgrade-key algorithms are unchanged, though the effective
local URL can produce a different URL digest from a previous source URL.

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
