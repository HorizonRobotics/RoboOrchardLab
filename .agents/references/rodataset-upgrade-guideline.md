# RODataset Upgrade Guideline

Use this reference when adding or reviewing RODataset metadata database table
upgrades, ORM version changes, or compatibility tests for historical
RODataset tables.

## Upgrade Ownership

- Treat table upgrades as explicit persisted-contract changes. When a field
  changes object identity, update the content-md5 fields, query helpers,
  merge behavior, and cache behavior in the same change.
- Keep the current ORM class on the canonical read and write path. Preserve
  old table shapes as dedicated deprecated ORM classes that inherit
  `DeprecatedDatasetORMBase`.
- Register upgrade functions through the table upgrade registry and construct
  the current ORM object from deprecated rows. Do not mix legacy compatibility
  branches into normal packaging or dataset read/write paths.
- For nullable fields added to an identity-bearing table, define the empty
  value explicitly. For task metadata, old rows upgrade with `info=None`, and
  empty dict inputs normalize to `None`.

## Legacy Fixtures

- Prefer fixed old-version dataset fixtures for upgrade tests that must prove
  compatibility with real historical tables.
- Build those fixtures with the legacy ORM/table shape rather than creating a
  current table and temporarily downgrading it.
- Keep legacy fixture databases immutable during tests. Tests may copy them or
  upgrade into a per-test temporary target database, but must not rewrite the
  original `meta_db.sqlite`.
- Include at least one boundary row that represents a historical edge case,
  such as JSON-looking text in a natural-language field, so upgrades prove
  they preserve old data instead of silently interpreting it as new metadata.

## Test Expectations

- Test both the table-shape migration and the caller-visible behavior after
  loading or packaging the upgraded dataset.
- Focused upgrade tests should call the upgrade path into a temporary target
  database when possible, so an existing auto-upgrade cache cannot hide a
  broken upgrade function.
- Verify that upgraded rows recompute and persist the current content md5
  when identity fields changed.
- Verify that querying, merging, and repacking use the current identity fields
  after upgrade and do not duplicate rows that should be equivalent.
- Verify that deprecated ORM classes are only used by upgrade tests or upgrade
  functions, not by new writer or packaging paths.
