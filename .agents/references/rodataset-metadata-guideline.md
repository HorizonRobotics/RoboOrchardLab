# RODataset Metadata Guideline

Use this reference when adding or reviewing RODataset metadata schema,
metadata persistence, metadata parsing, or metadata export behavior.

## Canonical Schema Ownership

- Keep canonical metadata schema models in
  `robo_orchard_lab.dataset.robot.metadata_schema`.
- Treat `schema_id` and `schema_version` as data-format tags, not Python class
  paths. Do not dynamically import schema classes from `schema_id`.
- Use `_ROBaseModel` for strict JSON-compatible nested payloads that do not
  need their own version tag.
- Use `ROVersionedMetadata` for nested or collection-owned records that need
  their own `schema_id` / `schema_version` and JSON dump helper but are not
  parsed as top-level metadata entries.
- Use `ROMetadataSchema` only for top-level canonical storage-entry payloads,
  such as `EpisodeData.info` or `InstructionData.json_content`, that should
  be registered for `parse_registered_metadata_schema(...)`.
- Keep shared episode-level metadata envelopes and polymorphic processing
  record base classes in `robo_orchard_lab.dataset.robot.metadata_schema`.
  Put transform-specific processing record subclasses next to the transform
  module that owns their fields.
- If a polymorphic schema field accepts registered subclasses, make subclass
  registration deterministic through a package or registry import boundary.
  Do not rely on arbitrary call sites happening to import a transform module
  before metadata parsing.
- Schema registries should resolve known tagged metadata locally.
  Dataset-specific legacy parsing belongs at dataset, converter, or transform
  boundaries, not in the canonical schema module.
- `parse_registered_metadata_schema(...)` should only parse canonical tagged
  metadata. `MetadataSchemaNotTagged` is the only parse error that a
  dataset-local adapter may use as the signal to try legacy parsing.
- Partial tags, unknown schema ids or versions, disallowed schema ids, and
  model validation failures mean the value already claims to be canonical but
  is invalid. These cases should fail hard instead of falling back to legacy
  parsing.
- Preserve the useful underlying validation message when wrapping validation
  failures in `MetadataSchemaValidationError`; callers need the field path and
  model error text to distinguish bad canonical data from legacy data.
- New writer paths should construct the Pydantic schema model and dump through
  its public JSON helper, rather than hand-writing tagged dictionaries.

## Field Ownership

- `TaskData.description` is human-readable natural-language text.
- `TaskData.info` stores structured task-level semantic identity. Non-empty
  task info participates in task md5, query, and dedupe behavior. Empty dicts
  should normalize to `None`.
- Do not store source provenance, raw source payloads, or temporary converter
  bookkeeping in `TaskData.info`.
- Do not introduce a shared `ROTaskInfo` schema or register a task-info schema
  id until the stable task-level semantic fields are known. The current
  default standard value is `None`.
- `InstructionData.json_content` is the preferred storage location for
  canonical instruction content, including equivalent descriptions and
  optional subtasks.
- Treat `ROInstructionContent.descriptions` as an unordered set of equivalent
  language expressions. It may be empty; the schema does not define a default
  training prompt. Dataloaders or transforms own any fixed-choice, random
  sampling, skip, or error policy for empty or multi-description inputs.
- Instruction subtasks may have gaps, but must not overlap. Keep
  per-subtask self-contained range validation in the schema; checks that need
  `EpisodeData.frame_num` belong in the dataset, converter, or packer layer.
- `EpisodeData.info` is the preferred storage location for episode-level
  metadata such as timing and media decoding contracts.
- Store sensor-specific media metadata by stable target names, such as camera
  name, rather than by source-internal order or position.

## Validation Ownership

- Keep canonical schema validation self-contained. Do not hide dataset,
  episode, or camera-list checks inside Pydantic context-dependent validators.
- Check external context at the boundary that owns it. For example, compare
  subtask frame ranges to `EpisodeData.frame_num` in the dataset or packer
  layer, and compare depth encoding keys to the actual depth camera list in
  the converter or packer layer.
- Persisted metadata must be strict JSON-compatible data. Reject non-string
  dict keys, non-finite floats, non-JSON objects, and implicit conversions at
  the schema boundary instead of waiting for database or exporter failures.
- Use `extras` only for explicit, JSON-compatible extension data. Keep source
  fields namespaced and do not use `extras` as an untyped second schema.
- If an `extras` collection needs durable typed records, give those records
  a `ROVersionedMetadata` model and parse them through a collection-specific
  helper instead of routing them through the top-level metadata registry.
- For polymorphic record lists, add JSON round-trip tests that store a
  subclass record through the base-field type and load it back as the
  intended subclass.

## Processing History Policy

- Treat processing history as durable episode-level provenance, not as a
  generic log for every transform or converter step.
- Write processing history only when downstream tools need traceability,
  idempotence checks, auditability, or user-visible provenance.
- Keep common processing-history containers in `metadata_schema.py`; keep the
  typed record payload for a specific transform in that transform's module.
- Avoid adding a new public helper in the shared metadata layer for each
  transform. Prefer a common record protocol or base type plus
  transform-owned record construction.

## Legacy Adapters

- Dataset-local adapters may canonicalize old values internally, but their
  caller-visible output must preserve the existing Dataset or transform
  contract unless the task explicitly changes that contract.
- Bare strings, `None`, blank strings, and source-family-specific legacy
  dictionaries are adapter-owned inputs, not shared schema inputs.
- Do not make storage objects such as `Instruction` or `Episode` part of the
  shared metadata schema contract. Boundary adapters should extract
  `json_content` or `info` before parsing.

## MCAP Episode Metadata Export

- `Dataset2Mcap` should preserve episode-level storage metadata through a
  JSON message on `/metadata/episode`.
- The message should use the existing JSON channel, align its log and publish
  time with the episode's first `timestamp_min`, and write exactly the
  caller-visible `Episode` ORM fields such as indices, frame count,
  `truncated`, `success`, and `info`.
- This export is a faithful storage metadata export. It should not parse
  `Episode.info` as `ROEpisodeInfo`, rewrite legacy values, or become a
  canonical metadata migration path.

## Upgrade And Compatibility

- When an ORM metadata field changes persisted identity, update md5/query
  behavior and add an explicit deprecated ORM shape plus table upgrade
  function.
- Do not add compatibility layers or bump schema versions for schema shapes
  that have not entered `master` or another released persistence boundary.
- For detailed table upgrade and legacy fixture rules, use
  `.agents/references/rodataset-upgrade-guideline.md`.
