# RODataset Reader Materialization Guideline

Use this reference when changing or reviewing `RODataset`,
`ROMultiRowDataset`, `RODatasetItem`, read-time image decoding, batch access,
or the boundary between stored features and user transforms.

## Storage Schema And Runtime Values

- Treat `RODataset.features` as the persisted frame-table schema. Reader-side
  materialization and user transforms may change returned Python values, but
  they do not rewrite this property into a post-transform schema.
- Keep the read order explicit: load the raw frame-table value, apply configured
  storage-feature materialization, expand metadata when requested, and only
  then run the user transform.
- Keep string column access as a raw inspection path. It should not silently
  decode storage features or apply a row transform to an entire column.
- Put storage-owned decode behavior behind the shared materialization seam.
  Subclasses should extend that seam instead of duplicating scalar, list,
  slice, and `__getitems__` entrypoints.

## ImageEncoded Materialization

- Reader-level image decoding is opt-in. Omitting
  `RODatasetImageDecodeOptions` must preserve encoded camera values and existing
  caller behavior.
- `columns=None` means every top-level `BatchCameraDataEncodedFeature` column.
  An explicit column list narrows the selection and must reject empty,
  duplicate, missing, or incompatible columns during reader construction.
- Keep backend choice and decode policy in the typed config so the same
  settings round-trip through JSON and `RODatasetItem` construction.
- ImageEncoded materialization is independent from video or other sidecar
  decoding. Dispatch from the stored feature type, not from column names or an
  unrelated decoder's enable flag.
- A downstream image transform must accept an already materialized
  `BatchCameraData` without decoding it a second time. The caller remains the
  owner of whether that transform is present.

## Batch And Multi-Row Reads

- Preserve one semantic read pipeline across scalar access, list or slice
  access, and PyTorch `__getitems__`. Batch optimization must not change
  transform count, row order, duplicate-index behavior, or `None` padding.
- For sampled columns, project only that stored column. Materialize only the
  sampled ImageEncoded column that was requested; an ordinary signal must not
  trigger camera decoding.
- Deduplicate non-current referenced row ids within a batch and reuse values
  already loaded for current rows. Restore the sampler's requested order after
  the shared read.
- Prefer batched source reads and per-row materialization over repeated scalar
  source access. Keep the materialization call count proportional to unique
  referenced rows rather than the number of sampler slots.
- Preserve the direct-access transform contract: direct scalar/list/slice
  access transforms the assembled result once, while `__getitems__` returns
  individually transformed rows for PyTorch batching.

## Construction And Resource Ownership

- Keep common reader options, including `image_decode_options`, as typed
  `RODatasetItem` fields. Reserve `reader_init_kwargs` for constructor keywords
  owned by a concrete reader and reject attempts to override common fields
  through that mapping.
- Validate the selected reader constructor both when possible during config
  validation and immediately before construction. Do not silently drop
  unsupported reader-specific options.
- A `ROMultiRowDataset` created with `from_dataset(...)` shares the source
  reader's metadata database lifecycle. Preserve that shared terminal close
  state across views and serialization rules.
- A subclass that owns extra resources must not inherit `from_dataset(...)`
  merely for convenience. It should reject cloning until it has an explicit
  ownership and reconstruction contract for every added resource.

## Validation

- Cover config and JSON round trips, automatic and explicit column selection,
  every supported backend, invalid feature selection, and default-off behavior.
- Exercise scalar, list, slice, `__getitems__`, transform, rename/select/view,
  serialization, and close behavior through the same materialization policy.
- Use disjoint current and sampled rows in MultiRow tests and assert decoder
  call patterns. Decoder caches can otherwise hide accidental full-row or
  repeated decode work.
- Pair semantic tests with an image-backed profile before claiming a data
  loading optimization. Video decode timing is not a substitute for measuring
  the ImageEncoded path being changed.
