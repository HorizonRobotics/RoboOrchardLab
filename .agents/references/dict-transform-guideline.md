# DictTransform Guideline

Use this reference for stable `DictTransform`-family guidance in this
repository.

## Applicability

Use this guideline when designing, implementing, reviewing, or testing:

- `DictRowTransform` and `DictRowTransformConfig`
- `DictTransform` and `DictTransformConfig`
- `DictTransformPipeline` and `DictTransformPipelineConfig`
- `ConcatDictTransform` and `ConcatDictTransformConfig`
- repository-owned transform subclasses under `robo_orchard_lab.transforms`
- downstream callers that consume transform outputs through `transform(...)`,
  `apply(...)`, or `__call__(...)`

## Family Contract

- For new repository-owned row transforms, prefer the `DictTransform` family
  over ad hoc callables so input-column mapping, `apply(...)`,
  `__call__(...)`, concat composition, and config-driven construction stay on
  one shared contract.
- For new transform composition, prefer explicit
  `DictTransformPipeline.from_transforms(...)`,
  `DictTransformPipelineConfig.from_configs(...)`, or direct pipeline config
  construction. Treat `ConcatDictTransform` as the legacy compatibility
  compose surface.
- `transform(...)` is the semantic transform stage. It may return a `dict`,
  dataclass, or `BaseModel`.
- When the transform output schema is stable, prefer returning a structured
  dataclass or `BaseModel` from `transform(...)`.
- Use a plain `dict` return only when the semantic output schema is genuinely
  dynamic, open-ended, or otherwise awkward to model as a stable structured
  type.
- `apply(...)` is the row-aware application stage. It returns both the
  structured semantic output and the final row dict.
- `__call__(...)` is the compatibility row path. It should stay aligned with
  `apply(...)[1]`.
- `DictRowTransform` is the shared weak interface for callers that only need
  row-aware behavior such as `apply(...)`, `__call__(...)`,
  `mapped_input_columns`, `mapped_output_columns`, or concat composition.
- `DictRowTransformConfig` is the shared weak config interface for callers
  that only need concat composition or pipeline normalization and do not
  depend on leaf-only config fields.
- Prefer `apply(...)[0]` when callers need the structured semantic output of
  the transform.
- Prefer `__call__(...)` or `apply(...)[1]` when callers need the final row
  dict after output-column mapping and row merge.
- Do not bypass row logic by calling `transform(...)` directly unless the
  caller intentionally wants to skip input-column mapping, missing-input
  handling, return-column validation, output mapping, and row merge. When
  doing so, make that boundary explicit near the call site or in tests.

## Schema And Mapping Boundary

- Use `semantic_output_to_dict(...)` when family internals need the
  pre-mapping dict view of a supported semantic output.
- `output_columns` describes the semantic pre-mapping output schema.
- `mapped_output_columns` describes the post-mapping transform-output schema.
- `check_return_columns` validates the pre-mapping dict view derived from the
  `transform(...)` result, not the final row dict.
- `keep_input_columns` affects only the final row dict returned by
  `__call__(...)` and `apply(...)[1]`. It does not change the structured
  semantic output returned by `apply(...)[0]`.

## Concat Contract

- `+` on `DictTransform`, `DictTransformPipeline`, `DictTransformConfig`, and
  `DictTransformPipelineConfig` returns the canonical
  `DictTransformPipeline` / `DictTransformPipelineConfig` surface by default.
- If a legacy `Concat*` instance participates in `+`, normalize the result
  back to the canonical pipeline surface instead of propagating the
  deprecated compatibility subtype.
- When a caller should accept either a leaf transform or a pipeline, type the
  boundary against `DictRowTransform` instead of branching on
  `DictTransform` versus `DictTransformPipeline`.
- When a caller should accept either a leaf transform config or a pipeline
  config, type the boundary against `DictRowTransformConfig` instead of
  maintaining a `DictTransformConfig | DictTransformPipelineConfig` union.
- Treat `DictTransformPipeline` as a lightweight container around existing
  child transforms, not a snapshotting wrapper.
- `+` on `DictTransform`, `DictTransformPipeline`, `DictTransformConfig`, and
  `DictTransformPipelineConfig` builds a new container object but keeps direct
  references to the existing child transforms or child configs.
- When a pipeline is constructed from config, each child transform should
  keep the exact child config object from `cfg.transforms`; child config
  updates still flow through `pipeline[i].cfg`.
- Column metadata is recomputed from the live child config on access. If
  callers mutate `input_columns` or `output_column_mapping`, the next
  metadata read and row application should reflect that new structure.
- Treat `DictTransformPipelineConfig.transforms` as construction metadata.
  Once a live pipeline exists, mutate the pipeline container itself instead of
  expecting `pipeline.cfg.transforms` reassignment to rewire runtime stages.
- Runtime composition and config composition are intentionally not identical
  for stateful transforms. Repeating the same runtime transform instance in a
  pipeline reuses that instance and its runtime state; repeating the same
  config in a pipeline creates one runtime instance per listed config entry,
  even when those entries share the same config object.
- If a transform owns mutable runtime state, decide explicitly whether callers
  should reuse one runtime instance or instantiate one stage per config
  entry, and add a focused test for the intended path.
- `DictTransformPipeline` is the preferred composition abstraction for new
  code.
- `DictTransformPipeline` intentionally stays on the row-aware boundary. Do
  not infer support for leaf-only APIs such as `transform(...)`,
  `input_columns`, or `output_columns` from its support for
  `DictRowTransform`.
- `mapped_input_columns` and `mapped_output_columns` are ordered metadata.
  Preserve the surviving chain order instead of re-emitting those columns in
  hash-dependent or otherwise unstable order.
- `ConcatDictTransform` and `ConcatDictTransformConfig` are thin legacy
  wrappers around `DictTransformPipeline` and `DictTransformPipelineConfig`.
- Legacy `Concat*` remains available from `robo_orchard_lab.transforms.base`
  as a deprecated compatibility surface.
- During the current package-root migration, the canonical
  `robo_orchard_lab.transforms` root and the deprecated
  `robo_orchard_lab.dataset.transforms` wrapper intentionally keep legacy
  concat and supporting base compatibility names in `__all__` so existing
  package-root imports and `import *` callers keep working.
- Treat those root-level compatibility exports as migration shims, not as
  the preferred new import path for repository-owned code.
- Treat `robo_orchard_lab.transforms` as the canonical package root.
  `robo_orchard_lab.dataset.transforms` is a deprecated compatibility wrapper
  that should be phased out over time rather than extended with new API.
- `ConcatDictTransform` remains runtime-compatible with `DictTransform`
  checks for compatibility, even though its implementation is pipeline-backed.
- That compatibility only covers the row-aware path. Do not treat
  `isinstance(concat, DictTransform)` as evidence that leaf-only APIs such as
  `input_columns`, `output_columns`, or `transform(...)` are supported.
- `ConcatDictTransformConfig` does not own leaf-style outer fields such as
  `missing_input_columns_as_none`, `output_column_mapping`,
  `check_return_columns`, or `keep_input_columns`; explicit default values
  remain tolerated for legacy config compatibility, but non-default policy
  still belongs on child transforms.
- Treat earlier child transforms in a concat chain as row-level preparation
  for later stages.
- `ConcatDictTransform.apply(...)[0]` is the structured semantic output from
  the last child transform in the chain.
- `ConcatDictTransform.apply(...)[1]` is the final row dict after the full
  chain.
- Do not treat concat `apply(...)[0]` as a trace of every intermediate child
  output. If a caller needs intermediate raw outputs, add that as a separate
  helper instead of widening the base family contract.

## Testing Expectations

- For new `DictTransform`-family classes, add a focused test that
  `__call__(...) == apply(...)[1]` at the row boundary.
- When a transform returns a dataclass or `BaseModel`, test the structured
  output contract through `apply(...)[0]`.
- When using `output_column_mapping` or `keep_input_columns`, add a focused
  test for the boundary between semantic output and final row dict.
- When mutable payload identity matters, assert identity with `is`, not only
  value equality.
- When concat-specific semantics matter, test the last-child structured-output
  contract explicitly.
- When relying on `ConcatDictTransform` compatibility, add a focused test for
  the exact legacy contract being preserved, including any intentionally
  unsupported leaf-only APIs.
- When maintaining both package roots during a migration, add a parity test
  so the compatibility wrapper cannot silently drift from the canonical root.
- When a deprecated compatibility seam remains supported, test both the
  warning behavior and the preserved runtime behavior.
- When mutable runtime state matters, add a focused test that makes the
  runtime-composition versus config-composition distinction explicit.

## Scaffold

When adding a new transform or transform-focused test, start from:

- `.agents/templates/dict-transform-scaffold.md`
