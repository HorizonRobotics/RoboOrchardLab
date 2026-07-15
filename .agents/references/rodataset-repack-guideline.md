# RODataset Repack Guideline

Use this reference when adding or reviewing RODataset dataset-to-dataset
copy, rewrite, filtering, or transform behavior, including `repack_dataset`,
`EpisodePackagingTransform`, and files under
`robo_orchard_lab.dataset.robot.packaging` or
`robo_orchard_lab.dataset.robot.re_packing`, and transform-style utilities
such as camera downscale.

## Ownership And Entry Points

- Treat `robo_orchard_lab.dataset.robot.re_packing` as the owner of
  RODataset copy, rewrite, filtering, and transform orchestration.
- Prefer extending the existing `repack_dataset` surface with optional
  transform inputs over adding parallel public dataset-processing entrypoints
  for behaviors that still read one RODataset and write another RODataset.
- Put transform-specific behavior in the transform module. Keep the generic
  repack runner limited to source reading, chunk iteration, episode buffering,
  index mapping, and transform dispatch.
- Expose transform schema and column requirements through
  `EpisodePackagingTransform.prepare_features(...)` instead of duplicating
  column matching or validation policy in each caller.
- Keep source-copy behavior available as the default path. Adding transform
  support should not make ordinary repack slower or more stateful.
- Keep ordinary copy and transform mode on the same canonical runner when
  practical. Do not keep a parallel helper path, legacy shim, or synonym
  public API unless compatibility for an already-released surface requires it.
- Treat `re_packing/__init__.py` as the public API surface for repack
  entrypoints. Keep orchestration and source helpers in underscore-prefixed
  modules such as `_runner.py` and `_source.py` when callers should not import
  them directly.
- Treat `EpisodePackaging` as the shared source interface and
  `EpisodePackagingTransform` as the shared transform interface. Import them
  from `robo_orchard_lab.dataset.robot.packaging`; do not re-export transform
  contract types from `robo_orchard_lab.dataset.robot.re_packing` or the
  `robo_orchard_lab.dataset.robot` root namespace.
- Prefer one shared episode packaging view or source adapter per semantic
  role. Do not create separate source/repack wrappers around the same
  `EpisodePackaging` behavior unless each wrapper owns a different invariant
  or caller-visible contract.
- For modules that are entirely internal to repack, use a private module name
  to carry that signal instead of mechanically prefixing every implementation
  class and function. Within a private module, still use symbol-level `_` when
  it adds an extra locality or privacy signal.

## Episode Packaging Surface

- Treat `robo_orchard_lab.dataset.robot.packaging` as the canonical package
  for episode packaging contracts and writer implementation. Keep the package
  split by semantic owner: episode contracts and transform composition in
  `_episode.py`, metadata dataclasses and ORM helpers in `_metadata.py`, and
  `DatasetPackaging` writer behavior in `_writer.py`.
- Keep `packaging.__all__` limited to the stable public surface.
  Developer-facing helpers such as `IdentityEpisodePackagingTransform`,
  `ComposedEpisodePackagingTransform`, and `EpisodePackagingView` may remain
  explicit imports from `robo_orchard_lab.dataset.robot.packaging` without
  being part of wildcard import compatibility.
- Do not reintroduce `robo_orchard_lab.dataset.robot._packaging_transform` as
  a parallel private entrypoint. Use
  `robo_orchard_lab.dataset.robot.packaging.ComposedEpisodePackagingTransform`
  when ordinary episode packaging transforms need composition.
- Prefer binding one-shot episode metadata into the existing
  `EpisodePackagingView` over adding a separate cached wrapper. Add another
  wrapper only when it owns a distinct invariant or caller-visible contract.

## Streaming And Frame Selection

- Preserve generator-based chunk and episode iteration. Do not resolve all
  chunks or all selected frames into a list when a streaming iterator can
  preserve the current memory profile.
- Do not pre-validate every chunk before `DatasetPackaging.packaging(...)` if
  doing so would materialize the generator or scan the full dataset twice.
  Prefer fail-fast execution plus cleanup of incomplete output.
- Frame selection belongs to `frame_indices` and the source reader. Keep
  `transform_frame(...)` as a frame-to-frame operation; it should not return
  `None` to skip individual frames.
- Follow the existing `repack_dataset` frame-index normalization and sorting
  policy. Do not add a separate transform-only ordering policy.
- Keep the grouped-by-episode constraint explicit in errors and tests when a
  path depends on processing all selected frames for one source episode
  together.
- Episode transform input and output frames should be `Iterable`, not
  `Sequence`, so transforms can stream through generator pipelines. The runner
  may still drain one transformed episode at the output boundary to preserve
  episode atomicity, but should not materialize the whole dataset.
- Treat `EpisodePackaging.generate_frames()` as the canonical
  developer-facing stream for episode transforms. The frames yielded by
  transforms are payload-only `DataFrame` objects; source frame-table rows and
  source frame indices stay internal to the repack owner.
- Do not add transform mode flags unless the runner consumes them with a
  tested semantic or performance effect. Prefer one clear dispatch path over
  no-op capability signals.

## Episode Atomicity And Failure Semantics

- Transform repack should build a complete target episode buffer before
  yielding it to `DatasetPackaging`. A transform failure should not allow a
  partially transformed episode to become package input.
- Keep this atomicity at the transform runner boundary. Do not change the
  baseline `DatasetPackaging(fail_fast=False)` partial-episode behavior as an
  incidental side effect of transform work.
- When transform mode is fail-fast, rely on packaging/output cleanup for
  incomplete datasets rather than global preflight scans that break streaming.
- If a future transform intentionally supports non-fail-fast partial output,
  redesign the transform-runner feedback contract first. Do not infer a
  written episode from a yielded buffer unless packaging has accepted it.
- `transform_episode(...)` may return `None` to skip the whole selected
  episode. `transform_frame(...)` must return a `DataFrame`; returning `None`
  from a frame-level hook is invalid.
- Episode transforms must preserve the selected row count and row order.
  Adding, dropping, or reordering rows belongs to selector or `frame_indices`
  behavior, not to transform execution. The repack runner validates row count
  before yielding an episode to `DatasetPackaging`; row order is a transform
  contract over the `DataFrame` stream because source-frame identity is not
  exposed through the shared transform interface.
- When a frame-stream transform fails after source episode and selected frame
  offset are known, wrap it at the repack runner boundary in
  `RepackFrameTransformError`. The wrapper should carry
  `source_episode_index`, `frame_offset`, and best-effort
  `source_frame_index`, while preserving the original exception through
  `__cause__` and `original_error`. Keep this exception out of
  `re_packing.__init__` until callers need a stable public import surface.

## Episode Index And Previous Links

- Treat `EpisodeData.index` and `EpisodeData.prev_episode_index` as target
  dataset indices at the packaging boundary.
- When setting `EpisodeData.prev_episode_index`, also set
  `EpisodeData.index` explicitly so packaging can validate the target-index
  space.
- Maintain a source-episode-index to target-episode-index map for episodes
  that were actually written.
- Preserve a previous-episode link only when the source previous episode was
  selected, fully transformed, and written successfully. Clear the link when
  the current source episode is a partial selection or the source previous
  episode was skipped, failed, or partially written.
- Do not assume source episode indices remain equal to target episode indices
  after filtering, skipping, or failed transforms.
- Transform runners may assume source episodes are complete unless a source
  reader explicitly documents partial-source behavior.

## Transform Metadata

- Put transform-specific processing records next to the transform module.
  Keep shared episode-level envelopes and polymorphic record base classes in
  `robo_orchard_lab.dataset.robot.metadata_schema`.
- Do not require every transform to write processing history. Write durable
  processing records only when downstream tools need traceability,
  idempotence checks, auditability, or user-visible provenance.
- Use `.agents/references/rodataset-metadata-guideline.md` for metadata schema
  ownership, registry, serialization, and compatibility rules.

## Validation

- Add focused tests for frame selection, transform failure, skipped episodes,
  partial selections, and previous-episode target-index remapping.
- Include at least one test where source indices and target indices differ,
  such as source episodes `0, 1, 2, 3` with episode `1` skipped and later
  episodes written.
- For transform metadata records, cover Pydantic serialization and
  deserialization through the same stored shape used by `EpisodeData.info`.
- For performance-sensitive transforms, test or inspect the code path that
  avoids unnecessary decode-then-reencode or full-dataset materialization.
- Test that frame-level and episode-level transforms run in user-provided
  order.
- Test that episode transform output cannot change row count.
- Test that old repack-specific transform contract names are not exported from
  `re_packing` and that shared transform contract types stay out of the
  package root namespace.
