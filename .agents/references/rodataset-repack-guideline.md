# RODataset Repack Guideline

Use this reference when adding or reviewing RODataset dataset-to-dataset
copy, rewrite, filtering, or transform behavior, including `repack_dataset`,
`RODatasetRepackTransform`, files under `robo_orchard_lab.dataset.robot.re_packing`,
and transform-style utilities such as camera downscale.

## Ownership And Entry Points

- Treat `robo_orchard_lab.dataset.robot.re_packing` as the owner of
  RODataset copy, rewrite, filtering, and transform orchestration.
- Prefer extending the existing `repack_dataset` surface with optional
  transform inputs over adding parallel public dataset-processing entrypoints
  for behaviors that still read one RODataset and write another RODataset.
- Put transform-specific behavior in the transform module. Keep the generic
  repack runner limited to source reading, chunk iteration, episode buffering,
  index mapping, and transform dispatch.
- Expose transform column requirements through the transform contract instead
  of duplicating column matching or validation policy in each caller.
- Keep source-copy behavior available as the default path. Adding transform
  support should not make ordinary repack slower or more stateful.

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
