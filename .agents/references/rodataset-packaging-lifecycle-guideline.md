# RODataset Packaging Lifecycle Guideline

Use this reference when changing or reviewing local RODataset write paths,
`DatasetPackaging`, `DatasetPackagingPaths`, packaging workspaces, same-target
coordination, cleanup, or final publication. For dataset-to-dataset transform
semantics, also read `rodataset-repack-guideline.md`.

## Path Ownership

- Treat `DatasetPackagingPaths.resolve(dataset_path)` as the single owner of
  local output normalization and all top-level packaging path derivation.
  Direct writers, repack staging, and external converter preflight should
  consume its `dataset_dir`, `output_roots`, or `write_roots` instead of
  recreating suffixes, cache paths, or normalization helpers.
- Reject URI-style output before expansion or absolute-path normalization.
  Preserve ordinary local path compatibility, including `os.PathLike`, local
  names containing `:`, and Windows-style drive paths.
- Reject symbolic-link write roots. Keep the caller's requested path and the
  canonical real target distinct long enough to detect parent-directory
  alias changes while waiting for the coordination lock.
- Keep path resolution and ordinary preflight non-mutating. The writer owns
  directory creation, stale-owned-path removal, and final cleanup.

## Workspace And Lock Placement

- Put Hugging Face output, cache, incomplete state, and HF-created lock files
  below the target-scoped packaging workspace. Ordinary success and handled
  failure paths must remove that workspace so the user-provided dataset
  directory and its parent remain clean. If cleanup itself fails, report that
  failure; do not relocate transient artifacts into the user output tree.
- Place the stable same-target coordination lock outside the user output tree,
  using `XDG_CACHE_HOME` or another writable cache location. Do not derive it
  beside the requested dataset merely for convenience.
- Writers that target the same shared filesystem from different hosts
  coordinate only when their lock-cache location is shared consistently. If
  that assumption cannot be satisfied, do not describe the local file lock as
  cross-host serialization.
- Acquire the coordination lock before mutating owned output roots. Recompute
  target identity and recheck overwrite preconditions under the held lock;
  an earlier preflight is diagnostic evidence, not write authorization.

## Cleanup And Failure Semantics

- Cleanup methods should remove only paths owned by the active packaging or
  staging lifecycle. Keep generic symlink-safe removal and atomic filesystem
  primitives in `robo_orchard_lab.utils.filesystem`; keep lifecycle policy in
  the writer or staging-session class that owns those paths.
- Always attempt workspace cleanup. When a packaging exception is already
  active, report cleanup failure without replacing the primary error. When no
  primary error exists, propagate cleanup failure rather than reporting a
  clean successful return.
- Do not conflate direct packaging overwrite with repack replacement. Direct
  `DatasetPackaging(..., force_overwrite=True)` may remove the old target
  before generation and does not promise rollback. Repack uses an outer
  sibling staging session and must restore an owned prior target if final
  replacement fails.

## Atomic Publication

- Publish completed same-filesystem staging output with an atomic no-replace
  rename. An existence check followed by ordinary POSIX `os.rename` is not a
  safe substitute because a concurrently created empty target directory may
  be replaced.
- Fail closed on platforms without a supported atomic no-replace primitive.
  Do not silently fall back to a clobbering rename.
- If the target appears after direct packaging starts, preserve it and fail.
  For repack, if the target appears, disappears, or changes identity after the
  outer staging session starts, preserve the unowned state and fail.
  `force_overwrite` authorizes repack to replace the target identity observed
  at session start; it does not authorize overwriting a different artifact
  created by another process.

## Hugging Face Config Identity

- The fixed `config_id="robo_orchard"` is safe only while every packaging call
  owns a fresh, disposable `hf_cache_dir` inside its workspace and same-target
  calls are serialized. Its purpose is to prevent Hugging Face from
  fingerprinting a streaming or unpickleable episode generator.
- If the HF cache becomes persistent, shared between runs, or reusable across
  different inputs, replace the fixed ID with an identity that distinguishes
  every cache-relevant input. Removing the explicit ID is not automatically
  equivalent: it restores Hugging Face generator fingerprinting and may fail
  for resource-owning or unpickleable episode iterables.

## Validation

- Cover local path forms, URI rejection, symbolic-link roots, and canonical
  aliases separately from mutation tests.
- Verify success and failure cleanup by checking the dataset parent and
  packaging workspace for transient HF artifacts. Do not assert only that the
  final dataset exists.
- Exercise same-target serialization and a target-appears race so publication
  proves no-replace behavior rather than only preflight behavior.
- Test direct overwrite and repack rollback as different contracts.
- When changing fixed config ID or cache placement, include a streaming or
  otherwise unpickleable episode source in the focused regression coverage.
