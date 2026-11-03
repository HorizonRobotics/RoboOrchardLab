# Iterable Dataset Lifecycle Guideline

Use this reference when changing `DictIterableDataset`, reader offload,
prefetch teardown, or prepared DataLoader worker cleanup. API details remain
in [dataset_ex.py](../../robo_orchard_lab/dataset/robot/dataset_ex.py) and the
[training tutorial](../../docs/tutorials/dataset_tutorial/nonb-04_dataset_training.py).

## Ownership And Progress

- Distinguish the physical reader, finite reader segment, shared immutable
  base selection, and private iteration plan/cursor. A cursor may retain
  progress, but must not keep an inactive reader or its tables and buffers
  alive. Snapshot declared item configuration without carrying runtime caches
  into fresh reader configurations.
- Reuse the existing selection, sharding and resampling semantics when
  reopening a reader. Offload divides one worker-local plan into finite
  residencies; it must not silently generate a different logical selection.
- Commit segment progress only after normal exhaustion and successful owned
  cleanup. Early close and failure do not commit that segment. This cursor is
  not an optimizer/checkpoint resume protocol. A short tail may be merged into
  its preceding residency, so a target batch count is not a strict maximum.
- Keep cached base selections immutable while they are reused. Their files
  may outlive an epoch independently of live readers; do not confuse retained
  index files with retained reader resources or give a reader ownership of
  other iterations' private plans.
- `max_dataset_concurrency` bounds active physical item iterators per mixture
  and worker. Logical groups expand into physical members. Rank/worker count,
  prefetched samples and batches retained by callers remain separate costs;
  this setting does not bound process RSS.
- Keep admission accounting, remaining rows in the active segment, and the
  EOF-committed cursor distinct. They describe different units or commit
  points; merging them can change scheduling or interruption semantics.

## Sampling Evidence

- Validate full-epoch quotas, finite-window logical-source balance, and
  cross-epoch ordering separately. Unchanged totals do not imply repeated
  sample order, and exact epoch quotas do not guarantee a fixed window ratio.
- Ratios attach to logical items or groups; physical members retain their own
  sharding and rounding. Multiple active members must not multiply their
  logical group's intended service share. Measure output rows, accounting for
  dataset-side batches and short or dropped tails, rather than counting yields.
- Compare ordered sample identities across epochs under the supported
  seed/generator contract. Include real restarted non-persistent workers and
  epoch notification when checking epoch reconstruction; retain explicit
  generator ownership rather than assuming every generator resets on epoch.

## Cleanup And Prepared Wrappers

- Close the child iterator stack and dataset owned by a materialized item on
  exhaustion, early close and failure, including partial construction. Keep
  the primary read exception when cleanup also fails.
- A prepared generator can be exhausted while its exception traceback still
  retains a native DataLoader iterator and workers. Use the existing
  `close_dataloader_resources` ownership boundary; close only resources proven
  to belong to that loader, preserving independent peer loaders.
- Do not turn third-party generator frame-local names into a general public
  cleanup protocol. Keep any implementation-specific recovery isolated and
  tested against the actual wrapper path.
- Treat close calls as requests with an observable outcome. They do not prove
  that a native operation can be cancelled or that every thread/worker has
  already terminated. State any blocking-operation limitation explicitly.

## Lifecycle And Memory Evidence

- Use more short physical datasets than the concurrency limit, with explicit
  small selections, so at least one reader must be replaced while iteration
  is still active. Inspect cleanup before final exhaustion; process exit or
  end-of-epoch cleanup alone does not prove mid-iteration release.
- Cover normal replacement, early break, and errors before and after the first
  yield. Retain an independent peer when testing failure cleanup and prove it
  remains usable. Observe the relevant reader, table, prefetch thread or
  worker rather than only counting wrapper generators.
- When changing prepared-loader behavior, exercise real DataLoader workers
  and `Accelerator.prepare()` or the actual prepare helper. Include the
  relevant shuffle/batching/sharding path and worker reuse across epochs;
  a wrapper-shaped mock is not evidence for native worker teardown.
- Keep process-global accelerator configuration out of unrelated lifecycle
  tests. When initialization is not under test, prefer a real prepare helper
  with explicit device/topology; preserve existing owners instead of clearing
  shared singleton state.
- Separate Python reachability/resource release, live allocation growth,
  allocator retention and RSS. Use object/resource liveness and allocation
  evidence to attribute a leak; RSS alone cannot establish one. Do not make
  forced GC or allocator trimming the only evidence for ordinary cleanup.
- Report the measured rank, worker and reader topology and observation window.
  A small lifecycle test does not establish a production RSS ceiling, and a
  Python reference test does not establish absence of native allocations.
- When close or join reports a worker failure, distinguish detection from
  cause. Record the failing PID, its parent and owning loader/iterator, plus
  native traceback or exit evidence where available; a process-wide worker
  signal handler may report a failure from another registered iterator.
- For intermittent worker failures, reproduce with the relevant framework
  versions, parallel test scheduling and coverage settings. Check the actual
  runtime and installed dependencies even when using the same image tag.
  Compare with and without diagnostic instrumentation when timing matters.
  Bounded repeated passes establish only that the failure was not reproduced;
  they do not identify its cause, prove recovery, or justify hiding the error.
