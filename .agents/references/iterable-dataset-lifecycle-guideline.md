# Iterable Dataset Lifecycle Guideline

Use this reference when changing `DictIterableDataset`, reader offload,
prefetch teardown, or prepared DataLoader worker cleanup. API details remain
in [dataset_ex.py](../../robo_orchard_lab/dataset/robot/dataset_ex.py),
[sampling_strategy.py](../../robo_orchard_lab/dataset/robot/sampling_strategy.py), and the
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

## Mixture Strategy Boundary

- `DictIterableDataset(sampling_strategy=None)` selects
  `DefaultMixtureSamplingStrategyConfig`. Shuffled defaults use live logical
  remaining-quota admissions and randomized initial finite-residency budgets.
  This intentionally changes shuffled output order and RNG consumption from
  the earlier default. Some tail windows can have larger mixture errors;
  exact epoch quotas do not imply fixed-window ratios or training benefits.
- Use the public `dataset.robot.sampling_strategy` module for extension.
  Subclass `MixtureSamplingStrategy` and give it a typed
  `MixtureSamplingStrategyConfig` subclass whose `class_type` uses `ClassType`.
  Pass configuration to the dataset; do not pass or share an active runtime.
  Keep custom runtime and config classes at importable module scope for the
  existing `ClassConfig` JSON serialization path.
- Each worker-local mixture iteration creates its own runtime as
  `MixtureSamplingStrategy(cfg, generator=generator, view=view)`. The injected
  generator is the existing scoped sampling stream, not a second seed or an
  independently reset stream. Source-local shuffle and resampling remain
  `DatasetItem` responsibilities.
- `choose_admission(view)` returns a pending physical item ID;
  `choose_active(view)` returns an active physical item ID. The live view is
  read-only: targets and logical membership are stable, while pending/active
  membership, remaining output and residency counts change during iteration.
  Strategies choose among those candidates; they do not open or close readers,
  change quotas, consume rows, advance cursors, or decide EOF.
- The optional `initial_residency_rows(item_id, *, rows, batch_size)` hook
  inherits an identity implementation. It requests a raw initial cursor
  budget, not a quota or cursor mutation. The engine validates and applies it
  only for positive-output candidates in the first shuffled native-offload
  fill. `view.initial_residency_slots` is zero outside that phase; without
  offload it is zero from construction. Non-identity budgets must be positive,
  bounded by the original budget and batch-aligned when it spans a batch.
  An original budget smaller than one batch stays intact, as does the tail
  behavior of an identity return.
  If all worker-output quotas are zero, slots must be zero and the default
  must not prefetch budget uniforms. Verify the strategy-initialization RNG
  state rather than empty output alone; zero-output reader initialization
  may still legitimately use the parent seed.
- The engine owns reader construction, the concurrency bound, output
  accounting, EOF checks and cleanup. It calls
  `residency_completed(item_id, requeue=...)` after normal residency completion
  and owned cleanup. The callback can update scheduling state; it does not
  commit progress or initiate reopening. Early close and failure are not
  normal completion.
- Custom strategies require `shuffle=True`; combining one with
  `shuffle=False` fails during construction. Ordered iteration keeps its
  existing loop and consumes no scheduling RNG. Source-local shuffle/resampling
  and full-epoch quotas retain their existing owners; sharing the scoped stream
  does not preserve the earlier policy's shuffled row sequence.

## Strategy And Epoch Replay

- `MixtureSamplingStrategyConfig.supports_epoch_replay` defaults to `False`.
  Treat custom strategy reconstruction as unknown until the implementation
  explicitly guarantees deterministic decisions from its configuration, the
  injected stream and the supplied view. Include every decision-affecting
  option in the serialized configuration; external mutable state or extra RNG
  streams do not become recoverable through this flag.
- The default config declares replay support. A custom declaration only makes
  the strategy eligible for existing worker-prefix recovery; the loader must
  still satisfy its seed, source identity, topology and other capability
  checks. Serialized strategy identity and configuration participate in
  recovery compatibility. Shuffled recovery also binds scheduling identity and
  revision; legacy shuffled layouts are rejected. Ordered states retain their
  earlier layout compatibility. The flag is not a runtime-state checkpoint API.
- Offload, sample prefetch, explicit dataset generators and other unsupported
  recovery configurations retain their existing unknown/best-effort status.
  A replay-capable strategy must not silently promote them to exact recovery.

## Sampling Evidence

- Before changing a mixture scheduler, record target proportions, the output
  unit, the global logical-batch definition, and primary and regression
  metrics. Keep physical quotas and semantic workload aligned across
  comparisons. Validate full-epoch quotas, finite-window source balance and
  ordering separately; unchanged totals do not imply repeated sample order.
- For an extraction that promises unchanged behavior for a selected policy,
  pin its reference and compare complete `(source, row)` traces, per-yield
  batch boundaries and final shared-generator states with the new version.
  Run both in the same dependency environment and record its identity.
  External-library random sequences can change across versions. CI should
  check real-backend repeatability, implicit/explicit defaults, exact row
  quotas and batch boundaries, RNG isolation and controlled-input numeric
  weights, rather than pin an external library's concrete random sequence.
- Aggregate source counts across ranks before evaluating global-batch
  proportions. Include representative consecutive-batch windows, the beginning
  and end of the epoch, and source-order correlation. State the actual batch
  and worker topology; not every configured worker contributes to one step.
- Separate global mixture quality from each worker's reader residency spans
  and intervals without active readers. Smooth interleaving can lengthen
  spans; use these as hard requirements only when that local guarantee is an
  agreed target. Reader event counts and clocks measured in output rows do
  not establish I/O latency, memory use or throughput.
- Before claiming improved balance, report each logical source's mean, P95 and
  maximum window error and any failed regression gates. Show source-level
  regressions alongside aggregate improvements. Over the same tail interval,
  report pooled source proportions separately from sliding-window deviations;
  neither substitutes for the other.
- Compare representative seeds and reader-slot limits under the same recipe.
  A larger pool is a comparator, not a promise of better finite-window balance
  or a substitute for validating the intended small pool.
- Ratios attach to logical items or groups; physical members retain their own
  sharding and rounding. Multiple active members must not multiply their
  logical group's intended service share. Measure output rows, accounting for
  dataset-side batches and short or dropped tails, rather than counting yields.
- Compare ordered sample identities across epochs under the supported
  seed/generator contract. Include real restarted non-persistent workers and
  epoch notification when checking epoch reconstruction; retain explicit
  generator ownership rather than assuming every generator resets on epoch.
- Validate actual source sequences and candidate ownership, not only distinct
  seeds. A deterministic scheduling priority can dominate random tie-breaking.
  For shuffled native shards, test rank and worker diversity; for
  `IterableDatasetShard` fallback, test a common stream for corresponding
  workers across ranks and exact positional coverage. Preserve ordered mixing
  and its scheduling-RNG behavior. Exercise both seed-managed and explicit
  generators through real prepared workers.
- Retain each candidate's source identity, inputs, metric definitions and raw
  counts, including rejected variants. Separate observations, hypotheses,
  controlled tests and ruled-out explanations. Whole-version comparisons do
  not isolate one cause when several behaviors changed.
- A vectorized or synthetic-reader replay establishes only its stated model.
  Keep RNG-consumption, worker-arrival and omitted-prefetch limits explicit;
  validate production boundaries separately before claiming training or
  resource benefits.

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
