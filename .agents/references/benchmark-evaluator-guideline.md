# Benchmark Evaluator Guideline

Use this reference when designing, implementing, or reviewing benchmark-level
evaluation under `robo_orchard_lab/policy/evaluator`.

Related guidance:

- Use `.agents/references/policy-evaluator-guideline.md` for single-episode
  evaluator, remote facade, metric surface, and episode-start contracts.
- Use `.agents/references/design-doc-guideline.md` when the benchmark work
  needs a design document or plan-readiness review.

## Layer Boundaries

Keep single-episode evaluator behavior separate from benchmark orchestration.

- `PolicyEvaluator` / `PolicyEvaluatorRemote` own one episode execution,
  environment starts, policy reset, metric runtime, remote transport, and
  evaluator-owned cleanup.
- Generic benchmark backends may own worker capacity, prepare/evaluate
  pipeline mechanics, worker lifecycle, scheduler state, and cancellation.
- Domain benchmark drivers own domain readiness, retry policy, logical
  episode identity, seed or offset advancement, artifact paths, and aggregate
  metric policy.
- Domain benchmark evaluators own user-facing configuration and compatibility
  entrypoints.

Do not put domain-specific task, seed, artifact, or metric aggregation rules
into generic evaluator or generic backend layers.

## User-Facing Concept Model

Introduce benchmark concepts by audience, not as three equal concepts that
every caller must learn upfront.

- `BenchmarkEvaluator` is the normal user entrypoint: provide one policy or
  policy config, run the full benchmark, and return `BenchmarkResult`.
- `BenchmarkDriver` is for benchmark implementers: define domain readiness,
  episode identity, retry, seed or offset advancement, artifact paths, and
  aggregate metric policy.
- Benchmark backends are for framework or advanced implementers: own worker
  capacity, prepare/evaluate scheduling, worker lifecycle, and cleanup. A
  remote backend may also own timeout isolation, worker replacement, and stale
  completion handling.

Public examples should start from a concrete domain evaluator and only mention
driver/backend when explaining how to implement or customize a benchmark.
While benchmark APIs are still evolving, prefer package/class docstrings and
short examples over a large standalone user guide that can drift.

## File Organization

Keep benchmark-level code under
`robo_orchard_lab/policy/evaluator/benchmark/` instead of adding one
`*_benchmark.py` module per environment to the evaluator root.

Use this package shape unless a domain needs a stronger local split:

- `core.py` for benchmark contracts, events, result records, and driver
  protocols.
- `backend.py` for generic backend implementations and backend-owned
  scheduler state.
- `<domain>.py` for domain drivers, benchmark evaluators, and domain config.

Compatibility shims may remain at older module paths, but new domain
benchmark implementations should live in the package.

## Backend And Driver Contract

A generic backend should not require downstream subclassing when callbacks or
protocols can express domain policy.

Prefer a driver protocol for domain policy:

- `has_unfinished_work()`
- `get_ready_jobs(max_jobs=...)`
- `make_attempt_request(job)`
- `on_attempt_prepared(event)`
- `on_terminal_event(event)`
- `result()`

Backend callbacks should be called from one owner context unless the design
explicitly requires thread-aware driver state.

## Backend Config Ownership

Generic backend configs should contain only fields that the backend itself
owns: worker capacity, worker factories, policy inputs, metric factories,
device placement, scheduler behavior, and cleanup policy.

Keep user-facing domain config fields on the domain evaluator config when
they express domain or UX policy. Map them to the true runtime owner during
backend construction. For example, remote reset/rollout timeout defaults
belong on the remote evaluator config, not duplicated on a generic benchmark
backend config.

Before adding a backend config field, check whether the field is only being
copied through to a lower layer. If so, configure the lower layer directly
unless the backend applies an independent default, validation, or per-call
override.

## Scheduler Ownership

For remote or threaded benchmark backends:

- The scheduler loop should be the only owner of worker state.
- Future callbacks should only enqueue completion events.
- Future callbacks should not release workers, replace workers, mutate pending
  jobs, or call driver callbacks directly.
- Use generation or equivalent stale-completion guards when workers can be
  replaced.
- Avoid fixed sleep polling when a queue or future completion can wake the
  scheduler.

## Backend Helper Boundaries

In benchmark backends, helper boundaries should make scheduler and worker
ownership easier to audit.

Good helper boundaries include:

- ready-work scheduling and capacity decisions
- worker future submission
- executor-thread prepare/evaluate bodies
- scheduler-thread event handling
- worker replacement and cleanup

Avoid helper boundaries that only forward one call, rename one operation, or
hide a single attribute conversion. Inline those unless the helper makes a
threading, lifecycle, or failure boundary explicit.

For worker-local cached state, document the concrete behavior behind the
state. For example, an initialized remote evaluator may reuse policy and
metric resources, but prepare must still make env configuration, metric reset,
and env reset behavior explicit.

## Prepare And Evaluate Separation

If a benchmark must reset the environment before rollout, model that as an
explicit prepare stage.

The prepare stage may:

- setup or reconfigure worker runtime
- reset worker metrics
- reset environment
- capture reset info needed by domain policy
- produce a prepared start payload

The evaluate stage should consume the prepared start and avoid a second env
reset.

Keep domain rules about whether prepare is per-task sequential, cross-task
parallel, or fully parallel in the domain driver.

## Worker Metrics

Worker runtime metrics should not be shared live objects across workers.

If aggregate metrics need worker results, prefer crossing the backend/domain
boundary as metric state or computed episode metrics rather than returning a
live mutable metric object.

## Failure Semantics

Separate these outcomes:

- prepare failure
- evaluate failure
- evaluate completed without infrastructure error
- domain/task success or failure

Do not name infrastructure success fields in a way that can be confused with
task success.

Failure events should carry enough identity to retry and report the attempt,
but should not include partial metrics unless the merge semantics are
explicitly designed.

For remote benchmark backends, timeout and worker-lost evaluate failures
should leave the worker actor non-reusable. Replace the worker before
returning the slot to the idle pool. Do not make the generic backend read or
restore metric state directly, and do not retry the logical benchmark attempt
inside the backend. Retry remains driver/domain policy.

## Compatibility

Benchmark migrations should classify old evaluator fields and methods as:

- retained with same semantics
- wrapper around new evaluator
- deprecated no-op
- explicit removal with migration error

Do not map an old timeout, retry, or polling field onto a new field if the
semantics are only approximate.

## Test Strategy

Benchmark tests should keep default validation fast and fake expensive
runtime dependencies.

- Use fake env, policy, metric, and remote evaluator objects for backend and
  scheduler unit tests.
- Test scheduler capacity, stale completion, cleanup, and failure events at
  the generic backend boundary without launching the domain simulator.
- Reserve real simulator, hardware, or remote-service runs for explicit
  integration tests or manual validation unless the task specifically targets
  that integration.
- When a domain evaluator has both local and remote backends, cover the
  default backend path and the explicitly selected non-default backend path.
