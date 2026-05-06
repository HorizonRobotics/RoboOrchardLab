# Policy Evaluator Guideline

Use this reference for `PolicyEvaluator`, `PolicyEvaluatorRemote`,
episode-level evaluation contracts, or evaluator-owned metric surfaces.

## Core Boundaries

- Keep public evaluator facades thin; shared episode execution semantics
  should live in a common runtime layer.
- Preserve local and remote parity for public success and failure semantics.
- Treat episode-level request and result contracts as the stable evaluator
  boundary. Do not let transport-specific details leak through that surface.
- `PolicyEvaluator` may close env instances it creates from env configs.
  Policy and metric contracts do not define `close()`, so generic evaluator
  close/reconfigure paths should detach them rather than duck-typing a close
  lifecycle.
- For env reconfiguration reuse, compare the requested config against the
  current runtime env's own config, for example
  `env.cfg.to_str(format="json")`. Do not maintain a separate evaluator-side
  env config snapshot, deepcopy, or serialized cache for reuse checks; that
  can drift from the true runtime env state.
- Env reuse paths should require an explicit config-backed env contract. If an
  env cannot expose a stable config, do not enable reuse for that path; force
  recreation or fail explicitly instead of adding generic duck-typed fallbacks
  in the evaluator core.

## Episode Start Inputs

- Keep evaluator episode starts explicit and single-shaped. If an env start
  input supports both reset kwargs and State, model it as
  `env_reset_input: dict[str, Any] | State | None`.
- Use `None -> env.reset()`, `dict -> env.reset(**input)`, and
  `State -> env.reset_from_state(state)`. Do not call `env.reset(...)` again
  after a State-backed start.
- Keep policy reset input independent from env reset input. Policy reset
  kwargs should not be bundled into env State or env reset payloads.
- Local `evaluate_episode(...)` and streaming `make_episode_evaluation(...)`
  should share the same episode-start preparation logic.
- Validate unsupported env State starts at the evaluator/env capability
  boundary, not after rollout has already begun.
- If a caller needs to reset the environment before calling
  `evaluate_episode(...)`, model the already-reset start explicitly with a
  prepared-start payload that carries observations and reset info. Do not use
  a bare `skip_env_reset` boolean when rollout still needs a concrete initial
  observation.
- A prepared-start input means "do not call env reset again"; it should not
  skip policy reset unless a separate policy-start contract is explicitly
  designed.
- Keep prepared-start payloads out of `reset_env(...)`; reset APIs should
  represent inputs that actually trigger env reset.

## Env Episode Finalization

- If an env supports episode-local finalization, the generic evaluator should
  call the optional env finalization helper once from the shared episode-loop
  `finally` boundary.
- Do not add evaluator-owned `episode_active`, `episode_finalized`, or
  equivalent lifecycle flags for env cleanup. Env implementations own
  idempotence, no-active safety, and what artifact cleanup means.
- Keep finalization best-effort in the generic evaluator path. Finalization
  failures may be logged, but they should not replace the episode result or
  public execution error from the episode itself.
- For streaming episode generators, document that callers must consume the
  generator to completion or explicitly close it to run timely episode
  finalization.

## Remote Wrapper Boundary

- Keep `PolicyEvaluatorRemote` as a thin Ray facade over `PolicyEvaluator`.
  The remote wrapper should forward evaluator methods except where the public
  remote contract explicitly composes calls, such as non-streaming
  `evaluate_episode(...)` consuming `make_episode_evaluation(...)` to preserve
  rollout-timeout semantics.
- `PolicyEvaluator` owns evaluator recovery through its private snapshot
  export and restore seam. `PolicyEvaluatorRemote` should not expose
  caller-controlled actor snapshot save/restore APIs; after timeout or
  worker loss, higher layers should close or replace the remote wrapper.
- Non-streaming `PolicyEvaluatorRemote.evaluate_episode(...)` may use
  evaluator-private metric checkpoint helpers to preserve local
  `evaluate_episode(...)` metric rollback parity after ordinary execution
  failures. Keep those helpers private and do not expose them as public
  metric recovery APIs.
- After timeout or worker loss, map the Ray failure to the public evaluator
  failure taxonomy and leave restore/replacement decisions to the caller or
  backend. Do not transparently replay the failed episode from the generic
  remote wrapper.
- Do not expose policy and metric child-state capture, composite snapshots,
  or actor snapshot restore as separate public remote concerns. Keep
  recovery either evaluator-private or explicitly owned by a higher-layer
  replace/retry design.
- Per-call timeout overrides should extend existing remote facade methods
  with a keyword argument such as `timeout_s`; do not add parallel
  timeout-specific methods for the same operation.
- Keep reset and rollout timeouts semantically separate when they bound
  different remote calls. Avoid introducing a single episode timeout that
  mixes reset, rollout, metric fetch, and worker lifecycle unless that total
  budget is explicitly the public contract.
- Timeout override is a remote transport concern. Do not encode it in
  episode identity, domain metadata, or benchmark logical episode records.
- `PolicyEvaluatorRemoteConfig` owns the default reset and rollout timeout
  values for remote evaluator calls. Higher-level orchestrators should not
  duplicate those defaults in their own configs; pass `timeout_s` only when
  they intentionally need a per-call override.

## Policy And Metric Recoverability

- Recoverability is an explicit evaluator-owned boundary contract, not a
  best-effort heuristic.
- Keep env recovery recreate-only in the generic evaluator path. Store env
  configuration in reconstruction metadata, but do not introduce generic env
  runtime `State` recovery here unless a dedicated env recoverability design
  exists first.
- Remote metric rollback is only an implementation detail for
  non-streaming `PolicyEvaluatorRemote.evaluate_episode(...)` ordinary
  failures. It should not become a public `get_metric_state()` /
  `set_metric_state()`-style recovery surface.
- The target runtime-state payload is `State`, but any future remote recovery
  should cross the transport boundary as one evaluator-owned recovery snapshot
  rather than separate public metric-state helper names.
- Do not silently treat a metric as recoverable only because it happens to
  expose helper methods. The evaluator should validate the actual
  `EvaluatorMetrics` reconstruction metadata plus delegated `MetricDict` and
  `State` runtime recovery path it intends to use.
- For repository-owned policy recovery, treat `PolicyMixin` plus a stable
  `cfg` as the evaluator-side recoverability contract.
- Capture and restore policy runtime state through
  `get_state() -> State` and `load_state(state)`; keep snapshot payloads as
  canonical `State` objects.
- Do not expand policy-specific runtime-state aliases on policy classes when
  the generic State seam already carries the contract.
- Timeout and worker-lost recovery should stay outside the public remote
  facade. If caller-controlled recovery is reintroduced later, design it
  around one evaluator-owned composite snapshot API rather than separate
  child-specific remote recovery methods.
- Fail early at configuration boundaries only for recovery contracts that the
  current evaluator surface actually owns.

## Metric Surface, Timing, And Merge Model

- Expose one canonical public metric input surface for evaluators:
  `EvaluatorMetrics`.
- Keep `PolicyEvaluator.setup(...)`, `PolicyEvaluatorRemote.setup(...)`, and
  `reconfigure_metrics(...)` single-shaped around `EvaluatorMetrics`. Do not
  keep direct `MetricProtocol | MetricDict` unions on those public APIs.
- Keep `get_metrics()` aligned with the same public surface by returning
  `EvaluatorMetrics`, or `None` before setup.
- Keep `compute_metrics()` single-shaped by always returning
  `dict[str, Any]`.
- Make evaluator timing fully evaluator-owned. Do not declare timing on
  member metrics through `metric_update_timing` or a similar metric-local
  contract.
- Keep the generic evaluator timing surface minimal. Prefer `STEP` and
  `TERMINAL`; do not retain an `EPISODE` timing path unless a repository
  use case truly needs whole-trajectory dispatch.
- Treat `EvaluatorMetrics.from_channels(...)` as the canonical timing source
  of truth.
- Keep `EvaluatorMetrics.update(timing=..., ...)` as the one public runtime
  dispatch entrypoint. Do not keep parallel public `dispatch_step(...)` or
  `dispatch_terminal(...)` surfaces.
- Keep metric-state merge as an optional consumer-facing seam, but do not let
  mergeability reshape evaluator core timing registration.
- Validate invalid channel layouts and normalized recovery setup at
  configuration time, not deep inside the runtime loop.
- Keep evaluator timing, recoverability, and dispatch policy in
  evaluator-owned contracts and runtime helpers rather than in generic metric
  containers.
- Do not require `MetricDict` to mirror evaluator capability validation or
  timing-specific dispatch rules. Generic metric containers may stay
  evaluator-agnostic even when evaluator-owned adapters apply stricter
  capability checks.
- If evaluator recovery relies on `MetricDict`, treat its runtime restore as
  replace-only unless the container explicitly documents a different contract.
  Do not rely on implicit preserve-state apply semantics for member metrics.
- If a higher-level orchestrator needs to merge evaluator-owned metrics across
  remote workers, use `get_metrics()` as a serialized metric snapshot
  boundary. Remote metric snapshots must be treated as copies; mutating them
  does not affect the worker actor. Do not add a parallel public
  `get_metric_state()` / `set_metric_state()` surface unless metric-state
  recovery becomes an explicit evaluator contract.

## Multi-Episode Orchestration

- Keep generic evaluator APIs centered on single-episode execution.
- Domain-specific multi-episode loops, retries, aggregation policy, or
  continue-on-error behavior should live in the owning script or higher-level
  orchestration component instead of a generic multi-episode helper on
  `PolicyEvaluator`.
- If a domain needs multi-episode rollback or checkpoint semantics later,
  design that around the domain's own batching and reporting needs instead of
  reintroducing a generic evaluator helper prematurely.
- Default callback or stop-condition behavior should be resolved at runtime
  rather than by binding class-level defaults to symbols that may not yet be
  defined during import.

## Remote Failure Semantics

- Keep timeout, worker-lost, and generic execution failures distinguishable
  through stable public exception types.
- Apply the same public failure taxonomy to sync APIs and generator
  iteration boundaries.
- `PolicyEvaluatorRemote` should stay a thin synchronous facade. Do not add a
  local FIFO dispatcher or background retry layer unless a new async contract
  is explicitly designed.
- Do not promise multi-thread concurrent use of the same
  `PolicyEvaluatorRemote` wrapper instance beyond Ray actor serialization.
- Keep non-streaming `PolicyEvaluatorRemote.evaluate_episode(...)` built on
  the remote `make_episode_evaluation(...)` stream when `rollout_timeout_s`
  is the public timeout contract. Consume the remote stream to completion and
  fetch metrics afterwards instead of wrapping reset, rollout, terminal
  updates, and metric computation in one remote timeout.
- Local `PolicyEvaluator.evaluate_episode(...)` should restore metric state
  after a failed attempt so later calls do not inherit polluted evaluator
  metrics.
- Remote `PolicyEvaluatorRemote.evaluate_episode(...)` should restore only
  metric state after an ordinary non-timeout, non-worker-lost execution
  failure, then re-raise the public execution error.
- Remote `PolicyEvaluatorRemote.evaluate_episode(...)` should not
  transparently replay after actor execution failure, timeout, or worker
  loss.
- Remote timeout and worker-lost failures should not trigger automatic
  restore inside the remote wrapper. Higher layers should treat the wrapper
  as disposable and close or replace the worker instead of restoring actor
  snapshots through the public remote facade.
- `make_episode_evaluation(...)` should stay explicit about not providing
  transparent rollback or replay semantics in the generic evaluator path.
- Remote wrappers that own disposable actors should define explicit close
  semantics. When an actor may be stuck inside reset or rollout, close paths
  may kill the actor directly instead of sending a graceful close RPC that can
  sit behind the stuck task.
- `KeyboardInterrupt` or caller cancellation should close the wrapper, make
  the actor non-reusable, and re-raise the original interruption rather than
  converting it into an episode failure.

## Validation Expectations

- Cover local and remote parity for evaluator-facing behavior.
- Add focused tests for:
  - timing and recovery validation at configuration boundaries
  - timeout and worker-lost public exception mapping
  - `dict`, `State`, and `None` episode-start inputs for both streaming and
    non-streaming evaluation when that surface changes
  - integration-script behavior when evaluator contract changes affect
    higher-level orchestration such as RobotWin evaluation
