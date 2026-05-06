# State Recovery Guideline

Use this reference for `State`, `StateSaveLoadMixin`, `obj2state`,
`state2obj`, runtime recovery payloads, checkpoint/recreate boundaries, and
cross-domain recoverability contracts.

## Applicability

Use this guideline when designing, implementing, or reviewing repository-owned
runtime recovery for policies, metrics, evaluators, envs, processors, or other
objects that need to capture and restore mutable runtime state.

Do not use this file for domain-local lifecycle policy by itself. Domain
guidance still owns when a policy, metric, env, or evaluator is allowed to
capture, refresh, discard, or apply recovery state.

## API Layering

- Keep live runtime recovery, filesystem persistence, and fresh-object
  materialization as separate concepts.
- Treat `StateRuntimeProtocol` as the narrow live-object contract:
  `get_state() -> State` and `load_state(state: State) -> None`.
- Keep persistence helpers such as `save(path)`, `load(path)`, and
  `load_state_from_path(path)` out of the runtime recovery protocol. If
  `load_state(path_or_state)` remains for compatibility, new canonical call
  sites should still pass a `State` object for runtime apply.
- Keep `StateMaterializeProtocol` focused on `state2obj(...)` fresh-object
  materialization. Its `allocate_state_instance()` and
  `apply_decoded_state(...)` hooks should not become evaluator/env runtime
  recovery APIs.
- When a class wants fresh materialization and live recovery to share code,
  share a protected implementation helper such as `_set_state(...)`; do not
  route materialization back through a public `load_state(...)` path that may
  re-decode payloads.

## Canonical Payload

- Treat `State` as the canonical repository-owned recovery payload.
- Treat `State` as a structured recovery snapshot, not as a general-purpose
  replacement for `pickle` / `cloudpickle` instance serialization. It should
  describe explicit capture and restore contracts, not promise transparent
  round-tripping of arbitrary Python object graphs.
- Before adding a new recovery snapshot type, protocol, helper module, or
  `get_xxx_runtime_state()` naming family, first check whether the existing
  `State` / `StateSaveLoadMixin` seam can carry the contract directly.
- Keep domain-specific recovery payloads as compatibility or capability
  wrappers only when they protect an existing public API or express a real
  domain capability boundary.
- Do not promote `cloudpickle` or another opaque serializer into the public
  canonical recovery format. If it is used as a fallback, keep the limitation
  explicit and do not make cross-domain recovery depend on inspecting opaque
  blobs.

## Capture And Apply

- Prefer `get_state() -> State` and `load_state(state: State) -> None` as the
  generic live-object capture/apply pair.
- For repository-owned policies, treat `PolicyMixin.get_state()` and
  `PolicyMixin.load_state(state)` as the canonical policy runtime recovery
  seam. Do not add policy-specific aliases that only rename those methods.
- For repository-owned evaluator metrics, prefer an evaluator-owned wrapper
  such as `EvaluatorMetrics.get_state()` / `load_state(...)` only when it
  delegates to the canonical underlying `MetricDict + State` seam instead of
  inventing a second recovery payload family.
- Treat `_get_state()` and `_set_state(...)` as protected implementation
  hooks, not public recovery surfaces for callers or evaluators.
- If `load_state(path)` exists for persistence compatibility, do not treat the
  path form as the runtime recovery payload contract. Runtime recovery should
  pass a `State` object across the boundary.
- Do not add parallel generic helpers that only rename `get_state()` or
  `load_state(state)` unless they enforce additional domain validation or
  preserve a supported compatibility surface.
- `StateSequence` is a State API transport/persistence support type for
  preserving `list` / `tuple` fidelity inside `State.state`; it is not a
  domain recovery payload. Domain-facing runtime recovery should still pass a
  top-level `State`.
- `StateList` is legacy compatibility only. Do not add new repository-owned
  call sites that construct `StateList` for canonical runtime capture; use a
  normal `list` and let `obj2state(...)` encode it as `StateSequence`.
- Treat nested `State` payloads without `class_type` as apply-only payloads:
  decode them into container-shaped data, then apply them to an existing live
  object through its `load_state(state)` / `_set_state(...)` path.
- Treat nested `State` payloads with `class_type` naming a
  `StateMaterializeProtocol` type as constructable payloads: decode may
  materialize a fresh object, and wrappers that own child replacement should
  document when they replace the existing live child versus applying into it.
- `StateSaveLoadMixin` is the common repository-owned live capture/apply seam,
  and it already satisfies `StateMaterializeProtocol`. The default
  `StateSaveLoadMixin` fallback only supports attribute-backed objects.
  Mutable container-backed objects should override `_get_state()` /
  `_set_state(...)` or provide compatible `__getstate__()` /
  `__setstate__(...)` hooks instead of relying on the fallback.
- If a recovery call must apply a constructable nested payload into an
  existing child instead of replacing it, keep that choice explicit at the
  recovery boundary. State API may preserve such nested payloads as `State`
  for apply, but the owner still decides how to handle the existing child.
- Owners and wrappers should make replace-only versus apply-into-existing
  restore semantics explicit at their public recovery boundary, and reject
  unsupported apply modes there instead of waiting for decoded payload shape
  failures deeper in the implementation.
- Compatibility wrappers that accept `State.state` without inheriting
  `StateSaveLoadMixin` should decode State API transport containers with
  `decode_state_payload_for_apply(...)` before handing payloads to domain
  implementations. Do not leak `StateSequence` into metric, policy, env, or
  evaluator domain state handlers.

## Env-Domain State

- Do not add an env-specific top-level State type. Env recovery should use the
  canonical `State` payload plus an env-domain lifecycle contract.
- Env State participation is explicit capability, not a heuristic inferred
  from `get_state()` / `load_state(...)` method presence.
- Keep generic `load_state(state) -> None` separate from episode-start APIs
  that need reset-shaped return values. Env domains may define
  `reset_from_state(state) -> (obs, info)` for reset-boundary rollout starts.
- Scope env runtime State explicitly, for example as `POST_RESET`, instead of
  assuming every captured env State is a valid rollout start.
- Generic evaluator recovery should not automatically capture env runtime
  State. Add env runtime participation only behind a dedicated env capability
  and evaluator contract.

## Persistence Profiles

- Treat `save_profile=None` as "no explicit profile selected". At a root save
  boundary it should select by profile `root_save_priority`; at a nested save
  boundary it should inherit the effective parent profile.
- Keep profile priorities separate: `root_save_priority` is only for default
  root save selection, while `load_priority` is only for artifact detection
  during load.
- Keep `StateSequence` as a transport type and `StateList` as legacy
  compatibility. They may remain import-compatible, but they should not be
  promoted as primary package-root `__all__` APIs.

## Recreate Boundary

- Separate live-object apply from fresh-instance recreate.
- A domain or wrapper that declares recoverability owns the recreate input,
  such as config, factory, checkpoint source, device, or other construction
  metadata.
- For evaluator recovery, prefer one evaluator-owned recovery snapshot and
  recovery manager over parallel wrapper-side policy-state and metric-state
  orchestration helpers.
- When evaluator replay semantics are needed, keep transient replay request
  context separate from the stable evaluator recovery snapshot. Replay inputs
  are operation context, not stable reconstruction metadata.
- For evaluator-owned metric wrappers such as `EvaluatorMetrics`, keep timing
  layout and channel membership as reconstruction metadata owned by the
  wrapper. Remote recreate may also require that wrapper metadata to be
  cloneable as evaluator-owned recovery input, while member runtime state
  still flows through the delegated `MetricDict + State` seam.
- In generic evaluator recovery, keep env participation reconstruction-only
  unless an explicit env runtime recoverability contract exists. Do not
  implicitly promote env runtime state into the generic evaluator snapshot
  shape.
- `state2obj(...)` may be used as an object materialization helper, but its
  presence does not make it the owner of every domain's recovery policy.
- Preserve reconstruction-critical metadata at the recovery boundary unless a
  live object is explicitly documented as the new source of truth.

## Capability Discipline

- Recoverability is explicit capability, not a heuristic inferred from
  method presence.
- Do not treat an object as recoverable only because it implements
  `get_state()`, `load_state(...)`, or inherits `StateSaveLoadMixin`.
- Policy recovery is the repository-owned exception: `PolicyMixin` plus a
  stable construction input such as `cfg` is the explicit recoverability
  contract for rollout policies in this repository.
- Callers that need rollback, timeout recovery, worker recreation, or
  cross-episode state preservation must require a domain-level recoverability
  contract at configuration or boundary validation time.
- For generic evaluator failed-episode handling, prefer restoring canonical
  metric `State` directly at the episode boundary rather than widening stable
  evaluator recovery payloads unnecessarily.

## Validation Expectations

- Test the canonical `State` round trip for changed capture/apply behavior.
- When a compatibility method remains supported, test that it delegates to or
  stays aligned with the canonical `State` path.
- When recoverability is capability-gated, test that method presence alone is
  not enough to enter the recovery path.
- For recreate paths, test that construction metadata and runtime state remain
  distinct and both survive the intended recovery boundary.
