# Robot Interactive Env Guideline

Use this reference for stable robot-interactive-env API and observation-contract
guidance in this repository.

## Applicability

Use this guideline when designing, implementing, reviewing, or testing:

- environment `reset()` / `step()` APIs
- observation payload shape and ownership
- env-owned runtime metadata or derived observation helpers
- boundaries between environment contracts and policy contracts

For rollout-policy ownership, caches, or `policy.act(...)` semantics, use
`.agents/references/policy-guideline.md`.

## Observation Contract

- Keep `reset()` and `step()` aligned on the same policy-facing observation
  schema unless one explicit adapter boundary intentionally normalizes them.
- Put runtime metadata that is part of the live env state or embodiment in the
  observation contract when downstream action selection depends on it.
- Keep `info` for auxiliary diagnostics, logging, or episode bookkeeping;
  do not make required action-selection inputs live only in `info`.

## Env Ownership

- Let the environment own observation fields that are derived from live env
  state, simulator state, or env-managed metadata.
- If deriving those fields is expensive, env-local caching is acceptable, but
  cache invalidation should stay attached to env lifecycle boundaries such as
  `reset()`, task changes, or other state rebuild points.
- Do not duplicate env-owned runtime metadata in downstream policy configs or
  other caller-side static config once the env itself can provide the source
  of truth.

## Validation Expectations

- Add focused tests for any new observation contract that downstream code
  depends on, especially when the change introduces new structured metadata or
  cached derived fields.
- When an env exposes compatibility-driven field naming or payload layout,
  document that compatibility rule near the public API so callers do not treat
  it as an accidental bug.
