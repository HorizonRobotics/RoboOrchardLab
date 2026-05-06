# Robot Interactive Env Guideline

Use this reference for stable robot-interactive-env API and observation-contract
guidance in this repository.

## Applicability

Use this guideline when designing, implementing, reviewing, or testing:

- environment `reset()` / `step()` APIs
- observation payload shape and ownership
- env-owned runtime metadata or derived observation helpers
- lab-side env base contracts or optional env capabilities
- boundaries between environment contracts and policy contracts

For rollout-policy ownership, caches, or `policy.act(...)` semantics, use
`.agents/references/policy-guideline.md`.

For RoboTwin-specific reset, seed, task config, or env State behavior, use
`.agents/references/robotwin-env-guideline.md`.

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

## Env Base Boundary

- Use `robo_orchard_lab.envs.base` as the lightweight lab-side supplement to
  `robo_orchard_core.envs.env_base`. Package-internal code should import
  core env base types and small optional env capabilities through this module.
- Keep `envs/base.py` free of concrete env imports and optional simulator
  dependencies. It should stay safe to import from evaluator, policy, and test
  code.
- Do not create a separate module for a single small env capability when the
  capability is just a structural protocol or helper that supplements the env
  base contract. Prefer `envs/base.py` until the topic grows into a distinct
  reusable guidance or implementation area.
- Keep `envs/__init__.py` from importing concrete envs or heavy env modules;
  env package discovery should not trigger simulator dependency imports.

## Env State Boundaries

- If an env supports State-backed episode starts, expose a reset-shaped API
  such as `reset_from_state(state) -> (obs, info)` instead of overloading
  `load_state(state)` to return reset data.
- For reset-boundary State, define the lifecycle scope precisely. A
  post-reset State means reset has completed and no `step(...)` has run yet.
- Validate env State payloads completely before closing or replacing live env
  resources.
- Do not infer env recoverability from generic State method presence alone;
  require an explicit env-domain capability declaration.

## Validation Expectations

- Add focused tests for any new observation contract that downstream code
  depends on, especially when the change introduces new structured metadata or
  cached derived fields.
- When an env exposes compatibility-driven field naming or payload layout,
  document that compatibility rule near the public API so callers do not treat
  it as an accidental bug.
- For State-backed env starts, test both the successful reset-shaped return
  path and negative cases where bad State input must leave the existing env
  usable.
