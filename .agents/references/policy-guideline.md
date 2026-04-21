# Policy Guideline

Use this reference for stable rollout-policy boundary guidance in this
repository.

For environment-owned observation contracts and env-side runtime metadata, use
`.agents/references/robot-interactive-env-guideline.md`.

## Applicability

Use this guideline when designing, implementing, reviewing, or testing:

- policy `act(...)` and `reset(...)` contracts
- policy-local caches or recurrent state
- boundaries between env or source adapters and policies

## Rollout Policy Boundary

- Treat rollout policies as observation-driven boundaries:
  `policy.act(...)` should consume the environment observation contract
  rather than hidden policy-local runtime side channels.
- Test or offline-call helpers may still accept explicit typed carrier inputs
  in addition to raw observations, but keep that support deliberate and avoid
  making hidden cfg fallbacks part of the main rollout contract.

## Policy State Ownership

- Let policies own rollout-control state such as action-chunk caches,
  recurrent hidden state, or other per-episode inference state.
- `reset(...)` should clear or rebuild policy-owned rollout state instead of
  relying on callers to reconstruct the policy object between episodes.
- Do not move source-local normalization or target reconstruction into the
  policy layer when an explicit env or source adapter already owns that work.

## Validation Expectations

- Validate policy-facing rollout invariants at setup time when possible, such
  as whether a wired runtime carries the required adapter family or typed
  contract.
- Add focused tests for the exact rollout contract a policy depends on:
  input contract, reset semantics, cache reuse, and any explicit typed
  carrier support.
