# RoboTwin Env Guideline

Use this reference when designing, implementing, reviewing, or testing
`RoboTwinEnv`, `RoboTwinEnvCfg`, RoboTwin reset/seed behavior, RoboTwin task
config handling, or RoboTwin-specific env State recovery.

For generic env `reset()` / `step()` contracts, use
`.agents/references/robot-interactive-env-guideline.md`. For generic `State`
API rules, use `.agents/references/state-recovery-guideline.md`. For evaluator
episode orchestration, use `.agents/references/policy-evaluator-guideline.md`.

## Reset And Seed Semantics

- Keep reset inputs explicit. Do not add shorthand seed modes that mutate
  hidden env context or make reset behavior depend on prior caller history.
- Keep `seed`, resolved start seed, retry offset, current seed, and
  `episode_id` conceptually separate. Derive current runtime seed from the
  resolved start seed plus offset instead of storing duplicate sources of
  truth.
- Retry or validation logic such as expert/init checking belongs to normal
  reset creation. State restore must not rerun retry logic that can choose a
  different seed or task setup.
- When changing reset arguments, update RoboTwin-specific callers under
  `projects/` as part of the same change.

## Recreate State

- RoboTwin currently supports only reset-boundary env State: after
  `reset()` and before the first `step()`.
- Use `State.config` for a deep-copied `RoboTwinEnvCfg`. Use `State.state`
  only for post-reset runtime payload that cannot be derived cleanly from the
  config.
- Keep the post-reset payload explicit and versioned. It should include the
  env state scope, retry offset, resolved task config, and instruction
  bookkeeping needed to recreate the reset boundary.
- Do not store live RoboTwin resources in State payloads: `_task`, viewer,
  video writer, cached FK helpers, cached robot metadata, raw observation
  frames, or file handles.
- Validate `State.class_type` exactly for recreate payloads. Do not accept a
  broad superclass match unless subclass compatibility is explicitly designed
  and tested.
- Validate outer State metadata and the RoboTwin payload before closing the
  current live task. Bad State input must not destroy the usable env.

## Restore Lifecycle

- Keep `load_state(state)` as a no-return runtime apply API.
- Keep `reset_from_state(state)` as the episode-start API that restores the
  reset boundary and returns the same shaped `(obs, info)` result as
  `reset()`.
- Share restore logic between `load_state(...)` and `reset_from_state(...)`,
  but do not route `reset_from_state(...)` through `reset(...)`.
- Recreate the RoboTwin task from the saved config and apply the saved
  resolved task config directly. Do not call `_check_and_update_seed()` during
  restore.
- Invalidate episode-local caches on reset, restore, and close, including FK
  transforms and observation robot metadata.
- Mark post-reset State capture unavailable after the first `step()` unless a
  simulator-level mid-episode checkpoint contract is explicitly introduced.
- Preserve runtime lifecycle flags in the RoboTwin State payload when they
  affect whether the restored env can step. Restore those flags from State
  instead of assuming a fixed post-reset active value inside shared restore
  logic.

## Episode Finalization

- Treat `_episode_finalized` as the env-local "no active stepable episode"
  state, not only as "a previously active episode was finalized". It may be
  true after construction, close, reset failure, or explicit finalization.
- `finalize_episode()` should be idempotent, stop only episode-local
  artifacts such as video recording, and keep the reusable RoboTwin runtime
  open.
- Mark the episode non-stepable before artifact cleanup starts so cleanup
  failures do not leave the env in an apparently active episode state.
- `step()` should reject no-active-episode states with wording that tells the
  caller how to start or restore an active episode, not wording that assumes
  the only cause was explicit finalization.
- A successful `reset()` may mark the episode active only after reset has
  built the return observation and info successfully.

## Compatibility And Validation

- Document RoboTwin compatibility conventions near the public API when they
  are observable, such as combined dual-arm metadata stored under a RoboTwin
  compatibility key.
- Add focused tests for State capture availability, payload validation,
  mismatched `class_type`, bad State not closing the current task, restore
  avoiding retry logic, and `reset_from_state(...)` observation/info parity.
- When evaluator or script-facing reset inputs change, cover both direct
  evaluator paths and RoboTwin/HoloBrain integration call sites.
