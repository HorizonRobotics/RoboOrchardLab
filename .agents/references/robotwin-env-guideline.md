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
- Treat a caller-provided `offset_seed` as the requested starting offset, not
  a guarantee that the episode will use that exact offset. Expert-trajectory
  validation may reject it and advance through later offsets; two different
  requested offsets can therefore converge to the same accepted runtime
  offset and the same first-frame scene.
- Return the accepted runtime `offset_seed` and seed in reset info. When
  diagnosing scene diversity, compare requested offset, accepted offset, and
  actual seed rather than only the reset argument. Per-candidate failures may
  stay at debug level, but any successful fallback should emit a concise
  summary that includes requested and actual seeds.
- Retry or validation logic such as expert/init checking belongs to normal
  reset creation. State restore must not rerun retry logic that can choose a
  different seed or task setup.
- When changing reset arguments, update RoboTwin-specific callers under
  `projects/` as part of the same change.

## Task Config And Embodiment

- Keep `RoboTwinEnvCfg.task_config_path` as the single public embodiment
  entrypoint. Resolve convenience presets or an explicit YAML path once at
  config construction, then pin the file content so reset, retry, State, and
  remote serialization do not silently observe later filesystem changes.
- Preserve RoboTwin's official embodiment syntax and registry as the
  construction source of truth. Lower one-item combined and three-item split
  configurations through the same validation path for each candidate seed;
  do not add a second RoboOrchard embodiment registry or a parallel public
  `embodiment` option.
- After setup, derive observation, action slicing, and FK topology from the
  actual runtime robot. A combined articulation keeps one compatibility
  `Robot` and one base; split articulations expose truthful left and right
  `Robot` descriptions and two absolute world-frame bases. Do not synthesize
  a combined URDF for split robots or expose the private runtime layout in the
  observation.
- Namespace split joint and frame identifiers inside the existing TF graph so
  identical left and right URDF link names cannot collide. Keep dataset and
  training schemas independent from this live-env topology contract.

## Real Embodiment And FK Validation

- Classify the renderer path before preflight. A GUI or X11-dependent path
  requires a non-empty `DISPLAY` verified with `xdpyinfo -display "$DISPLAY"`
  (or an equivalent X11 probe) in the same execution context. A headless
  Vulkan path does not require `DISPLAY`; construct `SapienRenderer()` in its
  actual execution context and use that result as the renderer gate.
- Run renderer and Vulkan validation outside the sandbox when the sandbox
  cannot reach the GPU or Vulkan device nodes. Preserve the variables used by
  the selected path: `DISPLAY` and `XAUTHORITY` for X11, and, when present,
  `VK_ICD_FILENAMES`, `SAPIEN_VULKAN_LIBRARY_PATH`, and related Vulkan/EGL
  variables. Do not add an X11 requirement to a known headless Vulkan CI path.
- Treat `failed to find a rendering device` or missing Vulkan devices as an
  execution-context failure first. An unset or unverified `DISPLAY` is
  diagnostic only for a headless-capable renderer; it is a failure for an
  explicitly X11-dependent path. Record X11 and renderer probes separately so
  a passing unit test is not mistaken for real-simulator validation.
- When embodiment lowering, runtime topology, joint slicing, EEF frames, or
  lifecycle behavior changes, use a real-simulator matrix that includes at
  least one combined articulation and one split-articulation configuration.
  Include a heterogeneous split pair when the changed path claims to support
  different left and right robots.
- Validate actual-articulation-qpos FK in two stages. First compare the URDF
  FK child-link pose with the SAPIEN child-link pose. Then map the FK pose
  through the end-effector joint's `pose_in_child` and RoboTwin's control-pose
  transforms before comparing it with the native RoboTwin EEF. A direct
  child-link-to-control-pose difference is diagnostic, not a gating error.
- Exercise more than one joint configuration. Use deterministic distributed
  samples inside safe joint-limit margins and assert per-joint coverage or
  span; a single home or nominal pose cannot expose all transform-direction,
  fixed-offset, arm-slicing, or embodiment-specific errors.
- Keep drive-target FK evidence separate from actual-qpos FK evidence. A
  target/runtime difference can be normal during tracking and must not be
  used to reinterpret RoboTwin target fields as measured joint state.
- Run process-global runtime patches, renderer state, and planner-worker
  combinations in isolated processes with bounded timeout and process-group
  cleanup. For an explicitly required real-integration gate, report a
  missing CUDA or renderer capability as a failing prerequisite instead of
  skipping the whole product matrix.
- Pair numerical pose checks with observable-contract checks: Robot and TF
  topology, action widths, reset/step, State recreate, evaluator entrypoints,
  and worker cleanup. Task success rate is not required to prove these
  integration contracts.

## Observation Step Metadata

- Let `RoboTwinEnv` own the live observation clock. Policy-facing
  observations use episode-local `step_index` and seconds-valued
  `step_timestamp`; do not expose dataset-layer names such as `frame_index` or
  `timestamp_min` from the env.
- A successful `reset()` or `reset_from_state()` establishes observation step
  zero. `get_obs()` reuses the current observation step without advancing it,
  and each successful `step()` commits exactly one next observation step.
- Advance the stored observation step only after the next raw observation has
  been formatted successfully. A formatting failure must not make later
  observations skip an index or report a timestamp for an observation that
  was never returned.
- Keep the step-to-time conversion in the env-owned
  `step_index_to_log_time_ns()` boundary, then expose seconds in the
  observation. Do not duplicate the RoboTwin FPS conversion in policies,
  processors, or evaluators.
- Keep observation-cache and clock lifecycle explicit. `reset()` and
  `reset_from_state()` return or establish step zero; `load_state()` may clear
  the cached observation while establishing the restored post-reset clock at
  step zero so a later `get_obs()` can materialize that observation without
  advancing it. Reset failure and close must invalidate both the cached
  observation and its clock. A closed or not-yet-reset env must not
  manufacture a current step.
- Downstream adapters may translate `step_index` and `step_timestamp` into
  dataset-specific frame and timestamp fields, including unit conversion.
  Keep that translation out of the env so the live observation contract and
  persisted dataset schema remain separate ownership layers.

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
- When the saved config enables expert or task-init checking, restore the
  official precheck lifecycle exactly once with the saved task config:
  `setup_demo()` with rendering disabled, `play_once()`, `close_env()`, then
  `setup_demo()` again on the same task Python object. Do not enter seed retry
  logic or run `play_once()` on the final evaluation scene.
- Invalidate episode-local caches on reset, restore, and close, including FK
  transforms and observation robot metadata.
- Mark post-reset State capture unavailable after the first `step()` unless a
  simulator-level mid-episode checkpoint contract is explicitly introduced.
- Persist lifecycle state that cannot be derived, such as whether the episode
  was finalized. Derive post-reset State availability from the restored
  reset-boundary contract instead of trusting a serialized availability flag;
  legacy payload fields may be accepted as ignored compatibility input.

## Runtime Ownership

- Let `RoboTwinEnv` own one full-dispose path for active tasks, setup
  candidates, expert probes, State candidates, planner workers, and task-local
  handles. Match RoboTwin's official evaluation lifecycle for an accepted
  precheck: run `setup_demo()`, `play_once()`, and `close_env()`, evaluate
  success, then run `setup_demo()` again on the same task Python object. This
  preserves task-local fields initialized by `play_once()` and lets split
  embodiments reuse their planner connections while rebuilding the scene.
  Do not create a replacement task, copy selected fields, copy `__dict__`, or
  rerun `play_once()` on the final evaluation scene.
- Fully dispose failed, rejected, cancelled, or otherwise abandoned precheck
  tasks. An accepted precheck remains env-owned through the second setup and
  then follows the ordinary active-task cleanup path.
- A task whose planner workers cannot be confirmed stopped remains under env
  ownership. Before reset or State restore creates another runtime, retry
  pending disposal and fail closed if any worker is still alive.
- Build the resolved task config once per candidate seed and pass that exact
  value through setup and expert validation. Do not lower the same seed twice
  or reconstruct accepted configuration from mutable source files.
- Preserve the primary setup, reset, restore, or close error when cleanup also
  fails; cleanup diagnostics should remain secondary while ownership of any
  unresolved runtime handle stays explicit.

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
- For compatibility fixes that patch external RoboTwin runtime behavior
  without modifying RoboTwin source, keep the patch installation at the env or
  demo-setup boundary. Avoid scattering the same patch across reset, task
  setup, planner calls, and script entrypoints. If the patch mutates
  process-wide classes, document the fresh-process requirement for disabling
  or comparing against original RoboTwin behavior.
- When adapting Curobo pose frames, parse fixed transforms from the same
  runtime artifacts RoboTwin passes to Curobo, such as the Curobo yml and URDF,
  instead of duplicating offsets in RoboOrchard config. Validate that legacy
  translation-only fields, such as `planner.frame_bias`, remain consistent
  with the parsed transform.
- Add focused tests for State capture availability, payload validation,
  mismatched `class_type`, bad State not closing the current task, restore
  avoiding retry logic, and `reset_from_state(...)` observation/info parity.
- When evaluator or script-facing reset inputs change, cover both direct
  evaluator paths and RoboTwin/HoloBrain integration call sites.
