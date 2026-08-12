---
description: Load these instructions when planning complex repository work, validating changes, or working with repository workflows, tests, documentation builds, or developer tooling.
---

# Workflow and Validation Instructions

## Sources of Truth

- Use `Makefile` when a relevant target exists.
- Use `pyproject.toml` and pytest config for tool behavior.
- Prefer source files over `build/`; use `build/` only for debugging generated output.
- If workflow files disagree, report the mismatch instead of guessing.

## Scoped Worktree Baseline

- Before implementing a non-mechanical or planned change, turn the approved
  scope into exact existing and new target paths. Capture the staged,
  unstaged, and untracked baseline for those paths; a repository-wide dirty
  summary is not a substitute for this check.
- If a target path already has changes, inspect the relevant diff and tell the
  user before editing. Call it a *pre-existing change*, not a user change:
  its author cannot be inferred reliably from status alone.
- Block for direction when a pre-existing change overlaps the intended hunk or
  prevents safe scope isolation. When it is demonstrably non-overlapping,
  record that it is excluded and leave it untouched; do not flood the user
  with unrelated worktree dirt.
- Repeat the scoped check before handing work to review or staging it. Stage
  only the approved paths, even when unrelated staged or unstaged changes are
  present. If an approved target path also contains excluded staged or
  unstaged hunks, do not use path-level `git commit --only`; isolate the
  index/worktree so it contains only approved content, verify the resulting
  commit tree, then verify the cached and uncached diffs separately after
  returning to the shared worktree.

## Documentation Builds

- Prefer `make doc` for a full documentation build.
- The docs `Makefile` defaults to serial Sphinx execution because parallel builds can hang in this repository. Only opt in to parallelism with an explicit `SPHINXJOBS=auto` or another positive job count when there is a clear reason.
- Prefer `make doc-debug-api API_TARGETS=...` or `make -C docs debug-api API_TARGETS=...` when validating `autoapi + autodoc + sphinx` output for a small set of Python modules.
- Prefer `make doc-debug-tutorial TUTORIAL_TARGETS=...` or `make -C docs debug-tutorial TUTORIAL_TARGETS=...` when validating Sphinx Gallery tutorials. This fast path intentionally skips AutoAPI generation to keep iteration cheap.
- If docs debug targets or their semantics change, keep contributor-facing examples in `CONTRIBUTING.md` aligned with the current `Makefile` entry points.

## Design And Delivery Loop

- For complex, cross-cutting, or high-uncertainty tasks, follow this minimal loop: design, develop, confirm, distill, then clean up.
- When the `feature-dev` skill applies, treat it as the detailed implementation of the design, develop, and confirm portions of this loop. The repository-level distill and clean-up requirements still apply after that skill's development flow completes.
- Before implementation, write a temporary design note in a disposable repository-local scratch path such as `.agents/scratch/designs/`; keep it uncommitted by default.
- Capture the problem, constraints, chosen approach, validation plan, and explicit non-goals in that temporary design note.
- For complex refactors, cross-layer features, public API changes, resource
  lifecycle changes, schedulers, or compatibility migrations, consult
  `.agents/references/design-doc-guideline.md`. Use
  `.agents/templates/design-doc-scaffold.md` only as an optional prompt list;
  do not force the note into that order.
- Keep design notes focused on architecture, ownership, contracts, failure
  semantics, compatibility, and testing boundaries. Put task ordering, worker
  assignment, and validation commands in a separate implementation plan.
- Before dispatching parallel implementation agents, make the plan state the
  shared interface freeze points, disjoint write scopes, dependency order,
  expected validation, and how conflicts should be reported back to the
  coordinator.
- Do not bake a per-task commit requirement into repository-local plans.
  Commit cadence should follow the user's request and this repository's git
  instructions, even when an external workflow recommends frequent commits.
- For user-reviewed phased development, finish implementation, validation,
  and cleanup for the phase, then stop at a review gate. Run delegated review
  at key checkpoints, such as phase completion with public API, persisted
  format, or boundary changes, or when the user explicitly asks for review.
  Do not commit that phase until the user explicitly says it can be committed.
- For manual review between phases, code, design notes, plans, and review
  findings may move at different speeds during active iteration, but before
  handing work back they must all describe the same current implementation,
  status, validation evidence, and remaining risk.
- At each human-review handoff, identify the current planned review gate and
  the number of planned gates remaining. Report issue-triggered extra review
  rounds as conditional instead of presenting them as fixed planned gates.
- Before committing, handing work back for human review, or retiring a scratch
  design after non-trivial work, run a distillation triage over the whole
  current conversation. Report candidate lessons with recommendations first,
  wait for the user's decision before editing durable guidance or memory, and
  record `none` when there is nothing worth preserving.
- If a Superpowers workflow is also in use, treat Superpowers as the
  collaboration and process guide, and this repository's design guideline as
  the repo-local content standard. User-specified scratch paths override
  Superpowers default `docs/superpowers/...` paths.
- A design is ready to become a plan only after public/internal boundaries,
  failure semantics, compatibility strategy, and test boundaries are explicit,
  and no unresolved user decision blocks implementation.
- Skip the temporary design note for small, local, or mechanical changes when the implementation path is already obvious.
- After implementation, run the smallest useful validation and confirm the result against the user's request before treating the task as complete.
- If part of the temporary design is durable project knowledge, promote only the stable subset into this repository's canonical design docs, `docs/`, package docs, or another established design-doc location instead of preserving the scratch note.
- If the work reveals durable agent-facing lessons, distill them into local guidance or other intentional local shared agent assets instead of copying the whole temporary design note into instructions.
- After confirmation, delete the temporary design note only after any required
  durable promotion or explicit no-durable-content decision is complete.

## Validation

- Choose the smallest validation that matches the changed files and impact.
- When Python changes affect rendered docs, autodoc or autoapi output, tutorial examples, or other documented behavior, run the smallest useful docs validation in addition to code validation.
- Prefer `make doc-debug-api ...` for API-doc impact and `make doc-debug-tutorial ...` for tutorial impact before escalating to a full `make doc` build.
- Add or update tests when behavior changes.
- Broaden validation for shared behavior, public APIs, packaging, or config changes.
- For dependency compatibility-window upgrades, keep reproducible lower and
  upper endpoint constraint files under `scm/constraints/` when the resolver
  surface is part of the validation story. Validate source compatibility at
  each endpoint before widening package metadata, then run a real packaging
  or import smoke with the affected optional extras and endpoint constraints.
- When broad pytest validation is large enough to justify xdist, prefer
  repository-owned entrypoints or an explicit worker count that matches the
  target's runtime profile; do not default to `-n auto`.
- If validation is partial or blocked, say what ran, what did not, and the remaining risk.

## Cleanup Gate

- Before finalizing non-trivial implementation, inspect `git diff --stat` and
  make sure the change size still matches the intended scope.
- Scan changed areas for expanding compatibility or unfinished work markers
  such as `TODO`, `legacy`, `compat`, `deprecated`, and newly added helper
  names when those terms are relevant to the task.
- Check each newly added `__all__` entry, public class, protocol, and helper:
  keep it only if it has a clear caller, compatibility purpose, or semantic
  boundary that cannot be handled by an existing seam.
- When delegating review for non-trivial Python changes that add or modify
  package roots, explicitly ask reviewers to check `__init__.py` and
  `__all__` root-surface minimality.
- Prefer deleting, merging, or downgrading compatibility-only code before
  adding another abstraction to explain it.
- For non-trivial implementation, run an explicit convergence pass after
  correctness findings are closed and before human review. Inventory added or
  materially changed public surfaces, helpers or wrappers, compatibility
  branches, validation or fallback paths, and test scaffolding. For each group,
  delete or merge it, or record its exact caller, compatibility commitment, or
  distinct owner/failure boundary; compare direct/no-new-helper and
  single-flow alternatives.
- If convergence changes executable code, rerun relevant validation, repeat
  the inventory, and complete a final correctness regression scan. Record the
  resulting convergence evidence in the active plan or human-review handoff;
  a one-pass helper scan is not a passed convergence gate.

## External CI Diagnosis

- When external CI logs contain both a primary build/test failure and later
  post-action, sandbox, or log-processing errors, identify the first real
  failure before acting on the tail error.
- For flaky-test diagnosis, inspect the repository's real parallel test
  entrypoints (for example `Makefile` targets, `tests/run_unit_tests.py`,
  xdist mode, and worker counts) before treating an isolated local rerun as
  representative.
