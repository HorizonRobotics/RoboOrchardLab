---
description: Load these instructions when planning complex repository work, validating changes, or working with repository workflows, tests, documentation builds, or developer tooling.
---

# Workflow and Validation Instructions

## Sources of Truth

- Use `Makefile` when a relevant target exists.
- Use `pyproject.toml` and pytest config for tool behavior.
- Prefer source files over `build/`; use `build/` only for debugging generated output.
- If workflow files disagree, report the mismatch instead of guessing.

## Documentation Builds

- Prefer `make doc` for a full documentation build.
- The docs `Makefile` defaults to serial Sphinx execution because parallel builds can hang in this repository. Only opt in to parallelism with an explicit `SPHINXJOBS=auto` or another positive job count when there is a clear reason.
- Prefer `make doc-debug-api API_TARGETS=...` or `make -C docs debug-api API_TARGETS=...` when validating `autoapi + autodoc + sphinx` output for a small set of Python modules.
- Prefer `make doc-debug-tutorial TUTORIAL_TARGETS=...` or `make -C docs debug-tutorial TUTORIAL_TARGETS=...` when validating Sphinx Gallery tutorials. This fast path intentionally skips AutoAPI generation to keep iteration cheap.
- If docs debug targets or their semantics change, keep contributor-facing examples in `CONTRIBUTING.md` aligned with the current `Makefile` entry points.

## Design And Delivery Loop

- For complex, cross-cutting, or high-uncertainty tasks, follow this minimal loop: design, develop, confirm, distill, then clean up.
- When the `feature-dev` skill applies, treat it as the detailed implementation of the design, develop, and confirm portions of this loop. The repository-level distill and clean-up requirements still apply after that skill's development flow completes.
- Before implementation, write a temporary design note in a disposable repository-local scratch path such as `.agents/temp/designs/`; keep it uncommitted by default.
- Capture the problem, constraints, chosen approach, validation plan, and explicit non-goals in that temporary design note.
- Skip the temporary design note for small, local, or mechanical changes when the implementation path is already obvious.
- After implementation, run the smallest useful validation and confirm the result against the user's request before treating the task as complete.
- After confirmation, delete the temporary design note.
- If part of the temporary design is durable project knowledge, promote only the stable subset into this repository's canonical design docs, `docs/`, package docs, or another established design-doc location instead of preserving the scratch note.
- If the work reveals durable agent-facing lessons, distill them into local guidance or other intentional local shared agent assets instead of copying the whole temporary design note into instructions.

## Validation

- Choose the smallest validation that matches the changed files and impact.
- When Python changes affect rendered docs, autodoc or autoapi output, tutorial examples, or other documented behavior, run the smallest useful docs validation in addition to code validation.
- Prefer `make doc-debug-api ...` for API-doc impact and `make doc-debug-tutorial ...` for tutorial impact before escalating to a full `make doc` build.
- Add or update tests when behavior changes.
- Broaden validation for shared behavior, public APIs, packaging, or config changes.
- If validation is partial or blocked, say what ran, what did not, and the remaining risk.
