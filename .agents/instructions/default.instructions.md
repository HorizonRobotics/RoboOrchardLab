---
description: Load these baseline instructions only for tasks that touch this repository's files, workflows, or behavior.
---

# Project Instructions

## Defaults

- Reply in the same language as the user unless asked otherwise.
- Prefer choosing the language used for internal reasoning and planning based on what fits the task best instead of forcing a single language.
- Read the relevant code, call sites, and tests before editing.
- For every newly created repository-owned source file, add the standard RoboOrchard Apache 2.0 license header at the top of the file using the comment style of that language.
- Follow `.agents/references/license-header-guideline.md` for the canonical templates, default year rendering, placement, and checker alignment.
- Exclude vendored or external-code directories from default code search unless the task explicitly targets them or requires cross-repository comparison.
- Treat `projects/` as repository-owned application-layer code. When tracing usage, call sites, integration flow, or end-to-end behavior, include `projects/` in default code search alongside `robo_orchard_lab/` unless the user explicitly limits the scope.
- Keep comments and docstrings aligned with the implementation.
- When changing public behavior or observable contracts, update the related
  docstrings, comments, and examples in the same change.
- Do not leave known docstring or comment drift behind after code changes.
- For public wrapper or adapter classes, the class docstring should state the
  main public methods or properties and include a minimal usage example when
  that improves discoverability.
- For public methods with structured inputs or outputs, document the required
  order, shape, and key conventions when they are not obvious from the type
  signature alone.
- This repository is published externally. In public-facing artifacts such as
	docs, READMEs, examples, commit messages, and MR or PR descriptions, use
	repository-relative paths or redacted placeholders instead of personal
	local absolute paths, and do not include company-internal or other
	non-public links.
- For poses, frame transforms, and spatial matrices, follow `.agents/references/spatial-transform-and-matrix-naming-guideline.md`.
- Prefer concise, minimally fragmented helper functions. Merge nearby
	single-purpose helpers when it keeps the main flow clear, and avoid
	introducing extra helpers unless they improve readability or reuse.
- Inline single-call helpers that only rename or forward one operation when
	they do not create a meaningful semantic boundary.
- Preserve IDE discoverability and static refactor safety when compressing
	call paths. Do not hide typed method names, parameter shapes, or public API
	forwarding behind string-dispatch or generic `getattr(...)` helpers only to
	reduce boilerplate; centralize only true shared boundary behavior.
- For non-trivial implementation, write down the negative design constraints
  before coding: do not add parallel abstractions, duplicate source-of-truth,
  synonym public APIs, or compatibility logic in canonical paths unless the
  task explicitly requires them.
- After non-trivial implementation, run a simplification pass before
  finalizing: remove single-use forwarding helpers, merge thin abstractions,
  check for duplicate sources of truth, and keep compatibility paths out of
  canonical execution paths.
- If documentation conflicts with code, treat the code as the source of
	truth unless the task says otherwise.

## Scope and Safety

- Stay within files and behavior directly related to the task.
- Avoid unrelated refactors, renames, structural changes, and edits under `build/` unless required.
- Call out risk before proposing or making breaking, destructive, or environment-dependent changes.
- If a user request seems unsafe, destructive, irreversible, high-cost, privacy- or security-sensitive, or materially misaligned with the stated goal, explain the concern and ask for explicit confirmation before proceeding.
- Do not assume external services, hardware, or network access are available.

## Reporting

- List changed files when code or docs are modified.
- Recommend only the minimum useful validation commands.
- If information is missing, ask only the most critical question.
- Separate completed changes, remaining risks, and optional follow-up work.
