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
- Keep comments and docstrings aligned with the implementation.
- For poses, frame transforms, and spatial matrices, prefer explicit direction-bearing names such as `a_to_b`, `a_to_b_tf`, `a_to_b_mat`, or `BatchFrameTransform(child=A, parent=B)`. In this repository, `BatchFrameTransform(child=A, parent=B)` and `a_to_b` share the same direction semantics. Do not assume compact forms such as `A|B`, `T_ab`, `T_a_b`, or `Tab` are acceptable default repository style without an explicit local mapping.
- Treat the repository convention above as the default for code owned by this repository. For external libraries, third-party APIs, protocol fields, dataset schemas, or compatibility layers, follow the external convention at the boundary and add an explicit local mapping before translating to repository-preferred names.
- Prefer concise, minimally fragmented helper functions. Merge nearby
	single-purpose helpers when it keeps the main flow clear, and avoid
	introducing extra helpers unless they improve readability or reuse.
- Inline single-call helpers that only rename or forward one operation when
	they do not create a meaningful semantic boundary.
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
