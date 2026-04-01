---
description: Load these baseline instructions only for tasks that touch this repository's files, workflows, or behavior.
---

# Project Instructions

## Defaults

- Reply in the same language as the user unless asked otherwise.
- Prefer choosing the language used for internal reasoning and planning based on what fits the task best instead of forcing a single language.
- Read the relevant code, call sites, and tests before editing.
- Exclude vendored or external-code directories from default code search unless the task explicitly targets them or requires cross-repository comparison.
- Keep comments and docstrings aligned with the implementation.
- Prefer concise, minimally fragmented helper functions. Merge nearby
	single-purpose helpers when it keeps the main flow clear, and avoid
	introducing extra helpers unless they improve readability or reuse.
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
