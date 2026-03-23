---
description: Load these instructions when validating changes or working with repository workflows, tests, documentation builds, or developer tooling.
---

# Workflow and Validation Instructions

## Sources of Truth

- Use `Makefile` when a relevant target exists.
- Use `pyproject.toml` and pytest config for tool behavior.
- Prefer source files over `build/`; use `build/` only for debugging generated output.
- If workflow files disagree, report the mismatch instead of guessing.

## Validation

- Choose the smallest validation that matches the changed files and impact.
- Add or update tests when behavior changes.
- Broaden validation for shared behavior, public APIs, packaging, or config changes.
- If validation is partial or blocked, say what ran, what did not, and the remaining risk.
