---
description: Load these instructions when modifying Python source files, tests, packaging metadata, or implementation-related documentation in this repository.
---

# Python Change Instructions

## Python

- Keep changes compatible with the project's Python version and public APIs unless the task allows otherwise.
- Reuse existing patterns, helpers, constants, and types before adding new ones.
- Preserve or add type annotations when touching function signatures or return values.
- Keep new logic focused; avoid abstraction added only for style.
- Do not silently swallow exceptions. If catching one, keep enough context to debug it.

## Style and Dependencies

- Follow the style of nearby code, including imports, naming, and file layout.
- Add comments only when they provide non-obvious context.
- Avoid new dependencies unless they are clearly necessary.
