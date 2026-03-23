---
description: Load these instructions when modifying Python source files, tests, packaging metadata, or implementation-related documentation in this repository.
---

# Python Change Instructions

## Core Expectations

- Keep changes compatible with the project's Python version and public APIs unless the task allows otherwise.
- Reuse existing patterns, helpers, constants, and types before adding new ones.
- Keep new logic focused; avoid abstraction added only for style.
- Do not silently swallow exceptions. If catching one, keep enough context to debug it.

## Typing

- Preserve or add type annotations when touching function signatures or return values.
- Prefer complete type hints for public APIs, key helpers, and newly added functions unless a clear local pattern or technical reason suggests otherwise.

## Documentation and Comments

- Follow the style of nearby code, including imports, naming, file layout, and docstring conventions.
- Add comments only when they provide non-obvious context.
- For key interface functions, public dataset/model/pipeline entrypoints, and helper functions whose behavior or parameters are not immediately obvious from the signature alone, add or update docstrings instead of leaving the interface undocumented.
- Follow the project's existing Google-style docstring format with `Args:` and `Returns:` when documenting functions.
- In `Args:`, use `name (Type): ...` for required parameters and `name (Type, optional): ... Default is ...` for optional parameters.
- Keep docstrings concise, with consistent indentation and defaults documented only for optional parameters.

## Dependencies

- Avoid new dependencies unless they are clearly necessary.
