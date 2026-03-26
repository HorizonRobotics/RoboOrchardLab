---
description: Load these instructions when creating, updating, or validating tests in this repository.
---

# Test Instructions

These rules specialize the general validation guidance in `workflow.instructions.md` for test-related changes.

## Test Design

- Follow the style of nearby tests before introducing a new pattern.
- Prefer the smallest test that proves the target behavior.
- Use real fixtures, datasets, models, and file paths for true integration behavior.
- Do not replace required real test inputs with fallback skip logic when the configured environment is expected to support the test.
- Use mocks or monkeypatch only when the test target is isolated assembly logic and real dependencies are not part of the behavior under test.

## Fixtures

- Keep reusable fixtures in the nearest `conftest.py` that matches their sharing scope.
- Move shared model paths, tokenizer paths, processor paths, and similar resources out of individual files when nearby tests can reuse them.
- Keep test-specific fixtures in the test module when they are only used by one file.

## Test Structure

- Match the local project convention for test organization; prefer class-based tests when nearby files use `Test...` classes.
- Keep assertions focused on the behavior under test instead of asserting incidental implementation details.
- When a test is meant to inspect real returned data, expose the key values so failures are easier to interpret.

## Validation

- Run the narrowest relevant `pytest` target for the changed test or module first.
- When running repository tests, disable `HTTP_PROXY`, `HTTPS_PROXY`, `http_proxy`, and `https_proxy` unless the task explicitly requires proxy access.
- Run `ruff check` on modified test files.
- Document any temporary pytest flags or environment variables required to run successfully.
