---
description: Load these instructions when creating, updating, or validating tests in this repository.
---

# Test Instructions

## Test Design

- Follow the style of nearby tests before introducing a new pattern.
- Prefer the smallest test that proves the target behavior.
- Use real fixtures, datasets, models, and file paths when the test is intended to validate actual integration behavior.
- Do not replace required real test inputs with fallback skip logic when the test is expected to prove correctness in the configured environment.
- Use mocks or monkeypatch only when the test target is isolated assembly logic and real dependencies are not part of the behavior under test.
- For processor, envelope, compose, or `pre_process`/`post_process` contract tests, follow `.agents/references/processor-guideline.md`.
- When adding a new `EnvelopeIOProcessor` or any new processor or adapter that reads or writes `processor_context`, add tests for the contract it relies on, including whether it returns fresh context objects or intentionally shares references.
- When deprecating or removing a processor or inference hook seam, test the canonical path and make the legacy seam's status explicit.
- If a deprecated compatibility entrypoint remains supported, test it directly.
- If the legacy seam is unsupported and ignored, test the warning and non-dispatch behavior instead, and do not describe it as direct-call compatibility.
- When a wrapper or adapter exposes both raw data and derived data for the
  same contract, test both the assembly logic and at least one end-to-end
  consistency path between the raw and derived representations.
- For frame or transform adapters, prefer one narrow unit test that checks the
  naming and graph assembly rules and one integration-style test that checks
  numerical consistency against the real runtime source.

## Fixtures

- Keep reusable fixtures in the nearest `conftest.py` that matches their sharing scope.
- Move shared model paths, tokenizer paths, processor paths, and other reusable test resources out of individual test files when multiple tests in the same directory can reuse them.
- Keep test-specific fixtures in the test module when they are only used by one file.

## Test Structure

- Match the local project convention for test organization; prefer class-based tests when nearby files use `Test...` classes.
- Keep assertions focused on the behavior under test instead of asserting incidental implementation details.
- When a test is meant to help inspect real returned data, print or otherwise expose the key returned values in the test run so failures and manual verification are easier to interpret.

## Validation

- Before running `python`, `pytest`, or `ruff`, load `.agents/instructions/environment.instructions.md`.
- Run the narrowest relevant `pytest` target for the changed test or module first.
- When running repository tests, disable `HTTP_PROXY`, `HTTPS_PROXY`, `http_proxy`, and `https_proxy` unless the task explicitly requires proxy access.
- For tests that spawn `accelerate launch`, `torchrun`, or similar
  distributed subprocesses under xdist, do not use a probe-and-release
  free-port helper. Prefer port `0` or another launcher-owned atomic port
  allocation path.
- If a subprocess helper does not need child-process coverage, clear
  inherited `COV_CORE_*` and `COVERAGE_PROCESS_START` variables before
  launch so pytest-cov noise does not mask the real failure.
- Run `ruff check` on modified test files.
- If the local pytest environment requires temporary flags or environment variables to run successfully, document the exact command used and why it was needed.
