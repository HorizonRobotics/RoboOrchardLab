# <Feature / Refactor Name> Design Notes

Use this as a prompt list, not a template. Keep only what helps; delete,
merge, rename, or reorder freely.

## Core Prompts

- Problem and success criteria:
- Non-goals:
- Current code facts and constraints:
- Chosen approach:
- Ownership and boundary decisions:
- Public, compatibility, and internal contracts:
- Failure, cleanup, timeout, or partial-output behavior:
- Test boundary:
- Open questions or user decisions:

## Optional Prompts

- Boundary split: `<layer>` owns `<responsibility>` and does not own
  `<excluded responsibility>`.
- Contract classification: `<API / type / schema>` is public,
  developer-facing, compatibility-only, or internal; produced by `<owner>` and
  consumed by `<consumer>`.
- State ownership: `<state>` is owned by `<owner>`, modified by `<path>`, and
  read through `<communication path>`.
- Flow sketch: write pseudocode only for lifecycle, scheduler, retry,
  callback, queue, or cleanup flow that prose may obscure.
- Data mapping: `<source>` maps to `<target>` with `<conversion / unit /
  encoding>`; `<field>` is intentionally excluded because `<reason>`.
- Compatibility: `<old surface>` is kept, wrapped, deprecated, rejected, or
  removed with `<behavior>`.
- Validation: fast tests cover `<behavior>`; fakes replace `<dependency>`;
  integration-only tests cover `<real boundary>`.

## Domain Hints

- Lifecycle / remote / scheduler work usually needs ownership, state, cleanup,
  timeout, cancellation, stale completion, and retry semantics.
- Data / schema / converter work usually needs input format, target output,
  field mapping, excluded fields, unit or timestamp semantics, and validation.
- API migration work usually needs current behavior, target behavior,
  compatibility strategy, migration path, and old/new surface tests.
