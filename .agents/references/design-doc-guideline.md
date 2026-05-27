# Design Doc Guideline

Use this reference when a change is complex enough that implementation should
not start from code edits alone.

## When To Write One

Write a temporary design note for changes that materially affect cross-module
contracts, public APIs, resource lifecycle, cleanup, cancellation, scheduling,
concurrency, compatibility, config semantics, or expensive integration seams.

Skip it for small, local, or mechanical changes when the implementation path,
failure behavior, and validation are already obvious.

## Design, Not Plan

A design explains the problem, ownership, contracts, invariants, failure
behavior, compatibility, and test boundary. An implementation plan explains
file edits, ordering, parallel work, and validation commands.

Keep those concerns separate. Do not use a plan to define unresolved
architecture contracts, and do not bury task sequencing inside the design.

## Design Judgment

- Treat repository guidance as the default constraint. If implementation facts
  reveal a better boundary, update the design or guidance instead of following
  stale text.
- Prefer ownership-first simplification: identify the real owner of state,
  defaults, lifecycle, and invariants before adding caches or mirror fields.
- Keep canonical paths low-dependency and explicit. Make heavier capabilities
  such as remote execution, retries, worker replacement, or compatibility
  wrappers intentional.
- Update or discard stale design conclusions after implementation feedback.

## Free Structure

Design documents do not need a fixed section order. Choose the structure that
best fits the problem. Use headings, tables, or pseudocode only when they make
the contract easier to review.

For non-trivial designs, make the applicable answers explicit:

- What problem is being solved, and what is out of scope?
- What current code facts, inputs, outputs, or constraints shape the design?
- Which layer or module owns each responsibility?
- Which contracts cross module boundaries, and who produces or consumes them?
- Which public, developer-facing, compatibility-only, and internal surfaces
  exist?
- Which states, invariants, or illegal states matter?
- Which flow is easy to misread and needs pseudocode or a step-by-step trace?
- What happens on failure, cleanup, cancellation, timeout, or partial output?
- Which old surfaces stay, migrate, warn, or disappear?
- Which behavior must be proven by fast tests, fakes, or integration tests?
- Which repository files or module families are expected to change, and which
  nearby public surfaces, generated files, or domain-specific callers are
  explicitly out of scope?
- Which user decisions or unresolved questions block implementation?

Scale the detail to the task. Small local changes may only need a short
scratch note or no design doc. Large designs may use several tables and
pseudocode blocks, but only where they clarify a real contract.

Use `.agents/templates/design-doc-scaffold.md` only as an optional prompt
list. Delete irrelevant prompts and reorganize freely.

## Useful Prompts

- Boundaries: for cross-layer work, state what each layer owns and does not
  own before listing helper types.
- Contracts: classify introduced APIs or data shapes as public,
  developer-facing, compatibility-only, or internal.
- State and flow: for schedulers, callbacks, retries, queues, remote work, or
  lifecycle cleanup, identify the state owner and sketch the hard-to-read
  flow.
- Data and schema: for converters, datasets, serializers, or migrations, show
  source-to-target mapping, excluded fields, units, timestamps, encoding, and
  validation.
- Compatibility: do not write only "keep compatibility"; say which old
  surface is kept, wrapped, deprecated, rejected, or removed.
- Code scope: for implementation-ready designs, list the expected code/test
  file families and explicit non-goals such as public exports, generated
  outputs, or domain-specific callers that should not change. Keep this as
  scope, not edit ordering.
- Testing: state the test boundary before individual test cases, especially
  what is faked and what is integration-only.

## Readiness

A design is ready to become an implementation plan when the relevant
ownership, contracts, failure behavior, compatibility, and test boundary are
clear enough that no unresolved user decision blocks implementation.

## Review

When reviewing a design, look for missing ownership, mixed responsibilities,
undefined failure semantics, unclear compatibility, vague test boundaries, and
open decisions that would block implementation.
