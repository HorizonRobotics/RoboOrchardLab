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

## User Decision Gate

When the user asks for a design, identify any design-relevant choice that
requires user decision, input, prioritization, or confirmation. Do not resolve
those choices on the user's behalf.

For each such choice, provide a recommendation with rationale and tradeoffs,
then mark it as `Decision Needed` until the user confirms it. After
confirmation, write the selected option as accepted design.

This is not a request to push decisions entirely to the user: the agent should
still analyze the options, recommend a concrete path, and explain why. The
user owns the final decision.

This applies to any design-relevant choice, including but not limited to
scope, schema, storage format, API surface, lifecycle, compatibility,
migration, fallback behavior, validation strength, and what belongs in the
first version versus a later TODO.

## Design Judgment

- Treat repository guidance as the default constraint. If implementation facts
  reveal a better boundary, update the design or guidance instead of following
  stale text.
- Prefer ownership-first simplification: identify the real owner of state,
  defaults, lifecycle, and invariants before adding caches or mirror fields.
- Run a minimality pass before accepting the design. State the smallest viable
  shape first, then explain why each additional method, field, protocol, DTO,
  adapter, registry, factory, cache, or snapshot is necessary.
- For persisted schemas and storage formats, run an exclusion pass before
  accepting the first version. Delete speculative fields until a concrete
  producer, consumer, validation rule, or migration need justifies storing
  them; capture deferred capabilities as explicit future work instead of
  widening the stored contract early.
- Before changing an existing datatype, payload, or public-ish schema, audit
  the current semantic contract and call sites. Do not overload a stable
  type with same-name different-meaning data just to reduce the number of
  new types; introduce a separate contract or adapter when the old meaning
  must stay intact.
- For artifacts, caches, and stored media or dataset sidecars, state who owns
  validation at each boundary: writer, packaging validator, opener, reader,
  migration tool, or caller. Do not assume that read-time code can always
  trust write-time validation, or that it must always fully revalidate; justify
  the choice from the trust boundary, mutability, cost, and failure impact.
- Run an implementation leverage pass before accepting the design. Check
  whether already-required external libraries or standard project
  dependencies can remove custom code, and explicitly evaluate whether adding
  a new dependency is worth its implementation savings, packaging cost,
  runtime constraints, licensing/security surface, and maintenance risk.
  Dependency choices that affect scope, compatibility, deployment, or user
  tradeoffs remain `Decision Needed` until confirmed.
- Prefer extending the existing semantic owner with one focused method or hook
  before introducing a parallel provider or adapter layer. Avoid adding a new
  abstraction only to avoid one function call, one local branch, or passing an
  already-owned value.
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
- Minimality: write the one-method / no-new-class version of the design, list
  which proposed surfaces can be deleted, and keep extra layers only when they
  remove real duplication or own a distinct lifecycle.
- Dependency leverage: list existing libraries or already-required
  dependencies that can simplify the implementation; for any proposed new
  dependency, state why it is worth the added packaging, runtime, licensing,
  security, and maintenance surface.
- Contracts: classify introduced APIs or data shapes as public,
  developer-facing, compatibility-only, or internal.
- State and flow: for schedulers, callbacks, retries, queues, remote work, or
  lifecycle cleanup, identify the state owner and sketch the hard-to-read
  flow.
- Data and schema: for converters, datasets, serializers, or migrations, show
  source-to-target mapping, excluded fields, units, timestamps, encoding,
  validation ownership, and any first-version assumptions such as single
  stream, single source of truth, or unsupported optional metadata.
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

When reviewing a design, start with a minimality pass before checking
completeness. Look for unnecessary provider/adapter/factory layers, DTOs that
only wrap an existing object, two-method protocols plus snapshots that avoid a
single build call, and abstractions with more names than real branch points.
Then review missing ownership, mixed responsibilities, undefined failure
semantics, unclear compatibility, vague test boundaries, and open decisions
that would block implementation.
