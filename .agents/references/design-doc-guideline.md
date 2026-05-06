# Design Doc Guideline

Use this reference when writing, reviewing, or distilling temporary design
documents for complex features, cross-layer refactors, public API changes,
resource-owning systems, schedulers, or compatibility migrations.

## Relationship To Workflows

This repository's design guidance defines the content standard for design
documents. External or plugin workflows such as Superpowers may define the
collaboration process, approval gates, or plan-writing process.

Do not duplicate workflow-specific rules here. Use this reference to decide
what a repository design document must make explicit.

## When To Use

Use a design document for changes that materially affect one or more of:

- cross-layer contracts or public APIs
- resource ownership, lifecycle, cleanup, or cancellation
- scheduler, callback, queue, or concurrent state ownership
- compatibility migrations or config semantics
- expensive integration boundaries that need fakeable test seams

Skip the design document for small, local, or mechanical changes when the
implementation path, failure behavior, and validation are already obvious.

## Design And Plan Separation

Keep design and implementation plan separate.

A design document answers:

- what problem is being solved
- which layers own which responsibilities
- what contracts cross module boundaries
- what states and invariants must hold
- how failure, cleanup, cancellation, and compatibility behave
- what test boundaries prove the design

An implementation plan answers:

- which files to edit
- in what order
- which tasks can run in parallel
- how to validate each slice

Do not bury task sequencing inside the design. Do not rely on the plan to
define unresolved architecture contracts.

## Execution Priority

A feature-level design is not an absolute override for repository guidance.
Repository guidance is the default constraint, but active refactors may reveal
that an existing guideline needs to change.

If a design or plan conflicts with `AGENTS.md`, `.agents/instructions/`, or
`.agents/references/`, decide based on:

- the current refactor goal
- existing code facts
- public API and compatibility impact
- long-term maintenance cost
- testability and failure clarity

If the better choice is clear, follow it and update the design, plan, or
guidance as appropriate. If the choice is unclear, public-facing, or would
change stable guidance, ask the user to decide.

## Design Methodology

Use these principles when a design or review is trying to simplify public
interfaces, duplicated state, or cross-layer config:

- Ownership-first simplification: find the true owner of a state, default, or
  invariant before deleting fields, caches, snapshots, or helpers. Move data
  back to the real owner instead of keeping a second source of truth in a
  nearby orchestrator.
- Default-light, explicit-heavy: prefer a canonical default path that is low
  dependency, low cost, and easy to debug. Make heavier capabilities such as
  remote execution, parallelism, timeout isolation, or worker replacement
  explicit choices. Compatibility wrappers may keep older defaults when that
  avoids breaking established callers.
- Design follows implementation feedback: if implementation or review reveals
  a clearer ownership boundary, default, or contract than the original design,
  update the design or plan. Do not leave stale design conclusions that future
  work could follow accidentally.

## Required Design Content

For non-trivial designs, include:

- goal and non-goals
- layer responsibility table
- terminology table for overloaded terms
- public, developer-facing, and internal contract classification
- core flow pseudocode for state machines, schedulers, retries, or lifecycle
- state ownership table when concurrency or callbacks are involved
- illegal states and invariants
- failure and cleanup semantics
- compatibility classification
- testing boundary before individual test cases
- plan readiness gate
- open questions or user decisions

Scale the detail to the task. Small local changes may only need a short
scratch note or no design doc.

## Boundary-First Structure

Start with layer boundaries before listing dataclasses, protocols, or helper
functions. Readers should understand what each layer owns before seeing field
details.

For each layer, make both sides explicit:

| Layer | Owns | Does Not Own |
| --- | --- | --- |

## Contract Classification

When a design introduces multiple structures, classify them:

| Type / API | Visibility | Producer | Consumer |
| --- | --- | --- | --- |

Recommended visibility values:

- public user-facing contract
- developer-facing contract
- driver-facing event
- backend internal message
- compatibility-only surface

Prefix internal runtime structures with `_` when they are not intended for
external use.

## State Ownership

For scheduler, worker, callback, retry, or queue-based designs, write a state
ownership table:

| State | Owner | Modified By | Read By | Communication |
| --- | --- | --- | --- | --- |

Future callbacks, threads, or remote completions should not mutate scheduler
state unless the design explicitly assigns them ownership.

## Pseudocode

Use short pseudocode for flows that are easy to misread in prose:

- scheduler loops
- retry handling
- offset or seed advancement
- close / cleanup lifecycle
- callback dispatch
- queue producer/consumer flow

Prefer comments in the same language as the design document. Pseudocode should
expose ownership and state transitions, not implementation noise.

## Failure And Cleanup Semantics

Resource-owning and remote designs must describe:

- who owns cleanup
- what `close()` means
- whether close is idempotent
- what happens on `KeyboardInterrupt`
- what happens to late callbacks or stale completions
- whether partial results are returned or discarded
- which failures become retries versus fatal errors
- which exceptions remain distinguishable

## Compatibility Classification

Do not write only "keep compatibility." Classify each old surface:

| Surface | Strategy | Behavior |
| --- | --- | --- |
| `<old API>` | keep / wrapper / deprecated no-op / explicit removal | `<details>` |

Do not approximate old semantics with a new field unless the mapping is exact
and intentional. Prefer explicit migration errors over silent semantic drift.

## Review Finding Resolution

Treat design review findings as signals about the design, not only text edits.

When a finding reveals contradictory terminology, unclear state ownership,
mixed responsibilities, or incompatible failure semantics, fix the underlying
design section first. Update terminology, ownership tables, pseudocode, or
compatibility classification so the document has one coherent contract.

Do not resolve a consequential finding by changing one local sentence while
leaving a conflicting flow, test plan, or contract table in place.

## Test Boundary

State the test boundary before listing test cases.

Designs should clarify:

- which behavior must be covered by fast unit tests
- which dependencies should be faked
- which real environment, simulator, hardware, or service tests are integration-only
- which boundary each test proves

For expensive domains, default tests should avoid real simulator startup unless
the task explicitly targets integration behavior.

## Plan Readiness Gate

A design is ready to become an implementation plan when:

- public and internal boundaries are stable
- failure semantics are defined
- compatibility strategy is explicit
- state ownership is clear
- test boundaries are clear
- no unresolved user decision blocks implementation

## Review Checklist

When reviewing a design document, check:

- are layer responsibilities and non-goals explicit
- are public, developer-facing, and internal contracts classified
- are overloaded terms defined before they are used
- is state ownership clear for callbacks, threads, queues, or retries
- are illegal states, failure semantics, and cleanup behavior defined
- is compatibility classified instead of described only as "supported"
- does the test boundary separate fast fake tests from integration tests
- is the design ready to become a plan, or are user decisions still open
