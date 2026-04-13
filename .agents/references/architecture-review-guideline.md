# Architecture Review Guideline

Use this reference for stable cross-domain architecture review guidance in
this repository.

This file captures the common review dimensions for module boundaries,
abstraction quality, contract design, dependency direction, compatibility
surfaces, evolvability, validation boundaries, and readability.

## Applicability

Use this guideline when the task is not limited to one domain's semantics and
the main question is whether the design keeps boundaries, contracts, public
surfaces, and extension cost clear and manageable.

Do not use this file to encode domain-local business or dataset semantics.
Those belong in the domain guidance family that owns them.

## Architecture Review Focus

Use this guideline to review or refactor the following architecture
dimensions:

- layer ownership and responsibility boundaries
- abstraction quality and whether an abstraction actually reduces complexity
- stable contract design and access patterns
- dependency direction and source-of-truth ownership
- compatibility surfaces and caller-facing public APIs
- evolvability, validation boundaries, and main-flow readability

Architecture review is not limited to immediate correctness bugs. Its goal is
to judge whether the design keeps shared-vs-local boundaries clear, contracts
stable, extension cost reasonable, validation attached to the right
boundaries, and the main execution path understandable.

## Review Scope Boundaries

- Treat code style, naming nits, and low-confidence cleanup suggestions as
  outside the core architecture-review scope unless they directly affect
  boundary clarity or contract correctness.
- Do not report one-off collaboration preferences or local workflow habits as
  architecture issues unless they were explicitly adopted as stable project
  rules.
- Keep domain semantics in the domain guidance family that owns them.
- Distinguish design weakness from immediate bug risk. A change may deserve
  an architecture note even when it is not yet a correctness bug.

## Layer Ownership And Responsibility

- Keep each layer responsible for one clear semantic stage.
- Keep shared layers focused on genuinely reusable primitives or mechanics.
- Keep caller-local, domain-local, or scenario-specific orchestration out of
  shared layers unless multiple callers truly share the entire contract.
- Avoid mixing normalization, orchestration, and core semantic conversion in
  one layer when those responsibilities can be separated cleanly.

## Abstraction Quality

- Prefer abstractions that remove duplicated policy or hide genuinely shared
  mechanics.
- Do not keep abstractions that only rename a loop, move branching behind
  callbacks, or centralize parameter passing without reducing cognitive load.
- If multiple callers still have to remember local policy after adopting a
  shared helper, that policy likely still belongs in the local layer.

## Contract Design

- Prefer explicit, stable contracts for structures that cross multiple layers.
- Avoid half-dynamic contracts where stable fields are still accessed through
  ad hoc string-based lookups or similar dynamic indirection in the main path.
- Keep optionality and defaults explicit instead of encoding critical
  semantics through missing keys, missing fields, or silent fallback logic.

## Dependency And Source-Of-Truth Discipline

- Keep dependency direction one-way: shared layers should not depend on
  caller-local or domain-local policy.
- Avoid duplicate sources of truth for shared runtime, transform, config, or
  business semantics across layers.
- When a lower layer already owns a semantic or invariant, configure and
  consume that owner instead of caching a second copy locally.

## Compatibility And Public Surface

- Preserve compatibility only for surfaces with clear external cost.
- Treat public exports and documented import paths as compatibility
  commitments rather than incidental implementation details.
- Do not keep dead shims or stale aliases once their supported behavior is
  gone.
- Keep caller-facing package or module exports deliberate and minimal.

## Evolvability, Validation, And Readability

- Prefer structures where a new caller, backend, or domain variant can be
  added by implementing local policy instead of copying a full orchestration
  stack.
- Enforce invariants at construction or boundary points when possible so
  invalid combinations fail before deep runtime logic.
- Attach validation and tests to the true design boundaries, not only to the
  easiest integrated path.
- Keep main flows readable without forcing the reader to chase many thin
  helpers that only rename or forward one step.

## Architecture Review Checklist

- Does each layer have one clear responsibility?
- Is the shared layer limited to genuinely reusable primitives rather than
  caller-local orchestration?
- Does each abstraction remove real complexity instead of turning policy into
  callbacks, indirection, or parameter plumbing?
- Are stable cross-layer contracts explicit rather than half-dynamic?
- Is dependency direction one-way, and is the source of truth for shared
  semantics unique?
- Are compatibility surfaces and public exports limited to intentionally
  supported caller commitments?
- Would adding one more caller, backend, or domain variant mostly require
  local policy changes rather than duplicated orchestration?
- Are validation hooks and tests aligned with the true boundaries of the
  design?
- Is the main execution path still readable without low-value indirection?
