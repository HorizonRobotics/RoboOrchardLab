# Interface Docstring Guideline

Use this reference for key public classes, key public functions, boundary
helpers, and resource-owning or stateful interfaces in this repository.

## Applicability

Use this guideline when the caller cannot use the interface safely from the
signature alone, especially for:

- public wrapper or adapter classes
- stateful or resource-owning interfaces
- file, process, network, model, device, or persistence boundaries
- public dataset, model, env, pipeline, and policy entrypoints
- helper functions with non-obvious contracts

## Goal

After reading a key interface docstring, a caller should be able to answer:

- what problem this interface helps solve
- when to use it instead of handling the boundary directly
- what it owns versus what the caller still owns
- what guarantees, limits, or failure semantics affect safe use

If a topic does not change caller decisions, omit it.

## General Rules

- Start from the caller's task and decision point, not from the
  implementation order.
- In the opening sentence or first short paragraph, explain what a caller is
  trying to get done and why they would reach for this interface.
- Explain the abstraction boundary before expanding `Args:` or `Returns:`.
- Keep implementation detail only when it explains an externally visible
  guarantee or limitation.
- Do not mechanically restate names, types, or obvious control flow.
- Keep examples minimal and representative of the canonical path.
- Keep the docstring as short as the contract allows; not every key
  interface needs every documentation topic.

## Topic Menu

Choose only the topics that materially affect safe use:

- purpose and fit: what caller problem this interface solves, when it
  applies, and when a caller should reach for it
- responsibility boundary: what the interface owns versus what the caller
  still owns
- lifecycle or state model: open/close, reuse, mutation, caching, or
  resource semantics
- input or output contract: shape, order, units, ownership, frame, or
  convention semantics
- side effects and failure semantics: writes, cleanup, rollback, partial
  output, or state changes
- guarantees and limits: invariants, compatibility limits, or publication
  semantics

## Classes

- Key classes usually need a one-line purpose and one or more topic blocks
  from the menu above.
- Key classes should usually open with user-facing purpose and fit before
  ownership or lifecycle details.
- Resource-owning or stateful classes usually need lifecycle/state language.
- Public wrapper or adapter classes should make the ownership boundary clear.

## Structured Data Fields

For public contract dataclasses, config objects, protocol payloads, and event
payloads, document fields whose meaning is not obvious from the type
signature.

- Prefer attribute docstrings placed immediately below the field when member
  discoverability in IDEs matters.
- Document fields that affect lifecycle, retry behavior, scheduler state,
  ownership, failure semantics, compatibility, or aggregation.
- Do not mechanically document every trivial field when the name and type are
  already sufficient.

For bool, enum, or literal fields and parameters that change behavior, do not
stop at "whether X". State each meaningful branch and the state transitions
that callers or maintainers must preserve.

## Functions And Special Callables

- Ordinary functions usually need only the non-obvious input, output,
  side-effect, or failure contract.
- If a function returns a resource-owning or stateful object, make the
  returned ownership or lifecycle contract explicit when it affects safe use.
- Do not force every function into a preconditions/postconditions shape if
  that is not the real contract.
- For functions, sync or async, document await or scheduling semantics when
  they affect safe use.
- For context managers, sync or async, document enter/exit semantics,
  resource ownership, and cleanup behavior.
- For generators or iterators, sync or async, document yield semantics,
  exhaustion, and cleanup behavior when they matter.
- For decorators or wrapper helpers, including async-target wrappers,
  document what boundary they preserve, add, or change, and what callable or
  helper they return. For async-target wrappers, spell out await or
  scheduling changes when they affect the public contract.
- For properties, classmethods, and staticmethods, use the ordinary
  function/class rules and document descriptor or class-vs-instance
  semantics only when they affect safe use.

## Review Checklist

Use this checklist when reviewing key interface docstrings:

- does the docstring make the interface purpose and fit clear
- after the first paragraph, can a new caller tell what task this
  interface helps them accomplish and why they would use it
- does it describe ownership, lifecycle, or state semantics when relevant
- does it document the caller-visible contract that is not obvious from the
  signature alone
- for special callables, does it document the right kind of contract rather
  than forcing an ordinary function shape
- for async boundaries or async-target wrappers, are await or scheduling
  changes documented when they affect safe use
- does it describe guarantees, limits, or failure semantics that affect safe
  use
- is the example, if present, the canonical path
- do `Args:`, `Returns:`, and `Raises:` stay aligned with the narrative
- is the docstring concise enough to avoid boilerplate
