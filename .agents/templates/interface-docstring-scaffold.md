# Interface Docstring Scaffold

Use this scaffold with
`.agents/references/interface-docstring-guideline.md` when drafting or
rewriting docstrings for key repository-owned interfaces.

This is a block menu, not a required outline. Pick the smallest set of
blocks that captures the caller-visible contract.

## Short Core Shape

Use this when one short paragraph plus normal section blocks is enough:

```python
"""<One-line purpose>.

<Only add the contract sentences that materially affect safe use.>

Args:
    <arg> (<type>): <meaning>.
Returns:
    <type>: <meaning>.
Raises:
    <ErrorType>: <when>.
"""
```

## Class Starting Shape

For classes, start with a user-facing purpose sentence. Resource-owning,
stateful, wrapper, or adapter classes often then need one ownership or
lifecycle sentence.

```python
"""<One-line purpose from the caller's task>.

Use this when <caller situation / why they would reach for it>.
It owns <owned behavior> and leaves <caller-owned behavior> to the caller.

Args:
  <arg> (<type>, optional): <meaning>. Default is <value>.
"""
```

## Optional Narrative Blocks

Pick only the caller-visible topics that matter from the topic menu in
`.agents/references/interface-docstring-guideline.md`.

If a short sentence starter helps, use one of these and then rewrite it to
fit the real contract:

- `Use this when <caller situation>.`
- `It helps the caller <complete task / avoid managing boundary detail>.`
- `It owns <owned behavior> and leaves <caller-owned behavior> to the caller.`
- `Its main contract is <lifecycle / state model>.`
- `On failure, <cleanup / rollback / publication behavior>.`

## Special Callable Blocks

Use these instead of forcing an ordinary function shape when they fit better.
Keep normal `Args:`, `Returns:`, or `Raises:` sections when the signature or
exception contract is still important to safe use.

Async variants usually follow the same block shapes; rewrite the sentences in
terms of await, async enter/exit, async yield, or scheduling semantics when
those details matter.

### Async Function

Use the short core shape for async functions too. Add one extra sentence only
when await, scheduling, cancellation, or task-boundary semantics affect safe
use, for example: `Awaiting this function <observable async behavior>.`

### Context Manager

```python
"""<One-line purpose>.

Entering the context <what becomes available>. Exiting the context
<cleanup / publish / rollback behavior>.

Args:
  <arg> (<type>): <meaning>.
Raises:
  <ErrorType>: <when>.
"""
```

### Async Context Manager

```python
"""<One-line purpose>.

Entering the async context <what becomes available>. Exiting the async
context <cleanup / publish / rollback behavior>.

Args:
  <arg> (<type>): <meaning>.
Raises:
  <ErrorType>: <when>.
"""
```

### Generator Or Iterator

```python
"""<One-line purpose>.

Yields <meaning of each item>. Iteration ends when <end condition>.
<Cleanup / ownership / side-effect sentence if needed>.

Args:
  <arg> (<type>): <meaning>.
Yields:
  <type>: <meaning>.
"""
```

### Async Generator

```python
"""<One-line purpose>.

Yields <meaning of each item> during async iteration. Iteration ends when
<end condition>.
<Cleanup / ownership / side-effect sentence if needed>.

Args:
  <arg> (<type>): <meaning>.
Yields:
  <type>: <meaning>.
"""
```

### Decorator Or Wrapper Helper

```python
"""<One-line purpose>.

Wraps <target boundary> to <added behavior>. Preserves <important preserved
contract> and changes <important changed contract>. For async targets,
spell out await or scheduling semantics when they change.

Args:
  <arg> (<type>): <meaning>.
Returns:
  <type>: <returned wrapper / callable / helper>.
"""
```

### Async Decorator Or Wrapper Helper

Use the same shape as the decorator block. Add one sentence when awaiting the
wrapped target or changing scheduling behavior becomes part of the contract.

### Property / Classmethod / Staticmethod

Use the short core shape. Add one sentence only when descriptor behavior or
class-vs-instance semantics affect safe use.

## Author Trim Pass

Before finalizing a key interface docstring, check:

- only blocks that materially affect caller decisions were kept
- examples, if present, show the standard path rather than an edge case
- parameter sections do not repeat narrative text without adding meaning
