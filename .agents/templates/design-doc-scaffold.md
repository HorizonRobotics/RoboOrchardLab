# <Feature / Refactor Name> Design Draft

## 1. Goal

- <What changes>
- <Why this is needed>
- <What success looks like>

## 2. Non-Goals

- <What this design intentionally does not solve>

## 3. Execution Priority

Follow `.agents/references/design-doc-guideline.md#execution-priority`.

Known guidance conflicts or user decisions:

- none yet

## 4. Layer Boundaries

| Layer | Owns | Does Not Own |
| --- | --- | --- |
| `<Layer>` | `<responsibility>` | `<excluded responsibility>` |

## 5. Terminology

| Term | Meaning | Notes |
| --- | --- | --- |
| `<term>` | `<definition>` | `<ambiguity / contrast>` |

## 6. Public And Internal Contracts

| Type / API | Visibility | Producer | Consumer |
| --- | --- | --- | --- |
| `<Name>` | public / developer-facing / internal | `<owner>` | `<consumer>` |

Illegal states:

- `<state that must not happen>`

## 7. Core Flow

```python
def main_flow(...):
    # Explain ownership and state transitions here.
    ...
```

## 8. State Ownership

| State | Owner | Modified By | Read By | Communication |
| --- | --- | --- | --- | --- |
| `<state>` | `<owner>` | `<callbacks / loop>` | `<consumer>` | `<queue / event / direct call>` |

## 9. Failure And Cleanup Semantics

- Failure classes:
- Retry / continue-on-error:
- Timeout semantics:
- Partial result behavior:
- Close / cancellation / KeyboardInterrupt:
- Late callback / stale completion:

## 10. Compatibility Strategy

| Surface | Strategy | Behavior |
| --- | --- | --- |
| `<old API>` | keep / wrapper / deprecated no-op / explicit removal | `<details>` |

## 11. Testing Boundary

- Fast unit tests:
- Fake / monkeypatch boundaries:
- Integration-only tests:
- Expensive tests excluded from default path:

## 12. Plan Readiness Gate

- [ ] Public/internal boundaries are stable.
- [ ] Failure semantics are defined.
- [ ] Compatibility strategy is explicit.
- [ ] State ownership is clear.
- [ ] Test boundary is clear.
- [ ] No unresolved user decision blocks implementation.

## 13. Open Questions

- `<question>`
