# Code Review Family Scope And Signal

Use this reference after `.agents/skills/codereview/SKILL.md` routes the
task here.

## Triggering Rules

- Heavy code review skills should trigger only on explicit review requests.
- Do not trigger them for routine self-checks, casual sanity scans, or
  ordinary debugging.
- Choose `changeset-codereview` for generic local changesets.
- Choose `prmr-codereview` only when the target is a GitHub PR or GitLab MR
  and PR/MR-specific behavior matters.
- Choose `architecture-review` only when architecture is the main review
  target.

## Signal Rules

- Report only validated, high-signal findings.
- Use the smallest useful scope; avoid reading unrelated files just to make
  the review look broader.
- Prefer concrete correctness, contract, compatibility, or scoped guidance
  findings over general quality opinions.
- Do not report style-only, taste-only, or low-confidence suggestions.
- Distinguish architecture weakness from immediate correctness bugs instead
  of collapsing them into one category.
- If multiple review skills contribute, keep report ownership with the active
  skill and use `report-composition.md` to summarize paired review inputs
  without merging unrelated findings together.
