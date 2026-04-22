# Code Review Family Scope And Signal

Use this reference after `.agents/skills/codereview/SKILL.md` routes the
task here.

## Triggering Rules

- Heavy code review skills should trigger on explicit review requests.
- They may also auto-trigger only when the task is already a review or
  evaluation request and a sub-skill-specific architecture auto-trigger rule
  applies.
- Do not trigger them for routine self-checks, casual sanity scans, or
  ordinary debugging.
- Choose `changeset-codereview` for generic local changesets.
- Choose `prmr-codereview` only when the target is a GitHub PR or GitLab MR
  and PR/MR-specific behavior matters.
- Choose `architecture-review` as the main skill only when architecture is
  the main review target.
- Auto-pair `architecture-review` under `changeset-codereview` or
  `prmr-codereview` when the reviewed scope materially changes layer
  ownership, cross-layer contracts, dependency direction, compatibility or
  public surfaces, export/import boundaries, or shared-vs-local
  responsibility splits.

## Signal Rules

- Report only validated, high-signal findings.
- Each reviewer should try to surface every high-signal issue it can find in
  scope, not only the single strongest one or two findings.
- Use the smallest useful scope; avoid reading unrelated files just to make
  the review look broader.
- For re-review of an updated changeset or PR/MR, default to the full current
  effective diff for the review target rather than only the incremental patch
  since the last review, unless the user explicitly asks for incremental-only
  validation.
- Do not drip-feed validated findings across rounds. Before reporting,
  consolidate newly validated issues with any still-relevant prior findings in
  the same review pass.
- Prefer concrete correctness, contract, compatibility, or scoped guidance
  findings over general quality opinions.
- Do not report style-only, taste-only, or low-confidence suggestions.
- Distinguish architecture weakness from immediate correctness bugs instead
  of collapsing them into one category.
- If multiple review skills contribute, keep report ownership with the active
  skill and use `report-composition.md` to summarize paired review inputs
  without merging unrelated findings together.
