---
name: codereview-family
description: Structured code review family for explicit heavy review requests. Route to PR/MR review, local changeset review, or architecture review based on the review target and intent. Do not use for routine self-checks or casual bug scans.
---
Use this skill family when the user explicitly asks for a heavy review flow.

Read `references/triggering-and-signal.md` first. When preparing a final
report, also use `references/report-composition.md`.

Then route to the smallest sub-skill that matches the target:

- `changeset-codereview/`
  Use for explicit review of a commit, branch diff, staged diff, working tree
  diff, patch, or local file set.
- `prmr-codereview/`
  Use for explicit GitHub PR or GitLab MR review when PR/MR-specific
  transport, metadata, or comment posting is required.
- `architecture-review/`
  Use for explicit architecture review or architecture-focused refactors
  where layering, contracts, compatibility surfaces, abstractions, or public
  API design are the main question.

Do not use this family for:

- routine implementation self-checks
- quick bug scans during development
- ordinary debugging
- style-only or taste-only feedback

Use the smallest review surface that satisfies the request. The heavier the
workflow, the more explicit the user request should be.
