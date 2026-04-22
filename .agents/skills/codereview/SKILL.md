---
name: codereview-family
description: Structured code review family for explicit heavy review requests and architecture-sensitive review/evaluation requests. Route to PR/MR review, local changeset review, or architecture review based on the review target and intent. Do not use for routine self-checks or casual bug scans.
---
Use this skill family when the user explicitly asks for a heavy review flow, or when the task is already a review/evaluation request and the auto-trigger rules in `references/triggering-and-signal.md` select a heavy review path.

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
  Use for explicit architecture review or for review/evaluation requests
  where layering, contracts, compatibility surfaces, abstractions, or public
  API design are the main question.

If the target is a PR/MR or local changeset and the user explicitly asks for
architecture review in addition to ordinary review dimensions, or if the
reviewed scope materially changes layering, ownership boundaries, dependency
direction, compatibility/public surfaces, or other architecture-review
dimensions, keep `prmr-codereview` or `changeset-codereview` as the active
skill. For PR/MR reviews, the delegated `changeset-codereview` workflow owns
any paired `architecture-review/` launch. In both cases, the active skill
owns the main report and must summarize the paired architecture results in
`Related review inputs`.

Do not use this family for:

- routine implementation self-checks
- quick bug scans during development
- ordinary debugging
- style-only or taste-only feedback

Use the smallest review surface that satisfies the request. The heavier the
workflow, the more explicit the user request should be.

These heavy review workflows are delegation-required. If the active tool
policy or current user authorization does not permit subagents for the chosen
sub-skill, stop and ask for explicit delegation permission instead of
silently collapsing the workflow into a single-agent review.
Treat the global subagent budget as shared with any other active user work.
Keep only the current review wave open: once triage, discovery, summary,
reviewer, validator, or paired-review outputs have been incorporated, close
those completed agents before launching the next review phase.
When a codereview sub-skill specifies a number of subagents or distinct
reviewer roles, treat that count and role split as the minimum required
workflow shape. Do not silently launch fewer subagents, merge distinct review
roles into one agent, or skip the validation pass.
