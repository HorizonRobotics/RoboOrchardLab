---
name: prmr-codereview
description: Review a GitHub PR or GitLab MR when the task is a PR/MR review or comment flow. Use this skill for PR/MR-specific transport, metadata, and comment workflows.
---
Provide a code review for the given pull request or merge request.

Use this skill after `.agents/skills/codereview/SKILL.md` routes the task
here. Read `../references/triggering-and-signal.md` and
`../references/review-depth-and-delegation.md` first. When composing the final
report, also read `../references/report-composition.md`.

Do not use this skill for commit, branch, staged, working-tree, patch, or
generic local review. Use `../changeset-codereview/SKILL.md` for those. If
architecture review is explicitly requested or materially applicable, pass
that dimension into the changeset workflow rather than launching a separate
architecture agent here. Keep this skill as the main report owner and
summarize paired logical review inputs in `Related review inputs`.

Follow `../references/review-depth-and-delegation.md`. The main agent owns
PR/MR triage, metadata, prior-comment inspection, and comment transport. The
delegated changeset review owns the one-reviewer-per-round budget and review
loop transitions.

To do this:

1. Check locally whether any stop condition applies:
   - The PR/MR is closed.
   - It is a draft.
   - It is automated or trivial and clearly does not need code review.
   - The current authenticated reviewer already reviewed the same head and
     the user did not request re-review.

   If a condition applies, stop. If the head advanced or the user explicitly
   requested re-review, continue with a fresh pass on the current head.

2. Fetch the title, description, source and target branches, current head,
   file list, and full current diff.
   - For re-review, also fetch the last reviewed head when available, the
     incremental diff, and prior findings or comment threads that need status
     verification.

3. If comment publication was requested, determine whether summary comments,
   inline comments, or both are required.

4. Run `../changeset-codereview/SKILL.md` against the full current PR/MR diff.
   Pass the title, description, prior-findings ledger, last reviewed head, and
   any architecture-review requirement as context. Do not launch another
   PR/MR-specific review agent; use the current round's changeset reviewer
   across all applicable review dimensions.

5. Convert the validated findings into `REPORT_TEMPLATE.md`.
   - Summarize paired logical review inputs without copying their findings
     into the main findings sections.
   - Include a required architecture summary when architecture review applied.

   If `--comment` was not provided, stop here. If it was provided and no issues
   were found, post the summary comment and stop.

6. Before posting issue comments, prepare and verify the complete comment set.
   - For re-review, separate fixed prior findings, unresolved prior findings,
     and newly introduced validated issues.
   - Run one convergence check against the current full diff.

7. Post validated issue comments using the platform-appropriate mechanism.
   - For GitHub, use the GitHub inline comment tool with `confirmed: true`.
   - For GitLab, use available GitLab review or note tooling. If inline
     comments are unavailable, post only the top-level summary and state that
     inline comments were skipped due to tool limitations.
   - Use committable suggestions only for small fixes that the suggestion
     resolves completely.
   - Reuse existing discussions for previously reported findings when
     possible; open new comments only for new findings.

Notes:

- Use `gh` for GitHub and `glab` for GitLab. Do not use web fetch.
- Keep PR/MR transport and publication logic here; keep generic diff review in
  `changeset-codereview`.
