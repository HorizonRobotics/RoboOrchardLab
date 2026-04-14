---
name: prmr-codereview
description: Review a GitHub PR or GitLab MR when the user explicitly asks for PR/MR review or PR/MR comment posting. Use this skill for PR/MR-specific transport, metadata, and comment workflows.
---
Provide a code review for the given pull request or merge request.

Use this skill after `.agents/skills/codereview/SKILL.md` routes the task
here. Read `../references/triggering-and-signal.md` first. When composing the
final report, also read `../references/report-composition.md`.

Do not use this skill for commit review, branch diff review, staged diff
review, working tree review, or generic local code review. Use
`../changeset-codereview/SKILL.md` for those. If the user explicitly wants
architecture review of a PR/MR, pass that requirement into the delegated
changeset review workflow rather than launching a separate architecture pass
here. Keep this skill as the main report owner and summarize any paired
review result in `Related review inputs`.

**Agent assumptions (applies to all agents and subagents):**
- All tools are functional and will work without error. Do not test tools or
  make exploratory calls. Make sure this is clear to every subagent that is
  launched.
- Only call a tool if it is required to complete the task. Every tool call
  should have a clear purpose.
- Choose subagents by capability tier rather than vendor-specific model
  names: use a lightweight reviewer for simple PR/MR triage or file
  discovery, a general reviewer for balanced summaries, and the strongest
  available reviewer for issue validation when needed.
- This workflow requires subagents. Do not silently collapse it into
  a single-agent review. If delegation is unavailable or not yet
  authorized, stop and ask the user for explicit delegation
  permission before proceeding.
- Treat each required subagent step from this skill and the delegated
  `changeset-codereview` workflow as mandatory. Do not reduce the required
  subagent count or merge distinct reviewer roles just to keep the flow
  moving.

To do this, follow these steps precisely:

1. Launch a lightweight review subagent to check if any of the following are true:
   - The pull request is closed
   - The pull request is a draft
   - The pull request does not need code review (for example automated PR or trivial change that is obviously correct)
   - A previous automated review from the current authenticated reviewer account has already commented on this review target

   If any condition is true, stop and do not proceed.

2. Fetch the PR/MR title, description, source branch, target branch, file
   list, and diff.

3. If the user asked to publish PR/MR comments, determine whether summary
   comments, inline comments, or both are required.

4. Load `../changeset-codereview/SKILL.md` and run its generic
   issue-discovery and validation workflow against the reviewed PR/MR diff.
   Pass the PR/MR title, description, and any explicit architecture-review
   request to that workflow as review context.

5. Convert the validated findings into the PR/MR-specific report format in
   `REPORT_TEMPLATE.md`.
   - If this workflow used any paired review skill, include a short
     `Related review inputs` summary for each paired skill. Keep those
     summaries concise and reference the paired report instead of copying its
     findings into the main findings sections.
   - If the delegated changeset review produced a paired architecture input,
     this summary is required; do not omit it from the final report.

   If `--comment` argument was NOT provided, stop here. Do not post any
   review comments.

   If `--comment` argument IS provided and NO issues were found, post a
   summary comment using the same report structure and stop.

   If `--comment` argument IS provided and issues were found, continue to
   step 6.

6. Create a list of all comments that you plan on leaving. This is only for
   you to make sure you are comfortable with the comments. Do not post this
   list anywhere.

7. Post inline comments for each issue using the platform-appropriate
   mechanism.
   - For GitHub PRs, use the GitHub inline comment tool with `confirmed:
     true`.
   - For GitLab MRs, use the available GitLab review or note tooling. If
     inline comments are not available in the current toolset, post the
     top-level summary comment only and explicitly note that inline comments
     were skipped due to tool limitations.
   - Provide a brief description of the issue.
   - For small, self-contained fixes, include a committable suggestion block
     only when the platform supports it.
   - For larger fixes, describe the issue and suggested fix without a
     suggestion block.
   - Never post a committable suggestion unless committing the suggestion
     fixes the issue entirely.

Notes:

- Use platform-appropriate CLI tools for the review target (`gh` for GitHub
  PRs, `glab` for GitLab MRs). Do not use web fetch.
- Keep PR/MR-specific logic here. Keep generic diff review logic in
  `changeset-codereview`.
