---
name: architecture-review
description: Review local code, diffs, modules, directories, or design changes for consequential architecture issues when the user explicitly asks for architecture review or when the task is primarily an architecture-focused refactor.
---
Provide an architecture review for the given local target.

Use this skill after `.agents/skills/codereview/SKILL.md` routes the task
here. Read `../references/triggering-and-signal.md` first.

Do not use this skill for ordinary PR/MR bug review when architecture is not
the main question. Use `../prmr-codereview/SKILL.md` for that flow. Do not
use this skill for routine implementation self-checks or casual cleanup
opinions.

**Agent assumptions (applies to all agents and subagents):**
- All tools are functional and will work without error. Do not test tools or
  make exploratory calls. Make sure this is clear to every subagent that is
  launched.
- Only call a tool if it is required to complete the task. Every tool call
  should have a clear purpose.
- Choose subagents by capability tier rather than vendor-specific model
  names: use a lightweight reviewer for scope discovery, a general reviewer
  for structural summaries, and the strongest available reviewer for issue
  finding or validation.

To do this, follow these steps precisely:

1. Determine the review target and scope.
   - Identify whether the review target is a local directory, file set, git
     diff, branch diff, or design-oriented change.
   - Identify the paths that actually belong to the requested review scope.
   - If the request is ambiguous, choose the smallest reasonable scope that
     satisfies the request.

2. Gather only the in-scope guidance.
   - Load the root `AGENTS.md` file, if it exists.
   - Load any directory-scoped `AGENTS.md` files that apply to the target.
   - Load the in-scope `.agents/instructions/`, `.agents/references/`, and
     `.agents/skills/` files referenced by those `AGENTS.md` files.
   - Always include `.agents/references/architecture-review-guideline.md`
     when it is in scope.
   - If an applicable package-local supplement exists, load that supplement
     only after the root architecture baseline.

3. Produce a concise structural summary before finding issues.
   - Summarize the main layers, contracts, and caller-facing boundaries in
     the target.
   - State which architecture dimensions are materially relevant to this
     review.

4. Launch 3 reviewers in parallel.
   - Reviewer 1: boundary and ownership reviewer
   - Reviewer 2: contract and compatibility reviewer
   - Reviewer 3: abstraction and evolvability reviewer

   Each reviewer should return only issues with:
   - a concrete architectural problem
   - clear impact on maintainability, compatibility, extension cost, or
     correctness boundary
   - enough evidence to validate the issue from the reviewed scope

5. Validate each proposed issue.
   - Launch validation reviewers in parallel for candidate issues.
   - Validate that the issue is real within the stated scope and that the
     claimed impact is concrete.
   - Validate any cited `AGENTS.md` / `.agents` guidance rule is actually in
     scope for the affected files.

6. Filter and de-duplicate.
   - Remove unvalidated issues.
   - Merge overlapping issues across reviewers.
   - Keep only high-signal findings.

7. Output a report using `REPORT_TEMPLATE.md`.
   - State the reviewed scope and the architecture dimensions actually
     reviewed.
   - If no issues were found, use the exact text:
     `No issues found. Checked the reviewed architecture dimensions.`
