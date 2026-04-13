---
name: changeset-codereview
description: Review a commit, branch diff, staged diff, working tree diff, patch, or file set for validated high-signal bugs and scoped guidance violations when the user explicitly asks for a code review on that changeset.
---
Provide a code review for the given local changeset.

Use this skill after `.agents/skills/codereview/SKILL.md` routes the task
here. Read `../references/triggering-and-signal.md` first. When composing the
final report, also read `../references/report-composition.md`.

If the user explicitly wants architecture review rather than bug or guidance
review, use `../architecture-review/SKILL.md` instead of broadening this
skill.

**Agent assumptions (applies to all agents and subagents):**
- All tools are functional and will work without error. Do not test tools or
  make exploratory calls. Make sure this is clear to every subagent that is
  launched.
- Only call a tool if it is required to complete the task. Every tool call
  should have a clear purpose.
- Choose subagents by capability tier rather than vendor-specific model
  names: use a lightweight reviewer for scope discovery or file lists, a
  general reviewer for balanced summaries or guidance checks, and the
  strongest available reviewer for bug finding or issue validation.

To do this, follow these steps precisely:

1. Determine the reviewed changeset.
   - Identify whether the target is a commit, branch diff, staged diff,
     working tree diff, patch, or explicit file set.
   - Choose the smallest reasonable diff or file scope that satisfies the
     request.
   - If the target is ambiguous, make a reasonable local choice and state it
     in the report metadata.

2. Launch a lightweight review subagent to return a list of file paths (not
   their contents) for all relevant repository guidance files including:
   - The root `AGENTS.md` file, if it exists
   - Any directory-scoped `AGENTS.md` files that apply to modified files
   - Any files under `.agents/instructions/`, `.agents/references/`, or
     `.agents/skills/` referenced by those `AGENTS.md` files and relevant to
     the review scope

3. Launch a general review subagent to summarize the reviewed changeset.

4. Launch 4 agents in parallel to independently review the changes. Each
   agent should return the list of issues, where each issue includes a
   description and the reason it was flagged.

   Agents 1 + 2: repository guidance compliance reviewers
   Audit changes for `AGENTS.md` / `.agents` guidance compliance in parallel.
   When evaluating guidance compliance for a file, only consider the guidance
   files that are in scope for that file, including applicable parent
   `AGENTS.md` files and the `.agents` instruction/reference/skill files
   they reference.

   Agent 3: deep bug reviewer
   Scan for obvious bugs. Focus only on the reviewed changeset. Flag only
   significant bugs; ignore nitpicks and likely false positives.

   Agent 4: deep bug reviewer
   Look for problems introduced by the reviewed changeset. This could be
   security issues, incorrect logic, or clear contract breakage.

   **CRITICAL: We only want HIGH SIGNAL issues.** Flag issues where:
   - The code will fail to compile or parse
   - The code will definitely produce wrong results regardless of inputs
   - Clear, unambiguous repository guidance violations where you can quote
     the exact rule being broken

5. For each issue found in the previous step, launch parallel subagents to
   validate the issue. The validator's job is to confirm that the issue is
   truly real with high confidence in the reviewed scope. For repository
   guidance issues, validate that the cited `AGENTS.md` / `.agents` rule is
   actually in scope and actually violated. Use the strongest available
   reviewer tier for bugs and logic issues, and a general review tier for
   guidance violations.

6. Filter out any issues that were not validated. De-duplicate overlapping
   issues across all reviewers, then assign a final severity to each
   remaining issue.

7. Output a summary using `REPORT_TEMPLATE.md`.
   - If this workflow used any paired review skill, include a short
     `Related review inputs` summary for each paired skill. Keep those
     summaries concise and reference the paired report instead of copying its
     findings into the main findings sections.
   - If no issues were found, use the exact text:
     `No issues found. Checked for bugs and scoped guidance compliance.`
   - If issues were found, include only validated, de-duplicated
     high-signal findings.
