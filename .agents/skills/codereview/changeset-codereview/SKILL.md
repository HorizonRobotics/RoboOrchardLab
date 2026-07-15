---
name: changeset-codereview
description: Review a commit, branch diff, staged diff, working tree diff, patch, or file set for validated high-signal bugs and scoped guidance violations in a local changeset review.
---
Provide a code review for the given local changeset.

Use this skill after `.agents/skills/codereview/SKILL.md` routes the task
here. Read `../references/triggering-and-signal.md` and
`../references/review-depth-and-delegation.md` first. When composing the final
report, also read `../references/report-composition.md`.

If the user explicitly asks for architecture review in addition to local
changeset review, if an upstream `prmr-codereview` workflow forwards that
requirement, or if the reviewed scope materially changes layering, ownership
boundaries, dependency direction, compatibility/public surfaces, or other
architecture-review dimensions, apply `../architecture-review/SKILL.md` as a
paired logical review against the same scope. Keep this skill as the main
report owner and summarize the paired architecture result in `Related review
inputs`. The paired review normally shares the same review subagent.
Use `../architecture-review/SKILL.md` by itself only when architecture is the
sole requested review dimension.

Follow `../references/review-depth-and-delegation.md`. The main agent owns
guidance discovery and candidate validation. Each heavy changeset-review round
defaults to one strong review subagent covering all applicable review
dimensions.

To do this:

1. Determine the reviewed changeset.
   - Identify whether the target is a commit, branch diff, staged diff,
     working tree diff, patch, or explicit file set.
   - Choose the smallest reasonable diff or file scope that satisfies the
     request.
   - For re-review, default to the full current effective changeset unless the
     user explicitly asks for incremental-only validation.
   - Keep a prior-findings ledger when needed and classify each item as fixed,
     unresolved, or no longer applicable.
   - If the target is ambiguous, make a reasonable local choice and state it
     in the report metadata.

2. Decide whether architecture review is also applicable.
   - Add the architecture-review dimensions when explicitly requested or when
     the scope materially changes layering, ownership, dependency direction,
     compatibility/public surfaces, or other architecture boundaries.
   - Keep architecture candidates separate so they can be summarized as a
     paired review input without launching another subagent by default.

3. Gather and read the relevant repository guidance locally, including:
   - This repository's root `AGENTS.md` file, if it exists
   - Any directory-scoped `AGENTS.md` files that apply to modified files
   - Relevant `.agents/instructions/`, `.agents/references/`, or
     `.agents/skills/` files referenced by those `AGENTS.md` files

4. Summarize the reviewed changeset locally before issue discovery.

5. Launch one strong review subagent for the current round by default. Give it
   the full effective changeset, applicable guidance paths, prior findings,
   and any paired architecture context. Require one structured candidate list
   that covers:
   - scoped repository-guidance compliance
   - bugs, correctness, security, and significant performance risk
   - contracts, compatibility, and caller-facing behavior
   - architecture boundaries and minimality when Step 2 applies

   The reviewer must exhaust the scope for high-signal issues rather than stop
   after the first finding. Keep only candidates with a concrete location,
   impact, evidence, and confidence. Ignore style nits and speculative issues.

6. Validate every candidate locally against the current code, call sites,
   tests, and scoped guidance. Confirm that each cited guidance rule is in
   scope and actually violated. Keep the current reviewer responsible for
   verifying fixes to its own findings. After its ledger is resolved, follow
   the shared review-loop rule and use a fresh reviewer for the next
   issue-discovery round. Do not launch separate validator agents by default.

7. Filter and converge.
   - Remove unvalidated issues.
   - De-duplicate overlapping issues and assign final severity.
   - Compare the validated set with reviewer output and any prior-findings
     ledger before reporting so one pass contains all current findings.

8. Output a summary using `REPORT_TEMPLATE.md`.
   - Summarize paired logical review inputs in `Related review inputs`; do not
     duplicate their full findings in the main findings sections.
   - If architecture review was paired, include its required summary.
   - If no issues were found, use the exact text:
     `No issues found. Checked for bugs and scoped guidance compliance.`
   - If issues were found, include only validated, de-duplicated high-signal
     findings.
