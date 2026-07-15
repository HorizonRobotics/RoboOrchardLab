---
name: architecture-review
description: Review local code, diffs, modules, directories, or design changes for consequential architecture issues when the user explicitly asks for architecture review, when an upstream review routes architecture-sensitive scope here, or when the current review/evaluation request is primarily architectural.
---
Provide an architecture review for the given local target.

Use this skill after `.agents/skills/codereview/SKILL.md` routes the task
here. Read `../references/triggering-and-signal.md` and
`../references/review-depth-and-delegation.md` first.

Do not use this skill for ordinary PR/MR bug review when architecture is not
the main or clearly material question. Use `../prmr-codereview/SKILL.md` for
that flow. Do not use this skill for routine implementation self-checks or
casual cleanup opinions.

Follow `../references/review-depth-and-delegation.md`. The main agent owns
scope, structural summary, guidance discovery, and candidate validation. Each
architecture-review round defaults to one strong review subagent covering all
applicable architecture dimensions.

To do this:

1. Determine the review target and scope.
   - Identify whether the target is a local directory, file set, git diff,
     branch diff, or design-oriented change.
   - Identify the paths that actually belong to the requested scope.
   - If the request is ambiguous, choose the smallest reasonable scope that
     satisfies it.

2. Gather only the in-scope guidance locally.
   - Load this repository's root `AGENTS.md` file, if it exists.
   - Load directory-scoped `AGENTS.md` files that apply to the target.
   - Load referenced `.agents/instructions/`, `.agents/references/`, and
     `.agents/skills/` files that are relevant to the review.
   - Always include `.agents/references/architecture-review-guideline.md`
     when it is in scope.
   - Load applicable package-local supplements only after the repository root
     baseline.

3. Produce a concise structural summary before finding issues.
   - Summarize the main layers, contracts, and caller-facing boundaries.
   - State which architecture dimensions are materially relevant.
   - For abstraction-heavy targets, name the smallest viable one-method or
     no-new-class alternative before accepting new protocols, providers,
     adapters, factories, registries, DTOs, snapshots, or caches.

4. Launch one strong architecture-review subagent for the current round by
   default. Require one structured candidate list covering:
   - boundaries, dependency direction, and ownership
   - contracts, compatibility, and public surfaces
   - abstraction minimality, readability, and evolvability

   Keep only candidates with a concrete architectural problem, clear impact,
   and enough evidence to validate within the reviewed scope. The reviewer
   must check whether proposed abstractions remove real complexity or only add
   indirection, and name deletable surfaces when a simpler owner method,
   helper, or existing contract would carry the same behavior.

5. Validate each candidate locally.
   - Confirm the problem and claimed impact against current call sites and
     ownership boundaries.
   - For overdesign findings, validate the proposed smaller shape.
   - Confirm that cited `AGENTS.md` / `.agents` rules are in scope.
   - Keep the current reviewer responsible for verifying fixes to its own
     findings. After its ledger is resolved, use a fresh reviewer for the next
     issue-discovery round. Do not launch separate validator agents by default.

6. Filter and de-duplicate.
   - Remove unvalidated issues.
   - Merge overlapping candidates.
   - Keep only high-signal findings.

7. Output a report using `REPORT_TEMPLATE.md`.
   - State the reviewed scope and architecture dimensions.
   - Include minimality or overdesign in the reviewed dimensions when
     material.
   - If no issues were found, use the exact text:
     `No issues found. Checked the reviewed architecture dimensions.`
