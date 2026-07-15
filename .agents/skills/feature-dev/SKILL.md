---
name: feature-dev
description: Guided feature development with codebase understanding and architecture focus. Use when implementing a non-trivial feature that benefits from structured discovery, exploration, architecture comparison, implementation, and review.
---

Use this skill to coordinate non-trivial feature work from discovery through
implementation, review, validation, and summary.

## Read Next

Before running this workflow, read:

- `references/phase-workflow.md`

Use the sub-skills in this directory only when their phase applies:

- `code-explorer`: deep codebase tracing and implementation analysis
- `code-architect`: architecture design and implementation blueprinting
- `code-reviewer`: high-signal review for correctness and project fit

## Role

This parent skill is the orchestration layer. Keep it focused on routing,
phase selection, and handoff between sub-skills. Concrete execution details
belong in the phase workflow reference or in the smallest applicable sub-skill.

## Core Principles

- Understand the codebase before changing it.
- Ask concrete clarifying questions when behavior, scope, compatibility, or
  integration points are underspecified.
- Read files surfaced by exploration or design passes before making
  architectural claims.
- Prefer simple, maintainable designs grounded in repository patterns.
- Track progress with a todo list when the work spans multiple steps.
- Compress the workflow for trivial tasks, but preserve the same reasoning
  discipline.

## Workflow Summary

The detailed actions, examples, and review rules live in
`references/phase-workflow.md`. The phase order is:

1. Discovery: understand the request and constraints.
2. Codebase exploration: map relevant implementation patterns.
3. Clarifying questions: resolve material ambiguity before designing.
4. Architecture design: compare viable approaches and choose one.
5. Implementation: build the approved or obvious approach.
6. Quality review: run risk-appropriate review, validation, and Cleanup Gate.
7. Summary: report decisions, changed files, validation, and residual risks.

## Delegation Rules

- Use `code-explorer` for broad or unfamiliar areas before designing.
- Use `code-architect` when multiple credible architecture choices exist.
- Use `code-reviewer` for non-trivial implementation review or explicit review
  loops.
- Default to at most one active delegated reviewer per review round. Treat
  architecture, correctness, simplicity, and project conventions as review
  dimensions that one reviewer can cover together.
- In a review loop, keep the current reviewer only long enough to verify fixes
  for its own findings, then start the next full-scope round with a fresh
  reviewer.
- Use parallel reviewers within one round only when the user explicitly
  requests them or a concrete high-risk scope justifies independent review.
- After delegated review, the main agent still owns consolidation,
  validation, and final handoff quality.

## When Not To Use

This workflow is usually overkill for:

- one-line bug fixes
- trivial refactors
- clearly scoped edits in a single file
- urgent hotfixes where architecture comparison adds no value
