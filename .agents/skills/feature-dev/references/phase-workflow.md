# Feature Development Phase Workflow

This reference contains the executable phase details for
`.agents/skills/feature-dev/SKILL.md`.

## Phase 1: Discovery

Goal: understand what needs to be built.

Actions:

1. Create a todo list with all phases that apply.
2. If the feature is unclear, ask the user for the problem being solved, the
   expected behavior, and known constraints.
3. Summarize the understanding and confirm it when uncertainty would affect
   design or implementation.

Keep questions specific. Avoid broad intake forms when a short repository
inspection would answer the question.

## Phase 2: Codebase Exploration

Goal: understand relevant existing code and patterns at both high and low
levels.

Actions:

1. Launch focused `code-explorer` style analysis passes when the area is broad
   or unfamiliar. Each pass should trace the code comprehensively and target a
   distinct aspect such as similar features, architecture, testing, UX, or
   integration points.
2. Require each exploration pass to return the most important files to read.
3. Read the identified files before making architectural claims.
4. Summarize entry points, execution flow, ownership boundaries, extension
   points, and existing test patterns.

Example exploration prompts:

- Find features similar to `<feature>` and trace their implementation.
- Map architecture and abstraction boundaries for `<feature area>`.
- Identify tests, validation commands, or integration points relevant to
  `<feature>`.

## Phase 3: Clarifying Questions

Goal: resolve material ambiguity before designing.

Do not skip this phase when behavior, integration, compatibility, performance,
failure handling, or ownership boundaries are underspecified.

Actions:

1. Review the codebase findings and original request.
2. Identify open questions that affect the design.
3. Present questions in one organized batch when possible.
4. Wait for answers before continuing when the answers materially affect the
   architecture or user-visible behavior.

If the user says "whatever you think is best", provide a recommendation and
get explicit confirmation when the choice has product or architectural impact.

## Phase 4: Architecture Design

Goal: choose an implementation approach grounded in repository patterns.

Actions:

1. Use `code-architect` style design passes when multiple credible approaches
   exist. Prefer distinct focuses such as minimal change, clean architecture,
   and pragmatic balance.
2. Compare approaches by files touched, ownership boundaries, compatibility,
   validation burden, and long-term complexity.
3. Recommend one approach with concrete reasoning.
4. Ask the user to choose when the decision is material.

Keep architecture choices scoped to the request. Do not invent a new
abstraction unless it removes real complexity, reduces meaningful duplication,
or matches an established local pattern.

## Phase 5: Implementation

Goal: build the selected feature.

Actions:

1. Wait for explicit approval if clarification or design approval is still
   pending.
2. Re-read relevant files identified in previous phases when needed.
3. Implement the chosen approach using repository conventions.
4. Keep edits scoped to the ownership boundaries implied by the task.
5. Add or update tests when behavior changes.
6. Update the todo list as progress changes.

Implementation should follow the patterns discovered in Phase 2 and the
architecture selected in Phase 4.

## Phase 6: Quality Review

Goal: ensure the result is simple, maintainable, project-fit, and correct.

This phase supplements the repository's normal review, validation, and Cleanup
Gate requirements in `.agents/instructions/workflow.instructions.md`.

Actions:

1. Decide review depth based on change risk and the user's request.
2. For small, local, low-risk changes, run a main-agent self-review,
   validation, and the repository Cleanup Gate.
3. For public API, persisted format, boundary, concurrency, lifecycle,
   compatibility, or explicit review-loop work, launch at most one focused
   `code-reviewer` style delegated review by default.
4. When delegating review, include all relevant Cleanup Gate, architecture,
   correctness, simplicity, and project-convention dimensions in the single
   reviewer prompt rather than splitting dimensions across agents. Start each
   fresh reviewer without inherited parent-thread history by default
   (`fork_context: false` or `fork_turns: "none"`) and give it a compact,
   self-contained scope that points to the current files, diff, guidance, and
   validation evidence.
5. Consolidate findings and identify material issues.
6. Ask the user before proceeding when material unresolved issues require a
   product or architecture decision.
7. Fix accepted material findings, then rerun focused validation.
8. For substantial changes or explicit review-loop requests, keep the current
   reviewer thread responsible for verifying fixes to its own findings rather
   than spawning a replacement verifier. Once that round's findings are
   resolved, close it and launch one fresh no-parent-context reviewer
   against the full current scope for the next issue-discovery round. Stop only
   when a fresh round returns no new must-fix findings or the user explicitly
   accepts the remaining risk. Keep at most one active reviewer per round
   unless parallel review was explicitly requested or justified by concrete
   high risk.
9. Before finalizing, run the repository Cleanup Gate from
   `.agents/instructions/workflow.instructions.md`.

Review findings should be high-signal. Prefer concrete file references,
validated risks, and specific fix directions.

## Phase 7: Summary

Goal: hand back the current state clearly.

Actions:

1. Mark completed todo items.
2. Summarize what was built.
3. List key decisions and preserved boundaries.
4. List files modified when useful.
5. Report validation commands and outcomes.
6. State blocked checks or residual risks.

Keep the handoff short enough to be useful, but include enough evidence for
the user or next agent to continue without reconstructing the session.

## Sub-skill Expected Outputs

`code-explorer` should return:

- entry points with file references
- execution flow and data transformations
- key components and responsibilities
- architecture patterns and extension points
- essential files to read next

`code-architect` should return:

- patterns and conventions found
- recommended architecture and rationale
- files to create or modify
- component responsibilities and data flow
- implementation phases and critical considerations

`code-reviewer` should return:

- high-confidence findings only
- severity and rationale
- affected files or areas
- concrete fix direction

## Compression Rules

- Use the full workflow for features spanning multiple files or requiring
  design trade-offs.
- Compress the workflow for trivial tasks.
- Do not ask broad, low-value questions before codebase exploration.
- Prefer one organized batch of clarifying questions over repeated
  interruptions.
- Ground recommendations in repository patterns rather than generic advice.
