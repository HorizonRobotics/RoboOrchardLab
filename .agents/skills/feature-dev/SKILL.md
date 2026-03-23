---
name: feature-dev
description: Guided feature development with codebase understanding and architecture focus. Use when implementing a non-trivial feature that benefits from structured discovery, exploration, architecture comparison, implementation, and review.
---
You are helping a developer implement a new feature. Follow a systematic approach: understand the codebase deeply, identify and ask about all underspecified details, design elegant architectures, then implement.

This parent skill orchestrates a finer-grained skill system under this directory:
- `code-explorer`: deep codebase tracing and implementation analysis
- `code-architect`: architecture design and implementation blueprinting
- `code-reviewer`: high-signal review for correctness and project fit

## Core Principles

- **Ask clarifying questions**: Identify ambiguities, edge cases, and underspecified behavior. Ask specific, concrete questions rather than making assumptions. Wait for user answers before implementation whenever those answers affect the design.
- **Understand before acting**: Read and comprehend existing code patterns first.
- **Read files identified by sub-skills**: When launching deeper analysis, require a list of the most important files to read, then read them before proceeding.
- **Simple and elegant**: Prioritize readable, maintainable, architecturally sound code.
- **Use a todo list**: Track progress throughout the workflow.

## Required 7-Phase Workflow

### Phase 1: Discovery

**Goal**: Understand what needs to be built.

Initial request: user feature request

**Actions**:
1. Create a todo list with all phases.
2. If the feature is unclear, ask the user for:
   - what problem they are solving
   - what the feature should do
   - constraints or requirements
3. Summarize your understanding and confirm it.

**Example**:
```
You: /feature-dev Add caching
Claude: Let me understand what you need...
   - What should be cached? (API responses, computed values, etc.)
   - What are your performance requirements?
   - Do you have a preferred caching solution?
```

### Phase 2: Codebase Exploration

**Goal**: Understand relevant existing code and patterns at both high and low levels.

**Actions**:
1. Launch 2-3 `code-explorer` style analysis passes in parallel. Each pass should:
   - trace through the code comprehensively
   - focus on abstractions, architecture, and flow of control
   - target a different aspect of the codebase such as similar features, high-level architecture, UX, testing, or integration points
   - include a list of 5-10 key files to read
2. After the analysis returns, read the identified files to build deep understanding.
3. Present a comprehensive summary of findings and patterns discovered.

**Example prompts**:
- Find features similar to [feature] and trace their implementation comprehensively.
- Map the architecture and abstractions for [feature area], tracing through the code comprehensively.
- Analyze the current implementation of [existing feature or area], tracing through the code comprehensively.
- Identify UI patterns, testing approaches, or extension points relevant to [feature].

**Example output**:
```
Found similar features:
- User authentication (src/auth/): Uses JWT tokens, middleware pattern
- Session management (src/session/): Redis-backed, 24hr expiry
- API security (src/api/middleware/): Rate limiting, CORS

Key files to understand:
- src/auth/AuthService.ts:45 - Core authentication logic
- src/middleware/authMiddleware.ts:12 - Request authentication
- src/config/security.ts:8 - Security configuration
```

### Phase 3: Clarifying Questions

**Goal**: Fill in gaps and resolve ambiguities before designing.

**CRITICAL**: Do not skip this phase when the request or integration behavior is underspecified.

**Actions**:
1. Review the codebase findings and original request.
2. Identify underspecified aspects such as edge cases, error handling, integration points, scope boundaries, backward compatibility, design preferences, and performance needs.
3. Present all questions to the user in a clear, organized list.
4. Wait for answers before proceeding to architecture design.

If the user says “whatever you think is best”, provide a recommendation and get explicit confirmation when the choice has product or architectural impact.

**Example**:
```
Before designing the architecture, I need to clarify:

1. OAuth provider: Which OAuth providers? (Google, GitHub, custom?)
2. User data: Store OAuth tokens or just user profile?
3. Existing auth: Replace current auth or add alongside?
4. Sessions: Integrate with existing session management?
5. Error handling: How to handle OAuth failures?
```

### Phase 4: Architecture Design

**Goal**: Design multiple implementation approaches with different trade-offs.

**Actions**:
1. Launch 2-3 `code-architect` style design passes in parallel with different focuses:
   - minimal changes: smallest change, maximum reuse
   - clean architecture: maintainability and elegant abstractions
   - pragmatic balance: speed plus quality
2. Review all approaches and decide which fits best for this task.
3. Present to the user:
   - a brief summary of each approach
   - trade-offs
   - your recommendation and reasoning
   - concrete implementation differences
4. Ask which approach they prefer when the choice is material.

**Example output**:
```
I've designed 3 approaches:

Approach 1: Minimal Changes
- Extend existing AuthService with OAuth methods
- Add new OAuth routes to existing auth router
- Minimal refactoring required
Pros: Fast, low risk
Cons: Couples OAuth to existing auth, harder to test

Approach 2: Clean Architecture
- New OAuthService with dedicated interface
- Separate OAuth router and middleware
- Refactor AuthService to use common interface
Pros: Clean separation, testable, maintainable
Cons: More files, more refactoring

Approach 3: Pragmatic Balance
- New OAuthProvider abstraction
- Integrate into existing AuthService
- Minimal refactoring, good boundaries
Pros: Balanced complexity and cleanliness
Cons: Some coupling remains

Recommendation: Approach 3 - gives you clean boundaries without
excessive refactoring, and fits your existing architecture well.

Which approach would you like to use?
```

### Phase 5: Implementation

**Goal**: Build the feature.

**Do not start without the necessary user approval.**

**Actions**:
1. Wait for explicit approval if clarification or design approval is still pending.
2. Read all relevant files identified in previous phases.
3. Implement the chosen architecture.
4. Follow codebase conventions strictly.
5. Write clean, well-documented code.
6. Update todos as progress is made.
7. Add or update tests when behavior changes.

**Notes**:
- Implementation only starts after the user approves.
- Follow patterns discovered in Phase 2.
- Use the architecture designed in Phase 4.
- Continuously track progress.

### Phase 6: Quality Review

**Goal**: Ensure the result is simple, DRY, elegant, easy to read, and functionally correct.

**Actions**:
1. Launch 3 `code-reviewer` style reviews in parallel with different focuses:
   - simplicity / DRY / elegance
   - bugs / functional correctness
   - project conventions / abstractions
2. Consolidate findings and identify all high-severity issues worth fixing.
3. Present findings to the user and ask what they want to do: fix now, fix later, or proceed as-is. Ask for explicit approval before proceeding when there are material issues to address.
4. Address issues based on the chosen path.
5. Run the smallest useful validation for the files changed.

**Example output**:
```
Code Review Results:

High Priority Issues:
1. Missing error handling in OAuth callback (src/auth/oauth.ts:67)
2. Memory leak: OAuth state not cleaned up (src/auth/oauth.ts:89)

Medium Priority:
1. Could simplify token refresh logic (src/auth/oauth.ts:120)
2. Consider extracting OAuth config validation

All tests pass. Code follows project conventions.

What would you like to do?
```

### Phase 7: Summary

**Goal**: Document what was accomplished.

**Actions**:
1. Mark all todos complete.
2. Summarize:
   - what was built
   - key decisions made
   - files modified
   - validation performed
   - suggested next steps or follow-up risks

**Example**:
```
Feature Complete: OAuth Authentication

What was built:
- OAuth provider abstraction supporting Google and GitHub
- OAuth routes and middleware integrated with existing auth
- Token refresh and session integration
- Error handling for all OAuth flows

Key decisions:
- Used pragmatic approach with OAuthProvider abstraction
- Integrated with existing session management
- Added OAuth state to prevent CSRF

Files modified:
- src/auth/OAuthProvider.ts (new)
- src/auth/AuthService.ts
- src/routes/auth.ts
- src/middleware/authMiddleware.ts

Suggested next steps:
- Add tests for OAuth flows
- Add more OAuth providers (Microsoft, Apple)
- Update documentation
```

## Sub-skill Usage

### `code-explorer`

Use this sub-skill to deeply analyze how an existing feature or subsystem works before implementation.

Expected outputs:
- entry points with file references
- execution flow and data transformations
- key components and responsibilities
- architecture patterns and extension points
- essential files to read next

### `code-architect`

Use this sub-skill to produce concrete implementation blueprints grounded in repository patterns.

Expected outputs:
- patterns and conventions found
- architecture decision and rationale
- files to create or modify
- component responsibilities and data flow
- implementation phases and critical considerations

### `code-reviewer`

Use this sub-skill after implementation to find high-confidence issues that materially matter.

Expected outputs:
- high-confidence findings only
- severity and rationale
- affected files or areas
- concrete fix direction

## Best Practices

- Use the full workflow for features spanning multiple files or requiring design trade-offs.
- Do not ask broad, low-value questions before codebase exploration.
- Prefer one organized batch of clarifying questions over repeated interruptions.
- Read the files surfaced by analysis before making architectural claims.
- Ground recommendations in repository patterns rather than generic advice.
- If the task turns out to be trivial, compress the workflow while preserving the same reasoning discipline.

## When Not to Use This Skill

This workflow is usually overkill for:
- one-line bug fixes
- trivial refactors
- clearly scoped edits in a single file
- urgent hotfixes where architecture comparison adds no value

## Troubleshooting

### Analysis takes too long

**Issue**: Code exploration or architecture analysis feels slow.

**Solution**:
- This is normal for large codebases.
- Analysis passes should run in parallel when possible.
- The extra depth usually pays off in better understanding and fewer wrong turns.

### Too many clarifying questions

**Issue**: Phase 3 asks too many questions.

**Solution**:
- Be more specific in the initial feature request.
- Provide constraints and preferences up front.
- Say "whatever you think is best" only when you truly have no preference.

### Architecture options feel overwhelming

**Issue**: Phase 4 presents too many choices.

**Solution**:
- Trust the recommendation when it is grounded in the codebase analysis.
- If still unsure, ask for more explanation of trade-offs.
- Pick the pragmatic option when in doubt.

## Tips

- **Be specific in the feature request**: More detail usually means fewer clarifying questions.
- **Trust the process**: Each phase builds on the previous one.
- **Review sub-skill outputs carefully**: They surface important insights about the codebase.
- **Do not skip phases**: Each phase exists to reduce mistakes and rework.
- **Use the workflow for learning**: The exploration phase is also a good way to understand your own codebase better.
