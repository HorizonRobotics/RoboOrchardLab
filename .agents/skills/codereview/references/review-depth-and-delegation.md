# Code Review Depth And Delegation

Use this reference after `.agents/skills/codereview/SKILL.md` routes a task
into the heavy review family.

## Review Budget

- Treat reviewer roles as logical review dimensions, not as one-agent-per-role
  requirements.
- Routine self-checks and small, local, low-risk changes stay with the main
  agent and should not enter this heavy review family.
- An explicit heavy review defaults to at most one active strong review
  subagent per round. That reviewer covers every applicable review dimension
  in the round.
- Use parallel review subagents within one round only when the user explicitly
  requests parallel review or when a concrete high-risk scope justifies
  independent review, such as security-critical behavior, concurrency,
  persisted-format compatibility, or a broad public contract migration. State
  the reason and keep the expansion bounded.
- If delegation is unavailable, the main agent may complete the review and
  report that no independent delegated pass was available, unless the user
  explicitly required an independent reviewer.

## Main-Agent Responsibilities

The main agent owns:

- target selection and scope control
- applicable `AGENTS.md` and `.agents` guidance discovery
- diff or structural summarization
- PR/MR triage, metadata, and comment transport
- candidate-finding validation against code, call sites, tests, and scoped
  guidance
- de-duplication, severity assignment, convergence, and final reporting

Do not launch separate discovery, triage, summary, or validator agents by
default. Reuse the one review subagent for targeted clarification when useful;
the main agent still validates the result before reporting it.

## Reviewer Context

- Start every fresh issue-discovery reviewer without inherited parent-thread
  history by default. Explicitly select `fork_context: false`, or
  `fork_turns: "none"` on interfaces that expose turn-count forking.
- Give the reviewer a compact self-contained prompt with the repository or
  workdir, target and baseline, effective scope, applicable guidance paths,
  intended behavior and constraints, completed validation, known unrelated
  failures, and expected output. Let it inspect the current files and diff.
- Do not pass the full parent transcript, prior review conclusions, suspected
  findings, or intended fixes to a fresh reviewer. Inherit context only when
  material non-repository information cannot be summarized compactly; state
  the reason and inherit the smallest available slice.
- For fix verification, continue or resume the reviewer thread that reported
  the findings instead of spawning a replacement verifier. After that ledger
  is resolved, close the reviewer and start the next discovery round with a
  fresh no-parent-context reviewer.

## Single-Reviewer Coverage

Give the review subagent the full effective review scope and the applicable
guidance paths. Require one structured candidate list covering every relevant
dimension:

1. scoped repository-guidance compliance
2. bugs, correctness, security, and significant performance risk
3. contracts, compatibility, and caller-facing behavior
4. architecture boundaries, ownership, dependency direction, abstraction
   minimality, and evolvability when architecture review is applicable

Each candidate should include its location, category, concrete impact,
evidence, and confidence. The reviewer should exhaust the scope for
high-signal issues instead of stopping after the first finding.

Paired review skills are logical report inputs and do not imply another
subagent. When architecture review is paired with changeset or PR/MR review,
the same review subagent should return a separate architecture candidate
section so report ownership remains clear.

## Validation And Review Loops

- Validate every candidate locally before reporting it. Reject issues whose
  cited rule is out of scope, whose impact is speculative, or whose evidence
  does not survive current call-site inspection.
- Keep the reviewer that found an issue responsible for verifying the fix and
  resolving its own round's finding ledger.
- Once the current round's findings are resolved, close that reviewer and
  launch one fresh reviewer against the full current effective scope to start
  the next issue-discovery round.
- Stop the review loop only when a fresh round returns no new must-fix
  findings, or when the user explicitly accepts the remaining risk. Do not
  reuse the previous round's reviewer as the issue-discovery reviewer for the
  next round.
- Distinguish a completed no-finding review from a reviewer that timed out,
  failed, or returned no usable result.
