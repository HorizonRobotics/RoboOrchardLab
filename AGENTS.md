# AGENTS.md

Repository instructions live in `.agents/instructions/`.
Repository references live in `.agents/references/`.
Repository skills live in `.agents/skills/`.

## Guidance Scope and Precedence

- For files in this repository, use only this local `AGENTS.md` and local `.agents/` as authoritative guidance.
- Do not inherit or fall back to guidance from any containing workspace or parent repository.
- Treat this repository independently.

## Read First

- Read `.agents/instructions/default.instructions.md` only when the task
  touches this repository's files, workflows, or behavior.
- When editing `AGENTS.md`, `.agents/instructions/`, `.agents/references/`, or `.agents/skills/`,
  also read `.agents/instructions/guidance-authoring.instructions.md`.
- Use `Quick Routing` below to load topic-specific instruction or reference files when relevant.

### Quick Routing

- Python implementation changes: `.agents/instructions/python.instructions.md`
- Architecture review, layering, abstraction seams, or public API surface design:
  `.agents/references/architecture-review-guideline.md`
- License header requirements and template: `.agents/references/license-header-guideline.md`
- Test creation, updates, or validation: `.agents/instructions/test.instructions.md`
- Validation scope and developer workflow decisions: `.agents/instructions/workflow.instructions.md`
- Commit messages, branches, merge requests, or pull requests: `.agents/instructions/git.instructions.md`
- Environment, runtime, hardware, or external-service constraints: `.agents/instructions/environment.instructions.md`
- Fresh environment bootstrap or local setup: `.agents/instructions/prepare_env.instructions.md`
- Spatial transform, pose, or matrix naming: `.agents/references/spatial-transform-and-matrix-naming-guideline.md`
- `AGENTS.md`, `.agents/instructions/`, `.agents/references/`, or `.agents/skills/` authoring and refactors: `.agents/instructions/guidance-authoring.instructions.md`

## Skills

- When a task matches a skill, read the relevant `.agents/skills/*/SKILL.md` file before proceeding.
- Current top-level skills:
  - `.agents/skills/codereview/SKILL.md`
  - `.agents/skills/feature-dev/SKILL.md`
- `feature-dev` is split into finer-grained sub-skills under `.agents/skills/feature-dev/`:
  - `code-architect`
  - `code-explorer`
  - `code-reviewer`
- `codereview` is split into finer-grained sub-skills under `.agents/skills/codereview/`:
  - `changeset-codereview`
  - `prmr-codereview`
  - `architecture-review`
- `feature-dev/code-reviewer` is the lighter-weight local implementation review sub-skill.

## Repository Notes

- Treat `.agents/instructions/` as the source of truth for agent guidance.
- Treat `.agents/references/` as the source of truth for stable agent-facing reference guidance.
- Treat `.agents/skills/` as the source of truth for task-specific skill workflows.
- Keep this file as a pointer; do not copy instruction content here.
- Use `Makefile` and `pyproject.toml` as the workflow and tool-config sources of truth when applicable.
- Use pytest config under `tests/` as the pytest source of truth.
- Prefer source files over generated copies under `build/`.
- Do not modify `build/` unless the task explicitly targets build outputs.
