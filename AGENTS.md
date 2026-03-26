# AGENTS.md

Repository instructions live in `.agents/instructions/`.
Repository skills live in `.agents/skills/`.

## Guidance Scope and Precedence

- For files in this repository, use only this local `AGENTS.md` and local `.agents/` as authoritative guidance.
- Do not inherit or fall back to guidance from any containing workspace or parent repository.
- Treat this repository independently.

## Read First

- Always read `.agents/instructions/default.instructions.md`.
- Also read these when relevant:
  - `.agents/instructions/python.instructions.md`
  - `.agents/instructions/test.instructions.md`
  - `.agents/instructions/workflow.instructions.md`
  - `.agents/instructions/git.instructions.md`
  - `.agents/instructions/environment.instructions.md`

### Quick Routing

- Python implementation changes: `.agents/instructions/python.instructions.md`
- Test creation, updates, or validation: `.agents/instructions/test.instructions.md`
- Validation scope and developer workflow decisions: `.agents/instructions/workflow.instructions.md`
- Commit messages, branches, merge requests, or pull requests: `.agents/instructions/git.instructions.md`
- Environment, runtime, hardware, or external-service constraints: `.agents/instructions/environment.instructions.md`

## Skills

- When a task matches a skill, read the relevant `.agents/skills/*/SKILL.md` file before proceeding.
- Current top-level skills:
  - `.agents/skills/remote-codereview/SKILL.md`
  - `.agents/skills/feature-dev/SKILL.md`
- `feature-dev` is split into finer-grained sub-skills under `.agents/skills/feature-dev/`:
  - `code-architect`
  - `code-explorer`
  - `code-reviewer`
- `remote-codereview` is for remote GitHub PR / GitLab MR review flows; `feature-dev/code-reviewer` is the lighter-weight local implementation review sub-skill.

## Repository Notes

- Treat `.agents/instructions/` as the source of truth for agent guidance.
- Treat `.agents/skills/` as the source of truth for task-specific skill workflows.
- Keep this file as a pointer; do not copy instruction content here.
- Use `Makefile` and `pyproject.toml` as the workflow and tool-config sources of truth when applicable.
- Use pytest config under `tests/` as the pytest source of truth.
- Prefer source files over generated copies under `build/`.
- Do not modify `build/` unless the task explicitly targets build outputs.
