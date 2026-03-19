# AGENTS.md

Repository instructions live in `.github/instructions/`.

## Read First

- Always read `.github/instructions/default.instructions.md`.
- Also read these when relevant:
  - `.github/instructions/python.instructions.md`
  - `.github/instructions/workflow.instructions.md`
  - `.github/instructions/git.instructions.md`
  - `.github/instructions/environment.instructions.md`

## Repository Notes

- Treat `.github/instructions/` as the source of truth for agent guidance.
- Keep this file as a pointer; do not copy instruction content here.
- Use `Makefile` and `pyproject.toml` as the workflow and tool-config sources of truth when applicable.
- Use pytest config under `tests/` as the pytest source of truth.
- Prefer source files over generated copies under `build/`.
- Do not modify `build/` unless the task explicitly targets build outputs.
