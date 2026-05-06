# AGENTS.md

Repository instructions live in `.agents/instructions/`.
Repository references live in `.agents/references/`.
Repository skills live in `.agents/skills/`.
Reusable agent-facing templates live in `.agents/templates/`.

## Guidance Scope and Precedence

- For files in this repository, use only this local `AGENTS.md` and local `.agents/` as authoritative guidance.
- Do not inherit or fall back to guidance from any containing workspace or parent repository.
- Treat this repository independently.

## Read First

- Read `.agents/instructions/default.instructions.md` only when the task
  touches this repository's files, workflows, or behavior.
- When editing `AGENTS.md`, `.agents/instructions/`, `.agents/references/`, `.agents/skills/`, or `.agents/templates/`,
  also read `.agents/instructions/guidance-authoring.instructions.md`.
- Use `Quick Routing` below to load topic-specific instruction or reference files when relevant.

### Quick Routing

- Python implementation changes: `.agents/instructions/python.instructions.md`
- `DictTransform`, `DictTransformPipeline`, `ConcatDictTransform`,
  `transform` / `apply` / `__call__`, output mapping, or reusable
  transform/test scaffolds:
  `.agents/references/dict-transform-guideline.md` and
  `.agents/templates/dict-transform-scaffold.md`
- Processor, envelope, compose, or `pre_process`/`post_process` contract changes: `.agents/references/processor-guideline.md`
- Complex feature/refactor design notes, design-doc structure, or plan
  readiness:
  `.agents/references/design-doc-guideline.md` and
  `.agents/templates/design-doc-scaffold.md`
- Policy, rollout boundary, or `act(...)` / `reset(...)` contract changes:
  `.agents/references/policy-guideline.md`
- Shared model-loading surfaces, `TorchModelRef`,
  `HFPretrainedModelRef`, `TorchModelLoadConfig`, or `hf://` compatibility:
  `.agents/references/model-loading-guideline.md`
- `State`, `StateSaveLoadMixin`, `obj2state` / `state2obj`,
  runtime recovery payloads, or cross-domain recoverability contracts:
  `.agents/references/state-recovery-guideline.md`
- `PolicyEvaluator`, `PolicyEvaluatorRemote`, episode contracts,
  metric capability dispatch, recovery snapshot behavior, or
  multi-episode orchestration guidance:
  `.agents/references/policy-evaluator-guideline.md`
- Benchmark evaluator, `LocalBenchmarkBackend`, `RemoteBenchmarkBackend`,
  benchmark driver contracts, benchmark retry, or worker scheduling:
  `.agents/references/benchmark-evaluator-guideline.md`
- Robot interactive env, `reset()` / `step()` observation contracts, or
  env-owned runtime metadata:
  `.agents/references/robot-interactive-env-guideline.md`
- `RoboTwinEnv`, RoboTwin reset/seed/task config behavior, post-reset
  recreate State, or RoboTwin compatibility rules:
  `.agents/references/robotwin-env-guideline.md`
- Key interface docstrings for public classes/functions or boundary helpers
  whose safe use is not obvious from the signature alone, including
  resource-owning or stateful interfaces: start with
  `.agents/references/interface-docstring-guideline.md`; use
  `.agents/templates/interface-docstring-scaffold.md` only when a short
  drafting scaffold is helpful
- Architecture review, layering, abstraction seams, or public API surface design:
  `.agents/references/architecture-review-guideline.md`
- License header requirements and template: `.agents/references/license-header-guideline.md`
- Test creation, updates, or validation: `.agents/instructions/test.instructions.md`
- Validation scope and developer workflow decisions: `.agents/instructions/workflow.instructions.md`
- Commit messages, branches, merge requests, or pull requests: `.agents/instructions/git.instructions.md`
- Code runtime environment, hardware, or external-service constraints:
  `.agents/instructions/environment.instructions.md`
- Fresh environment bootstrap or local setup: `.agents/instructions/prepare_env.instructions.md`
- Spatial transform, pose, or matrix naming: `.agents/references/spatial-transform-and-matrix-naming-guideline.md`
- Distilling stable implementation, review, or design-process lessons into local guidance:
  `.agents/instructions/experience-distillation.instructions.md`
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
- Treat `.agents/templates/` as the source of truth for reusable
  agent-facing implementation and test scaffolds.
- Keep this file as a pointer; do not copy instruction content here.
- Use `Makefile` and `pyproject.toml` as the workflow and tool-config sources of truth when applicable.
- Use pytest config under `tests/` as the pytest source of truth.
- Prefer source files over generated copies under `build/`.
- Do not modify `build/` unless the task explicitly targets build outputs.
