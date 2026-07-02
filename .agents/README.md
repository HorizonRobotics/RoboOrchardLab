# Agent Asset Index

This directory stores repository-local agent guidance and reusable
agent-facing assets for `robo_orchard_lab`.

## Local Assets

- `instructions/`: repository-local guidance and workflow entrypoints.
- `references/`: stable agent-facing reference guidance and decision
  summaries.
- `skills/`: task-specific local workflows.
- `templates/`: reusable fill-in skeletons for repeated implementation,
  design, and documentation work.

Use `AGENTS.md` for routing and precedence. Keep durable rules in
`references/`, workflow expectations in `instructions/`, and repeated output
or planning shapes in `templates/`.

## Current References

- Architecture and documentation:
  `references/architecture-review-guideline.md`,
  `references/design-doc-guideline.md`,
  `references/interface-docstring-guideline.md`, and
  `references/license-header-guideline.md`.
- Data transforms and processors:
  `references/dict-transform-guideline.md`,
  `references/processor-guideline.md`, and
  `references/model-loading-guideline.md`.
- Transformers upgrade and compatibility:
  `references/transformers-upgrade-guideline.md`.
- Policy, evaluation, and benchmark workflows:
  `references/policy-guideline.md`,
  `references/policy-evaluator-guideline.md`, and
  `references/benchmark-evaluator-guideline.md`.
- Environment and runtime state:
  `references/robot-interactive-env-guideline.md`,
  `references/robotwin-env-guideline.md`, and
  `references/state-recovery-guideline.md`.
- RODataset metadata, repack, and persistence:
  `references/rodataset-metadata-guideline.md`,
  `references/rodataset-repack-guideline.md`, and
  `references/rodataset-upgrade-guideline.md`.
- Experimental MCAP export:
  `references/mcap-guideline.md`.
- Naming conventions:
  `references/spatial-transform-and-matrix-naming-guideline.md`.

## Current Instructions

- `instructions/default.instructions.md`: baseline scope, safety, and
  reporting rules.
- `instructions/python.instructions.md`: Python implementation guidance.
- `instructions/environment.instructions.md` and
  `instructions/prepare_env.instructions.md`: runtime, hardware,
  external-service, and setup guidance.
- `instructions/test.instructions.md`: test creation, updates, and
  validation expectations.
- `instructions/workflow.instructions.md`: validation scope and developer
  workflow decisions.
- `instructions/git.instructions.md`: commit, branch, merge-request, and
  pull-request guidance.
- `instructions/guidance-authoring.instructions.md`: guidance authoring,
  placement, and consistency rules.
- `instructions/experience-distillation.instructions.md`: local experience
  distillation routing for guidance assets and memory.

## Current Skills

- `skills/codereview/SKILL.md`: local code-review family.
- `skills/feature-dev/SKILL.md`: local feature development workflow.

## Current Templates

- `templates/design-doc-scaffold.md`: design note scaffold.
- `templates/dict-transform-scaffold.md`: `DictTransform` implementation and
  test scaffold.
- `templates/interface-docstring-scaffold.md`: interface docstring drafting
  scaffold.

## Maintenance Rules

- Check this inventory before adding a new local shared reference or
  template.
- Update this file whenever local shared assets are added, removed, moved, or
  substantially re-scoped.
- Keep temporary design notes under `.agents/scratch/`; do not list them as
  shared assets.
