---
description: Load these instructions when creating, editing, or reorganizing AGENTS.md files, .agents/instructions/*.md files, .agents/references/* assets, .agents/skills/* guidance, or .agents/templates/* assets in this repository.
---

# Guidance Authoring Instructions

## Guidance Model

- Treat this repository as an independent guidance root.
- Keep local `AGENTS.md`, `.agents/instructions/`, `.agents/references/`, `.agents/skills/`, and `.agents/templates/`
  self-contained; do not rely on any containing workspace or parent
  repository to make the wording complete.
- When borrowing a rule from another repository, rewrite it so it remains
  accurate for this repository's own layout, tooling, and ownership model.

## AGENTS.md Responsibilities

- Use `AGENTS.md` for scope, precedence, routing, and discovery entrypoints.
- Keep detailed behavior rules in `.agents/instructions/` or
  `.agents/skills/`, not in `AGENTS.md`.
- Keep `Quick Routing` limited to topics owned by this repository.
- If `AGENTS.md` says a local `.agents/instructions/`, `.agents/references/`, `.agents/skills/`, or `.agents/templates/`
  tree exists, make sure the referenced paths actually exist.
- Keep `source of truth` and `independent repository` wording consistent
  with the no-parent-fallback model.

## Instruction Files

- Use `.agents/instructions/*.md` for detailed rules, constraints, and
  workflow expectations.
- Keep `.agents/*` and `AGENTS.md` as concise as possible while preserving
  scope, precedence, routing, and discoverability.
- Keep each file focused on repository-specific behavior or a clearly
  scoped topic instead of duplicating broad guidance without need.
- For high-frequency entrypoint instructions such as
  `.agents/instructions/python.instructions.md`, prefer compact direct rules
  plus narrow local routing over broad cross-reference-only indirection when
  that keeps routine tasks from loading a much larger reference file.
- Prefer one clear routing entry over repeated topic lists across `Read
  First`, `Quick Routing`, and repository notes when the repeated text does
  not add repository-specific meaning.
- Keep the front-matter `description` aligned with the actual scope of the
  file after edits.
- Prefer updating or deleting stale guidance over leaving broader text that
  no longer matches the body.

## Skill Files

- Use `.agents/skills/*` for independent workflows or task playbooks, not
  for restating general instruction text.
- Before adding or renaming a local skill or skill family, check for naming
  collisions with globally available skills and existing local skills.
- Prefer specific names for family roots instead of claiming a generic name
  that may already belong to another visible skill surface.
- Reference skills from `AGENTS.md` so discovery stays explicit.
- If multiple local skills overlap, make the intended routing clear in
  `AGENTS.md`.
- When a workflow needs multiple related entrypoints, prefer a skill family:
  keep the family root focused on routing and applicability, keep reusable
  workflow rules in family-local references, keep reusable report skeletons in
  family-local templates or report templates, and keep concrete execution logic
  inside the smallest applicable sub-skill.
- Do not turn a family root into another heavy execution skill when
  sub-skills already own the concrete workflows.

## Reference Files

- Use `.agents/references/*` for stable, agent-facing terminology, naming guidance, checklists, and short decision summaries.
- Keep reference files concise and specific enough that `AGENTS.md` or a skill can link to them directly.

## Template Files

- Use `.agents/templates/*` for reusable fill-in scaffolds, intake forms, or recurring implementation and test skeletons.
- Keep templates action-oriented and focused on repeated capture structure instead of durable rules that belong in instructions or references.
- Prefer a shared `.agents/templates/` path when multiple local workflows would otherwise duplicate the same scaffold.

## Asset Fit And Extraction

- Do not default to appending new guidance to the nearest existing file just
  because that file is already open.
- When a stable lesson introduces a new reusable topic, a new routing target,
  or a distinct ownership boundary, prefer a dedicated instruction,
  reference, or template over expanding an only-partially-related file.
- If the new content would turn one file into a mixed-scope catch-all or
  duplicate the same rule across guidance families, split it out and leave
  short cross-links instead. Temporary stopgap additions should be extracted
  once the topic proves stable.

## Temporary Design Drafts And Retrospectives

- Temporary design drafts and one-off task retrospectives are not shared agent guidance.
- Keep in-progress design notes in a disposable scratch path such as `.agents/scratch/designs/` while the task is active.
- Before deleting temporary design notes or other temporary development documents, get explicit user confirmation.
- Before that deletion, run a local experience distillation pass. If durable lessons remain, update the appropriate local instructions, references, templates, or skills and record concise repo memory notes when that will help future tasks.
- Do not treat temporary scratch notes as routed shared guidance while they remain task-local.
- Promote stable project-facing knowledge into `docs/`, package docs, or another established design-doc location for this repository.
- Distill stable agent-facing lessons into local instructions, references, templates, or other intentional local shared agent assets as appropriate; do not preserve one-off retrospectives when nothing durable should remain.

## Consistency Checks

- After editing guidance, read the affected `AGENTS.md`, instruction files,
  reference files, and skill references together to check for duplicate rules,
  contradictory precedence, and mismatched scope.
- For guidance-only or skill-only changes, run a minimal static validation
  pass: verify referenced paths exist, scan for stale routing or naming
  residues, confirm inventory classification still matches asset type, and
  check that new skill names do not collide with other visible skill surfaces.
- Verify that every referenced file path exists and that every listed local
  tree actually exists.
- Check that `Quick Routing`, `Read First`, and `Repository Notes` all
  describe the same ownership and fallback model.
- Confirm the guidance still reads coherently when this repository is
  viewed on its own, without any parent-workspace context.
