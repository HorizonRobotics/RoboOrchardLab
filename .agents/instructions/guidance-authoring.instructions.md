---
description: Load these instructions when creating, editing, or reorganizing AGENTS.md files, .agents/instructions/*.md files, .agents/references/* assets, or .agents/skills/* guidance in this repository.
---

# Guidance Authoring Instructions

## Guidance Model

- Treat this repository as an independent guidance root.
- Keep local `AGENTS.md`, `.agents/instructions/`, `.agents/references/`, and `.agents/skills/`
  self-contained; do not rely on any containing workspace or parent
  repository to make the wording complete.
- When borrowing a rule from another repository, rewrite it so it remains
  accurate for this repository's own layout, tooling, and ownership model.

## AGENTS.md Responsibilities

- Use `AGENTS.md` for scope, precedence, routing, and discovery entrypoints.
- Keep detailed behavior rules in `.agents/instructions/` or
  `.agents/skills/`, not in `AGENTS.md`.
- Keep `Quick Routing` limited to topics owned by this repository.
- If `AGENTS.md` says a local `.agents/instructions/`, `.agents/references/`, or `.agents/skills/`
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

## Temporary Design Drafts And Retrospectives

- Temporary design drafts and one-off task retrospectives are not shared agent guidance.
- Keep in-progress design notes in a disposable scratch path such as `.agents/scratch/designs/` while the task is active, then delete them after confirmation.
- Promote stable project-facing knowledge into `docs/`, package docs, or another established design-doc location for this repository.
- Distill stable agent-facing lessons into local instructions or other intentional local shared agent assets as appropriate, not raw scratch notes.

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
