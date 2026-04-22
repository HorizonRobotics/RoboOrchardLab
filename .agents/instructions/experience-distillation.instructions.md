---
description: Use this guidance when distilling stable implementation, review, or design-process lessons into local guidance assets in this repository.
---

# Experience Distillation Instruction

Use this instruction for local guidance distillation work in this repository.

- Update assets by layer:
  - `AGENTS.md` for scope, precedence, and routing only
  - `.agents/instructions/*.md` for applicability and read-first behavior
  - `.agents/references/*.md` for stable rules, checklists, and decision summaries
  - `.agents/templates/*.md` for reusable fill-in scaffolds when a repeated capture structure emerges
  - `.agents/skills/*` for reusable workflow-specific execution logic or reporting expectations
- During distillation, re-evaluate asset shape as well as content. If the
  lesson deserves a dedicated asset or a cleaner split, follow the extraction
  rules in `.agents/instructions/guidance-authoring.instructions.md` instead
  of automatically appending it to the file already in hand.
- Prefer updating an existing local guidance family when the lesson is
  clearly tied to an established domain or workflow in this repository.
- Promote a lesson into local shared guidance only when it is validated,
  likely to recur, and discoverable from the local `AGENTS.md` routing.
- If the task used temporary design notes or temporary development docs,
  finish the distillation pass before those documents are deleted.
- Do not promote one-off bug fixes, temporary drafts, or unvalidated opinions
  into shared guidance.
- Do not promote turn-local collaboration preferences or one-off workflow
  requests into shared guidance unless this repository explicitly adopts them
  as stable policy.