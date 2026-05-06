---
description: Use this guidance when distilling stable implementation, review, or design-process lessons into local guidance assets in this repository.
---

# Experience Distillation Instruction

Use this instruction for local guidance distillation work in this repository.

## Distillation Trigger

Run a distillation pass when the user asks for "沉淀", "复盘",
"总结到 .agents", "有什么可以沉淀", or equivalent wording after
implementation, design, review, debugging, or workflow iteration.

When the user asks what can be distilled, first report candidates and
recommended destinations. Edit guidance files only when the user explicitly
asks to apply the changes, or when the request already clearly asks to update
local guidance.

## Candidate Dimensions

Consider each of these dimensions before deciding that there is nothing
durable to capture:

- best practices
- methodology
- project decision records
- terminology
- interface boundaries
- anti-patterns
- test strategy
- migration strategy
- naming conventions
- tool or workflow experience

## Promotion Criteria

Promote a lesson into shared guidance only when it is:

- validated by implementation, review, or repeated use
- likely to recur in future work
- useful to future agents or maintainers
- discoverable from existing local routing, or worth adding routing for
- not already covered by existing guidance

Do not promote:

- one-off bug fixes
- temporary drafts
- unvalidated opinions
- turn-local collaboration preferences
- historical task notes that do not change future behavior

## Placement Decision Tree

Do not force every distilled lesson into the nearest existing guidance file.
Choose the destination by audience, lifecycle, and reuse mode:

- `AGENTS.md`: routing, scope, precedence, and discovery only.
- `.agents/instructions/*.md`: when to load guidance, required workflow
  behavior, validation expectations, or execution constraints.
- `.agents/references/*.md`: stable principles, interface boundaries,
  terminology, anti-patterns, checklists, and decision summaries.
- `.agents/templates/*.md`: reusable fill-in structures, not durable rules.
- `.agents/skills/*`: independent reusable workflows with routing, execution
  steps, and reporting expectations.
- `docs/` or package docs: stable user-facing architecture, API, migration, or
  usage knowledge.
- `.agents/scratch/`: task-local, evolving, or historical context that should
  not be routed as shared guidance.

Before editing guidance, ask who will consume the lesson, whether it is a
rule, principle, workflow, template, or historical decision, whether it is
repository-wide or domain-specific, and whether an authoritative file already
exists. If the existing destination would become mixed-scope or repetitive,
create a focused new asset instead. If a lesson fits multiple places, put the
durable rule in the narrowest reusable file and add only short routing
pointers where discoverability requires them.

## Candidate Report Shape

When reporting candidate distillation items, include:

- proposed content
- recommended destination
- why it is worth keeping
- whether it should be applied now or left as a candidate
- any existing guidance it may overlap with

## Maintenance

- During distillation, re-evaluate asset shape as well as content. Follow
  `.agents/instructions/guidance-authoring.instructions.md` for extraction,
  splitting, and guidance consistency checks.
- Prefer updating an existing local guidance family when the lesson is
  clearly tied to an established domain or workflow in this repository.
- If the task used temporary design notes or temporary development docs,
  finish the distillation pass before those documents are deleted.
