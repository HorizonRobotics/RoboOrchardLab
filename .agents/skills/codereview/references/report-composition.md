# Code Review Family Report Composition

Use this reference when a `codereview` sub-skill prepares a final report and
especially when more than one review skill contributed input.

## Main Report Ownership

- The active skill owns the main report.
- The main findings sections should contain only findings that belong to the
  active skill's primary review responsibility.
- Do not merge findings from a paired review skill into the active skill's
  main findings sections just to keep everything in one place.

## Related Review Inputs

- If no paired review skill was used, write `none`.
- If one or more paired review skills were used, summarize them briefly in
  `Related review inputs`.
- Keep each summary short and factual:
  - paired skill name
  - reviewed scope or dimensions
  - outcome summary
  - whether a separate detailed report exists

Good summary shape:

- `architecture-review`: reviewed layering and compatibility surface for
  `python/foo`; found 1 medium-risk issue; separate report attached.

## When To Keep A Separate Report

- Keep a separate report when the paired review has its own distinct review
  target, dimensions, or severity logic.
- Keep a separate report when merging findings would blur ownership between
  correctness review and architecture review.
- Keep a separate report when the paired review is materially larger than a
  short supporting note.

## When A Short Summary Is Enough

- A short summary is enough when the paired review only adds small context to
  the active report.
- A short summary is enough when the paired review found no issues and only
  needs to record that the extra pass happened.

## Composition Rules

- Keep the active skill's recommendation grounded primarily in its own
  validated findings.
- If a paired review materially affects the final recommendation, mention that
  effect in `Related review inputs` and in the overall assessment.
- Do not duplicate the same issue in both the main findings sections and the
  paired review summary.
- Do not copy full findings text from the paired review into the active
  report.
