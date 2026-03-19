---
description: Load these instructions when working with git history, commit messages, branches, or GitLab merge requests.
---

# Git and Merge Request Instructions

## Git Conventions

- Commit message format: `<type>(<scope>): <Description>.`
- Allowed types: `feat`, `fix`, `bugfix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `scm`.
- Keep the scope short and the full line within 128 characters.

## GitLab Merge Requests

- When drafting a GitLab merge request description, compare against `master` unless told otherwise.
- Base the squash message and description on the full branch diff, not only the latest commit.
- Use this section structure unless told otherwise: `Summary`, `Why`, `Impact`, `Validation`, `Risks / Notes`.
- Keep each section concise and factual.
- For `Impact`, cover user-visible behavior, API/CLI/config impact, and workflow or internal maintenance impact when relevant.
- Prefer explicit `none` / `not applicable` markers over leaving expected bullets ambiguous.
- Use this minimal template as the default:

  ```md
  ## Summary

  - Scope: <component/module>
  - Main changes:
    - <change 1>
    - <change 2>

  ## Why

  - <problem or motivation>

  ## Impact

  - User-facing: none / <change>
  - API / CLI / config: none / <change>
  - Internal / workflow: none / <change>

  ## Validation

  - Passed: `<command>`
  - Not run: <reason>, if applicable

  ## Risks / Notes

  - None identified / <known risk, limitation, migration note, or follow-up>
  ```
- When creating a GitLab merge request, preserve the intended multiline title and description format.
- If creating a GitLab merge request with `glab`, enable `--remove-source-branch` by default unless told otherwise.
- Do not flatten line breaks or section structure just to fit push options or other transport shortcuts; use a method that keeps the final MR text intact.
