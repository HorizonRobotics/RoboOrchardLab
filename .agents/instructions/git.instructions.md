---
description: Load these instructions when working with git history, commit messages, branches, or GitLab merge requests.
---

# Git and Merge Request Instructions

## Git Conventions

- Commit message format: `<type>(<scope>): <Description>.`
- Allowed types: `feat`, `fix`, `bugfix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `scm`.
- Keep the scope short and the full title line within 128 characters.
- Do not require local commits to be squashed unless explicitly instructed.
- Commit messages should use a multiline body after a blank line.
- Keep the first line exactly in the title format above, and use the body for details.
- The commit message body should use the same section structure as the default GitLab merge request description: `Summary`, `Why`, `Impact`, `Validation`, `Risks / Notes`.
- Prefer writing the commit body so it can be reused directly, or with minimal edits, as the GitLab merge request description.
- Use this default commit message template unless told otherwise:

  ```md
  <type>(<scope>): <Description>.

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

## GitLab Merge Requests

- When drafting a GitLab merge request, compare against the target branch; use `master` by default unless told otherwise.
- Base the squash commit message and merge request description on the full branch diff, not only the latest commit.
- Require squash on merge for GitLab merge requests unless explicitly instructed otherwise.
- The final squash commit message must follow the same commit message format and body template defined in `Git Conventions` above.
- Use the merge request title as the first line of the final squash commit message unless told otherwise.
- Use the merge request description to mirror the body of the final squash commit message unless told otherwise.
- Keep each section concise and factual.
- For `Impact`, cover user-visible behavior, API/CLI/config impact, and workflow or internal maintenance impact when relevant.
- Prefer explicit `none` / `not applicable` markers over leaving expected bullets ambiguous.
- Keep the merge request title to a single line, and preserve the multiline section structure in the description.
- If creating a GitLab merge request with `glab`, enable `--remove-source-branch` by default unless told otherwise.
- Do not flatten line breaks or section structure just to fit push options or other transport shortcuts; use a method that keeps the final MR text intact.
