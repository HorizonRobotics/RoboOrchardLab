---
description: Load these instructions when working with git history, commit messages, branches, GitLab merge requests, or GitHub pull requests.
---

# Git and Merge Request Instructions

## Git Conventions

- Commit message format: `<type>(<scope>): <Description>.`
- Allowed types: `feat`, `fix`, `bugfix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `scm`.
- Keep the scope short and the full title line within 128 characters.
- Do not require local commits to be squashed unless explicitly instructed.
- Apply the same commit message conventions to GitHub commits as well; do not use ad-hoc one-line commit messages there.
- Commit messages should use a multiline body after a blank line.
- Keep the first line exactly in the title format above, and use the body for details.
- The commit message body should use the same section structure as the default GitLab merge request description: `Summary`, `Why`, `Impact`, `Validation`, `Risks / Notes`.
- Prefer writing the commit body so it can be reused directly, or with minimal edits, as the GitLab merge request description.
- Apply the public-artifact rule from `default.instructions.md` to commit
  messages.
- In public-facing commit text, keep `Validation` commands repository-relative
  or generic, and strip local paths, machine-specific interpreter locations,
  usernames embedded in paths, proxy wrappers, and internal-only URLs.
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

## GitLab Branch Workflow

- For GitLab-managed work, use this default workflow unless the user explicitly asks for a different base branch or to continue an existing branch.
- Start from the latest `master`.
- Create a new branch from `master` for the task before making changes.
- Before creating a new branch, refresh `master` when needed and check whether the current branch already contains unrelated work or an open merge request.
- Do not append unrelated changes to an existing feature branch or merge request by default.
- If the current branch already contains in-scope work and the user clearly intends to continue it, stay on that branch; otherwise create a fresh branch from the latest `master`.
- Do the implementation work on that new branch.
- Before submitting a merge request, pull the latest remote state of the target branch and rebase the task branch onto it; use `master` as the default target branch unless the user or repository workflow says otherwise.
- Submit a merge request from the task branch into `master`.
- Use `master` as the default target branch unless the user or repository workflow says otherwise.
- After the merge request is merged, remove the remote source branch.
- After the merge request is merged, update the local target branch to the latest remote state; use `master` by default unless the user or repository workflow says otherwise.
- After switching back to the updated local target branch, delete the local source branch when it is no longer needed.
- Do not force-delete a local source branch if it may still contain local-only work or if the user asked to keep it.

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
- Apply the public-artifact rule from `default.instructions.md` to merge
  request descriptions.
- Keep the merge request title to a single line, and preserve the multiline section structure in the description.
- If the MR description is auto-filled from commit text, review and scrub the
  exact final title and description before sending, using the same public-artifact
  checks as for commit text.
- If creating a GitLab merge request with `glab`, enable `--remove-source-branch` by default unless told otherwise.
- Do not flatten line breaks or section structure just to fit push options or other transport shortcuts; use a method that keeps the final MR text intact.

## GitHub Pull Requests

- Apply the same title, description, and squash-commit requirements used for GitLab merge requests.
- Compare against the target branch; use `master` by default unless told otherwise.
- Base the squash commit message and pull request description on the full branch diff, not only the latest commit.
- Require squash on merge unless explicitly instructed otherwise.
- Apply the public-artifact rule from `default.instructions.md` to pull
  request descriptions.
- Keep the pull request title to a single line, and preserve the multiline section structure in the description.
