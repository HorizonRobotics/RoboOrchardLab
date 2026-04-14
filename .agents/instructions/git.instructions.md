---
description: Load these instructions when working with git history, commit messages, branches, GitLab merge requests, or GitHub pull requests.
---

# Git and Merge Request Instructions

## Git Conventions

- Commit message format: `<type>(<scope>): <Description>.`
- Allowed types: `feat`, `fix`, `bugfix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `scm`.
- For commit, MR, and PR titles that use this format, keep the description
  capitalized: the first character after `: ` must be uppercase.
- Keep the scope short and the full title line within 128 characters.
- Commit messages should use a multiline body after a blank line.
- Use body sections `Summary`, `Why`, `Impact`, `Validation`, and `Risks / Notes`.
- In `Impact`, cover user-visible behavior, API or config changes, and internal or workflow effects when relevant.
- Prefer commit text that can be reused directly as the merge request or pull request description and final squash commit body.
- When pending changes contain more than one coherent topic, prefer separate
  commits grouped by topic.
- If the grouping is ambiguous or the worktree contains changes that may be
  unrelated to the current task, ask the user before choosing a commit split.
- Do not silently bundle unrelated changes into a single commit.
- Do not require local checkpoint commits to be squashed unless explicitly instructed.
- Do not force-add ignored scratch files or temporary design notes such as `.agents/scratch/**` unless the user explicitly asks for a versioned snapshot. If the content should become durable project knowledge, move or distill it into a tracked location such as `docs/` instead.
- Apply the public-artifact rule from `default.instructions.md` to commit
  messages.
- In public-facing commit text, keep `Validation` commands repository-relative
  or generic, and strip local paths, machine-specific interpreter locations,
  usernames embedded in paths, proxy wrappers, and internal-only URLs.
- Apply the public-artifact rule from `default.instructions.md` to merge
  request and pull request text as well.

## Branch Workflow

- Default target branch is `master` unless the user or repository workflow says otherwise.
- Branch from the latest remote target branch, not a stale local copy.
- If continuing an existing in-scope branch is clearly intended, stay on it; otherwise create a fresh task branch.
- Do not append unrelated work to an existing feature branch or open review by default.
- Before the first push, before opening or updating a review, and after the target branch moves materially, fetch the latest remote target branch again.
- If the task branch has not been pushed or shared yet, prefer rebasing it onto the latest remote target branch.
- If the task branch has already been pushed or shared, do not rewrite history silently. Ask before force-pushing, and otherwise prefer merging the latest remote target branch into the task branch to keep published history stable.
- After merge, remove the remote source branch, refresh the local target branch, and then delete the local source branch when safe.
- Before deleting an archive, backup, or pre-reset local branch, compare its content against the branch you plan to keep using `git diff`, tree equality, or equivalent content-level checks; do not rely only on `git cherry` or ancestry.
- Do not force-delete a local source branch if it may still contain local-only work or if the user asked to keep it.

## Merge Requests and Pull Requests

- Base the title, description, and final squash commit on the full branch diff, not only the latest commit.
- Compare against the target branch and keep the title to a single line.
- For GitLab merge requests targeting `master`, keep the title compatible with
  `scm/qac/check_mr_title.py`: `<type>(<scope>): <Description>` with an
  uppercase first letter in the description.
- Keep the description concise and preserve the multiline section structure used in commit bodies.
- Use the review title as the first line of the final squash commit message unless told otherwise, and mirror the final squash body in the review description unless told otherwise.
- Prefer explicit `none` / `not applicable` markers over ambiguous omissions.
- If the description is auto-filled from commit text, review and scrub the exact final title and description before sending.
- Require squash on merge unless explicitly instructed otherwise.
- For GitLab with `glab`, enable `--remove-source-branch` by default unless told otherwise.
- Do not flatten multiline descriptions just to fit push options or other transport shortcuts.
