# CLAUDE.md

## Commands

**Lint:** `make check-lint` / `make auto-format`

**Tests:**
```bash
make test_ut   # unit tests
make test_it   # integration tests
pytest -x -c ../../tests/test_robo_orchard_lab/path/to/test_file.py  # single file
```

**Docs:** `make doc`

## Git Commit Convention

Format: `<type>(<scope>): <Description>`

- **Types:** `feat` | `fix` | `bugfix` | `docs` | `style` | `refactor` | `perf` | `test` | `chore` | `scm`

- Example: `feat(api): Implement new authentication module`

## Git Workflow

**Daily development (on feature branch):**
```bash
git add <files>
git commit -m "<type>(<scope>): <Description>"
git push origin <branch>
```

**Before creating a Merge Request:**
```bash
git fetch origin
git merge origin/master
# resolve conflicts if any, then:
git push origin <branch>
```

## Workflows

**Fixing bugs:** First write a standalone script that reproduces the bug and verify it fails as expected. Use it to validate your fix. Once passing, add the test case to the appropriate file under `tests/test_robo_orchard_lab/`.

**Running tests:** Never run the full test suite. Always run a single file or test case using the single-file command above.
