---
description: Load these instructions when preparing a fresh local development environment for this repository.
---

# Environment Preparation Instructions

## When to Use

- Use this guide only for first-time setup, intentional environment
  rebuilds, or explaining how to prepare the repository from an empty
  Python environment.
- For routine task execution, runtime debugging, or dependency triage
  inside an already prepared environment, use
  `.agents/instructions/environment.instructions.md` instead.

## Baseline Requirements

- Follow the Python version required by the project configuration instead of
  assuming a fixed interpreter version.
- Confirm system-level prerequisites only when the task actually depends on
  them, such as CUDA, ROS, simulator stacks, or external services.
- Do not assume optional developer tools are already installed.
- Use `README.md`, `Makefile`, `.env.example`, `pyproject.toml`, and
  `scm/requirements.txt` as the primary setup sources of truth.

## Recommended Setup Flow

- Start from the repository root.
- If this repository is being used inside a containing workspace whose
  root already has `.venv/`, prefer that shared workspace-root
  environment instead of creating another local virtual environment.
- In that shared-workspace case, use explicit executables from the
  workspace-root environment for repository commands.
- Resolve the executable path from the containing workspace root instead
  of assuming a fixed relative depth from this repository.
- For example, if the containing workspace root is available as
  `<workspace-root>`, use:

  ```bash
  <workspace-root>/.venv/bin/python -V
  <workspace-root>/.venv/bin/pip -V
  <workspace-root>/.venv/bin/python -m pip install -e .
  ```

- Otherwise, create a project-local virtual environment with system site
  packages:

  ```bash
  python -m venv .venv --system-site-packages
  ```

- When using a project-local virtual environment, use explicit
  executables from that environment for repository Python commands:

  ```bash
  .venv/bin/python -V
  .venv/bin/pip -V
  ```

- Treat the shared workspace-root `.venv` as the default development path
  when this repository is used inside the parent workspace that owns that
  environment.
- Otherwise, treat a project-local `.venv` plus local editable install as
  the default standalone development path unless the task or confirmed
  runtime constraints require another environment.
- Do not create `.env` by default.
- Copy `.env.example` to `.env` only when `make` targets need local
  overrides such as a non-default package installer or command runner.
- Treat `.env` overrides as optional `make` configuration, not as a
  replacement for the project-local `.venv` flow.
- If using a project-local `.venv`, keep any `.env` overrides that select
  Python, pip, or command runners aligned with that environment unless the
  task explicitly requires another runtime.
- Install the project in editable mode with the selected environment's
  explicit Python executable. For a project-local `.venv`, for example:

  ```bash
  .venv/bin/python -m pip install -e .
  ```

- Install extra development dependencies only when needed by the task. Use
  repository sources of truth such as `Makefile`, `pyproject.toml`, and
  `scm/requirements.txt`.

## Cache and Data Expectations

- Do not override `HF_HOME` by default.
- If the existing environment relies on pre-downloaded Hugging Face models,
  datasets, or artifacts, preserve the current `HF_HOME` so setup and tests
  can reuse them.
- If a Hugging Face component needs a writable cache path, prefer setting a
  narrower cache variable such as `HF_DATASETS_CACHE` or `HF_HUB_CACHE`
  instead of changing `HF_HOME`.

## Validation

- Verify that the selected environment resolves to the expected
  interpreter. For a project-local `.venv`, for example:

  ```bash
  .venv/bin/python -c "import sys; print(sys.executable)"
  ```

- If the task intentionally uses another environment, verify the actual
  interpreter that will run the task before installing or validating.
- Verify that the repository imports from the local checkout after editable
  install. For a project-local `.venv`, for example:

  ```bash
  .venv/bin/python -c "import robo_orchard_lab; print(robo_orchard_lab.__file__)"
  ```

- Run the smallest relevant validation for the task after setup instead of
  defaulting to the full test suite.

## Common Pitfalls

- Do not rely on shell activation persistence across commands; prefer the
  selected environment's explicit `python` and `pip` executables.
- Do not treat `.env` overrides as a second environment-management system;
  use them only to steer `make` when the default command wiring is not
  sufficient.
- Do not use `prepare_env` as the default guide for ordinary task
  execution after the environment is already working.
- Do not create a second project-local `.venv` by default when this
  repository is already being used from a parent workspace with a shared
  root `.venv`.
- If the repository already has cached data or environment-specific tooling,
  preserve those paths unless the task explicitly requires a clean rebuild.
- If validation fails because a cache path, shared-memory helper, or site
  package location is not writable, report the environment limitation before
  treating it as a code defect.
