---
description: Load these instructions when tasks depend on the active Python environment, optional extras, external services, hardware, or other runtime-specific conditions.
---

# Environment and Runtime Instructions

## Environment

- Unless the user explicitly specifies another environment, if this
  repository is opened inside a containing workspace whose root contains
  `.venv/`, use that shared workspace-root environment first for
  Python-related commands (for example `python`, `pip`, `pytest`, and
  editable installs).
- Otherwise, if this repository root contains `.venv/`, use that
  environment.
- If no project-local `.venv/` is available at either level, fall back to
  the active environment.
- If task requirements or confirmed runtime dependencies require a
  different environment, state that reason before switching.
- Prefer explicit executables from the selected virtual environment over
  relying on shell activation persistence across commands.
- When using a shared workspace-root `.venv/`, use that environment's
  explicit `bin/python` and `bin/pip` paths instead of assuming a
  submodule-local `.venv/` exists.
- Do not override `HF_HOME` by default. If tests or validation depend on
  pre-downloaded Hugging Face data, models, or datasets under the existing
  `HF_HOME`, preserve it so cached artifacts remain available.
- If Hugging Face cache writes are blocked, prefer redirecting the
  narrowest writable cache path needed by the failing tool (for example
  `HF_DATASETS_CACHE` or `HF_HUB_CACHE`) instead of changing `HF_HOME`.
- Do not assume optional extras, developer tools, or external services are installed.
- Check environment-dependent requirements before running related validation.
- If running repository code fails because `robo_orchard_lab` is not
  installed from the local checkout, prefer `make install-editable` or the
  selected environment's explicit `python -m pip install -e .` command
  before changing code or treating it as an external dependency.

## Runtime and Reporting

- Do not assume network access, hardware, display servers, or background services are available.
- Treat optional services such as `ray` as unavailable until confirmed.
- If a task materially depends on GPU execution and the sandbox cannot access CUDA or NVIDIA devices, request escalated execution for the smallest command that requires GPU access.
- Treat signals such as `torch.cuda.is_available()` returning `False`, `nvidia-smi` not seeing devices, or CUDA/NVML initialization failures as environment-access issues first, not immediate proof of a code bug.
- Do not escalate for code reading, CPU-only validation, or steps that do not require GPU.
- If validation is blocked by the environment, state what ran, what was unavailable, and the remaining risk.
- If GPU-dependent validation is rerun with escalated permissions, report what failed in the sandbox, what was rerun outside it, and any remaining risk.
