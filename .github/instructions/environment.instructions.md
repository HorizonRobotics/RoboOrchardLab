---
description: Load these instructions when tasks depend on the active Python environment, optional extras, external services, hardware, or other runtime-specific conditions.
---

# Environment and Runtime Instructions

## Environment

- Use the active environment unless the task requires a change.
- Do not assume optional extras, developer tools, or external services are installed.
- Check environment-dependent requirements before running related validation.

## Runtime and Reporting

- Do not assume network access, hardware, display servers, or background services are available.
- Treat optional services such as `ray` as unavailable until confirmed.
- If validation is blocked by the environment, state what ran, what was unavailable, and the remaining risk.
