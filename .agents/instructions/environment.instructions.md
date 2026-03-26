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
- If a task materially depends on GPU execution and the sandbox cannot access CUDA or NVIDIA devices, request escalated execution for the smallest command that requires GPU access.
- Treat signals such as `torch.cuda.is_available()` returning `False`, `nvidia-smi` not seeing devices, or CUDA/NVML initialization failures as environment-access issues first, not immediate proof of a code bug.
- Do not escalate for code reading, CPU-only validation, or steps that do not require GPU.
- If validation is blocked by the environment, state what ran, what was unavailable, and the remaining risk.
- If GPU-dependent validation is rerun with escalated permissions, report what failed in the sandbox, what was rerun outside it, and any remaining risk.
