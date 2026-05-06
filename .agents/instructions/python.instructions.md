---
description: Load these instructions when modifying Python source files, tests, packaging metadata, or implementation-related documentation in this repository.
---

# Python Change Instructions

## Core Expectations

- Keep changes compatible with the project's Python version and public APIs unless the task allows otherwise.
- Reuse existing patterns, helpers, constants, and types before adding new ones.
- Keep new logic focused; avoid abstraction added only for style.
- Do not silently swallow exceptions. If catching one, keep enough context to debug it.
- For reusable wrappers around subprocesses, file handles, sockets, or other
  OS-backed resources, keep `open()` / `write()` / `close()` /
  context-manager semantics coherent: entering the context should represent
  an active resource session, not a closed object that fails only later.
- After startup, write, or finalize failures in an OS-resource wrapper,
  return the object to a stable and diagnosable state before raising. Best-
  effort cleanup should reap child processes or handles when possible and
  keep cleanup-failure details in the surfaced error.
- When forgetting `close()` is a plausible caller misuse path for a
  long-lived OS-resource wrapper, prefer adding a best-effort warning or
  finalizer cleanup path in addition to the explicit lifecycle API. Treat
  that fallback as a safety net, not the primary contract.
- Prefer explicit exceptions over `assert` for production-facing runtime
  contracts such as config values, user inputs, and environment settings.
- For processor, envelope, compose, or `pre_process`/`post_process` contract work, follow `.agents/references/processor-guideline.md`.
- For shared model-loading surfaces, `TorchModelRef`,
  `HFPretrainedModelRef`, `TorchModelLoadConfig`, or `hf://`-compatible path
  handling, follow `.agents/references/model-loading-guideline.md`.

## Module Organization

- When reorganizing a Python module and no stronger local convention exists,
  prefer a public-first layout: module constants/config, core public types,
  public entrypoints, public adapters or mixins, then private helpers.
- Keep canonical types and code paths before legacy or compatibility ones. If
  a legacy type belongs to the same public type family, keep it near that
  family but after the canonical type it complements.
- Keep private persistence, validation, conversion, and compatibility helpers
  after public API definitions unless a local file pattern or readability need
  clearly justifies an earlier placement.

## Helper Granularity

- Prefer helpers that represent a real semantic boundary: validation,
  conversion, lifecycle, scheduling, resource cleanup, error handling, or
  reusable domain behavior.
- Inline single-call helpers that only rename, forward, or wrap one operation
  unless they make a non-obvious boundary explicit.
- Avoid splitting one local flow into many tiny helpers when the helper names
  only restate the next line of code.
- Keep helpers near the narrowest layer that owns the behavior. Do not promote
  local policy into shared helpers only because two call sites look similar.
- For bool, enum, or literal parameters that change control flow, resource
  state, threading, or failure behavior, document each meaningful branch in
  the helper docstring.
- Prefer state names that include the owned object or state target, such as
  `evaluator_configured`, `env_prepared`, or `worker_replaced`, instead of
  generic names such as `configured`, `ready`, or `valid`.

## Logging

- For newly created repository-owned Python modules under `robo_orchard_lab/`,
  prefer
  `LoggerManager().get_child(__name__)` for module-level loggers.
- When editing an existing file, keep the established nearby logger surface
  unless the task explicitly includes logging cleanup or refactoring.
- Do not rewrite existing files from `logging.getLogger(__name__)` to
  `LoggerManager().get_child(__name__)` as incidental cleanup.
- Keep a framework-native logger only when the surrounding module family
  already depends on framework-specific logging behavior, such as
  `accelerate.logging.get_logger(__name__)`.

## Package Export Surfaces

- Treat repository-owned package `__init__.py` files and their `__all__`
  lists as curated public API surfaces, not as mirrors of every submodule
  symbol.
- Prefer exporting only the most common intended entrypoints from package
  roots. Keep compatibility-only, adapter, resolver, and type-alias symbols
  in their defining submodules unless the package root is intentionally the
  supported import path.
- When an old package-level import path must remain for compatibility,
  prefer a deprecated compatibility re-export and update repository-owned
  imports to the defining submodule instead of growing the root surface
  further.
- If preserving package-root `import *` or `__all__` parity is an explicit
  compatibility requirement during a migration, document that exception in
  package-specific guidance and keep repository-owned imports on the
  defining submodule instead of treating the compatibility export as the
  preferred new entrypoint.

## Spatial Transform And Matrix Naming

- Use explicit source/target naming for poses, transforms, and spatial matrices.
- Prefer `a_to_b`, `a_to_b_tf`, `a_to_b_mat`, and `BatchFrameTransform(child=A, parent=B)`.
- Treat `a_to_b`, `a_to_b_tf`, `a_to_b_mat`, `a2b`, and `BatchFrameTransform(child=A, parent=B)` as the same direction semantics.
- `a2b` may remain for historical compatibility, but new code should prefer `a_to_b`.
- Do not introduce `A|B`, `T_ab`, `T_a_b`, or `Tab` as the primary naming style.
- Keep legacy names such as `T_world2cam`, `world2cam`, and `cam2world` only when external data, third-party APIs, or existing public interfaces require them, and add a nearby mapping comment to the preferred form.
- For non-frame spatial matrices, still use explicit semantic names such as `cam_intrinsic_mat`, `world_to_img_proj_mat`, `base_jacobian_mat`, or `pose_cov_mat`.
- For external-library code, third-party APIs, protocol fields, dataset schemas, or compatibility layers, follow the external convention at the boundary instead of force-renaming it to the repository convention. Add an explicit local mapping when bridging between the external convention and repository-owned code.

## Typing

- Preserve or add type annotations when touching function signatures or return values.
- Prefer complete type hints for public APIs, key helpers, and newly added functions unless a clear local pattern or technical reason suggests otherwise.
- For persisted, deserialized, or schema-contract records that need runtime
  validation, prefer Pydantic models.
- For internal runtime structures, builder intermediates, and context objects
  that do not need dict compatibility, prefer `dataclass(slots=True)`.
- For fixed value sets, prefer `Enum` over scattering many string
  `Literal[...]` annotations. Keep explanatory comments next to the enum
  members they describe.
- Use `TypedDict` when the value really remains a plain dict and only static
  typing is needed. Do not use it as the primary representation for contracts
  that need runtime validation.
- When a Pydantic config field is meant to accept a config base class and its
  subclasses, annotate it with `ConfigInstanceOf[BaseConfig]` instead of the
  raw base config type.
- Do not rely on a raw base-config annotation such as `BaseConfig` or
  `BaseConfig[Any]` to express "this base config or any subclass config":
  Pydantic may then serialize only the base fields or emit subtype
  serialization warnings when a subclass config instance is provided.
- For config `class_type` fields that must serialize and deserialize through
  the repository config JSON path, annotate them with `ClassType[T]` instead
  of a bare `type[T]`.
- When validation normalizes a field into a narrower stable runtime type,
  annotate the field with that stored invariant rather than the full raw
  input surface.
- If callers still need to provide a wider input type, keep that wider
  shape at the validation boundary and expose a small conversion helper or
  separate input alias for runtime use.
- Do not leave field annotations implying that normalized-away input types
  still remain available after validation.

## Documentation and Comments

- Follow the style of nearby code, including imports, naming, file layout, and docstring conventions.
- Add comments only when they provide non-obvious context.
- For local control-flow or data-shape decisions that are easy to misread, add a short adjacent comment at the decision point rather than relying only on a function-level docstring.
- For coordinate-frame transforms, matrix transforms, inversions, or
  convention-bridging code, add a short adjacent comment when the direction
  of the transform or the frame handoff is not obvious from the code alone.
- For public APIs, boundary helpers, or non-obvious interfaces that accept
  or return TF-like structures such as `BatchFrameTransform`,
  `BatchFrameTransformGraph`, camera-pose wrappers, or other frame-bearing
  pose containers, document the frame contract explicitly in the docstring.
- For those TF-like interface values, state the relevant
  `parent_frame_id` and `child_frame_id`. For graph-like values, document
  only the root frame and the specific edge, path, or static edge contract
  that callers rely on.
- Prefer code-near comments for one-off shape normalization, batch unwrapping, side-channel filtering, compatibility branches, or similar logic whose intent is not obvious from names alone.
- For key interface functions, public dataset/model/pipeline entrypoints, and helper functions whose behavior or parameters are not immediately obvious from the signature alone, add or update docstrings instead of leaving the interface undocumented.
- For key public classes/functions and boundary helpers where safe use is
  not obvious from the signature alone, keep the docstring contract-first
  and user-task-first: start by explaining what the caller is trying to do
  and why they would use the interface, then document ownership,
  lifecycle, and other caller-visible constraints. Use
  `.agents/templates/interface-docstring-scaffold.md` only when a short
  drafting scaffold is helpful.
- Follow the project's existing Google-style docstring format with `Args:` and `Returns:` when documenting functions.
- In `Args:`, use `name (Type): ...` for required parameters and `name (Type, optional): ... Default is ...` for optional parameters.
- Keep docstrings concise, with consistent indentation and defaults documented only for optional parameters.
- For public Python docstrings that render into API docs, prefer standard
  Sphinx / Napoleon-friendly section shapes.
- Do not introduce ad hoc section headers followed by lists or examples
  unless the local docs toolchain already uses that pattern safely.
- Prefer `Example::`, standard `Examples:` sections, or explicit
  `.. code-block:: python` blocks for usage examples in public docstrings.

## Dependencies

- Avoid new dependencies unless they are clearly necessary.
