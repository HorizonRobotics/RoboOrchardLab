---
description: Load these instructions when modifying Python source files, tests, packaging metadata, or implementation-related documentation in this repository.
---

# Python Change Instructions

## Core Expectations

- Keep changes compatible with the project's Python version and public APIs unless the task allows otherwise.
- Reuse existing patterns, helpers, constants, and types before adding new ones.
- Keep new logic focused; avoid abstraction added only for style.
- Do not silently swallow exceptions. If catching one, keep enough context to debug it.

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
- Follow the project's existing Google-style docstring format with `Args:` and `Returns:` when documenting functions.
- In `Args:`, use `name (Type): ...` for required parameters and `name (Type, optional): ... Default is ...` for optional parameters.
- Keep docstrings concise, with consistent indentation and defaults documented only for optional parameters.

## Dependencies

- Avoid new dependencies unless they are clearly necessary.
