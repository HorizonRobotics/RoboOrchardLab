# Spatial Transform And Matrix Naming Guideline

Preferred forms:
- `a_to_b`
- `a_to_b_tf`
- `a_to_b_mat`
- `BatchFrameTransform(child=A, parent=B)`

Direction equivalence:
- `a_to_b`
- `a_to_b_tf`
- `a_to_b_mat`
- `a2b`
- `BatchFrameTransform(child=A, parent=B)`
all use the same direction semantics.

Historical but allowed:
- `a2b`

Avoid as repository style:
- `A|B`
- `T_ab`
- `T_a_b`
- `Tab`

Use explicit names for spatial matrices:
- `world_to_cam_mat`
- `cam_intrinsic_mat`
- `world_to_img_proj_mat`

Legacy names such as `T_world2cam` may be kept only for compatibility. When kept, map them locally to the preferred form.

Repository scope:
- The preferred forms in this guideline apply to repository-owned code.
- For external libraries, third-party APIs, protocol fields, dataset schemas, or compatibility layers, follow the external convention at the boundary and map it locally before translating to repository-preferred names.

Matrix and composition conventions for repository-owned transform types:
- For `Transform3D_M`, `BatchTransform3D`, and `BatchFrameTransform`, transform matrices are interpreted as acting on homogeneous column vectors.
- Point tensors are typically stored as row-wise batches, so point application is implemented in the equivalent row-wise form `points_h @ M.T`.
- `A.compose(B)` applies `A` first and `B` second.
- `A.compose(B, C)` stores the same transform as `C @ B @ A`.
- `B @ A` is equivalent to `A.compose(B)`, so `@` follows matrix multiplication order and applies the right operand first.

TF-like interface contracts for repository-owned APIs:
- Apply this at API boundaries, not everywhere. Public APIs, boundary
	adapters, and non-obvious frame-bearing interfaces should document their
	frame contract; purely generic helpers do not need to repeat it when the
	frame semantics are already obvious from the local context.
- For transform inputs and outputs, state the relevant expected or returned
	`parent_frame_id` and `child_frame_id`.
- For graph-like inputs or outputs, state the required root frame and the
	specific edge, path, or static edge contract that callers rely on.
- If an API changes frame ownership across the boundary, for example from
	`world -> eef` input to `control -> eef` output, state that handoff
	explicitly.
- If wrapper fields such as `current_pose`, `target_pose`, `camera_pose`, or
	`tf_world` hide the concrete frames, document the frame contract at that
	field or API boundary.

Example:
```python
# `T_world2cam` is equivalent to `world_to_cam_mat`.
T_world2cam = sample["T_world2cam"]
```
