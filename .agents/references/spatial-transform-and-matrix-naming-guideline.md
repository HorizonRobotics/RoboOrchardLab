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

Example:
```python
# `T_world2cam` is equivalent to `world_to_cam_mat`.
T_world2cam = sample["T_world2cam"]
```
