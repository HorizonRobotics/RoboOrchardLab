# URDF Alignment Notes

This note records lessons from aligning the InternData-A1 (Genie G1, AgileX
Split Aloha, ARX Lift-2) and RoboTwin (aloha_v1/v2, ur5_wsg, arx_x5a,
franka_panda, piper) URDFs. It is intentionally blunt about mistakes, because
the same failure modes are easy to repeat when a URDF looks visually
plausible.

## What Worked

- Keep origin assets separate from aligned assets. Use
  `urdf_origin/<dataset>/<embodiment>/...` as the immutable reference and
  `urdf_align/<dataset>/<embodiment>/...` as the edited target.
- Colocate one `alignment.yaml` manifest with each aligned URDF at
  `urdf_tools/manifests/<dataset>/<embodiment>/alignment.yaml`. The manifest
  owns the
  per-embodiment validation contract (dataset-config wiring, arm link chains,
  EE spec, camera-reference frames). One embodiment per folder; when two
  dataset-config keys are backed by exactly the same physical robot and same
  dataset asset (RoboTwin aloha_v1 / aloha_v2), reuse one folder by listing
  both keys under `config.key`.
- Copy visualization mesh sidecars into the aligned URDF folder, but ignore
  mesh directories in git. The aligned URDF should open in a viewer without
  depending on the legacy source tree.
- Validate before trusting visual inspection. A URDF can look reasonable at
  zero pose while still reversing a joint action at nonzero values.

## Alignment Stages

Each pass over one origin URDF runs through four stages in fixed order:

1. **Normalize joint axes.** Every actuated (`revolute`, `continuous`,
   `prismatic`) joint whose unit axis is not `+ẑ` is rewritten so the axis
   becomes `0 0 1`. The joint frame is rotated by the compensating `S` and
   the immediate child link's local origins are re-expressed by `S⁻¹` so the
   same control values still produce the same physical motion. See
   *Axis Compensation Rule* below.
2. **Insert semantic `*_ee` children.** For each arm in the manifest whose
   `ee.rotate_z_deg` is nonzero, a fixed child link named `<parent>_ee` is
   inserted with a joint that rotates by `rotate_z_deg` around the parent's
   `+Z`. The parent link's `<inertial>`, `<visual>`, and `<collision>`
   children move to the `_ee` child, with their `<origin>` re-expressed by
   `S⁻¹` so world-frame poses stay invariant. This gives the model the
   right/down/front semantic EE frame while leaving the unsuffixed parent
   link available as the mechanical / camera-calibration frame.
3. **Insert `<parent>_gripper_end` children.** One fixed child per arm,
   attached to `<parent>_ee` when the `_ee` link was inserted in stage 2,
   otherwise attached to `<parent>` directly. Its origin is
   `xyz="0 0 gripper_forward"` (default 20 cm, per-embodiment override in
   the manifest) and identity rotation in the parent frame. The link
   represents the physical tip of the closed gripper and becomes the single
   `finger_keys` entry consumed by `MultiArmKinematics` at runtime.
4. **Insert `*_camera_mount_compat` compensator children where needed.**
   Compare each declared `camera_references` link's zero-pose FK matrix in
   the origin URDF against the in-progress aligned URDF. If they match to
   floating-point tolerance, no compat alias is inserted; if they differ (a
   consequence of stage 1 rotating an upstream frame), a fixed child named
   `<link>_camera_mount_compat` is inserted whose transform recovers the
   origin pose. Existing calibrated camera extrinsics can then keep working
   by targeting the compat link.

A final canonical whitespace pass re-indents the aligned URDF with 2-space
indentation and one element per line so re-align runs produce clean diffs.

## Manifest Schema

`alignment.yaml` at `urdf_tools/manifests/<dataset>/<embodiment>/alignment.yaml`:

```yaml
adapter: robotwin                 # dataset adapter family
config:
  module: data_configs.config_robotwin_dataset
  getter: get_dataset_config
  key: aloha_v1                   # str or list[str] for shared embodiments
  aligned_key: aloha_v1_urdf_v2
mesh_search_roots: []             # optional
arms:
  - arm_link_keys:                # ordered chain from arm base to last link
      - fl_link1
      - fl_link2
      - fl_link3
      - fl_link4
      - fl_link5
      - fl_link6
    ee:
      parent: fl_link6            # must equal arm_link_keys[-1]
      rotate_z_deg: 270           # optional, default 0 (no _ee child inserted)
      gripper_forward: 0.20       # optional, default 0.20 m
  - arm_link_keys:
      - fr_link1
      - fr_link2
      - fr_link3
      - fr_link4
      - fr_link5
      - fr_link6
    ee:
      parent: fr_link6
      rotate_z_deg: 270
camera_references:                # optional bare list
  - fl_link6
  - fr_link6
```

Notes:

- `arms[].ee.rotate_z_deg == 0` means the arm's last link is already the
  semantic EE frame — no `_ee` child is inserted. `<parent>_gripper_end` is
  still emitted, attached directly under `<parent>`.
- `arms[].ee.gripper_forward` is the distance in meters from the EE frame
  origin to the tip of the closed gripper along `+Z`. The default of 0.20 m
  is a reasonable starting point; override per embodiment when needed.
- `arms[].gripper_end` (optional per-arm block) overrides the default
  attach link + origin when `+Z of the EE frame` is not the gripper
  approach axis. Behavior R1 Pro is the canonical example: the last arm
  joint rotates about X (stage-1 compensated to Z) and the gripper
  extends along `+X` from `<side>_gripper_link`, so the manifest declares
  `gripper_end: {attach_link: <side>_gripper_link, xyz: [...], rpy: [pi, 0, ...]}`.
  Rx(pi) preserves the invariant that `gripper_end +Z == approach direction`;
  in-plane yaw is allowed since runtime records the full quaternion. See
  `README.md` for the schema block and the R1 Pro test for the pinned
  invariants.
- `camera_references` are URDF link names whose FK pose in the origin URDF
  must be recoverable in the aligned URDF (directly if unchanged, otherwise
  via a `*_camera_mount_compat` compensator child). The old
  `[{stream, link}]` object form is dropped — `stream` was metadata only.

Fields no longer in the schema and not needed:

- `link_renames` — the alignment pipeline never renames existing links; the
  `*_ee` child is a sibling, so this field was always redundant.
- `joints:` with semantic/origin/aligned names — the alignment pipeline
  never renames actuated joints (only inserts new fixed joints), so origin
  and aligned URDFs share the same actuated-joint name list. FK regression
  tests derive that list from `chain.get_joint_parameter_names()` directly
  and apply the same sample vector to both URDFs.
- `ee_frames[].child` — always `<parent>_ee`, no reason to author it.
- `camera_references[].stream` — informational metadata never read by the
  transformer.

## Model-Facing Contract

- **State supervision:** the model consumes convention-compliant EE frames.
  For arms with `rotate_z_deg != 0` this is `<parent>_ee`; for arms with
  `rotate_z_deg == 0` this is `<parent>` directly.
- **Gripper supervision:** the model consumes `<parent>_gripper_end` as the
  single-element `finger_keys` entry. The runtime
  `results[i * 2 + 1].mean(dim=0, keepdim=True)` collapses this 1-row
  tensor to a 1-row tensor unchanged — no code branch needed. The gripper
  frame has identity rotation relative to the EE frame and a manifest-
  configured forward offset along `+Z`, so its quaternion is exactly the
  EE frame's quaternion.
- **Camera projection:** the model consumes the origin-compatible reference
  link — either the unchanged same-name link, or its
  `*_camera_mount_compat` alias when alignment rotated an upstream frame.

Runtime transforms (`interna1/transforms.py`, `robotwin/transforms.py`) do
not override the finger-group rotation with the arm-last-link rotation. The
single `<parent>_gripper_end` element already lives in the correct EE
frame, so the componentwise 7-vector mean is a no-op and returns the exact
gripper pose in `[pos, quat]` form.

## Axis Compensation Rule

For a joint with old unit axis `a`, choose a fixed rotation `S` such that
`S @ ẑ = a`. Then:

1. Set the aligned joint axis to `0 0 1`.
2. Post-multiply the joint origin rotation by `S`.
3. Re-express immediate child link local `<inertial>`, `<visual>`, and
   `<collision>` origins by `S⁻¹` (left-multiplied on both xyz and rpy).
4. Re-express every outgoing child joint's `<origin>` from that child link
   by `S⁻¹`.
5. Do not rewrite inertia tensor components: the inertial `<origin>` rpy
   already absorbs the frame change, so the tensor stays expressed in the
   same physical frame.

The `_ee` insertion stage reuses the same `S⁻¹` re-expression rule on the
parent link's moved `<inertial>/<visual>/<collision>` origins so the mesh /
collision / inertial world-frame pose stays invariant across the frame
change into the rotated `*_ee` child.

## Mistakes To Avoid

- Do not guess the origin URDF. The correct G1 origin for InternData-A1 is
  `projects/holobrain_internal/common/urdf/InternData-A1_urdf/G1_120s/G1_120s.urdf`,
  not `G1_120s_dual.urdf`. If a path is a bucket symlink that is not mounted
  locally, verify it on the remote source before declaring it missing.
- Do not treat axis normalization as a text replacement. Changing
  `<axis xyz="0 0 -1">` to `<axis xyz="0 0 1">` changes `Rz(-q)` into
  `Rz(q)`. Preserve same-value action semantics by rotating the joint frame
  and compensating the immediate child-side origins.
- Do not validate only arm links. The first G1 test case missed gripper
  regressions because it sampled body/arm joints and compared only arm /
  gripper center links. Include the `<parent>_gripper_end` link and
  representative gripper samples when a gripper has actuated or mimic
  joints.
- Do not decide camera compatibility from URDF camera links alone. AgileX
  Split Aloha has `hand_left` and `hand_right` camera streams and
  extrinsics in the dataset metadata even though the URDF only models
  `front_camera_link` and `top_camera_link`.
- Do not tie `*_camera_mount_compat` only to links whose names contain
  "camera". The trigger is: a calibrated camera extrinsic references a
  robot link frame, and axis normalization rotated that reference frame's
  coordinates. In that case, keep the aligned EE / reference link as the
  convention-compliant frame and add a fixed compat child that preserves
  the origin extrinsic frame.
- Do not override the finger-group rotation in runtime transforms. Under
  the current pipeline, `<parent>_gripper_end` is the only element of
  `finger_keys` per arm and is authored to live in the semantic EE frame
  directly. Any override that steals the rotation from the last arm-group
  link becomes a redundancy at best and a source of drift when the yaml is
  edited.
- Do not use a virtual EE overlay for visualization. Aligned URDFs carry
  the semantic EE frame explicitly via `*_ee` and `*_gripper_end` links;
  overlays that re-derive an EE frame in Python are a code path to keep in
  sync and a source of divergence.

## Validation Checklist

- Run the axis-normalization test: every actuated joint's `<axis>` in the
  aligned URDF must be `0 0 1`.
- Run motion consistency with nonzero samples for every changed joint
  family. Positions of every link in the manifest's `arm_link_keys` plus
  each `<parent>_gripper_end` must agree between origin and aligned URDFs
  (with the aligned URDF's last link swapped to `<parent>_ee` when
  `rotate_z_deg != 0`).
- Run camera compatibility checks for every declared `camera_references`
  entry: origin-URDF FK of the reference link must equal aligned-URDF FK of
  either the same-named link or the `*_camera_mount_compat` child.
- Run a mesh-resolution check for every aligned URDF: every relative
  `<mesh filename="...">` should exist under the aligned URDF directory.
- Run focused pytest and ruff in the remote environment when local
  dependencies are missing.
