# URDF Alignment Tools

This folder contains scripts and documentation for the URDF alignment
workflow. Two parallel trees hold the aligned URDF assets and their
per-embodiment validation manifests:

```text
projects/holobrain_internal/common/
├── urdf_tools/manifests/<dataset>/<embodiment>/
│   └── alignment.yaml    # git-tracked, per-embodiment validation contract
└── urdf_align/<dataset>/<embodiment>/
    └── <robot>.urdf      # aligned URDF asset — symlink to a shared bucket,
                          # not git-tracked
```

Manifests are the canonical source of alignment intent and are committed
alongside the code. The aligned URDF assets themselves live in the shared
bucket (`urdf_align/` is a symlink); the case loader tolerates a missing
bucket by returning the `UrdfAlignmentCase` with `resolved=False`, and the
pytest suite skips the case cleanly rather than failing.

Do not place scripts, reports, baselines, or other working logs in either
tree.

Working logs are written under the ignored workspace folder:

```text
projects/holobrain_internal/common/workspace_test/urdf_align_logs/
```

See `ALIGNMENT_NOTES.md` for lessons learned from earlier alignment passes,
including mistakes that should not be repeated, and for the full derivation
of the axis-compensation rule.

## Quick Start

Three common paths, tersest command first.

**Add a new embodiment.** Land a manifest, run alignment, wire the
config, run tests.

```bash
# 1. Write the manifest (see "Manifest Schema" below).
$EDITOR projects/holobrain_internal/common/urdf_tools/manifests/<dataset>/<embodiment>/alignment.yaml

# 2. Emit the aligned URDF into urdf_align/<dataset>/<embodiment>/.
conda run -n sem python -m projects.holobrain_internal.common.urdf_tools.cli align \
  --repo-root . --case <embodiment>

# 3. Add the aligned dataset-config entry (or cut over the base key).
$EDITOR projects/holobrain_internal/common/configs/data_configs/config_<dataset>_dataset.py

# 4. Verify.
PYTHONPATH=. conda run -n sem pytest -c tests/pytest.ini \
  tests/test_robo_orchard_lab/projects/holobrain_internal/common/urdf_tools/ \
  -k <embodiment> -q
```

**Re-align every embodiment after a pipeline change.** Idempotent by
contract — a no-op when nothing upstream moved.

```bash
conda run -n sem python -m projects.holobrain_internal.common.urdf_tools.cli align \
  --repo-root .
```

**Inspect an aligned URDF visually.** Emits a per-case contact sheet
plus a JSON checklist under the output dir.

```bash
conda run -n sem python -m projects.holobrain_internal.common.urdf_tools.cli visual-verify \
  --repo-root . \
  --output-dir workspace_test/visual_verify \
  --case <embodiment>
```

## Repository Layout

Files a new contributor should know about, in the order they matter:

```text
urdf_tools/
├── manifests/<dataset>/<embodiment>/alignment.yaml
│                            # per-embodiment validation contract (git-tracked)
├── cases.py                 # loads manifests into UrdfAlignmentCase objects
├── transform/               # four-stage pipeline (axis / ee / gripper_end / camera)
│   └── __init__.py::apply_alignment
├── config_wiring.py         # stage-4 wiring check dispatched from tests + CLI
├── fk_baseline.py           # motion + camera consistency assertions
├── visual_verify.py         # renders per-case contact sheets
├── cli.py                   # `origin`, `align`, `visual-verify` subcommands
└── README.md, ALIGNMENT_NOTES.md
```

Tests mirror the pipeline concerns one file each — see
[Test Suite Layout](#test-suite-layout) below.

## Control Semantics

The alignment pipeline must preserve control semantics: existing joint
values and actions must map to the same physical robot motion. Alignment
changes are limited to coordinate conventions and must be validated by
FK / action-equivalence pytest tests before replacing legacy dataset config
entries.

## Alignment Goals

The final target is to make multi-robot pretraining easier by aligning
URDF coordinate definitions across datasets without changing FK
correctness or control semantics. Aligned URDFs expose two convention-
compliant fixed children per arm:

- `<parent>_ee` — right/down/front semantic EE frame (only inserted when
  the manifest's `rotate_z_deg != 0`).
- `<parent>_gripper_end` — physical tip of the closed gripper, at
  `xyz="0 0 gripper_forward"` in the EE frame (or in `<parent>` directly
  when no `_ee` child exists). Identity rotation relative to the EE frame.

Model-facing state/action supervision consumes these two link families;
camera projection consumes the origin-compatible reference link (either the
unchanged same-name link, or a `*_camera_mount_compat` alias when
alignment rotated an upstream frame).

## Alignment Workflow

Use this staged process for each new embodiment:

1. **Copy assets.** Copy the source URDF and visualization mesh sidecars
   into `urdf_align/<dataset>/<embodiment>/`. Keep immutable references
   under `urdf_origin/<dataset>/<embodiment>/`; keep generated reports and
   baselines under `workspace_test/urdf_align_logs/`.
2. **Trace consumers.** Read dataset configs, dataloader/packer code, and
   real metadata to identify which URDF links are used as FK state links,
   EE links, and camera-extrinsic reference frames. Wrist cameras are
   common but not guaranteed; use actual metadata when available.
3. **Author `alignment.yaml`.** Colocated with the aligned URDF. Declares
   the dataset-config wiring, the ordered arm link chains, the per-arm EE
   spec (`parent`, `rotate_z_deg`, `gripper_forward`), and the
   camera-reference link list. See the schema section below.
4. **Run the transformer pipeline.** Four fixed-order stages:
   1. Normalize actuated joint axes to `0 0 1` with the axis-compensation
      rule (`ALIGNMENT_NOTES.md § Axis Compensation Rule`).
   2. Insert `<parent>_ee` fixed children per arm where
      `rotate_z_deg != 0`; move `<inertial>/<visual>/<collision>` from
      parent to `_ee` child with `S⁻¹` re-expression.
   3. Insert `<parent>_gripper_end` fixed children per arm — attached to
      `<parent>_ee` when present, otherwise to `<parent>` — at
      `xyz="0 0 gripper_forward"` with identity rotation in the parent
      frame.
   4. Insert `*_camera_mount_compat` compensator children only where the
      declared reference link's zero-pose FK moved between origin and
      aligned URDFs (i.e. stage 1 rotated an upstream frame).
5. **Wire the dataset config.** Add a `_urdf_v2` companion entry that
   points at the aligned URDF and swaps the last-arm-link key to
   `<parent>_ee` where an `_ee` child was inserted, and sets
   `finger_keys=[["<parent>_gripper_end"], ...]` per arm. `arm_joint_id`
   stays inline (packer-specific, not URDF-side).

The learning model consumes convention-compliant semantic links when they
exist, such as `fl_link6_ee` for arm FK state and `fl_link6_gripper_end`
for gripper state. Camera projection code uses the origin-compatible
reference link directly, or its `*_camera_mount_compat` alias only when
the reference link itself had to be rotated for alignment.

## Manifest Schema

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

**Optional per-arm `gripper_end` override.** When the default
`<parent>_ee` (or `<parent>`) attach with `xyz="0 0 gripper_forward"` is
wrong for an embodiment — for example when the last arm joint rotates
about a non-Z axis and the gripper extends perpendicular to `+Z` — the
manifest can override attach link and origin per arm:

```yaml
arms:
  - arm_link_keys: [...]
    ee:
      parent: link7
      rotate_z_deg: 0
    gripper_end:
      attach_link: left_gripper_link   # attach under the real gripper link
      xyz: [0.00, 0.00, -0.08]         # look-ahead grasp point past finger tip
      rpy: [3.14159, 0.0, -1.5708]     # Rx(180) so +Z == approach direction
```

The Behavior R1 Pro manifest is the canonical example — see
`test_urdf_align_embodiment_overrides.py::test_behavior_r1_pro_gripper_end_override_geometry`
and the `r1-gripper-end-geometry` memory for the load-bearing invariants
this override must preserve.

**Behavior anchored to the schema:**

- The `arms[]` block is authoritative for the aligned URDF's per-arm
  kinematic identity. `arm_link_keys[-1]` must equal `ee.parent`.
- `ee.rotate_z_deg == 0` (or absent) skips `<parent>_ee` insertion; the
  last arm link remains the EE frame directly.
- `ee.gripper_forward` (meters, default `0.20`) is applied as the origin
  `xyz` of the `<parent>_gripper_end` fixed joint.
- `camera_references` is a bare list of URDF link names whose FK pose in
  the origin URDF must remain reachable in the aligned URDF. When the
  aligned FK equals the origin FK, no compat child is inserted; when it
  differs, `<link>_camera_mount_compat` is inserted with a compensator
  transform.
- Shared embodiments: when two dataset-config keys back the same physical
  robot with the same dataset asset (RoboTwin `aloha_v1` / `aloha_v2`),
  authors one folder and lists both keys under `config.key: [...]`.

## Validation Policy

- Pytest is the source of truth for whether an aligned URDF preserves
  origin FK / action semantics.
- Generated JSON baselines and copy reports are debugging logs only; they
  live under `workspace_test/urdf_align_logs/` and are ignored by git.
- Tests are case-driven and reusable across datasets / robots. Add new
  robots by writing a new `alignment.yaml` under the appropriate
  `urdf_tools/manifests/<dataset>/<embodiment>/` folder; the case loader
  auto-discovers manifests by walking the directory.
- The manifest is the per-robot validation contract. Fields it does *not*
  carry, and why:
  - **Joint names:** the alignment pipeline never renames actuated joints
    (only adds new fixed joints), so `chain.get_joint_parameter_names()`
    returns the same list for origin and aligned URDFs. FK regression
    tests derive that list from either URDF directly.
  - **Aligned link renames:** the pipeline never renames existing links;
    `<parent>_ee` is always a sibling. No rename table is required.
  - **Camera stream names:** metadata only, not consumed by the
    transformer. If a stream-to-link mapping is needed for a debug tool,
    it lives in the dataset config, not the alignment manifest.

## Idempotency Contract

Every stage in `transform/` is idempotent: running the pipeline on an
already-aligned URDF is a byte-level no-op.

- **Stage 1 (axis).** Naturally idempotent — actuated joints whose axis is
  already `0 0 1` short-circuit without touching the parent frame.
- **Stage 2 (ee_frames).** Detects an existing `<parent>_ee_joint` whose
  parent/child links and rotation origin match the manifest and skips the
  geometry move.
- **Stage 3 (gripper_end).** Detects an existing `<parent>_gripper_end`
  fixed joint whose parent/child and full `xyz`/`rpy` origin match the spec
  and skips insertion.
- **Stage 4 (camera).** Naturally idempotent — the compensator is only
  inserted when origin-vs-aligned FK differs; after alignment the two FKs
  are equal by construction.

The `test_apply_alignment_on_aligned_urdf_is_a_no_op` case in the pytest
suite enforces this contract per embodiment: it applies the pipeline once,
feeds the aligned bytes back in, and asserts that a second pass produces
the same bytes and reports zero newly inserted joints or compat links.

## Tooling

- `sync_assets.py` copies a URDF plus local visualization mesh sidecars.
  Meshes are copied for local visualization but ignored by git.
- `cases.py` walks
  `urdf_tools/manifests/<dataset>/<embodiment>/alignment.yaml` files and
  loads them into typed metadata used by pytest and the
  transformer runner.
- `transform/__init__.py` orchestrates the four alignment stages
  (`axis`, `ee_frames`, `gripper_end`, `camera`) and returns serialized
  aligned URDF bytes plus a per-stage report.
- `fk_baseline.py` can write optional FK baseline JSON logs for manual
  debugging.
- `cli.py` exposes the utilities.

## Test Suite Layout

All tests live under
`tests/test_robo_orchard_lab/projects/holobrain_internal/common/urdf_tools/`.
Each file is single-topic and opens with a docstring stating the invariant it
defends. Shared helpers live in `conftest.py`:

| File | What it pins |
|---|---|
| `conftest.py` | `alignment_cases`, `alignment_case(adapter, config_key)`, `require_resolved(case)`, `wrist_axis_alignment_cases` — imported by every case-driven file. `require_resolved(case)` is the uniform skip helper called as the first line of any test that touches `case.aligned_urdf` or `case.origin_urdf`. |
| `test_manifest_loader.py` | `alignment.yaml` → `UrdfAlignmentCase` shape. Loader parses a real manifest into the schema fields the rest of the pipeline consumes. |
| `test_alignment_plan.py` | `UrdfAlignManifest.propose_entry` / `from_inventory_json` — the hierarchical `urdf_align/<dataset>/<embodiment>/` layout convention. |
| `test_config_wiring.py` | Parametrized per case: every packer config actually references its aligned URDF (companion `*_urdf_v2` key or base-cutover). |
| `test_cli_inventory.py` | `cli origin` subprocess writes an inventory JSON operators can act on. |
| `test_fk_baseline.py` | `compute_fk_baseline` determinism + `copy_urdf_asset_tree` mesh-copy contract. |
| `test_urdf_align_transform.py` | `insert_ee_children` re-expresses moved geometry so world pose stays fixed under `rotate_z_deg != 0`. |
| `test_urdf_align_contract.py` | **The big one.** Seven parametrized invariants every aligned URDF must satisfy — motion consistency, camera consistency, actuated-axis normalization, compat-link necessity, byte-idempotence, apply-on-aligned no-op, per-arm `gripper_end` FK, wrist-axis extension. |
| `test_urdf_align_embodiment_overrides.py` | Bespoke invariants that don't generalize — R1 Pro `gripper_end` override geometry (per the `r1-gripper-end-geometry` memory) and G1 camera-mount ordering. |

**Running the suite.**

```bash
PYTHONPATH=. conda run -n sem pytest -c tests/pytest.ini \
  tests/test_robo_orchard_lab/projects/holobrain_internal/common/urdf_tools/ -q
```

**Expected output when the bucket is unmounted:** ~9 pure-unit passes,
~256 skips. Every skip is a per-case parametrization whose aligned URDF
lives in the shared bucket. When the bucket is mounted, those skips
convert to passes.

**Running one embodiment only.** `-k <substring>` filters by case name,
which is derived from `<adapter>/<config_key>`:

```bash
PYTHONPATH=. conda run -n sem pytest -c tests/pytest.ini \
  tests/test_robo_orchard_lab/projects/holobrain_internal/common/urdf_tools/ \
  -k behavior -q
```

**Adding a new embodiment.** No new test file. Land the manifest and
the parametrized tests in `test_config_wiring.py` /
`test_urdf_align_contract.py` pick it up automatically via
`alignment_cases()`. Only add a file under
`test_urdf_align_embodiment_overrides.py` when the embodiment carries a
load-bearing invariant that the generic contract can't express.
