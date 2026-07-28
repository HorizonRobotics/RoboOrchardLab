"""Diagnose whether TRAINING_DATASETS ends up opening the same
`instructions_v2/agilex` LMDB more than once, which triggers
`lmdb.Error: The environment ... is already open in this process.`.

Usage:
    cd projects/holobrain_internal/common
    python diagnose_agilex_instruction_dup.py

You can also pass a custom dataset_specs module path:
    python diagnose_agilex_instruction_dup.py --specs configs/dataset_specs.py

The script does not import torch / accelerate / cuda; it only loads the
dataset spec module, so it is safe to run on any machine.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path


AGILEX_INSTRUCTION_TOKEN = "instructions_v2/agilex"


def _load_specs_module(specs_ref: str):
    """Load a Python module either by import name or file path."""
    p = Path(specs_ref)
    if specs_ref.endswith(".py") or p.exists():
        p = p.resolve()
        spec = importlib.util.spec_from_file_location(p.stem, p)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load specs module from: {p}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(specs_ref)


def _stringify(value) -> str:
    """Convert instruction_paths (which may be a list/str/None) to a string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(x) for x in value)
    return str(value)


def _is_agilex(spec: dict) -> bool:
    return spec.get("dataset_type") == "agilex"


def _uses_agilex_instruction(spec: dict) -> bool:
    return AGILEX_INSTRUCTION_TOKEN in _stringify(spec.get("instruction_paths"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specs",
        default="dataset_specs",
        help=(
            "Either a Python import name (default: 'dataset_specs') or a "
            "path to a dataset_specs.py file. When using the default, run "
            "this script from `projects/holobrain_internal/common/`."
        ),
    )
    args = parser.parse_args()

    # Make sure `configs/` is importable when running from `common/`.
    here = Path(__file__).resolve().parent
    for candidate in (here / "configs", here):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    print(f"[info] cwd            : {os.getcwd()}")
    print(f"[info] loading specs  : {args.specs}")
    module = _load_specs_module(args.specs)
    print(f"[info] loaded from    : {getattr(module, '__file__', '<?>')}")
    print()

    if not hasattr(module, "training_datasets"):
        print("[error] dataset specs module has no `training_datasets` attr.")
        return 2

    enabled = list(module.training_datasets)
    raw = list(getattr(module, "TRAINING_DATASETS", enabled))

    print(f"Total TRAINING_DATASETS entries : {len(raw)}")
    print(f"Enabled after filter_list       : {len(enabled)}")
    print()

    # ---- agilex-family stats ---------------------------------------------
    agilex_raw = [s for s in raw if _is_agilex(s)]
    agilex_enabled = [s for s in enabled if _is_agilex(s)]
    uses_inst = [s for s in enabled if _is_agilex(s) and _uses_agilex_instruction(s)]

    print("== agilex-family (dataset_type == 'agilex') ==")
    print(f"  defined in TRAINING_DATASETS      : {len(agilex_raw)}")
    print(f"  enabled after filter_list         : {len(agilex_enabled)}")
    print(f"  enabled AND open agilex instr lmdb: {len(uses_inst)}   "
          f"<-- this is the number that must be <= 1")
    print()

    print("Enabled agilex datasets:")
    for s in agilex_enabled:
        marker = "  [OPENS instructions_v2/agilex]" if _uses_agilex_instruction(s) else ""
        print(f"  - {s.get('dataset_name'):<40s} "
              f"setting_type={s.get('setting_type')}{marker}")
    print()

    # ---- verdict ---------------------------------------------------------
    if len(uses_inst) >= 2:
        print("[VERDICT] >=2 enabled datasets share "
              f"'{AGILEX_INSTRUCTION_TOKEN}'. This WILL trigger the")
        print("          'lmdb.Error: The environment ... is already open in "
              "this process.'")
        print("          error during training-set construction, regardless "
              "of CUDA / torch / py-lmdb version.")
        return 1

    if len(uses_inst) == 1:
        print("[VERDICT] Only 1 enabled dataset opens "
              f"'{AGILEX_INSTRUCTION_TOKEN}'. No duplicate open, safe.")
        return 0

    print("[VERDICT] No enabled dataset opens "
          f"'{AGILEX_INSTRUCTION_TOKEN}'. No duplicate open, safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
