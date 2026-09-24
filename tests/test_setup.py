# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import ast
import runpy
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parents[1]
_SETUP_PATH = _LAB_ROOT / "setup.py"


def _setup_install_requires() -> list[str]:
    """Return the literal base dependency list declared by ``setup.py``."""
    module = ast.parse(_SETUP_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "install_requires"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("setup.py does not declare install_requires")


_CORE_MASTER_DEPENDENCY = (
    "robo_orchard_core[robotics]@git+https://github.com/"
    "HorizonRobotics/robo_orchard_core.git@"
    "1b6df5bd5659e2deaa679b8354b68f324af3da9c"
)


def test_setup_pins_core_master_commit_with_robotics_profile() -> None:
    assert _CORE_MASTER_DEPENDENCY in _setup_install_requires()


def test_local_dependency_rewrite_preserves_core_extras(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "robo_orchard_core").mkdir()
    monkeypatch.setenv("ROBO_ORCHARD_USE_LOCAL", "1")

    setup_globals = runpy.run_path(
        str(_SETUP_PATH),
        run_name="setup_under_test",
    )

    assert setup_globals["handle_local_dependency"](
        [_CORE_MASTER_DEPENDENCY],
        base_dir=str(tmp_path),
    ) == [f"robo_orchard_core[robotics]@file://{tmp_path}/robo_orchard_core"]
