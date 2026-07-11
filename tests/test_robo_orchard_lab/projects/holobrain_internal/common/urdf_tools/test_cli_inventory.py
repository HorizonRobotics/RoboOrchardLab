# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

"""``cli origin`` writes an inventory JSON operators can act on.

Guards the subprocess contract: given a stub configs tree and a stub
urdf tree, ``cli origin`` walks both and emits a JSON with the config
references and URDF assets it discovered. The rest of the CLI
(visual-verify) is exercised through direct-function tests; only the
inventory writer has a subprocess test because it's the entry point
operators run by hand.
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path


def _write_tiny_urdf(path: Path) -> None:
    path.write_text(
        """
<robot name="tiny_arm">
  <link name="base" />
  <link name="link1" />
  <joint name="joint1" type="revolute">
    <parent link="base" />
    <child link="link1" />
    <origin xyz="1 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
  </joint>
</robot>
""".strip(),
        encoding="utf-8",
    )


def test_inventory_cli_writes_json(tmp_path: Path):
    """``cli origin`` emits a JSON linking configs to their URDFs."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    urdf_dir = tmp_path / "urdf"
    urdf_dir.mkdir()
    _write_tiny_urdf(urdf_dir / "tiny.urdf")
    (config_dir / "config_tiny_dataset.py").write_text(
        'kinematics_config = {"urdf": "./urdf/tiny.urdf"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "inventory.json"

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "projects.holobrain_internal.common.urdf_tools.cli",
            "origin",
            "--repo-root",
            str(tmp_path),
            "--config-root",
            "configs",
            "--urdf-root",
            "urdf",
            "--output",
            str(output_path),
        ]
    )

    inventory = json.loads(output_path.read_text(encoding="utf-8"))
    assert inventory["config_references"][0]["urdf"] == "./urdf/tiny.urdf"
    assert inventory["urdf_assets"][0]["path"].endswith("tiny.urdf")
