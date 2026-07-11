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

"""``UrdfAlignManifest`` alignment-plan proposal shape.

The plan tells operators *where the emitted URDF is going to land*
before any file is written. The hierarchical
``urdf_align/<dataset>/<embodiment>/<file>`` layout is the durable
convention every downstream tool (packer configs, visual verify, bucket
sync) reads. If a proposal ever silently changes shape, the wrong dir
gets written into and consumers break. These tests pin the three
inputs the plan builder receives: a direct call, an inventory that
carries a dict_path, and an already-aligned entry that must be
skipped.
"""

from __future__ import annotations
import json
from pathlib import Path

from projects.holobrain_internal.common.urdf_tools.aligned import (
    UrdfAlignManifest,
)


def test_propose_entry_uses_hierarchical_urdf_align_path():
    """Direct proposal drops URDFs under dataset/embodiment/file."""

    manifest = UrdfAlignManifest.default()

    entry = manifest.propose_entry(
        dataset="robotwin",
        embodiment="ur5_wsg",
        source_urdf=Path("projects/holobrain_internal/common/urdf/robot.urdf"),
    )

    assert entry.target_urdf == Path(
        "projects/holobrain_internal/common/urdf_align/robotwin/"
        "ur5_wsg/robot.urdf"
    )
    assert entry.status == "inventory"
    assert entry.control_semantics == "preserve_joint_values_and_actions"


def test_from_inventory_json_uses_dataset_key_for_embodiment(tmp_path: Path):
    """A one-level ``dict_path`` uses ``dataset`` as the embodiment slug."""

    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "config_references": [
                    {
                        "config_path": "projects/holobrain_internal/common/"
                        "configs/data_configs/config_agibot_dataset.py",
                        "dataset": "agibot",
                        "dict_path": ["agibot", "urdf"],
                        "urdf": "./urdf/G1_120s_dual.urdf",
                    }
                ],
                "urdf_assets": [],
            }
        ),
        encoding="utf-8",
    )

    manifest = UrdfAlignManifest.from_inventory_json(inventory_path)

    assert manifest.entries[0].target_urdf == Path(
        "projects/holobrain_internal/common/urdf_align/agibot/agibot/"
        "G1_120s_dual.urdf"
    )


def test_from_inventory_json_uses_nested_dict_key_for_singleton_variables(
    tmp_path: Path,
):
    """A nested ``dict_path`` takes the first key as the embodiment slug.

    Confirms that configs which nest their variants under a shared root
    variable (``dataset_lmdb_config``) still yield one embodiment
    directory per variant instead of collapsing them all under one.
    """

    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "config_references": [
                    {
                        "config_path": "projects/holobrain_internal/common/"
                        "configs/data_configs/config_interna1_dataset.py",
                        "dataset": "dataset_lmdb_config",
                        "dict_path": ["interna1_genieg1", "urdf"],
                        "urdf": "./urdf/InternData-A1_urdf/G1_120s/"
                        "G1_120s.urdf",
                    }
                ],
                "urdf_assets": [],
            }
        ),
        encoding="utf-8",
    )

    manifest = UrdfAlignManifest.from_inventory_json(inventory_path)

    assert manifest.entries[0].target_urdf == Path(
        "projects/holobrain_internal/common/urdf_align/interna1/"
        "interna1_genieg1/G1_120s.urdf"
    )


def test_from_inventory_json_skips_entries_already_pointing_at_aligned(
    tmp_path: Path,
):
    """Config entries already under ``urdf_align/`` yield no proposals."""

    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "config_references": [
                    {
                        "config_path": "projects/holobrain_internal/common/"
                        "configs/data_configs/config_interna1_dataset.py",
                        "dataset": "dataset_lmdb_config",
                        "dict_path": ["interna1_genieg1_urdf_v2", "urdf"],
                        "urdf": "./urdf_align/interna1/interna1_genieg1/"
                        "G1_120s.urdf",
                    }
                ],
                "urdf_assets": [],
            }
        ),
        encoding="utf-8",
    )

    manifest = UrdfAlignManifest.from_inventory_json(inventory_path)

    assert manifest.entries == []
