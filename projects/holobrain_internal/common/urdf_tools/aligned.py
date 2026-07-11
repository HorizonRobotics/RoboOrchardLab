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

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class UrdfAlignEntry:
    """Manifest row for one legacy URDF and its aligned target asset."""

    dataset: str
    embodiment: str
    source_urdf: Path
    target_urdf: Path
    status: str = "inventory"
    control_semantics: str = "preserve_joint_values_and_actions"
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict:
        ret = asdict(self)
        ret["source_urdf"] = str(self.source_urdf)
        ret["target_urdf"] = str(self.target_urdf)
        ret["notes"] = list(self.notes)
        return ret


@dataclass
class UrdfAlignManifest:
    """Path policy for aligned URDF assets."""

    root: Path
    entries: list[UrdfAlignEntry] = field(default_factory=list)

    @classmethod
    def default(cls) -> "UrdfAlignManifest":
        return cls(root=Path("projects/holobrain_internal/common/urdf_align"))

    def propose_entry(
        self,
        dataset: str,
        embodiment: str,
        source_urdf: Path,
        status: str = "inventory",
        notes: tuple[str, ...] = (),
    ) -> UrdfAlignEntry:
        """Create a manifest row using `urdf_align/<dataset>/<embodiment>`."""

        # The aligned tree mirrors the conceptual dataset/embodiment grouping
        # instead of the vendor source folder. This keeps future robot cases
        # predictable while preserving the original URDF filename for
        # traceability.
        entry = UrdfAlignEntry(
            dataset=dataset,
            embodiment=embodiment,
            source_urdf=source_urdf,
            target_urdf=self.root / dataset / embodiment / source_urdf.name,
            status=status,
            notes=notes,
        )
        self.entries.append(entry)
        return entry

    def to_json_dict(self) -> dict:
        return {
            "root": str(self.root),
            "control_semantics": "preserve_joint_values_and_actions",
            "entries": [entry.to_json_dict() for entry in self.entries],
        }

    @classmethod
    def from_inventory_json(
        cls,
        inventory_path: Path,
        root: Path | None = None,
    ) -> "UrdfAlignManifest":
        """Build aligned path proposals from an origin scan JSON file."""

        import json

        data = json.loads(inventory_path.read_text(encoding="utf-8"))
        manifest = cls(root=root or cls.default().root)
        seen: set[tuple[str, str, Path]] = set()
        for ref in data.get("config_references", []):
            # One URDF can be referenced by multiple config keys. De-duplicate
            # after deriving the user-facing dataset/embodiment path so the
            # plan contains one aligned asset target per unique origin asset.
            dataset = _dataset_from_config_path(ref["config_path"])
            embodiment = _embodiment_from_ref(ref)
            source_urdf = Path(ref["urdf"])
            if _is_already_aligned(source_urdf):
                # Re-running the planner on an inventory that includes
                # urdf_align should not recursively propose aligned copies of
                # already-aligned assets.
                continue
            key = (dataset, embodiment, source_urdf)
            if key in seen:
                continue
            seen.add(key)
            manifest.propose_entry(dataset, embodiment, source_urdf)
        return manifest


def _slug(value: str) -> str:
    return (
        value.replace("/", "_").replace(".", "_").replace("-", "_").strip("_")
    )


def _dataset_from_config_path(config_path: str) -> str:
    name = Path(config_path).name
    if name.startswith("config_"):
        name = name[len("config_") :]
    if name.endswith("_dataset.py"):
        name = name[: -len("_dataset.py")]
    elif name.endswith(".py"):
        name = name[:-3]
    return _slug(name)


def _embodiment_from_ref(ref: dict) -> str:
    dict_path = ref.get("dict_path", [])
    if len(dict_path) > 1:
        return _slug(dict_path[0])
    return _slug(ref["dataset"])


def _is_already_aligned(source_urdf: Path) -> bool:
    # Treat the transformer output tree as a terminal root so the planner
    # never proposes recursive aligned copies of it.
    return any(part == "urdf_align" for part in source_urdf.parts)
