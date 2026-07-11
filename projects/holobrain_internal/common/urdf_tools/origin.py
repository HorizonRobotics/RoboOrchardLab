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
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class ConfigUrdfReference:
    config_path: Path
    dataset: str
    urdf: str
    dict_path: tuple[str, ...]

    def to_json_dict(self) -> dict:
        ret = asdict(self)
        ret["config_path"] = str(self.config_path)
        ret["dict_path"] = list(self.dict_path)
        return ret


@dataclass(frozen=True)
class MeshReference:
    filename: str
    resolved_path: Path | None

    def to_json_dict(self) -> dict:
        return {
            "filename": self.filename,
            "resolved_path": (
                None if self.resolved_path is None else str(self.resolved_path)
            ),
        }


@dataclass(frozen=True)
class UrdfAsset:
    path: Path
    mesh_references: tuple[MeshReference, ...]

    def to_json_dict(self) -> dict:
        return {
            "path": str(self.path),
            "mesh_references": [
                mesh_ref.to_json_dict() for mesh_ref in self.mesh_references
            ],
        }


@dataclass(frozen=True)
class UrdfInventory:
    config_references: tuple[ConfigUrdfReference, ...]
    urdf_assets: tuple[UrdfAsset, ...]

    def to_json_dict(self) -> dict:
        return {
            "config_references": [
                ref.to_json_dict() for ref in self.config_references
            ],
            "urdf_assets": [
                asset.to_json_dict() for asset in self.urdf_assets
            ],
        }

    def write_json(self, path: Path) -> None:
        """Write origin scan results as deterministic JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def collect_urdf_inventory(
    repo_root: Path,
    config_roots: list[Path],
    urdf_roots: list[Path],
) -> UrdfInventory:
    """Collect URDF config references and mesh references under URDF roots."""

    # Config references answer "which URDFs are in use by dataset configs?".
    # URDF assets answer "which URDF files exist under the scanned roots?".
    # Keeping both views in one JSON makes gaps obvious during migration.
    config_references: list[ConfigUrdfReference] = []
    for config_root in config_roots:
        if not config_root.exists():
            continue
        for config_path in sorted(config_root.rglob("*.py")):
            config_references.extend(
                _collect_config_references(repo_root, config_path)
            )

    urdf_assets: list[UrdfAsset] = []
    for urdf_root in urdf_roots:
        if not urdf_root.exists():
            continue
        for urdf_path in sorted(urdf_root.rglob("*.urdf")):
            urdf_assets.append(_collect_urdf_asset(urdf_path))

    return UrdfInventory(
        config_references=tuple(config_references),
        urdf_assets=tuple(urdf_assets),
    )


def _collect_config_references(
    repo_root: Path, config_path: Path
) -> list[ConfigUrdfReference]:
    try:
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    references: list[ConfigUrdfReference] = []
    rel_config_path = config_path.relative_to(repo_root)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not _is_mapping_node(
            node.value
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Most configs use a top-level `dataset_config` mapping; some
                # files assign named dictionaries directly. Preserve that name
                # as a fallback dataset label for nested URDF references.
                dataset = None
                if target.id != "dataset_config":
                    dataset = target.id
                references.extend(
                    _walk_literal_dict(
                        node.value,
                        rel_config_path,
                        (),
                        dataset=dataset,
                    )
                )
    return references


def _walk_literal_dict(
    node: ast.AST,
    config_path: Path,
    dict_path: tuple[str, ...],
    dataset: str | None,
) -> list[ConfigUrdfReference]:
    if not _is_mapping_node(node):
        return []

    references: list[ConfigUrdfReference] = []
    for key, value_node in _iter_mapping_items(node):
        if key is None:
            continue
        next_dataset = dataset if dataset is not None else key
        next_path = (*dict_path, key)
        if key == "urdf" and isinstance(value_node, ast.Constant):
            # The alignment inventory intentionally records only literal URDF
            # paths. Dynamic config expressions need manual review before they
            # can be migrated safely.
            if isinstance(value_node.value, str) and value_node.value.endswith(
                ".urdf"
            ):
                references.append(
                    ConfigUrdfReference(
                        config_path=config_path,
                        dataset=next_dataset,
                        urdf=value_node.value,
                        dict_path=next_path,
                    )
                )
        references.extend(
            _walk_literal_dict(
                value_node,
                config_path,
                next_path,
                next_dataset,
            )
        )
    return references


def _is_mapping_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Dict) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    )


def _iter_mapping_items(node: ast.AST) -> list[tuple[str | None, ast.AST]]:
    if isinstance(node, ast.Dict):
        return [
            (_literal_key(key_node), value_node)
            for key_node, value_node in zip(
                node.keys,
                node.values,
                strict=False,
            )
        ]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    ):
        return [(keyword.arg, keyword.value) for keyword in node.keywords]
    return []


def _literal_key(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_urdf_asset(urdf_path: Path) -> UrdfAsset:
    try:
        root = ElementTree.parse(urdf_path).getroot()
    except ElementTree.ParseError:
        # A malformed URDF is still an asset worth listing; leave mesh refs
        # empty so the caller can decide whether to fix or skip it.
        return UrdfAsset(path=urdf_path, mesh_references=())

    mesh_references: list[MeshReference] = []
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh_references.append(
            MeshReference(
                filename=filename,
                resolved_path=_resolve_mesh_path(urdf_path, filename),
            )
        )
    return UrdfAsset(path=urdf_path, mesh_references=tuple(mesh_references))


def _resolve_mesh_path(urdf_path: Path, filename: str) -> Path | None:
    if "://" in filename:
        # package:// and other URI-like mesh references are not local files, so
        # there is no repo-relative path to resolve or copy automatically.
        return None
    mesh_path = Path(filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()
