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

"""Contract tests for local packaging paths."""

from __future__ import annotations
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from robo_orchard_lab.dataset.packaging_paths import (
    DatasetPackagingPaths,
    normalize_local_dataset_path,
)


@pytest.fixture(autouse=True)
def _use_test_lock_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path.parent / "lock-cache"))


def test_normalizes_local_paths_and_rejects_uris(tmp_path: Path) -> None:
    relative_path = Path("local:dataset")
    assert normalize_local_dataset_path(relative_path).endswith(
        os.path.join("local:dataset")
    )
    with pytest.raises(ValueError, match="URI paths are not supported"):
        normalize_local_dataset_path("s3://bucket/dataset")
    with pytest.raises(TypeError, match="string or os.PathLike"):
        normalize_local_dataset_path(b"not-a-text-path")
    paths = DatasetPackagingPaths.resolve(tmp_path / "dataset")
    assert paths.dataset_dir == str(tmp_path / "dataset")


def test_empty_lock_cache_uses_default_cache_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path.parent / "home"))

    paths = DatasetPackagingPaths.resolve(tmp_path / "dataset")

    assert paths.coordination_lock_path.startswith(
        str(tmp_path.parent / "home" / ".cache")
    )


def test_default_cache_accepts_home_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    paths = DatasetPackagingPaths.resolve("~/dataset")

    assert paths.dataset_dir == str(tmp_path / "home" / "dataset")
    assert paths.coordination_lock_path.startswith(
        str(tmp_path / "home" / ".cache")
    )


def test_resolves_same_parent_workspace_and_stable_alias_lock(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    paths = DatasetPackagingPaths.resolve(real_parent / "dataset")
    alias_paths = DatasetPackagingPaths.resolve(alias_parent / "dataset")

    expected_workspace = f"{paths.dataset_dir}.__robo_orchard_packaging__"
    assert paths.workspace_dir == expected_workspace
    assert Path(paths.workspace_dir).parent == Path(paths.dataset_dir).parent
    assert paths.builder_output_dir == os.path.join(
        paths.workspace_dir,
        "dataset",
    )
    assert paths.hf_cache_dir == os.path.join(paths.workspace_dir, "hf_cache")
    assert paths.coordination_lock_path == alias_paths.coordination_lock_path
    assert not paths.coordination_lock_path.startswith(str(real_parent))
    with pytest.raises(FrozenInstanceError):
        paths.dataset_dir = "mutated"  # type: ignore[misc]


def test_reservation_prevents_workspace_from_becoming_a_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dataset"
    reserved = Path(f"{target}.__robo_orchard_packaging__")

    with pytest.raises(ValueError, match="reserved workspace suffix"):
        DatasetPackagingPaths.resolve(reserved)
    with pytest.raises(ValueError, match="reserved workspace suffix"):
        DatasetPackagingPaths.resolve(reserved / "nested")


def test_preconditions_reject_links_and_require_force(
    tmp_path: Path,
) -> None:
    paths = DatasetPackagingPaths.resolve(tmp_path / "dataset")
    Path(paths.workspace_dir).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symbolic links"):
        paths.validate_preconditions(force_overwrite=True)

    Path(paths.workspace_dir).unlink()
    Path(paths.dataset_dir).mkdir()
    with pytest.raises(FileExistsError, match="force_overwrite"):
        paths.validate_preconditions(force_overwrite=False)
    paths.validate_preconditions(force_overwrite=True)


def test_lock_cannot_be_inside_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "dataset"))
    with pytest.raises(ValueError, match="must be outside"):
        DatasetPackagingPaths.resolve(tmp_path / "dataset")


def test_locked_identity_rechecks_canonical_target_and_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = DatasetPackagingPaths.resolve(tmp_path / "dataset")

    paths.validate_locked_target_identity()

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path.parent / "other-cache"))
    with pytest.raises(RuntimeError, match="identity changed"):
        paths.validate_locked_target_identity()
