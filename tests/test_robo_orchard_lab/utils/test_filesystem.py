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

"""Compatibility tests for shared Lab filesystem primitives."""

from __future__ import annotations
from pathlib import Path

import pytest

from robo_orchard_lab.utils.filesystem import remove_path, rename_noreplace


def test_remove_path_handles_file_directory_and_symlink(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("value", encoding="utf-8")
    directory_path = tmp_path / "directory"
    directory_path.mkdir()
    (directory_path / "child").write_text("value", encoding="utf-8")
    link_path = tmp_path / "link"
    link_path.symlink_to(directory_path, target_is_directory=True)

    remove_path(file_path)
    remove_path(link_path)
    remove_path(directory_path)

    assert not file_path.exists()
    assert not link_path.exists()
    assert not directory_path.exists()


def test_rename_noreplace_preserves_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("new", encoding="utf-8")
    target = tmp_path / "target"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        rename_noreplace(source, target)

    assert source.read_text(encoding="utf-8") == "new"
    assert target.read_text(encoding="utf-8") == "old"
