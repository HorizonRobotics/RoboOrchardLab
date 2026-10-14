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


"""Coverage for pretrained directory resolution and its export contract."""

import pytest

from projects.holobrain_0_5.configs import config_holobrain_0_5
from robo_orchard_lab.models.holobrain_0_5.resources import (
    CKPT_BASE_ENV,
    DEFAULT_CKPT_BASE,
    pretrained_export_path,
    pretrained_source,
)

__all__ = []


def test_bare_name_uses_env_base(tmp_path, monkeypatch):
    base = tmp_path / "base"
    (base / "Qwen3.5-2B").mkdir(parents=True)
    monkeypatch.setenv(CKPT_BASE_ENV, str(base))
    monkeypatch.chdir(tmp_path)

    resolved = pretrained_source("Qwen3.5-2B")

    assert resolved == str(base / "Qwen3.5-2B")


def test_bare_name_defaults_to_local_ckpt(tmp_path, monkeypatch):
    (tmp_path / "ckpt" / "Qwen3.5-2B").mkdir(parents=True)
    monkeypatch.delenv(CKPT_BASE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = pretrained_source("Qwen3.5-2B")

    assert resolved == str(tmp_path / "ckpt" / "Qwen3.5-2B")


def test_exported_reference_ignores_env_base(tmp_path, monkeypatch):
    """The exported ``ckpt/<name>`` form must stay relative to the cwd.

    Model loading enters the export directory via ``in_cwd``, so redirecting a
    separator-containing path to the base would load foreign weights.
    """
    export = tmp_path / "export"
    (export / "ckpt" / "Qwen3.5-2B").mkdir(parents=True)
    base = tmp_path / "base"
    (base / "ckpt" / "Qwen3.5-2B").mkdir(parents=True)
    monkeypatch.setenv(CKPT_BASE_ENV, str(base))
    monkeypatch.chdir(export)

    resolved = pretrained_source("ckpt/Qwen3.5-2B")

    assert resolved == str(export / "ckpt" / "Qwen3.5-2B")


def test_explicit_paths_are_verbatim(tmp_path, monkeypatch):
    absolute = tmp_path / "elsewhere" / "Qwen3.5-2B"
    absolute.mkdir(parents=True)
    (tmp_path / "ckpt" / "Qwen3.5-2B").mkdir(parents=True)
    unrelated = tmp_path / "base"
    (unrelated / "Qwen3.5-2B").mkdir(parents=True)
    monkeypatch.setenv(CKPT_BASE_ENV, str(unrelated))
    monkeypatch.chdir(tmp_path)

    assert pretrained_source(str(absolute)) == str(absolute)
    assert pretrained_source("./ckpt/Qwen3.5-2B") == str(
        tmp_path / "ckpt" / "Qwen3.5-2B"
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    assert pretrained_source("~/ckpt/Qwen3.5-2B") == str(
        tmp_path / "ckpt" / "Qwen3.5-2B"
    )


def test_resolution_is_idempotent(tmp_path, monkeypatch):
    """``structure.py`` resolves the value twice before building the VLM."""
    base = tmp_path / "base"
    (base / "Qwen3.5-2B").mkdir(parents=True)
    (tmp_path / "ckpt" / "Qwen3.5-2B").mkdir(parents=True)
    monkeypatch.setenv(CKPT_BASE_ENV, str(base))
    monkeypatch.chdir(tmp_path)

    for source in ("Qwen3.5-2B", "ckpt/Qwen3.5-2B", str(base / "Qwen3.5-2B")):
        once = pretrained_source(source)
        assert pretrained_source(once) == once


def test_error_lists_candidates_and_env_var(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setenv(CKPT_BASE_ENV, str(base))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        pretrained_source("Qwen3.5-2B")

    message = str(excinfo.value)
    assert str(base / "Qwen3.5-2B") in message
    assert CKPT_BASE_ENV in message


def test_config_default_matches_legacy_path(tmp_path, monkeypatch):
    """The shipped default must still resolve to ``./ckpt/Qwen3.5-2B``."""
    source = config_holobrain_0_5.config["vlm_pretrain"]
    legacy = tmp_path / DEFAULT_CKPT_BASE / source
    legacy.mkdir(parents=True)
    monkeypatch.delenv(CKPT_BASE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = pretrained_source(source)

    assert resolved == str(legacy)
    assert pretrained_export_path(resolved) == f"ckpt/{source}"
