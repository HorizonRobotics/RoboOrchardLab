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


"""Coverage for the deployment processor catalog."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from projects.holobrain_0_5.configs import (
    config_holobrain_0_5,
    dataset_factory,
    dataset_specs,
    deploy_specs,
)

__all__ = []


def test_config_points_at_the_deploy_catalog():
    name = config_holobrain_0_5.config["deploy_specs"]
    assert importlib.import_module(name) is deploy_specs


def test_deploy_entries_carry_no_data_paths():
    """A processor needs an embodiment, not a dataset location."""
    for spec in deploy_specs.deploy_datasets:
        assert set(spec) == {"dataset_type", "dataset_name", "setting_type"}


def test_default_deploy_catalog_preserves_existing_settings():
    """Default exports do not acquire Panda's additional URDF dependency.

    Panda's processor is selected by explicit per-run overrides. Existing
    settings remain in the default catalog.
    """
    trainable = {
        (spec["dataset_type"], spec["setting_type"])
        for spec in dataset_specs.training_datasets
    }
    deployable = {
        (spec["dataset_type"], spec["setting_type"])
        for spec in deploy_specs.deploy_datasets
    }
    assert deployable == {
        ("robotwin", "aloha_v2"),
        ("robotwin", "ur5_wsg"),
        ("libero", "libero"),
    }
    assert trainable - deployable == {("isaac", "panda_droid")}
    assert deployable <= trainable


def test_one_processor_per_deployed_setting():
    """A duplicate setting would export the same processor twice."""
    settings = [
        (spec["dataset_type"], spec["setting_type"])
        for spec in deploy_specs.deploy_datasets
    ]
    assert len(settings) == len(set(settings))


def test_libero_deploys_a_single_processor():
    libero = [
        spec
        for spec in deploy_specs.deploy_datasets
        if spec["dataset_type"] == "libero"
    ]
    assert [spec["dataset_name"] for spec in libero] == ["libero"]


def test_inline_panda_exports_do_not_construct_other_bodies(monkeypatch):
    """Fake only processor assembly, without model/URDF dependencies."""
    spec = dict(
        dataset_type="isaac",
        dataset_name="panda_droid_rigid",
        setting_type="panda_droid",
    )
    config = dict(config_holobrain_0_5.config, deploy_datasets=[spec])
    original = copy.deepcopy(config)
    calls = []
    processor = object()

    def builder(data_type):
        assert data_type == "isaac"

        def build(settings, name, setting):
            calls.append((name, setting))
            return processor

        return SimpleNamespace(build_processors=build)

    monkeypatch.setattr(dataset_factory, "_builder", builder)
    # The inline path should not import even an invalid default catalog.
    config["deploy_specs"] = "unused.missing.module"
    assert dataset_factory.build_processors(config) == {
        "panda_droid_rigid": processor
    }
    assert calls == [("panda_droid_rigid", "panda_droid")]
    assert config["deploy_datasets"] == original["deploy_datasets"]
    assert "deploy_datasets" not in config_holobrain_0_5.config


def test_default_processor_dispatch_is_unchanged(monkeypatch):
    calls = []

    def builder(data_type):
        def build(config, name, setting):
            calls.append((data_type, name, setting))
            return name

        return SimpleNamespace(build_processors=build)

    monkeypatch.setattr(dataset_factory, "_builder", builder)
    output = dataset_factory.build_processors(config_holobrain_0_5.config)
    assert calls == [
        (spec["dataset_type"], spec["dataset_name"], spec["setting_type"])
        for spec in deploy_specs.deploy_datasets
    ]
    assert list(output) == [
        spec["dataset_name"] for spec in deploy_specs.deploy_datasets
    ]


def test_empty_inline_catalog_disables_exports():
    assert dataset_factory.build_processors({"deploy_datasets": []}) == {}


@pytest.mark.parametrize("specs", [None, "panda_droid_rigid", {}])
def test_invalid_inline_catalog_fails(specs):
    with pytest.raises(TypeError, match="deploy_datasets"):
        dataset_factory.build_processors({"deploy_datasets": specs})


def test_smoke_overrides_select_only_panda_without_mutating_defaults():
    defaults = copy.deepcopy(config_holobrain_0_5.config)
    path = Path(config_holobrain_0_5.__file__).with_name(
        "panda_droid_smoke.json"
    )
    overrides = json.loads(path.read_text())
    config = dict(defaults, **overrides)
    assert config["training_datasets"] == ["panda_droid_rigid"]
    assert config["deploy_datasets"] == [
        dict(
            dataset_type="isaac",
            dataset_name="panda_droid_rigid",
            setting_type="panda_droid",
        )
    ]
    (selection,) = dataset_factory._specs(config, "training")
    assert selection["dataset_type"] == "isaac"
    assert selection["setting_type"] == "panda_droid"
    assert config["batch_size"] == 1 and config["max_step"] == 20
    assert config_holobrain_0_5.config == defaults
