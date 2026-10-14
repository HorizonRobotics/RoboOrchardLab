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

import importlib

from projects.holobrain_0_5.configs import (
    config_holobrain_0_5,
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


def test_every_trainable_setting_is_deployable():
    """Training suites may share a body; each body needs one processor.

    The four LIBERO suites differ only by ``data_paths``, so they collapse to a
    single deployed processor. What must hold is that no trainable
    (type, setting) pair is left without one.
    """
    trainable = {
        (spec["dataset_type"], spec["setting_type"])
        for spec in dataset_specs.training_datasets
    }
    deployable = {
        (spec["dataset_type"], spec["setting_type"])
        for spec in deploy_specs.deploy_datasets
    }
    assert trainable == deployable


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
