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

"""Focused tests for the projects.holobrain.policy package-root lazy import."""

import sys
import types

import pytest


def _reset_policy_package(monkeypatch):
    import projects.holobrain.policy as pol

    monkeypatch.delitem(
        sys.modules,
        "projects.holobrain.policy.robotwin_policy",
        raising=False,
    )
    monkeypatch.delitem(pol.__dict__, "HoloBrainRoboTwinPolicy", raising=False)
    monkeypatch.delitem(
        pol.__dict__,
        "HoloBrainRoboTwinPolicyCfg",
        raising=False,
    )
    return pol


class TestPolicyPackageImportCompat:
    """Test that the historical package-root import path still works."""

    def test_package_root_import_lazy_bridge(self, monkeypatch):
        pol = _reset_policy_package(monkeypatch)

        assert "HoloBrainRoboTwinPolicy" in pol.__all__
        assert "HoloBrainRoboTwinPolicyCfg" in pol.__all__
        assert (
            "projects.holobrain.policy.robotwin_policy" not in sys.modules
        ), "Package import should not eagerly load robotwin_policy"

    def test_direct_package_root_import_resolves_with_lazy_bridge(
        self,
        monkeypatch,
    ):
        _reset_policy_package(monkeypatch)

        class _FakePolicy:
            pass

        class _FakePolicyCfg:
            pass

        fake_module = types.ModuleType(
            "projects.holobrain.policy.robotwin_policy"
        )
        fake_module.HoloBrainRoboTwinPolicy = _FakePolicy
        fake_module.HoloBrainRoboTwinPolicyCfg = _FakePolicyCfg
        monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

        from projects.holobrain.policy import (
            HoloBrainRoboTwinPolicy,
            HoloBrainRoboTwinPolicyCfg,
        )

        assert HoloBrainRoboTwinPolicy is _FakePolicy
        assert HoloBrainRoboTwinPolicyCfg is _FakePolicyCfg

    def test_unknown_attr_raises(self):
        import projects.holobrain.policy as pol

        with pytest.raises(AttributeError):
            _ = pol.NonExistentSymbol
