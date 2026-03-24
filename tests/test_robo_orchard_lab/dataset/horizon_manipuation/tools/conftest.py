# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import pytest

import robo_orchard_lab.dataset.horizon_manipulation.tools.app as app_module


@pytest.fixture(autouse=True)
def env_configured(monkeypatch):
    """Bypass the /setup redirect so tests run without a .env file."""
    monkeypatch.setattr(app_module, "ENV_CONFIGURED", True)
