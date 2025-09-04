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
import os
import tempfile

import pytest


@pytest.fixture()
def PROJECT_ROOT() -> str:
    """Fixture to provide the project root directory."""

    # Get the absolute path of the current file
    current_file = os.path.abspath(__file__)

    # Traverse up the directory tree to find the project root
    project_root = os.path.dirname(current_file)
    for _ in range(2):
        project_root = os.path.dirname(project_root)

    return project_root


@pytest.fixture(scope="module")
def ROBO_ORCHARD_TEST_WORKSPACE() -> str:
    return os.environ["ROBO_ORCHARD_TEST_WORKSPACE"]


@pytest.fixture(scope="session")
def tmp_local_folder():
    with tempfile.TemporaryDirectory(dir=os.path.abspath("./")) as temp_dir:
        yield temp_dir
