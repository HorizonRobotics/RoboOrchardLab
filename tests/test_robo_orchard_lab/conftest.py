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
import logging
import os
import tempfile
import warnings

import pytest

# import multiprocessing as mp
# try:
#     # Prefer 'fork' to avoid spawn/forkserver pickling/fd issues in tests
#     mp.set_start_method("fork", force=True)
# except Exception:
#     pass

try:
    # Use filesystem-backed sharing to avoid resizing fds on some filesystems
    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass


warnings.filterwarnings(
    "ignore",
    message=".*register_feature.*experimental.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=(
        "Failed to load dataset using `datasets.load_from_disk`\\. "
        "Falling back to use wrapped version\\."
    ),
    category=UserWarning,
)
logging.getLogger("curobo").setLevel(logging.ERROR)


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


@pytest.fixture(autouse=True)
def env_vars(tmp_local_folder):
    os.environ["ROBO_ORCHARD_HOME"] = os.path.abspath(tmp_local_folder)
    yield
