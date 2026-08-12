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
import subprocess
import sys
import tempfile

import pytest


def test_resnet50_imagenet(PROJECT_ROOT: str):
    """Test ResNet50 training on ImageNet dataset."""
    example_file_path = os.path.join(
        PROJECT_ROOT, "examples", "resnet50_imagenet", "train.py"
    )

    assert os.path.exists(example_file_path), (
        f"File not found: {example_file_path}"
    )
    example_file_path = os.path.relpath(example_file_path, os.getcwd())

    # Run the training script with the specified parameters
    with tempfile.TemporaryDirectory() as workspace_root:
        ret_code = subprocess.check_call(
            [
                sys.executable,
                example_file_path,
                "--dataset.pipeline_test",
                "True",
                "--dataset.dummy_train_imgs",
                "8192",
                "--max_epoch",
                "2",
                "--workspace_root",
                workspace_root,
            ],
        )

        # Check if the script ran successfully
        assert ret_code == 0, f"Script failed with return code: {ret_code}"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
