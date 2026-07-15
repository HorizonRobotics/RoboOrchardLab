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


import json
import os
import subprocess
import tempfile

import pytest


def test_sem_robotwin(PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str):
    """Test SEM training on RoboTwin dataset."""
    project_dir = os.path.join(PROJECT_ROOT, "projects/sem/robotwin")
    assert os.path.exists(project_dir), f"File not found: {project_dir}"

    project_test_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/robo_orchard_lab_projects_ut/sem_robotwin",
    )
    kwargs = dict(
        checkpoint=None,
        bert_checkpoint=os.path.join(project_test_path, "bert-base-uncased"),
        data_path=os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/robotwin/main_branch/lmdb",
        ),
        urdf=os.path.join(
            project_test_path, "urdf/arx5_description_isaac.urdf"
        ),
        step_log_freq=1,
        max_step=3,
        # Multiprocessing cleanup is covered separately from this training
        # smoke.
        num_workers=0,
        lr=0.0,
        task_names=["shoe_place"],
    )
    kwargs = json.dumps(kwargs)
    os.chdir(project_dir)
    with tempfile.TemporaryDirectory() as workspace_root:
        cmd = (
            " ".join(
                [
                    "python3",
                    "train.py",
                    f"--workspace {workspace_root}",
                    "--config config_sem_robotwin.py",
                    f"--kwargs '{kwargs}'",
                ]
            ),
        )
        ret_code = subprocess.check_call(cmd, shell=True)

        # Check if the script ran successfully
        assert ret_code == 0, f"Script failed with return code: {ret_code}"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
