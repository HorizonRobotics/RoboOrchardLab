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


def test_bip3d_grounding(PROJECT_ROOT: str, ROBO_ORCHARD_TEST_WORKSPACE: str):
    """Test BIP3D training on EmbodiedScan dataset."""
    project_dir = os.path.join(PROJECT_ROOT, "projects/bip3d_grounding")
    assert os.path.exists(project_dir), f"File not found: {project_dir}"

    project_test_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/robo_orchard_lab_projects_ut/bip3d_grounding",
    )
    kwargs = dict(
        checkpoint=os.path.join(
            project_test_path,
            "groundingdino_swint_ogc_mmdet-822d7e9d-rename.pth",
        ),
        bert_checkpoint=os.path.join(project_test_path, "bert-base-uncased"),
        anchor_file=os.path.join(
            project_test_path, "embodiedscan_kmeans_det_cam_log_z-0.2-3.npy"
        ),
        data_root=os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/embodiedscan",
        ),
        step_log_freq=1,
        max_num_views=12,
        val_max_num_views=18,
        max_epoch=3,
        epoch_eval_freq=3,
        lr=0.0,
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
                    "--config config_bip3d_det.py",
                    f"--kwargs '{kwargs}'",
                ]
            ),
        )
        ret_code = subprocess.check_call(cmd, shell=True)

        # Check if the script ran successfully
        assert ret_code == 0, f"Script failed with return code: {ret_code}"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
