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

import sys
from pathlib import Path

import numpy as np

from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationInput,
)
from robo_orchard_lab.utils.build import build


def test_robocasa_holobrain_config_supports_validation_and_deploy_modes():
    repo_root = Path(__file__).resolve().parents[3]
    config_dir = repo_root / "projects/holobrain_internal/common/configs"
    sys.path.insert(0, str(config_dir))
    try:
        from data_configs.config_robocasa_dataset import (
            build_processors,
            build_transforms,
        )
    finally:
        sys.path.remove(str(config_dir))

    config = {
        "hist_steps": 2,
        "pred_steps": 4,
        "with_depth": False,
        "dst_wh": (16, 12),
    }

    training_transforms = build_transforms(config, "training", cam_names=None)
    validation_transforms = build_transforms(
        config, "validation", cam_names=None
    )
    deploy_transforms = build_transforms(config, "deploy", cam_names=None)

    assert "state_loss_weights" in training_transforms[-1]["keys"]
    assert "fk_loss_weight" in training_transforms[-1]["keys"]
    assert "uuid" in validation_transforms[-1]["keys"]
    assert "pred_mask" in validation_transforms[-1]["keys"]
    assert "uuid" not in deploy_transforms[-2]["keys"]
    assert "pred_mask" not in deploy_transforms[-2]["keys"]
    assert deploy_transforms[-1]["type"].__name__ == "UnsqueezeBatch"

    for transforms in (
        training_transforms,
        validation_transforms,
        deploy_transforms,
    ):
        assert all(callable(build(transform)) for transform in transforms)

    processor = build_processors(config, dataset_name="robocasa")
    expected_transform_names = [
        type(build(transform)).__name__ for transform in deploy_transforms
    ]
    assert [
        type(transform).__name__ for transform in processor.transforms
    ] == expected_transform_names

    cam_names = ["robot0_agentview_left", "robot0_eye_in_hand"]
    sample = MultiArmManipulationInput(
        intrinsic={cam_name: np.eye(4) for cam_name in cam_names},
        t_base2cam={cam_name: np.eye(4) for cam_name in cam_names},
        history_joint_state=[
            np.zeros((1, 8), dtype=np.float32),
            np.zeros((1, 8), dtype=np.float32),
        ],
        image={
            cam_name: [np.zeros((8, 10, 3), dtype=np.uint8)]
            for cam_name in cam_names
        },
        instruction="Open the oven door.",
    )

    input_data = processor.struction_to_dict(sample)
    assert "T_base2cam" in input_data
    assert "T_world2cam" not in input_data

    batch = processor.pre_process(sample)

    assert batch["imgs"].shape == (1, 2, 12, 16, 3)
    assert batch["hist_robot_state"].shape == (1, 2, 1, 8)
    assert batch["pred_robot_state"].shape == (1, 4, 1, 8)
    assert batch["projection_mat"].shape == (1, 2, 4, 4)
    assert batch["embodiedment_mat"].shape == (1, 4, 4)
    assert batch["text"] == ["Open the oven door."]
